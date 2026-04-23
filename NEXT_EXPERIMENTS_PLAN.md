# Next Experiments Plan — DistServeSystem

## Context

The current system benchmarks colocated vs disaggregated inference on same-node GPUs. The next experiments expand in two directions:

1. **Comprehensive sweeps** — cross batch size × workload dimensions to expose where disaggregation helps most
2. **Cross-node KV transfer** — simulate real-world deployment where prefill GPU and decode GPU are on separate physical machines (requires NCCL-based transfer instead of `.to(cuda:N)`)

---

## Track 1 — Comprehensive Workload × Batch Sweep

### What's already done
- `run_batch_sweep_gpu.py` sweeps `batch_sizes=[1,2,4,8]` at a fixed workload
- `run_gpu_comparison.py` accepts `--arrival-rate`, `--num-requests`, `--output-low`, `--output-high` individually

### What's missing
A sweep that crosses **batch_size × arrival_rate × output_length** on the real GPU path. No such script exists.

### New file: `src/experiments/run_workload_sweep_gpu.py`

**Sweep dimensions:**

| Axis | Values |
|---|---|
| `batch_size` | 1, 4, 8 |
| `arrival_rate` | 0.5, 1.0, 2.0, 4.0 (req/s) |
| `output_high` | 32, 64, 128 (tokens; `output_low` stays 8) |

Total: 3 × 4 × 3 = **36 conditions**, run for both colocated and disaggregated → **72 runs**.

**Design:**
- Load model once at startup (shared across all batch sizes for that GPU pair)
- For each `(batch_size, arrival_rate, output_high)` combination:
  - Rebuild ShareGPT requests with those parameters
  - Run `run_baseline_gpu` and `run_disaggregated_gpu`
  - Collect metrics
- Write all results to `results/milestone/workload_sweep_gpu.csv`

**CSV columns:**
```
batch_size, arrival_rate, output_high, design,
goodput, mean_ttft, p95_ttft, p99_ttft,
mean_tpot, p95_tpot, mean_e2e, throughput_req_s
```

**Reuses:**
- `run_baseline_gpu` from `src/runtime/baseline_gpu.py`
- `run_disaggregated_gpu` from `src/runtime/disaggregated_gpu.py`
- `build_requests_from_sharegpt` from `src/stage_b/workload_sharegpt.py`
- `summarize_results` + `SLOConfig` from `src/core/metrics.py`
- `load_model_and_tokenizer` / `load_two_models` from the two runtime files
- `write_csv`, `write_json` from `src/experiments/io_utils.py`

**CLI flags (mirrors run_batch_sweep_gpu.py):**
```
--model, --prefill-gpu, --decode-gpu, --num-requests,
--max-prompt-tokens, --output-low, --arrival-rates (comma-separated),
--batch-sizes (comma-separated), --output-highs (comma-separated),
--ttft-slo, --tpot-slo, --seed, --sharegpt-jsonl
```

---

## Track 2 — Cross-Node KV Transfer (NCCL)

### Architecture change

Current: one Python process, two GPUs on the same node, `.to(cuda:N)` transfer.

Target: two Python processes on two machines, each with one GPU. Prefill node (rank 0) runs prefill, then sends KV tensors over NCCL. Decode node (rank 1) receives KV tensors, reconstructs DynamicCache, then decodes.

### Changes required

#### A. `src/runtime/inference_core.py` — add `time_transfer_nccl`

New function alongside existing `time_transfer` (which stays untouched for same-node path):

```python
def time_transfer_nccl(past, device, dist_group, src_rank, dst_rank, is_sender):
```

- Extracts layer K/V tensors from the DynamicCache (reuses `_extract_kv_rows_from_past`)
- If `is_sender` (prefill node, rank 0):
  - For each layer's K tensor: `dist.send(tensor.contiguous(), dst=dst_rank, group=dist_group)`
  - For each layer's V tensor: `dist.send(tensor.contiguous(), dst=dst_rank, group=dist_group)`
- If receiver (decode node, rank 1):
  - Pre-allocate tensors of matching shape (shapes must be communicated first via a small metadata send)
  - For each layer's K: `dist.recv(tensor, src=src_rank, group=dist_group)`
  - For each layer's V: `dist.recv(tensor, src=src_rank, group=dist_group)`
  - Reconstruct `DynamicCache` from received tensors (reuses `prepare_past_for_decode` logic)
- Returns `(transfer_seconds, reconstructed_past_or_None)`
- Timing is measured on the **receiver side** (decode node) since network latency should be captured at the receiving end

Shape metadata exchange: send a small int tensor `[num_layers, batch, heads, seq, dim]` via `dist.broadcast` before the KV tensors.

#### B. `src/runtime/disaggregated_gpu_distributed.py` — new file

Two entry-point functions, one per rank:

**`run_prefill_node(rank, world_size, args, dist_group)`:**
- Loads only model_A on `cuda:0` (the prefill GPU on this machine)
- Loads tokenizer
- Receives the request list from rank 1 (or reads from a shared file / CLI-specified workload file)
- For each batch: tokenize → prefill → `time_transfer_nccl(..., is_sender=True)` → send timing back to rank 1
- Sends per-batch `(prefill_s, transfer_s, first_token_times)` to rank 1 via `dist.send`

**`run_decode_node(rank, world_size, args, dist_group)`:**
- Loads only model_B on `cuda:0` (the decode GPU on this machine)
- Loads tokenizer
- Holds the request list (either via shared workload file or generated locally with same seed)
- For each batch: `time_transfer_nccl(..., is_sender=False)` → decode → record `RequestResult`
- Collects all results and writes CSVs/summary JSON

#### C. `src/experiments/run_distributed_gpu.py` — new launcher

```
python -m src.experiments.run_distributed_gpu \
  --rank 0 \            # or 1
  --master-addr HOST \  # prefill node IP
  --master-port 29500 \
  --model TinyLlama/... \
  --num-requests 30 \
  --batch-size 4 \
  ...same workload flags as run_gpu_comparison.py...
```

- Calls `torch.distributed.init_process_group(backend='nccl', init_method='tcp://MASTER_ADDR:PORT', rank=rank, world_size=2)`
- Dispatches to `run_prefill_node` (rank 0) or `run_decode_node` (rank 1)
- Rank 1 writes results (it holds the full timing picture)

Alternatively, can be launched via `torchrun --nproc-per-node=1 --nnodes=2` on each machine.

#### D. No changes to existing files

`baseline_gpu.py`, `disaggregated_gpu.py`, `inference_core.py` (existing `time_transfer`), `run_gpu_comparison.py`, `run_batch_sweep_gpu.py` — all stay exactly as-is. Cross-node is an additive parallel path.

---

## Verification

### Track 1 — Workload sweep
```bash
python -m src.experiments.run_workload_sweep_gpu \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-requests 30 --prefill-gpu 0 --decode-gpu 1 \
  --batch-sizes 1,4,8 --arrival-rates 0.5,1.0,2.0,4.0 --output-highs 32,64,128
# Check results/milestone/workload_sweep_gpu.csv has 72 rows
```

### Track 2 — Cross-node
On machine A (prefill node):
```bash
python -m src.experiments.run_distributed_gpu --rank 0 \
  --master-addr <machine-A-IP> --master-port 29500 \
  --model TinyLlama/... --num-requests 15 --batch-size 1
```
On machine B (decode node):
```bash
python -m src.experiments.run_distributed_gpu --rank 1 \
  --master-addr <machine-A-IP> --master-port 29500 \
  --model TinyLlama/... --num-requests 15 --batch-size 1
```
Both processes rendezvous → run → rank 1 writes `results/*-distributed-summary.json`.

To validate correctness before having two nodes: run both ranks on the same machine using two different GPUs (`cuda:0` prefill, `cuda:1` decode) and compare results to `run_gpu_comparison.py` output — TTFT should be slightly higher due to NCCL overhead vs direct P2P copy.

---

## File summary

| File | Action |
|---|---|
| `src/experiments/run_workload_sweep_gpu.py` | **New** — cross-dimension sweep runner |
| `src/runtime/inference_core.py` | **Edit** — add `time_transfer_nccl` (existing functions untouched) |
| `src/runtime/disaggregated_gpu_distributed.py` | **New** — rank-split prefill/decode runners |
| `src/experiments/run_distributed_gpu.py` | **New** — NCCL launcher with `--rank` flag |
