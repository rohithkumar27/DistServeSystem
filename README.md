# DistServeSystem (Course Reimplementation)

This repo is our **reimplementation of the DistServe idea** under course/HPC constraints:

- **No Ray** (we will not depend on Ray actors/placement groups)
- **No SwiftTransformer** (we will not depend on the upstream C++ inference backend)

Instead, we reimplement the **architecture and control-plane concepts** from upstream DistServe using a lightweight
Python-first design that can run on **1 node / 2–4 GPUs** and still support the **same evaluation story**:
tail latency, TTFT, goodput under SLOs, and the overhead of KV-cache handoff.

## What we are reimplementing (from upstream DistServe)

Upstream DistServe’s core claim is architectural:

- LLM inference has two phases:
  - **Prefill**: process full prompt, produce KV cache
  - **Decode**: autoregressively generate tokens using KV cache
- Colocating them in one worker pool causes **prefill–decode interference**, hurting tail latency and SLO goodput.
- Disaggregating them into **separate worker pools** allows independent batching/scheduling and better goodput.

This repo focuses on reproducing that idea and its measurable effects, not the exact upstream execution backend.

## High-level architecture (no Ray, no SwiftTransformer)

We implement two serving “systems”:

1. **Baseline (monolithic / colocated)**:
   - One worker pool does prefill + decode.
   - One shared queue; batching mixes prefill and decode work.
2. **Disaggregated (DistServe-style)**:
   - **Prefill pool** produces KV caches.
   - **Decode pool** consumes KV caches and generates tokens.
   - A **KV handoff layer** transfers KV (or a KV-sized proxy) from prefill to decode.
   - Separate batching/scheduling knobs per phase + admission control for SLOs.

Two evaluation modes:

| Mode | Where | What it uses |
|------|--------|----------------|
| **Simulator** | `src/simulator/baseline.py`, `disaggregated.py` | `TimingModel` only (defaults or fitted from profiling). **No GPU** in these files. |
| **Real GPU** | `src/runtime/baseline_gpu.py`, `disaggregated_gpu.py` | HuggingFace `forward` on CUDA: prefill → decode with KV cache; disaggregated uses **two model copies** and **moves `past_key_values`** between devices (DistServe-style separation). |

## Mapping to upstream DistServe repo (concept-to-module)

This mapping is intentionally “idea-level” because we are not porting upstream’s Ray/SwiftTransformer code.

- **Upstream “worker pool” / “cluster”** → `src/simulator/{baseline,disaggregated}.py` (queueing + `TimingModel`) or `src/runtime/{baseline_gpu,disaggregated_gpu}.py` (real PyTorch forwards)
- **Upstream scheduling knobs (separate for prefill/decode)** → our per-phase batching + queueing parameters
- **Upstream KV communication/memory management** → our KV handoff model (simulated now; measured IPC later)
- **Upstream evaluation scripts** → our `src/experiments/` scripts + plots (added as needed)

If you want upstream’s simulation methodology for inspiration, look at `SYSFORDL/DistServe/simdistserve/`:
it contains estimators and ablation scripts that are conceptually aligned with what we will do here.

## Repository structure

- `src/core/`
  - `request.py`: request representation (prompt length, gen length, timestamps)
  - `metrics.py`: TTFT/TPOT/latency distributions, throughput, goodput/SLO accounting
- `src/simulator/`
  - `workload.py`: mixed workload generator (prompt/gen distributions, arrival processes)
  - `timing.py`: service-time models (phase-dependent timing + batching effects)
  - `baseline.py`: colocated prefill+decode simulator
  - `disaggregated.py`: disaggregated simulator (prefill queue + decode queue + KV handoff cost)
- `src/experiments/`
  - `run_comparison.py`: simulator baseline vs disaggregated (`TimingModel` / optional fitted JSON)
  - `run_gpu_comparison.py`: **real GPU** baseline vs disaggregated (ShareGPT prompts; 2 GPUs for disaggregated, or `--baseline-only` on 1 GPU)
- `src/runtime/` (real inference — **not** simulated)
  - `inference_core.py`: timed prefill + decode steps; KV `past_key_values` move between devices
  - `baseline_gpu.py`: one GPU, FIFO prefill then decode (colocated)
  - `disaggregated_gpu.py`: prefill model on GPU A, copy KV to GPU B, decode model on B
- `src/stage_b/` (Stage B — measured timing + ShareGPT)
  - `sharegpt_loader.py`: load ShareGPT-style conversations from Hugging Face or local JSONL
  - `profile_sharegpt.py`: GPU timing of prefill + decode steps, fit `TimingModel`, write `fitted_timing.json`
  - `fitted_timing.py`: save/load fitted timing; linear fit for prefill vs prompt length
  - `workload_sharegpt.py`: build `Request` list with real tokenizer lengths from ShareGPT text
- `tests/`
  - sanity tests for workload, timing, and simulator invariants

## Recommended first steps (to verify the scaffold)

1. Create a virtual environment.
2. Install the requirements from `requirements.txt`.
3. Run the comparison script:

```bash
python -m src.experiments.run_comparison
```

You should see two printed summaries (baseline vs disaggregated) and a new `results/` folder containing:

- `*-colocated-requests.csv`
- `*-disaggregated-requests.csv`
- `*-summary.json`

## Milestone experiments (recommended commands)

Run a small suite to generate evidence artifacts for the milestone report:

```bash
# 1) Baseline vs disaggregated comparison + per-request CSVs
python -m src.experiments.run_comparison

# 2) Ablations (KV transfer cost sweep + capacity split sweep)
python -m src.experiments.run_ablations
```

The ablations script writes `results/*-ablations.csv` (table form, easy to plot in a notebook).

### Stage B — Real GPU measurements + ShareGPT (implemented)

1. **Install** Stage B dependencies (GPU machine recommended):

```bash
pip install -r requirements.txt
```

2. **Profile** an open-weight causal LM on ShareGPT prompts (downloads model + dataset on first run):

```bash
# Default: TinyLlama on HF ShareGPT Vicuna unfiltered (first N rows)
python -m src.stage_b.profile_sharegpt \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-samples 80 \
  --max-new-tokens 32 \
  --out results/fitted_timing.json
```

This writes `results/fitted_timing.json` with:

- `meta`: model id, dataset, sample counts, device
- `timing`: fitted `TimingModel` fields (`prefill_base` + `prefill_per_token` from least squares; `decode_base` = mean decode step time; `decode_per_token` = 0)

3. **Run the simulator** using **ShareGPT-derived token counts** and **measured** timing:

```bash
python -m src.experiments.run_comparison \
  --workload sharegpt \
  --timing-json results/fitted_timing.json \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-requests 300 \
  --arrival-rate 2.0
```

**What is measured vs simulated:** profiling measures **wall-clock prefill forward** and **per-step decode forward** on GPU. The queueing simulator still combines these into end-to-end latency under load (same as upstream `simdistserve` style: measured kernels + system-level scheduling).

**Local ShareGPT JSONL:** pass `--jsonl /path/to/sharegpt.jsonl` to `profile_sharegpt.py` (and use a matching workflow if you host data offline).

4. **Optional — no `TimingModel` at all:** run **actual** colocated vs disaggregated forwards (same HF model weights on 1 vs 2 GPUs):

```bash
# Needs 2 GPUs for disaggregated (prefill GPU + decode GPU). Use --baseline-only on a 1-GPU node.
python -m src.experiments.run_gpu_comparison \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-requests 15 \
  --prefill-gpu 0 --decode-gpu 1
```

This uses `src/runtime/baseline_gpu.py` (one GPU, prefill+decode) and `src/runtime/disaggregated_gpu.py` (prefill on GPU 0, **KV tensor copy** to GPU 1, decode on GPU 1). `src/simulator/baseline.py` and `disaggregated.py` are **not** used here.

## Working process (what to implement, in order)

This order keeps the project “always runnable” and makes experiments reproducible.

### Stage A — Simulator correctness and evaluation parity (fast iteration)

- Ensure both systems produce:
  - **TTFT** (time to first token)
  - **TPOT** (time per output token) or equivalent decode rate
  - **p50/p95/p99** end-to-end latency
  - **goodput** under explicit SLOs (e.g., TTFT SLO, end-to-end SLO)
- Add workload mixes that stress the DistServe claim:
  - long prompts + short interactive prompts
  - bursty arrivals
  - varied generation lengths
- Add ablations:
  - GPU split (modeled now as “capacity split”)
  - per-phase batch sizes
  - KV handoff overhead sweep

### Stage B (status) — Measured prefill/decode → fitted `TimingModel`

Implemented via `src/stage_b/profile_sharegpt.py` + `--timing-json` on `run_comparison.py`.

Optional next step for KV realism:

- **Two-process / two-GPU prototype**: run prefill on GPU0, decode on GPU1, and measure KV handoff overhead via IPC, then set `transfer_per_prompt_token` in the fitted JSON.

### Stage C — Minimal runtime prototype (optional, but closest to the paper)

Goal: preserve the DistServe shape with a runnable “online” system.

- A tiny API layer that accepts requests and routes them through:
  - baseline queue, or
  - prefill queue → KV handoff → decode queue
- Admission control policies to hit SLOs under load

This is still feasible **without Ray** by using Python multiprocessing + explicit GPU binding.

## What “done” looks like for this course project

- A runnable comparison script + saved results showing:
  - tail latency reduction and/or goodput improvement under mixed workloads
  - measured (or carefully modeled) KV handoff overhead
  - ablations over GPU split and batching knobs
- A short writeup connecting these results back to the upstream DistServe claim.
