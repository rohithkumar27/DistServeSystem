# DistServeSystem — Complete Architecture Explanation

This document walks through the entire system step by step, from how a request enters the system to how metrics are computed and reported.

---

## Table of Contents

1. [The Core Problem](#1-the-core-problem)
2. [Request Representation](#2-request-representation)
3. [Timing Model](#3-timing-model)
4. [Workload Generation](#4-workload-generation)
5. [Colocated (Baseline) Simulator](#5-colocated-baseline-simulator)
6. [Disaggregated Simulator](#6-disaggregated-simulator)
7. [Real GPU Runtime — Shared Inference Core](#7-real-gpu-runtime--shared-inference-core)
8. [Real GPU Runtime — Baseline (1 GPU)](#8-real-gpu-runtime--baseline-1-gpu)
9. [Real GPU Runtime — Disaggregated (2 GPUs)](#9-real-gpu-runtime--disaggregated-2-gpus)
10. [Stage B — Profiling and Fitting Real GPU Timings](#10-stage-b--profiling-and-fitting-real-gpu-timings)
11. [Metrics Computation](#11-metrics-computation)
12. [Experiment Runners](#12-experiment-runners)
13. [End-to-End Data Flow Diagram](#13-end-to-end-data-flow-diagram)

---

## 1. The Core Problem

LLM inference has two distinct phases:

- **Prefill**: The model processes the entire input prompt in one forward pass. This is compute-intensive and scales with prompt length.
- **Decode**: The model autoregressively generates one token at a time, reusing the KV cache from prefill. This is memory-bandwidth-intensive and sequential.

When both phases share the same GPU(s), **prefill disrupts decode**. A long prefill from one request steals GPU cycles from the decode steps of many other in-flight requests, causing spikes in tail latency and poor SLO goodput.

DistServe's solution: **separate prefill and decode into dedicated worker pools**, giving each phase its own scheduling lane and eliminating the interference.

This repo implements and evaluates both designs: colocated and disaggregated.

---

## 2. Request Representation

**File:** `src/core/request.py`

Every inference request is a `Request` object:

```
Request
  request_id        unique integer ID
  arrival_time      simulated or real wall-clock arrival (seconds)
  prompt_tokens     number of input tokens
  output_tokens     number of tokens to generate
  prompt_text       optional: actual text (used in GPU runs for tokenization)
```

After a request completes, it becomes a `RequestResult`:

```
RequestResult
  request_id
  arrival_time
  first_token_time   wall-clock time when first output token was produced
  finish_time        wall-clock time when all output tokens were produced
  output_tokens

  Properties (computed):
    ttft  = first_token_time - arrival_time      (Time To First Token)
    tpot  = (finish_time - first_token_time)     (Time Per Output Token)
              / max(output_tokens - 1, 1)
    e2e   = finish_time - arrival_time           (End-to-end latency)
```

**Step-by-step lifecycle of a request:**

1. Created by the workload generator with an arrival time and token counts.
2. Queued in the simulator or placed in a real-GPU pipeline.
3. Prefill runs: first token time is recorded.
4. Decode runs: finish time is recorded.
5. Becomes a `RequestResult` and is passed to the metrics layer.

---

## 3. Timing Model

**File:** `src/simulator/timing.py`

The simulator does not run actual GPU kernels. Instead it uses a `TimingModel` — a parametric model of how long each operation takes.

```
TimingModel fields (all in seconds):
  prefill_base              = 0.020       base overhead per batch
  prefill_per_token         = 0.00020     extra cost per input token
  decode_base               = 0.004       base overhead per decode step
  decode_per_token          = 0.0025      extra cost per token in batch
  transfer_per_prompt_token = 0.000015    KV transfer cost per prompt token
  colocated_interference    = 1.35        slowdown multiplier for colocated runs
  prefill_capacity_mult     = 1.0         models under-provisioned prefill GPU
  decode_capacity_mult      = 1.0         models under-provisioned decode GPU
  prefill_batch_alpha       = 0.20        batch efficiency exponent for prefill
  decode_batch_alpha        = 0.10        batch efficiency exponent for decode
```

**How each timing function works:**

### `prefill_time(prompt_tokens, batch_size)`

```
speedup = batch_size ^ prefill_batch_alpha
          (larger batches share overhead → sub-linear scaling)

time = (prefill_base + prefill_per_token * prompt_tokens)
       / speedup
       * prefill_capacity_multiplier
```

The batch exponent (0.20) captures that doubling the batch does not double the prefill time — GPU parallelism absorbs some of the overhead.

### `decode_step_time(batch_size)`

```
speedup = batch_size ^ decode_batch_alpha

time = (decode_base + decode_per_token)
       / speedup
       * decode_capacity_multiplier
```

Decode is memory-bandwidth bound. The exponent is smaller (0.10) because batching helps less here.

### `transfer_time(prompt_tokens)`

```
time = transfer_per_prompt_token * prompt_tokens
```

Linear in prompt length because the KV cache size scales with the number of prompt tokens.

These coefficients can be set from the defaults or replaced with values fitted from real GPU measurements (Stage B).

---

## 4. Workload Generation

**File:** `src/simulator/workload.py`

The workload generator produces a list of `Request` objects that simulate clients sending requests over time.

### Arrival process

Two modes:

**Poisson:**
```
inter-arrival gap = -log(uniform(0,1)) / arrival_rate
```
This gives exponentially distributed gaps — the standard model for independent client arrivals.

**Bursty:**
```
for each request:
  with probability burst_prob:
    use arrival_rate * burst_rate_multiplier
  else:
    use arrival_rate
  inter-arrival gap = exponential(chosen_rate)
```
This creates clusters of arrivals separated by quieter periods, stressing the system more realistically.

### Token sizes

**Simple (uniform):**
```
prompt_tokens = randint(prompt_low, prompt_high)
output_tokens = randint(output_low, output_high)
```

**Mixed (interactive + long-prompt):**
```
with probability interactive_frac:
  prompt_tokens = randint(32, 128)     # short interactive prompt
  output_tokens = randint(16, 64)      # short output
else:
  prompt_tokens = randint(256, 1024)   # long prompt
  output_tokens = randint(64, 256)     # longer output
```

The mixed workload is the critical one for demonstrating DistServe's value: long prefills from the second class interfere with the low-latency demands of the first class.

All randomness is seeded, so results are reproducible.

---

## 5. Colocated (Baseline) Simulator

**File:** `src/simulator/baseline.py`

This simulates a single GPU pool where prefill and decode share the same execution lane.

### State

```
pending    sorted list of requests by arrival_time
now        current simulated clock (seconds)
results    list of RequestResult
```

### Main loop (step by step)

**Step 1 — Advance time to next batch**

```
next_arrival = pending[0].arrival_time
now = max(now, next_arrival)

# optionally wait batch_wait_s to form a larger batch
if batch_wait_s > 0:
  now = max(now, next_arrival + batch_wait_s)
```

**Step 2 — Form a batch**

Take up to `max_batch_size` requests that have arrived by `now`:
```
batch = [r for r in pending if r.arrival_time <= now][:max_batch_size]
remove batch from pending
```

**Step 3 — Prefill the batch (all at once)**

```
# The interference multiplier slows down prefill on the colocated GPU
# because decode from other in-flight requests is competing
prefill_s = timing.prefill_time(avg_prompt_tokens, batch_size)
           * colocated_interference

prefill_done = now + prefill_s
```

**Step 4 — First decode step**

```
# Also slowed by interference
step_s = timing.decode_step_time(batch_size) * colocated_interference

first_token_time = prefill_done + step_s  # same for all requests in batch
```

**Step 5 — Remaining decode steps**

```
for each request in batch:
  remaining = request.output_tokens - 1
  finish_time = first_token_time + remaining * step_s
  record RequestResult(arrival, first_token_time, finish_time)
```

**Step 6 — Advance clock**

```
now = max(finish_time for all requests in batch)
```

Repeat until `pending` is empty.

### Key property

All requests in the same batch share the same `prefill_done` and the same `step_s`. The colocated interference multiplier (1.35) inflates both prefill and decode times globally — a simplified model of GPU resource contention.

---

## 6. Disaggregated Simulator

**File:** `src/simulator/disaggregated.py`

This simulates two separate pools: a **prefill pool** and a **decode pool**, with a KV transfer in between.

### State

```
prefill_q    pending requests sorted by arrival_time
decode_q     requests ready for decode, sorted by eligibility_time
prefill_now  clock for the prefill pool
decode_now   clock for the decode pool
results      list of RequestResult
```

### Phase 1 — Prefill Pool

**Step 1 — Advance prefill clock**

```
next_arrival = prefill_q[0].arrival_time
prefill_now = max(prefill_now, next_arrival)
```

**Step 2 — Form prefill batch** (same logic as colocated, using `prefill_max_batch`)

**Step 3 — Run prefill (no interference)**

```
prefill_s = timing.prefill_time(avg_prompt_tokens, batch_size)
# No colocated_interference multiplier — dedicated GPU
prefill_done = prefill_now + prefill_s
```

**Step 4 — Account for KV transfer and queue for decode**

```
for each request in prefill_batch:
  transfer_s = timing.transfer_time(request.prompt_tokens)
  eligible_for_decode = prefill_done + transfer_s
  add to decode_q with eligibility_time = eligible_for_decode
```

**Step 5 — Advance prefill clock**

```
prefill_now = prefill_done
```

Repeat until `prefill_q` is empty. This populates `decode_q`.

### Phase 2 — Decode Pool

Processes requests from `decode_q` in eligibility order.

**Step 1 — Advance decode clock**

```
next_eligible = decode_q[0].eligibility_time
decode_now = max(decode_now, next_eligible)
```

**Step 2 — Form decode batch**

Take up to `decode_max_batch` requests eligible by `decode_now`.

**Step 3 — Run decode steps (no interference)**

```
step_s = timing.decode_step_time(batch_size)
first_token_time = decode_now + step_s

for each request in decode_batch:
  remaining = request.output_tokens - 1
  finish_time = first_token_time + remaining * step_s
  record RequestResult(
    arrival_time = original_arrival_time,  # not the eligibility_time
    first_token_time = first_token_time,
    finish_time = finish_time
  )
```

**Step 4 — Advance decode clock**

```
decode_now = max(finish_time for requests in batch)
```

### Why this is better

- Prefill runs with no decode interference → faster prefill → lower TTFT
- Decode runs with no prefill interference → more consistent step time → lower TPOT
- The cost is the KV transfer delay, which adds to TTFT but is typically small
- Separate batch size knobs: `prefill_max_batch` and `decode_max_batch` can be tuned independently

---

## 7. Real GPU Runtime — Shared Inference Core

**File:** `src/runtime/inference_core.py`

This module provides the building blocks used by both GPU runtimes — both single-request and batched variants.

### `timed_prefill(model, input_ids, device)` — single request

```
torch.cuda.synchronize(device)       # flush any pending GPU work
t0 = time.perf_counter()

outputs = model(
  input_ids=input_ids,
  use_cache=True                     # tells HuggingFace to return past_key_values
)

torch.cuda.synchronize(device)       # wait for GPU to finish
t1 = time.perf_counter()

return (t1 - t0), outputs.past_key_values, outputs.logits[:, -1:, :]
```

Returns wall time, the KV cache, and the first output token logit position.

### `timed_decode_steps(model, past, first_token, num_tokens, device)` — single request

```
for step in range(num_tokens):
  sync → t0
  out = model(input_ids=current_token, past_key_values=p, use_cache=True)
  sync → step_times.append(t1 - t0)
  current_token = out.logits[:, -1:, :].argmax(dim=-1)
  p = out.past_key_values

return step_times, sum(step_times)
```

### `tokenize_batch(tokenizer, prompts, device, max_prompt_tokens)` — batched

```
tokenizer.padding_side = "left"    # left-pad so last real token sits at position -1
                                   # for every sequence in the batch

enc = tokenizer(prompts, padding=True, truncation=True, max_length=max_prompt_tokens)
return enc["input_ids"].to(device), enc["attention_mask"].to(device)  # [B, L] each
```

Left-padding is critical: it ensures that `logits[:, -1:, :]` picks the correct next-token prediction for every sequence regardless of prompt length.

### `timed_prefill_batch(model, input_ids, attention_mask, device)` — batched

```
sync → t0
out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
sync → prefill_s = t1 - t0

next_tokens = out.logits[:, -1:, :].argmax(dim=-1)   # [B, 1]
return prefill_s, out.past_key_values, next_tokens
```

A single forward call processes all `B` prompts in parallel. The returned `past_key_values` has shape `[B, num_heads, seq_len, head_dim]` per layer.

### `timed_decode_steps_batch(model, past, first_tokens, max_new_tokens, device)` — batched

```
next_tokens = first_tokens    # [B, 1]

for step in range(max_new_tokens):
  sync → t0
  out = model(input_ids=next_tokens, past_key_values=p, use_cache=True)
  sync → step_times.append(t1 - t0)
  next_tokens = out.logits[:, -1:, :].argmax(dim=-1)
  p = out.past_key_values

return step_times, sum(step_times)
```

All `B` requests in the batch decode together for `max_new_tokens` steps (the maximum across the batch). Each step is one GPU forward. The caller derives per-request finish times by slicing `step_times[:output_tokens]` per request.

### `time_transfer(past, target_device, source_device)`

```
sync(source)
t0 = time.perf_counter()

# Extract all layer KV tensors from DynamicCache (handles [B,...] batched shape)
# Move every tensor: tensor.to(target_device)
# Rebuild DynamicCache on target device

sync(target)
return (t1 - t0), new_past_on_target_device
```

Works for both single-request and batched KV caches since it moves tensors generically. For a batch of `B` requests, the entire batched cache is transferred in one call.

**Important constraint:** Both GPUs must be in the same process on the same node. Cross-node transfer is not supported.

---

## 8. Real GPU Runtime — Baseline (1 GPU)

**File:** `src/runtime/baseline_gpu.py`

Runs requests on a single GPU in arrival order, grouped into batches of up to `batch_size`. This is the colocated reference point for real GPU experiments.

### Setup

```
device = cuda:prefill_gpu (or cpu if no GPU)
model = load model on device (float16 or float32)
tokenizer = load tokenizer
requests sorted by arrival_time
gpu_free_time = 0.0    (when the GPU last became free)
batch_size = N         (configurable, default 1)
```

### Per-batch loop

Requests are sliced into consecutive groups of up to `batch_size`:

```
while requests remain:

  batch = next min(batch_size, remaining) requests

  # Batch starts when GPU is free or first request in batch has arrived.
  t_start = max(batch[0].arrival_time, gpu_free_time)

  if batch_size == 1:
    ── single-request fast path (no padding) ──
    input_ids = tokenize_prompt(prompt_text)
    prefill_s, past, next_t = timed_prefill(model, input_ids, device)
    step_times = timed_decode_steps(model, past, next_t, output_tokens, device)

    first_token_time = t_start + prefill_s + step_times[0]
    finish_time      = t_start + prefill_s + sum(step_times)
    gpu_free_time    = finish_time

  else:
    ── batched path ──
    input_ids, attn_mask = tokenize_batch(prompts)    # [B, L] left-padded
    prefill_s, past, next_ts = timed_prefill_batch(model, input_ids, attn_mask, device)

    max_out = max(r.output_tokens for r in batch)
    step_times = timed_decode_steps_batch(model, past, next_ts, max_out, device)
    # step_times has length max_out; all B requests share each step's forward pass

    first_token_time = t_start + prefill_s + step_times[0]  # same for all in batch

    for each request r in batch:
      finish_time = t_start + prefill_s + sum(step_times[:r.output_tokens])
      record RequestResult

    gpu_free_time = t_start + prefill_s + sum(step_times)   # after slowest request
```

**Key properties:**
- All requests in a batch share the same `first_token_time` — this mirrors the simulator's behaviour.
- Per-request `finish_time` uses only the decode steps that request actually needs, so shorter requests finish earlier on the timeline even though the GPU runs the full `max_out` steps.
- TTFT includes queuing time (`t_start - arrival_time`). If arrivals are faster than processing, the queue grows and TTFT rises — the same pressure the simulator models.

---

## 9. Real GPU Runtime — Disaggregated (2 GPUs)

**File:** `src/runtime/disaggregated_gpu.py`

Uses two GPUs: one for prefill, one for decode. Two separate model instances are loaded. Supports configurable batch sizes.

### Setup

```
prefill_device = cuda:prefill_gpu
decode_device  = cuda:decode_gpu

model_A = load model on prefill_device  (float16)
model_B = load model on decode_device   (float16)
tokenizer = load tokenizer

pipeline_free = 0.0
batch_size    = N   (configurable, default 1)
```

### Per-batch loop (step by step)

```
while requests remain:

  batch = next min(batch_size, remaining) requests
  t_start = max(batch[0].arrival_time, pipeline_free)

  if batch_size == 1:
    ── single-request fast path ──
    input_ids = tokenize_prompt(prompt_text, prefill_device)

    prefill_s, past_A, next_t_A = timed_prefill(model_A, input_ids, prefill_device)
    transfer_s, past_B = time_transfer(past_A, decode_device, source=prefill_device)
    next_t_B = next_t_A.to(decode_device)
    step_times = timed_decode_steps(model_B, past_B, next_t_B, output_tokens, decode_device)

    first_token_time = t_start + prefill_s + transfer_s + step_times[0]
    finish_time      = t_start + prefill_s + transfer_s + sum(step_times)
    pipeline_free    = finish_time

  else:
    ── batched path ──
    input_ids, attn_mask = tokenize_batch(prompts, prefill_device)   # [B, L] left-padded

    # Prefill: single forward produces batched KV on GPU A
    prefill_s, past_A, next_ts_A = timed_prefill_batch(
      model_A, input_ids, attn_mask, prefill_device
    )

    # Transfer: move the entire batched KV ([B, heads, seq, dim] per layer) to GPU B
    transfer_s, past_B = time_transfer(past_A, decode_device, source=prefill_device)
    next_ts_B = next_ts_A.to(decode_device)

    # Decode: single batched forward per step, runs max_out steps
    max_out = max(r.output_tokens for r in batch)
    step_times = timed_decode_steps_batch(model_B, past_B, next_ts_B, max_out, decode_device)

    first_token_time = t_start + prefill_s + transfer_s + step_times[0]

    for each request r in batch:
      finish_time = t_start + prefill_s + transfer_s + sum(step_times[:r.output_tokens])
      record RequestResult

    pipeline_free = t_start + prefill_s + transfer_s + sum(step_times)
```

### What this measures vs the simulator

| Component | Simulator | GPU runtime (batch_size=1) | GPU runtime (batch_size>1) |
|---|---|---|---|
| Prefill time | `prefill_base + prefill_per_token * N` | Actual HF forward, 1 request | Actual HF forward, B requests padded |
| Decode step | `decode_base + decode_per_token` | Actual HF forward, 1 request | Actual HF forward, B requests |
| KV transfer | `transfer_per_prompt_token * N` | Actual `.to(device)`, 1 KV | Actual `.to(device)`, batched KV |
| Interference | Multiplier (1.35) | None | None |

The pipeline is sequential across batches (one batch completes before the next starts). Within a batch, prefill and decode are parallelised across the `B` requests. Full pipelining overlap between batches is a next step.

---

## 10. Stage B — Profiling and Fitting Real GPU Timings

**Files:** `src/stage_b/profile_sharegpt.py`, `src/stage_b/fitted_timing.py`, `src/stage_b/sharegpt_loader.py`, `src/stage_b/workload_sharegpt.py`

Stage B replaces the synthetic timing model with coefficients measured from real GPU runs.

### Step 1 — Load ShareGPT prompts

`sharegpt_loader.py` handles two sources:

```
HuggingFace hub:  iter_sharegpt_from_hf(num_rows)
  → streams Aeala/ShareGPT_Vicuna_unfiltered dataset
  → extracts first user message from each conversation

Local JSONL file: iter_sharegpt_jsonl(path)
  → parses .jsonl with conversation format
  → same extraction logic
```

### Step 2 — Build requests with real token counts

`workload_sharegpt.py` → `build_requests_from_sharegpt()`:

```
for each prompt_text from ShareGPT:
  input_ids = tokenizer(prompt_text)
  prompt_tokens = len(input_ids)
  output_tokens = randint(min_new_tokens, max_new_tokens)

  arrival_time = cumulative sum of exponential(arrival_rate) gaps

  → Request(prompt_tokens, output_tokens, prompt_text, arrival_time)
```

### Step 3 — Measure GPU timings

`profile_sharegpt.py` runs on a single GPU:

```
for each request (up to max_samples):
  input_ids = tokenize(prompt_text, max_tokens=max_prompt_tokens)

  prefill_s, past, next_token = timed_prefill(model, input_ids, device)
  step_times, _ = timed_decode_steps(model, past, next_token, max_new_tokens, device)

  collect:
    prompt_tokens[i] = len(input_ids)
    prefill_times[i] = prefill_s
    decode_steps[i]  = step_times   (list of per-step times)
```

### Step 4 — Fit the linear prefill model

`fitted_timing.py` → `fit_prefill_linear(prompt_tokens, prefill_times)`:

```
Least-squares fit: prefill_s ≈ a + b * prompt_tokens

n = len(samples)
sum_x  = sum(prompt_tokens)
sum_y  = sum(prefill_times)
sum_xy = sum(prompt_tokens[i] * prefill_times[i])
sum_xx = sum(prompt_tokens[i]^2)

b = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x^2)
a = (sum_y - b * sum_x) / n
```

The slope `b` = `prefill_per_token`, the intercept `a` = `prefill_base`.

### Step 5 — Write fitted_timing.json

```json
{
  "timing": {
    "prefill_base":              a,
    "prefill_per_token":         b,
    "decode_base":               mean(all_step_times),
    "decode_per_token":          0.0,
    "transfer_per_prompt_token": 1.5e-5
  },
  "meta": {
    "model":                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "dataset":              "ShareGPT",
    "num_samples_profiled": N,
    "device":               "cuda"
  }
}
```

### Step 6 — Feed back into simulator

`run_comparison.py --timing-json results/fitted_timing.json` loads this file and replaces the synthetic `TimingModel` defaults with the measured values. The simulator then runs with real-GPU-calibrated coefficients.

---

## 11. Metrics Computation

**File:** `src/core/metrics.py`

### SLO configuration

```
SLOConfig
  ttft_slo   = 0.8 s    (default)
  tpot_slo   = 0.03 s   (default)
  e2e_slo    = None      (optional)
```

### `summarize_results(results, slo)` — step by step

**Step 1 — Compute per-request values**

```
for each RequestResult r:
  ttft[i]  = r.first_token_time - r.arrival_time
  tpot[i]  = (r.finish_time - r.first_token_time) / max(r.output_tokens - 1, 1)
  e2e[i]   = r.finish_time - r.arrival_time
```

**Step 2 — Check SLO compliance**

```
for each request i:
  within_slo[i] = (ttft[i] <= ttft_slo) AND (tpot[i] <= tpot_slo)
  if e2e_slo is set:
    within_slo[i] = within_slo[i] AND (e2e[i] <= e2e_slo)
```

**Step 3 — Compute aggregates**

```
goodput    = sum(within_slo) / total_requests
throughput = total_requests / (max_finish_time - min_arrival_time)

mean_ttft  = mean(ttft)
p50_ttft   = percentile(ttft, 50)
p90_ttft   = percentile(ttft, 90)
p95_ttft   = percentile(ttft, 95)
p99_ttft   = percentile(ttft, 99)
(same for tpot and e2e)
```

**Step 4 — Return summary dict**

All values are returned as a dictionary and written to a `*-summary.json` file. Per-request values are written to a `*-requests.csv` file with columns: `request_id, arrival_time, first_token_time, finish_time, ttft, tpot, e2e, within_slo`.

---

## 12. Experiment Runners

### `run_comparison.py` — Simulator comparison

```
1. Parse args (workload type, timing source, num_requests, arrival_rate, SLOs, batching)
2. Build TimingModel:
   - if --timing-json: load measured coefficients from Stage B
   - else: use synthetic defaults
3. Generate requests:
   - if --workload sharegpt: build_requests_from_sharegpt(model, num_requests)
   - else: generate_requests(WorkloadConfig)
4. Run colocated simulator → results_colocated
5. Run disaggregated simulator → results_disaggregated
6. Compute metrics for both
7. Print comparison table
8. Write CSVs and summary JSON to results/ with timestamp prefix
```

### `run_ablations.py` — Sensitivity sweeps

```
1. Build base TimingModel and WorkloadConfig
2. Sweep 1: KV transfer cost
   for transfer_per_prompt_token in [0.0, 5e-6, 1e-5, 1.5e-5, 2e-5, 3e-5, 4.5e-5, 6e-5]:
     build TimingModel with this value
     run colocated and disaggregated simulators
     record: goodput, p95_ttft, p99_e2e for both
3. Sweep 2: capacity split
   for (prefill_mult, decode_mult) in [(0.8,1.2), (1.0,1.0), (1.2,0.8), (1.4,0.7)]:
     build TimingModel with these multipliers
     run both simulators
     record same metrics
4. Write combined CSV and meta JSON
```

### `run_gpu_comparison.py` — Real GPU comparison

```
1. Parse args (model, num_requests, prefill_gpu, decode_gpu, baseline_only, batch_size)
2. Run baseline_gpu.run(model, num_requests, gpu=prefill_gpu, batch_size=batch_size)
   → results_colocated
3. if not baseline_only:
   Run disaggregated_gpu.run(model, num_requests, prefill_gpu, decode_gpu, batch_size=batch_size)
   → results_disaggregated
4. Compute metrics for both
5. Print comparison table
6. Write CSVs and summary JSON (includes batch_size) to results/ with timestamp prefix
```

`--batch-size` (default `1`) controls how many requests are grouped into a single padded forward call for both prefill and decode. `batch_size=1` is identical to the original sequential behaviour.

---

## 13. End-to-End Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│                  Workload Generator                 │
│  (workload.py / workload_sharegpt.py)               │
│                                                     │
│  Arrival times (Poisson / Bursty)                   │
│  Token counts (Uniform / Mixed / ShareGPT)          │
└─────────────────────────────┬───────────────────────┘
                              │  List[Request]
              ┌───────────────┼───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│   SIMULATOR PATH        │   │   REAL GPU PATH         │
│                         │   │                         │
│  TimingModel            │   │  inference_core.py      │
│  (timing.py)            │   │  - timed_prefill()      │
│  - prefill_time()       │   │  - timed_decode_steps() │
│  - decode_step_time()   │   │  - time_transfer()      │
│  - transfer_time()      │   │                         │
│                         │   │  baseline_gpu.py        │
│  baseline.py            │   │  - 1 GPU, sequential    │
│  - single pool          │   │  - prefill → decode     │
│  - interference × 1.35  │   │                         │
│                         │   │  disaggregated_gpu.py   │
│  disaggregated.py       │   │  - GPU A: prefill       │
│  - prefill pool         │   │  - .to(GPU B): KV copy  │
│  - KV transfer cost     │   │  - GPU B: decode        │
│  - decode pool          │   │                         │
└────────────┬────────────┘   └────────────┬────────────┘
             │                             │
             │  List[RequestResult]        │  List[RequestResult]
             └───────────────┬─────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │      metrics.py          │
              │  summarize_results()     │
              │                          │
              │  Per-request:            │
              │    TTFT, TPOT, E2E       │
              │                          │
              │  Aggregates:             │
              │    mean, p50/95/99       │
              │    goodput, throughput   │
              └──────────────┬───────────┘
                             │
                   ┌─────────┴──────────┐
                   ▼                    ▼
         results/*-requests.csv    results/*-summary.json


┌───────────────────────────────────────┐
│         STAGE B PIPELINE              │
│                                       │
│  ShareGPT prompts                     │
│        ↓                              │
│  profile_sharegpt.py                  │
│  - timed_prefill() on real GPU        │
│  - timed_decode_steps()               │
│        ↓                              │
│  fit_prefill_linear()                 │
│  - least-squares: a + b * tokens      │
│        ↓                              │
│  fitted_timing.json                   │
│        ↓                              │
│  run_comparison.py --timing-json      │
│  - replaces synthetic TimingModel     │
│  - simulator runs with real numbers   │
└───────────────────────────────────────┘
```

---

## Summary of Design Decisions

| Decision | Rationale |
|---|---|
| No Ray, no SwiftTransformer | Course constraints; Ray actors/placement groups not available |
| Python-only, PyTorch + HuggingFace | Portable, inspectable, works on 1–2 GPU nodes |
| Simulator + real GPU two-track | Simulator enables large controlled sweeps; GPU run validates direction |
| `colocated_interference = 1.35` | Empirical estimate of prefill/decode contention on shared GPU |
| Batch exponents (0.20, 0.10) | Sub-linear batch speedup: GPU parallelism partially absorbs overhead |
| Stage B fitting pipeline | Grounds simulator in real measurements rather than synthetic constants |
| Same-node KV transfer only | Sufficient for demonstrating the architectural idea; cross-node is follow-on |
| Left-padding for batched tokenization | Ensures `logits[:, -1:, :]` picks the correct next token for every sequence in the batch regardless of prompt length |
| Batched KV transfer as a unit | Moving all `B` requests' KV in one `time_transfer` call avoids per-request round-trip overhead and keeps the transfer measurement representative |
| Sequential batched pipeline | Batches complete one at a time; full inter-batch pipelining (prefill overlapping with decode) is a follow-on step |
