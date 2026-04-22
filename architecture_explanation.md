# DistServeSystem — Complete Architecture Explanation

This document walks through the real-GPU inference system step by step, from how a ShareGPT prompt enters the pipeline to how metrics are computed and reported.

---

## Table of Contents

1. [The Core Problem](#1-the-core-problem)
2. [Request Representation](#2-request-representation)
3. [ShareGPT Workload Generation](#3-sharegpt-workload-generation)
4. [Real GPU Runtime — Shared Inference Core](#4-real-gpu-runtime--shared-inference-core)
5. [Real GPU Runtime — Baseline (1 GPU, Colocated)](#5-real-gpu-runtime--baseline-1-gpu-colocated)
6. [Real GPU Runtime — Disaggregated (2 GPUs)](#6-real-gpu-runtime--disaggregated-2-gpus)
7. [Metrics Computation](#7-metrics-computation)
8. [Experiment Runner](#8-experiment-runner)
9. [End-to-End Data Flow Diagram](#9-end-to-end-data-flow-diagram)

> **Note on simulator code:** The repo also contains a parametric simulator (`src/simulator/`) used for controlled sweeps and a Stage B profiling pipeline (`src/stage_b/profile_sharegpt.py`, `fitted_timing.py`) that fits simulator coefficients from GPU measurements. Those are secondary; this document focuses on the real-GPU path.

---

## 1. The Core Problem

LLM inference has two distinct phases:

- **Prefill**: The model processes the entire input prompt in one forward pass. This is compute-intensive and scales with prompt length.
- **Decode**: The model autoregressively generates one token at a time, reusing the KV cache from prefill. This is memory-bandwidth-intensive and sequential.

When both phases share the same GPU(s), **prefill disrupts decode**. A long prefill from one request steals GPU cycles from the decode steps of many other in-flight requests, causing spikes in tail latency and poor SLO goodput.

DistServe's solution: **separate prefill and decode into dedicated worker pools**, giving each phase its own scheduling lane and eliminating the interference. The KV cache is transferred between GPUs after prefill completes.

This repo implements and evaluates both designs on real hardware: **colocated** (single GPU) vs **disaggregated** (two GPUs).

---

## 2. Request Representation

**File:** `src/core/request.py`

Every inference request is a `Request` object:

```
Request
  request_id        unique integer ID
  arrival_time      wall-clock arrival (seconds)
  prompt_tokens     number of input tokens (from tokenizer)
  output_tokens     number of tokens to generate
  prompt_text       actual prompt string (from ShareGPT)
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

1. Built by the ShareGPT workload generator with an arrival time, tokenized prompt, and sampled output length.
2. Queued in the GPU runtime (colocated or disaggregated).
3. Prefill runs on GPU: first token time is recorded.
4. Decode runs on GPU: finish time is recorded.
5. Becomes a `RequestResult` and is passed to the metrics layer.

---

## 3. ShareGPT Workload Generation

**Files:** `src/stage_b/sharegpt_loader.py`, `src/stage_b/workload_sharegpt.py`

The workload generator produces a list of `Request` objects with **real prompts** drawn from the ShareGPT conversation corpus.

### Step 1 — Load ShareGPT conversations

`sharegpt_loader.py` supports two sources:

**HuggingFace hub (default):**
```
iter_sharegpt_from_hf(
  dataset_name = "Aeala/ShareGPT_Vicuna_unfiltered",
  split        = "train",
  max_samples  = num_requests * 4
)
```
Falls back across mirrors (`anon8231489123/ShareGPT_Vicuna_unfiltered`) if the primary dataset fails to load. Rows pulled are capped at `max_samples * 10` to avoid huge downloads.

**Local JSONL file:**
```
iter_sharegpt_jsonl(path, max_samples)
```
Parses a `.jsonl` file with the same `{conversations: [...]}` schema.

### Step 2 — Extract the prompt from each conversation

Each ShareGPT row has a `conversations` array of turns like:
```json
{"from": "human", "value": "Explain how gradient descent works..."}
{"from": "gpt",   "value": "Gradient descent is..."}
```

`_conversation_to_prompt()` walks the turns:
```
for turn in conversation:
  role  = turn["from"] or turn["role"]
  value = turn["value"] or turn["content"]
  if role in {human, user, system} and value non-empty:
    append value
  if role in {human, user}:
    break    # stop after first user turn → stable "chat prompt"
```

This produces a single prompt string per conversation (no model turns, no multi-turn context).

### Step 3 — Build requests with real token counts

`build_requests_from_sharegpt()`:

```
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token

rng = Random(seed)
t = 0.0
for sample in sharegpt_samples:
  enc = tokenizer(
    sample.text,
    truncation  = True,
    max_length  = max_prompt_tokens
  )
  prompt_tokens = len(enc["input_ids"])
  if prompt_tokens < 6:
    skip    # drop trivially short prompts

  # Poisson arrival: exponential inter-arrival gaps
  if arrival_rate > 0:
    t += rng.expovariate(arrival_rate)

  # Sample decode length uniformly
  output_tokens = rng.randint(output_low, output_high)

  Request(
    request_id    = rid,
    arrival_time  = t,
    prompt_tokens = prompt_tokens,
    output_tokens = output_tokens,
    prompt_text   = sample.text
  )
```

### Key workload properties

| Knob | Purpose | Typical value |
|---|---|---|
| `tokenizer_name` | Must match the serving model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `max_prompt_tokens` | Truncates long ShareGPT prompts | 1024–2048 |
| `output_low`, `output_high` | Uniform range for generated tokens per request | 16, 64 |
| `arrival_rate` | Poisson rate in req/s | 1.0–4.0 |
| `num_requests` | Total requests in the run | 15–300 |
| `seed` | RNG seed for reproducibility | 7 |

**Why ShareGPT?** Real conversational prompts have naturally skewed length distributions — a mix of short interactive turns and long context-heavy prompts. That skew is exactly the workload that makes disaggregation valuable: long prefills from a few requests would otherwise stall decode for everyone else.

---

## 4. Real GPU Runtime — Shared Inference Core

**File:** `src/runtime/inference_core.py`

This module provides the building blocks used by both GPU runtimes — both single-request and batched variants.

### `timed_prefill(model, input_ids, device)` — single request

```
torch.cuda.synchronize(device)       # flush any pending GPU work
t0 = time.perf_counter()

outputs = model(
  input_ids = input_ids,
  use_cache = True                   # tells HuggingFace to return past_key_values
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

## 5. Real GPU Runtime — Baseline (1 GPU, Colocated)

**File:** `src/runtime/baseline_gpu.py`

Runs requests on a single GPU in arrival order, grouped into batches of up to `batch_size`. This is the colocated reference point — prefill and decode share the same device and queue.

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
- All requests in a batch share the same `first_token_time`.
- Per-request `finish_time` uses only the decode steps that request actually needs, so shorter requests finish earlier on the timeline even though the GPU runs the full `max_out` steps.
- TTFT includes queuing time (`t_start - arrival_time`). If arrivals are faster than processing, the queue grows and TTFT rises.

---

## 6. Real GPU Runtime — Disaggregated (2 GPUs)

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

### What each component measures

| Component | GPU runtime (batch_size=1) | GPU runtime (batch_size>1) |
|---|---|---|
| Prefill time | Actual HF forward, 1 request | Actual HF forward, B requests left-padded |
| Decode step | Actual HF forward, 1 request | Actual HF forward, B requests |
| KV transfer | Actual `.to(device)`, 1 KV | Actual `.to(device)`, batched KV |
| Interference | None | None |

The pipeline is sequential across batches (one batch completes before the next starts). Within a batch, prefill and decode are parallelised across the `B` requests. Full pipelining overlap between batches is a next step.

---

## 7. Metrics Computation

**File:** `src/core/metrics.py`

### SLO configuration

```
SLOConfig
  ttft_slo   = 0.8 s    (default)
  tpot_slo   = 0.03 s   (default)
  e2e_slo    = None     (optional)
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

## 8. Experiment Runner

### `run_gpu_comparison.py` — Real GPU comparison

```
1. Parse args (model, num_requests, prefill_gpu, decode_gpu, baseline_only, batch_size,
               sharegpt dataset/jsonl, arrival_rate, SLOs)
2. Build requests from ShareGPT:
   requests = build_requests_from_sharegpt(
     tokenizer_name    = model,
     dataset_name      = "Aeala/ShareGPT_Vicuna_unfiltered",
     num_requests      = N,
     max_prompt_tokens = 1024,
     output_low        = 16,
     output_high       = 64,
     arrival_rate      = rate,
     seed              = 7
   )
3. Run baseline_gpu.run(model, requests, gpu=prefill_gpu, batch_size=batch_size)
   → results_colocated
4. if not baseline_only:
   Run disaggregated_gpu.run(model, requests, prefill_gpu, decode_gpu, batch_size=batch_size)
   → results_disaggregated
5. Compute metrics for both with SLOConfig
6. Print comparison table
7. Write CSVs and summary JSON (includes batch_size) to results/ with timestamp prefix
```

`--batch-size` (default `1`) controls how many requests are grouped into a single padded forward call for both prefill and decode. `batch_size=1` is identical to the original sequential behaviour.

---

## 9. End-to-End Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│         ShareGPT Workload Generator                 │
│  (sharegpt_loader.py + workload_sharegpt.py)        │
│                                                     │
│  1. Load Aeala/ShareGPT_Vicuna_unfiltered from HF   │
│  2. Extract first user turn from each conversation  │
│  3. Tokenize with model tokenizer (TinyLlama)       │
│  4. Assign Poisson arrivals, uniform output lengths │
└─────────────────────────────┬───────────────────────┘
                              │  List[Request]
                              │  (real prompt_text + token counts)
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│   BASELINE GPU          │   │   DISAGGREGATED GPU     │
│   (baseline_gpu.py)     │   │   (disaggregated_gpu.py)│
│                         │   │                         │
│  Single GPU, FIFO       │   │  GPU A: prefill pool    │
│  ├─ tokenize_batch      │   │  ├─ tokenize_batch      │
│  ├─ timed_prefill_batch │   │  ├─ timed_prefill_batch │
│  └─ timed_decode_steps  │   │  │                      │
│     _batch              │   │  time_transfer          │
│                         │   │  └─ .to(GPU B) KV copy  │
│  Prefill + decode share │   │                         │
│  the same device queue  │   │  GPU B: decode pool     │
│                         │   │  └─ timed_decode_steps  │
│                         │   │     _batch              │
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
```

---

## Summary of Design Decisions

| Decision | Rationale |
|---|---|
| Python-only, PyTorch + HuggingFace | Portable, inspectable, works on 1–2 GPU nodes |
| ShareGPT prompts for the workload | Real conversational length distribution exposes the prefill/decode skew that motivates disaggregation |
| First-user-turn-only extraction | Stable single-prompt-per-request semantics; no multi-turn context drift |
| Two model instances in disaggregated mode | Simpler than sharding — each GPU owns a full copy so prefill and decode are fully independent |
| Left-padding for batched tokenization | Ensures `logits[:, -1:, :]` picks the correct next token for every sequence in the batch regardless of prompt length |
| Batched KV transfer as a unit | Moving all `B` requests' KV in one `time_transfer` call avoids per-request round-trip overhead |
| Same-node KV transfer only | Sufficient for demonstrating the architectural idea; cross-node is follow-on |
| Sequential batched pipeline | Batches complete one at a time; full inter-batch pipelining (prefill overlapping with decode) is a follow-on step |
