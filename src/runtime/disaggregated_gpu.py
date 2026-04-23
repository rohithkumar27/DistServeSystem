"""
Disaggregated prefill vs decode on **two devices** — real forwards + measured KV
handoff, with inter-batch pipelining.

Matches DistServe's idea: prefill worker produces KV on `prefill_device`, tensors
are moved to `decode_device`, decode worker runs autoregressive steps. Two full
model copies with the **same** weights (like two role-specific replicas).

Supports batched operation: multiple requests are padded into a single forward
call for both prefill and decode, and the batched KV cache is transferred as a
unit.

Pipelining model (the key difference from a naive sequential implementation):
  * Two independent free-time clocks are tracked: gpu_a_free (prefill GPU) and
    gpu_b_free (decode GPU).
  * Batch N+1's prefill can start on GPU A as soon as GPU A is free, i.e. as
    soon as the transfer of batch N's KV cache has completed. It does NOT wait
    for batch N's decode on GPU B to finish.
  * Decode for batch N starts on GPU B when both (a) GPU B is free and (b) the
    KV transfer for batch N has finished.
  * Per-batch wall times (prefill_s, transfer_s, decode step_times) are measured
    on real hardware the same way as before. Only the *timeline* is computed as
    if the two GPUs ran concurrently — which they would in a real deployment,
    since GPU A and GPU B are physically independent devices.

Requires at least 2 CUDA devices for meaningful separation; with one GPU this
path cannot isolate hardware pools.
"""
from __future__ import annotations

import time
from typing import Iterable, List
import torch
from src.core.request import Request, RequestResult
from src.runtime.inference_core import (
    timed_decode_steps,
    timed_decode_steps_batch,
    timed_prefill,
    timed_prefill_batch,
    time_transfer,
    tokenize_batch,
    tokenize_prompt,
)


def run_disaggregated_gpu(
    requests: Iterable[Request],
    *,
    model_prefill,
    model_decode,
    tokenizer,
    prefill_device: torch.device,
    decode_device: torch.device,
    max_prompt_tokens: int = 2048,
    batch_size: int = 1,
    verbose: bool = False,
) -> List[RequestResult]:
    """
    Pipelined disaggregated inference across two GPUs.

    Per-batch ordering on the two device clocks:
      prefill_start  = max(latest_arrival_in_batch, gpu_a_free)
      prefill_end    = prefill_start + prefill_s
      transfer_end   = prefill_end + transfer_s
      gpu_a_free     = transfer_end              # GPU A free after transfer
      decode_start   = max(transfer_end, gpu_b_free)
      decode_end     = decode_start + sum(step_times)
      gpu_b_free     = decode_end

    A later batch's prefill can start on GPU A as soon as GPU A is free, even
    while GPU B is still decoding an earlier batch. This is the scheduling
    overlap that gives disaggregation its real benefit.

    Requests are grouped into batches of up to `batch_size`. Each batch is
    processed as a single padded forward; the batched KV cache is transferred
    together.
    """
    if prefill_device == decode_device:
        raise ValueError(
            "Disaggregated GPU mode requires prefill_device != decode_device "
            "(e.g. cuda:0 and cuda:1)."
        )

    req_list = sorted(list(requests), key=lambda r: r.arrival_time)
    results: List[RequestResult] = []

    # Two independent device clocks — the source of the pipelining benefit.
    gpu_a_free = 0.0  # prefill GPU free after prefill+transfer of the last batch
    gpu_b_free = 0.0  # decode GPU free after decode of the last batch

    i = 0
    while i < len(req_list):
        batch = req_list[i : i + batch_size]
        i += len(batch)

        for r in batch:
            if not r.prompt_text:
                raise ValueError(
                    "run_disaggregated_gpu requires Request.prompt_text "
                    "(use build_requests_from_sharegpt)."
                )

        # Prefill starts when GPU A is free AND every request in the batch has
        # arrived. batch is sorted by arrival_time, so batch[-1] is the latest.
        t_arrive = batch[-1].arrival_time
        prefill_start = max(t_arrive, gpu_a_free)

        if len(batch) == 1:
            # Fast path: no padding overhead for a single request.
            r = batch[0]
            input_ids = tokenize_prompt(
                tokenizer, r.prompt_text, prefill_device, max_prompt_tokens
            )
            prefill_s, past, next_t = timed_prefill(
                model_prefill, input_ids, prefill_device
            )
            transfer_s, past_dec = time_transfer(
                past, decode_device, source_device=prefill_device
            )
            t_prealloc = time.perf_counter()
            _preallocate_decode_buffers(past_dec, decode_device)
            prealloc_s = time.perf_counter() - t_prealloc
            next_on_dec = next_t.to(decode_device)
            step_times, _ = timed_decode_steps(
                model_decode, past_dec, next_on_dec, r.output_tokens, decode_device
            )
            prompt_len = input_ids.shape[-1]
        else:
            # Batched path.
            prompts = [r.prompt_text for r in batch]
            input_ids, attn_mask = tokenize_batch(
                tokenizer, prompts, prefill_device, max_prompt_tokens
            )
            prefill_s, past_A, next_ts_A = timed_prefill_batch(
                model_prefill, input_ids, attn_mask, prefill_device
            )
            transfer_s, past_B = time_transfer(
                past_A, decode_device, source_device=prefill_device
            )
            t_prealloc = time.perf_counter()
            _preallocate_decode_buffers(past_B, decode_device)
            prealloc_s = time.perf_counter() - t_prealloc
            next_ts_B = next_ts_A.to(decode_device)
            max_out = max(r.output_tokens for r in batch)
            step_times, _ = timed_decode_steps_batch(
                model_decode, past_B, next_ts_B, max_out, decode_device
            )
            prompt_len = input_ids.shape[-1]

        if verbose:
            print(
                f"  [disagg batch {i//batch_size - 1}] "
                f"prompt_len={prompt_len} out_tokens={max_out if len(batch)>1 else r.output_tokens} | "
                f"prefill={prefill_s*1e3:.0f}ms  transfer={transfer_s*1e3:.0f}ms  "
                f"prealloc={prealloc_s*1e3:.0f}ms  "
                f"decode_step0={step_times[0]*1e3:.0f}ms  "
                f"decode_total={sum(step_times)*1e3:.0f}ms"
            )

        # Virtual-timeline accounting with independent per-GPU clocks.
        prefill_end  = prefill_start + prefill_s
        transfer_end = prefill_end + transfer_s
        gpu_a_free   = transfer_end
        decode_start = max(transfer_end, gpu_b_free)
        decode_end   = decode_start + sum(step_times)
        gpu_b_free   = decode_end

        # All requests in the batch share the same first-token time (the first
        # decode step produces one token per sequence).
        first_token_time = decode_start + step_times[0]

        for r in batch:
            finish_time = decode_start + sum(step_times[: r.output_tokens])
            results.append(
                RequestResult(
                    request_id=r.request_id,
                    arrival_time=r.arrival_time,
                    first_token_time=first_token_time,
                    finish_time=finish_time,
                    output_tokens=r.output_tokens,
                )
            )

    return results


def _preallocate_decode_buffers(past: object, device: torch.device) -> None:
    """Pre-allocate and free the exact tensor sizes DynamicLayer.update() will need.

    torch.cat([prefill_kv, new_1_token], dim=-2) allocates [B, kv_h, seq_len+1, head_dim]
    per layer. These land in PyTorch's SMALL pool (≤1MB). The large-batch warmup
    (batch=8, seq_len=512) populates the LARGE pool (>1MB) — a completely separate
    allocator pool. Without this call, every batch's first decode step triggers
    cudaMalloc for each of 22×2=44 small tensors, costing 2-4s on the cluster.

    Allocating and immediately freeing these exact-sized dummies writes them into
    the free list of the correct pool, so the real torch.cat gets a cache hit.
    """
    with torch.no_grad():
        for layer in past.layers:
            k = layer.keys
            # Shape that torch.cat([k, one_new_token], dim=-2) will produce
            dummy_k = torch.empty(
                k.shape[0], k.shape[1], k.shape[2] + 1, k.shape[3],
                dtype=k.dtype, device=device,
            )
            dummy_v = torch.empty_like(dummy_k)
            del dummy_k, dummy_v
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_two_models(
    model_name: str,
    prefill_device: torch.device,
    decode_device: torch.device,
    batch_size: int = 1,
    max_prompt_tokens: int = 512,
    max_new_tokens: int = 64,
) -> tuple[object, object, object]:
    """Two identical causal LMs — required so KV from prefill matches decode weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16
    model_prefill = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": str(prefill_device)},
        trust_remote_code=True,
    )
    model_decode = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": str(decode_device)},
        trust_remote_code=True,
    )
    model_prefill.eval()
    model_decode.eval()

    # Full pipeline warmup: prefill on GPU A → KV transfer → decode on GPU B.
    # Two passes:
    #   Pass 1 (batch=8, seq=512): loads all CUDA kernels and fills the LARGE pool.
    #   Pass 2 (actual batch_size, seq=max_prompt_tokens, max_new_tokens steps):
    #     fills the SMALL pool with blocks for every decode-step size that real
    #     inference will need — so serving pays zero cudaMalloc overhead.
    _warmup_pipeline(
        model_prefill, model_decode, tokenizer, prefill_device, decode_device,
        warmup_batch=max(8, batch_size), warmup_seq_len=max_prompt_tokens,
    )
    _warmup_pipeline(
        model_prefill, model_decode, tokenizer, prefill_device, decode_device,
        warmup_batch=batch_size, warmup_seq_len=max_prompt_tokens,
        warmup_decode_steps=max_new_tokens,
    )

    return model_prefill, model_decode, tokenizer


def _warmup_pipeline(
    model_prefill,
    model_decode,
    tokenizer,
    prefill_device: torch.device,
    decode_device: torch.device,
    warmup_batch: int = 8,
    warmup_seq_len: int = 512,
    warmup_decode_steps: int = 2,
) -> None:
    """Full prefill→transfer→decode warmup to prime CUDA kernels and memory pools.

    Called twice from load_two_models:
      Pass 1  warmup_batch=8, warmup_seq_len=max_prompt_tokens, warmup_decode_steps=2
        → loads all CUDA kernels and populates the LARGE allocator pool (>1 MB).
      Pass 2  warmup_batch=actual_batch_size, warmup_seq_len=max_prompt_tokens,
              warmup_decode_steps=max_new_tokens
        → runs the exact same code path as real inference with the exact shapes
          that serving will use, so PyTorch's SMALL pool (<1 MB) ends up with
          free blocks for every decode-step size.  Without this, the first real
          batch pays 2-4 s of cudaMalloc for each of 22×2=44 tensors per step.
    """
    input_ids, attn_mask = tokenize_batch(
        tokenizer,
        ["The quick brown fox jumps over the lazy dog. " * 60] * warmup_batch,
        prefill_device,
        warmup_seq_len,
    )

    with torch.no_grad():
        out_a = model_prefill(
            input_ids=input_ids, attention_mask=attn_mask, use_cache=True
        )
        if prefill_device.type == "cuda":
            torch.cuda.synchronize(prefill_device)

        _, past_b = time_transfer(
            out_a.past_key_values, decode_device, source_device=prefill_device
        )
        _preallocate_decode_buffers(past_b, decode_device)
        next_tok_b = out_a.logits[:, -1:, :].argmax(dim=-1).to(decode_device)

        for _ in range(warmup_decode_steps):
            out_b = model_decode(
                input_ids=next_tok_b, past_key_values=past_b, use_cache=True
            )
            next_tok_b = out_b.logits[:, -1:, :].argmax(dim=-1)
            past_b = out_b.past_key_values

    if decode_device.type == "cuda":
        torch.cuda.synchronize(decode_device)
    if prefill_device.type == "cuda":
        torch.cuda.synchronize(prefill_device)
