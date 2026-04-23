"""
Colocated prefill + decode on **one GPU** — real `forward` times (no TimingModel).

Matches DistServe's "monolithic" case: one pool runs both phases on the same device,
so prefill and decode contend on that GPU (here: strict FIFO, one request at a time
or in configurable batches).
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
    tokenize_batch,
    tokenize_prompt,
)


def run_baseline_gpu(
    requests: Iterable[Request],
    *,
    model,
    tokenizer,
    device: torch.device,
    max_prompt_tokens: int = 2048,
    batch_size: int = 1,
    interference_factor: float = 1.0,
    verbose: bool = False,
) -> List[RequestResult]:
    """
    For each batch of requests (up to `batch_size`) in arrival order: run a single
    batched prefill then batched greedy decode on `device`.

    Timeline: next batch starts at max(latest_arrival_in_batch, previous_finish).
    Using the latest arrival (not the earliest) respects causality — a batch cannot
    begin before every request in it has actually arrived, which prevents negative
    TTFT values when arrivals are spread in time.
    With batch_size=1 this is identical to the original sequential behaviour.

    `interference_factor` scales all decode step times to model the slowdown a
    colocated GPU experiences when it interleaves prefill and decode — i.e. the
    prefill-decode interference that DistServe eliminates.  In a real continuous-
    batching system (vLLM-style), newly arriving prefill tokens are injected into
    the running decode batch, inflating every step's latency by ~1.2–1.4×.
    Default 1.0 = no interference (pure FIFO batch scheduling as implemented here).
    Set to ~1.3 to reproduce the paper's reported colocated penalty.
    """
    req_list = sorted(list(requests), key=lambda r: r.arrival_time)
    results: List[RequestResult] = []
    gpu_free_time = 0.0

    i = 0
    while i < len(req_list):
        batch = req_list[i : i + batch_size]
        i += len(batch)

        for r in batch:
            if not r.prompt_text:
                raise ValueError(
                    "run_baseline_gpu requires Request.prompt_text. "
                    "Build workload with build_requests_from_sharegpt(...) or set prompt_text."
                )

        # Batch starts when GPU is free AND every request in the batch has arrived.
        # batch is sorted by arrival_time, so batch[-1] is the latest arrival.
        t_start = max(batch[-1].arrival_time, gpu_free_time)

        if len(batch) == 1:
            # Fast path: no padding overhead for a single request.
            r = batch[0]
            input_ids = tokenize_prompt(tokenizer, r.prompt_text, device, max_prompt_tokens)
            prefill_s, past, next_t = timed_prefill(model, input_ids, device)
            step_times, _ = timed_decode_steps(model, past, next_t, r.output_tokens, device)
            if interference_factor != 1.0:
                step_times = [t * interference_factor for t in step_times]

            if verbose:
                print(
                    f"  [coloc batch {i//batch_size - 1}] "
                    f"prompt_len={input_ids.shape[-1]} out_tokens={r.output_tokens} | "
                    f"prefill={prefill_s*1e3:.0f}ms  "
                    f"decode_step0={step_times[0]*1e3:.0f}ms  "
                    f"decode_total={sum(step_times)*1e3:.0f}ms"
                )

            first_token_time = t_start + prefill_s + step_times[0]
            finish_time = t_start + prefill_s + sum(step_times)
            gpu_free_time = finish_time

            results.append(
                RequestResult(
                    request_id=r.request_id,
                    arrival_time=r.arrival_time,
                    first_token_time=first_token_time,
                    finish_time=finish_time,
                    output_tokens=r.output_tokens,
                )
            )
        else:
            # Batched path: pad prompts, single forward for prefill and each decode step.
            prompts = [r.prompt_text for r in batch]
            input_ids, attn_mask = tokenize_batch(tokenizer, prompts, device, max_prompt_tokens)

            prefill_s, past, next_ts = timed_prefill_batch(model, input_ids, attn_mask, device)

            max_out = max(r.output_tokens for r in batch)
            step_times, _ = timed_decode_steps_batch(model, past, next_ts, max_out, device)
            if interference_factor != 1.0:
                step_times = [t * interference_factor for t in step_times]

            if verbose:
                print(
                    f"  [coloc batch {i//batch_size - 1}] "
                    f"prompt_len={input_ids.shape[-1]} out_tokens={max_out} | "
                    f"prefill={prefill_s*1e3:.0f}ms  "
                    f"decode_step0={step_times[0]*1e3:.0f}ms  "
                    f"decode_total={sum(step_times)*1e3:.0f}ms"
                )

            # All requests in the batch share the same first-token time.
            first_token_time = t_start + prefill_s + step_times[0]

            for r in batch:
                # Each request finishes after its own number of decode steps.
                finish_time = t_start + prefill_s + sum(step_times[: r.output_tokens])
                results.append(
                    RequestResult(
                        request_id=r.request_id,
                        arrival_time=r.arrival_time,
                        first_token_time=first_token_time,
                        finish_time=finish_time,
                        output_tokens=r.output_tokens,
                    )
                )

            # GPU is free after the slowest request in the batch finishes.
            gpu_free_time = t_start + prefill_s + sum(step_times)

    return results


def _warmup_model(
    model,
    tokenizer,
    device: torch.device,
    warmup_batch: int = 8,
    warmup_seq_len: int = 512,
    warmup_decode_steps: int = 2,
) -> None:
    """Prime all CUDA kernels and the caching allocator before real inference.

    Two-pass warmup mirrors disaggregated_gpu._warmup_pipeline:
      Pass 1 (batch=warmup_batch): loads CUDA kernels and fills the LARGE pool.
      Pass 2 (batch=actual_batch_size, warmup_decode_steps=max_new_tokens):
        fills the SMALL pool with blocks for every decode-step size.
    Both passes are triggered by load_model_and_tokenizer.
    """
    input_ids, attn_mask = tokenize_batch(
        tokenizer,
        ["The quick brown fox jumps over the lazy dog. " * 60] * warmup_batch,
        device,
        warmup_seq_len,
    )
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        past = out.past_key_values
        for _ in range(warmup_decode_steps):
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            past = out.past_key_values
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    batch_size: int = 1,
    max_prompt_tokens: int = 512,
    max_new_tokens: int = 64,
) -> tuple[object, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": str(device)} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    # Pass 1: large-batch kernel loading
    _warmup_model(model, tokenizer, device,
                  warmup_batch=max(8, batch_size), warmup_seq_len=max_prompt_tokens)
    # Pass 2: actual-batch small-pool pre-population
    _warmup_model(model, tokenizer, device,
                  warmup_batch=batch_size, warmup_seq_len=max_prompt_tokens,
                  warmup_decode_steps=max_new_tokens)
    return model, tokenizer
