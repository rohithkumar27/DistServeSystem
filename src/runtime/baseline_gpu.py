"""
Colocated prefill + decode on **one GPU** — real `forward` times (no TimingModel).

Matches DistServe's "monolithic" case: one pool runs both phases on the same device,
so prefill and decode contend on that GPU (here: strict FIFO, one request at a time
or in configurable batches).
"""
from __future__ import annotations

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
) -> List[RequestResult]:
    """
    For each batch of requests (up to `batch_size`) in arrival order: run a single
    batched prefill then batched greedy decode on `device`.

    Timeline: next batch starts at max(earliest_arrival_in_batch, previous_finish).
    With batch_size=1 this is identical to the original sequential behaviour.
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

        # Batch starts when GPU is free or the first request has arrived.
        t_start = max(batch[0].arrival_time, gpu_free_time)

        if len(batch) == 1:
            # Fast path: no padding overhead for a single request.
            r = batch[0]
            input_ids = tokenize_prompt(tokenizer, r.prompt_text, device, max_prompt_tokens)
            prefill_s, past, next_t = timed_prefill(model, input_ids, device)
            step_times, _ = timed_decode_steps(model, past, next_t, r.output_tokens, device)

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


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
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
    return model, tokenizer
