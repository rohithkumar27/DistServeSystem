"""
Disaggregated prefill vs decode on **two devices** — real forwards + measured KV handoff.

Matches DistServe's idea: prefill worker produces KV on `prefill_device`, tensors are
moved to `decode_device`, decode worker runs autoregressive steps. Two full model
copies with the **same** weights (like two role-specific replicas).

Supports batched operation: multiple requests are padded into a single forward call for
both prefill and decode, and the batched KV cache is transferred as a unit.

Requires at least 2 CUDA devices for meaningful separation; with one GPU this path
cannot isolate hardware pools.
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
) -> List[RequestResult]:
    """
    Sequential pipeline: prefill on GPU A → copy KV to GPU B → decode on B.
    Requests are grouped into batches of up to `batch_size`. Each batch is processed
    as a single padded forward; the batched KV cache is transferred together.

    With batch_size=1 this is identical to the original sequential behaviour.
    """
    if prefill_device == decode_device:
        raise ValueError(
            "Disaggregated GPU mode requires prefill_device != decode_device "
            "(e.g. cuda:0 and cuda:1)."
        )

    req_list = sorted(list(requests), key=lambda r: r.arrival_time)
    results: List[RequestResult] = []
    pipeline_free = 0.0

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

        t_start = max(batch[0].arrival_time, pipeline_free)

        if len(batch) == 1:
            # Fast path: no padding overhead for a single request.
            r = batch[0]
            input_ids = tokenize_prompt(
                tokenizer, r.prompt_text, prefill_device, max_prompt_tokens
            )

            prefill_s, past, next_t = timed_prefill(model_prefill, input_ids, prefill_device)
            transfer_s, past_dec = time_transfer(past, decode_device, source_device=prefill_device)
            next_on_dec = next_t.to(decode_device)

            step_times, _ = timed_decode_steps(
                model_decode, past_dec, next_on_dec, r.output_tokens, decode_device
            )

            first_token_time = t_start + prefill_s + transfer_s + step_times[0]
            finish_time = t_start + prefill_s + transfer_s + sum(step_times)
            pipeline_free = finish_time

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
            # Batched path: pad prompts, single prefill forward, transfer full batched
            # KV cache to decode device, then single batched decode.
            prompts = [r.prompt_text for r in batch]
            input_ids, attn_mask = tokenize_batch(
                tokenizer, prompts, prefill_device, max_prompt_tokens
            )

            prefill_s, past_A, next_ts_A = timed_prefill_batch(
                model_prefill, input_ids, attn_mask, prefill_device
            )

            # Transfer the full batched KV cache (all B requests) from GPU A to GPU B.
            transfer_s, past_B = time_transfer(
                past_A, decode_device, source_device=prefill_device
            )
            next_ts_B = next_ts_A.to(decode_device)

            max_out = max(r.output_tokens for r in batch)
            step_times, _ = timed_decode_steps_batch(
                model_decode, past_B, next_ts_B, max_out, decode_device
            )

            # All requests in the batch share the same first-token time.
            first_token_time = t_start + prefill_s + transfer_s + step_times[0]

            for r in batch:
                finish_time = t_start + prefill_s + transfer_s + sum(step_times[: r.output_tokens])
                results.append(
                    RequestResult(
                        request_id=r.request_id,
                        arrival_time=r.arrival_time,
                        first_token_time=first_token_time,
                        finish_time=finish_time,
                        output_tokens=r.output_tokens,
                    )
                )

            pipeline_free = t_start + prefill_s + transfer_s + sum(step_times)

    return results


def load_two_models(
    model_name: str,
    prefill_device: torch.device,
    decode_device: torch.device,
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
    return model_prefill, model_decode, tokenizer
