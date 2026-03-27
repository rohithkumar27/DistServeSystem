"""
Disaggregated prefill vs decode on **two devices** — real forwards + measured KV handoff.

Matches DistServe’s idea: prefill worker produces KV on `prefill_device`, tensors are
moved to `decode_device`, decode worker runs autoregressive steps. Two full model
copies with the **same** weights (like two role-specific replicas).

Requires at least 2 CUDA devices for meaningful separation; with one GPU this path
cannot isolate hardware pools.
"""
from __future__ import annotations

from typing import Iterable, List
import torch
from src.core.request import Request, RequestResult
from src.runtime.inference_core import (
    timed_decode_steps,
    timed_prefill,
    time_transfer,
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
) -> List[RequestResult]:
    """
    Sequential pipeline: prefill on GPU A → copy KV to GPU B → decode on B.
    Next request starts when the previous decode finishes (single stream; no overlap).
    """
    if prefill_device == decode_device:
        raise ValueError(
            "Disaggregated GPU mode requires prefill_device != decode_device "
            "(e.g. cuda:0 and cuda:1)."
        )

    req_list = sorted(list(requests), key=lambda r: r.arrival_time)
    results: List[RequestResult] = []
    pipeline_free = 0.0

    for r in req_list:
        if not r.prompt_text:
            raise ValueError(
                "run_disaggregated_gpu requires Request.prompt_text "
                "(use build_requests_from_sharegpt)."
            )

        t_start = max(r.arrival_time, pipeline_free)

        input_ids = tokenize_prompt(
            tokenizer, r.prompt_text, prefill_device, max_prompt_tokens
        )

        prefill_s, past, next_t = timed_prefill(model_prefill, input_ids, prefill_device)
        transfer_s, past_dec = time_transfer(past, decode_device, source_device=prefill_device)
        next_on_dec = next_t.to(decode_device)

        step_times, _ = timed_decode_steps(
            model_decode,
            past_dec,
            next_on_dec,
            r.output_tokens,
            decode_device,
        )

        # First token emitted after prefill + transfer + first decode step (on decode pool).
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
