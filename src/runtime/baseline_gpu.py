"""
Colocated prefill + decode on **one GPU** — real `forward` times (no TimingModel).

Matches DistServe’s “monolithic” case: one pool runs both phases on the same device,
so prefill and decode contend on that GPU (here: strict FIFO, one request at a time).
"""
from __future__ import annotations

from typing import Iterable, List

import torch

from src.core.request import Request, RequestResult
from src.runtime.inference_core import timed_decode_steps, timed_prefill, tokenize_prompt


def run_baseline_gpu(
    requests: Iterable[Request],
    *,
    model,
    tokenizer,
    device: torch.device,
    max_prompt_tokens: int = 2048,
) -> List[RequestResult]:
    """
    For each request in arrival order: run prefill then greedy decode on `device`.
    Timeline: next request starts at max(arrival, previous finish) — single GPU queue.
    """
    req_list = sorted(list(requests), key=lambda r: r.arrival_time)
    results: List[RequestResult] = []
    gpu_free_time = 0.0

    for r in req_list:
        if not r.prompt_text:
            raise ValueError(
                "run_baseline_gpu requires Request.prompt_text. "
                "Build workload with build_requests_from_sharegpt(...) or set prompt_text."
            )

        t_start = max(r.arrival_time, gpu_free_time)
        input_ids = tokenize_prompt(tokenizer, r.prompt_text, device, max_prompt_tokens)

        prefill_s, past, next_t = timed_prefill(model, input_ids, device)
        step_times, _ = timed_decode_steps(
            model, past, next_t, r.output_tokens, device
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
