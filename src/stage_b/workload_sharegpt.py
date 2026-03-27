from __future__ import annotations

import random
from typing import List

from src.core.request import Request
from src.stage_b.sharegpt_loader import iter_sharegpt_from_hf, iter_sharegpt_jsonl


def build_requests_from_sharegpt(
    *,
    tokenizer_name: str,
    dataset_name: str | None,
    jsonl_path: str | None,
    split: str,
    num_requests: int,
    seed: int,
    max_prompt_tokens: int,
    output_low: int,
    output_high: int,
    arrival_rate: float,
) -> List[Request]:
    from transformers import AutoTokenizer

    if jsonl_path:
        samples = list(iter_sharegpt_jsonl(jsonl_path, max_samples=num_requests * 4))
    else:
        dataset_name = dataset_name or "Aeala/ShareGPT_Vicuna_unfiltered"
        samples = list(
            iter_sharegpt_from_hf(
                dataset_name=dataset_name,
                split=split,
                max_samples=num_requests * 4,
            )
        )

    rng = random.Random(seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    requests: List[Request] = []
    t = 0.0
    rid = 0
    for s in samples:
        if len(requests) >= num_requests:
            break
        enc = tokenizer(s.text, truncation=True, max_length=max_prompt_tokens, return_tensors=None)
        ids = enc["input_ids"]
        prompt_tokens = len(ids)
        if prompt_tokens < 6:
            continue
        if arrival_rate > 0:
            t += rng.expovariate(arrival_rate)
        else:
            t += 0.0
        out_tok = rng.randint(output_low, output_high)
        requests.append(
            Request(
                request_id=rid,
                arrival_time=float(t),
                prompt_tokens=prompt_tokens,
                output_tokens=out_tok,
                prompt_text=s.text,
            )
        )
        rid += 1

    return requests
