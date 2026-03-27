"""
Stage B: measure prefill and decode step times on GPU using an open causal LM,
fit TimingModel coefficients, and save JSON for the simulator.

Requires: torch, transformers, datasets (see requirements.txt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.runtime.inference_core import timed_decode_steps, timed_prefill, tokenize_prompt
from src.stage_b.fitted_timing import (
    fit_prefill_linear,
    save_fitted_timing,
)
from src.stage_b.sharegpt_loader import iter_sharegpt_from_hf, iter_sharegpt_jsonl
from src.simulator.timing import TimingModel


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: CUDA not available; measurements will not reflect GPU timing.", file=sys.stderr)
    return torch.device("cpu")


def measure_one_sample(
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int,
    max_prompt_tokens: int,
) -> tuple[int, float, list[float]]:
    """Returns (prompt_token_count, prefill_seconds, decode_step_seconds_list)."""
    input_ids = tokenize_prompt(tokenizer, prompt_text, device, max_prompt_tokens)
    prompt_len = int(input_ids.shape[1])
    prefill_s, past, next_token = timed_prefill(model, input_ids, device)
    decode_times, _ = timed_decode_steps(
        model, past, next_token, max_new_tokens, device
    )
    return prompt_len, prefill_s, decode_times


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile prefill/decode on ShareGPT + fit TimingModel")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace causal LM id (open weights)",
    )
    parser.add_argument("--dataset", type=str, default="Aeala/ShareGPT_Vicuna_unfiltered")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--jsonl", type=str, default=None, help="Optional local ShareGPT JSONL instead of HF")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Decode steps to time per sample")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--out", type=str, default="results/fitted_timing.json")
    args = parser.parse_args()

    device = _pick_device()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)

    prompt_tokens_list: list[int] = []
    prefill_s_list: list[float] = []
    all_decode_steps: list[float] = []

    if args.jsonl:
        samples = list(iter_sharegpt_jsonl(args.jsonl, max_samples=args.max_samples))
    else:
        samples = list(
            iter_sharegpt_from_hf(
                dataset_name=args.dataset,
                split=args.split,
                max_samples=args.max_samples * 3,  # loader may skip short
            )
        )[: args.max_samples]

    if not samples:
        print("No ShareGPT samples loaded; check dataset name or --jsonl path.", file=sys.stderr)
        sys.exit(1)

    for i, s in enumerate(samples):
        try:
            plen, pre_s, dec = measure_one_sample(
                model,
                tokenizer,
                s.text,
                device,
                max_new_tokens=args.max_new_tokens,
                max_prompt_tokens=args.max_prompt_tokens,
            )
        except Exception as e:
            print(f"skip sample {i}: {e}", file=sys.stderr)
            continue
        prompt_tokens_list.append(plen)
        prefill_s_list.append(pre_s)
        all_decode_steps.extend(dec)

    if len(prompt_tokens_list) < 5:
        print("Too few successful samples; relax --max-samples or max length.", file=sys.stderr)
        sys.exit(1)

    prefill_base, prefill_per_token = fit_prefill_linear(prompt_tokens_list, prefill_s_list)
    mean_decode = sum(all_decode_steps) / max(len(all_decode_steps), 1)

    # Map mean decode step to decode_base; decode_per_token left 0 (constant per step with KV cache).
    timing = TimingModel(
        prefill_base=prefill_base,
        prefill_per_token=prefill_per_token,
        decode_base=mean_decode,
        decode_per_token=0.0,
        transfer_per_prompt_token=1.5e-5,  # placeholder; override with microbenchmark if needed
        colocated_interference=1.35,
        prefill_capacity_multiplier=1.0,
        decode_capacity_multiplier=1.0,
    )

    meta = {
        "model": args.model,
        "dataset": args.dataset if not args.jsonl else args.jsonl,
        "num_samples_profiled": len(prompt_tokens_list),
        "decode_steps_per_sample": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "device": str(device),
    }
    out_path = Path(args.out)
    save_fitted_timing(out_path, timing, meta)
    print(f"Wrote {out_path}")
    print(f"  prefill_base={timing.prefill_base:.6f} prefill_per_token={timing.prefill_per_token:.9f}")
    print(f"  decode_base (mean step)={timing.decode_base:.6f}")


if __name__ == "__main__":
    main()
