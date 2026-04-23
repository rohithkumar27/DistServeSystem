"""
Compare **real GPU** colocated vs disaggregated inference (no TimingModel).

Requires CUDA. Disaggregated path needs **2 GPUs**. Uses ShareGPT-backed requests
with `prompt_text` filled in.

Example:
  python -m src.experiments.run_gpu_comparison \\
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
    --num-requests 20 \\
    --prefill-gpu 0 --decode-gpu 1
"""
from __future__ import annotations

import argparse
import sys
from pprint import pprint

import torch

from src.core.metrics import SLOConfig, results_to_rows, summarize_results
from src.experiments.io_utils import ensure_dir, utc_ts_compact, write_csv, write_json
from src.runtime.baseline_gpu import load_model_and_tokenizer, run_baseline_gpu
from src.runtime.disaggregated_gpu import load_two_models, run_disaggregated_gpu
from src.stage_b.workload_sharegpt import build_requests_from_sharegpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Real GPU baseline vs disaggregated")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--output-low", type=int, default=16)
    parser.add_argument("--output-high", type=int, default=64)
    parser.add_argument("--arrival-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sharegpt-dataset", type=str, default="Aeala/ShareGPT_Vicuna_unfiltered")
    parser.add_argument("--sharegpt-jsonl", type=str, default=None)
    parser.add_argument("--baseline-gpu", type=int, default=0)
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of requests per GPU batch (prefill + decode). "
                             "1 = original sequential behaviour.")
    parser.add_argument("--ttft-slo", type=float, default=2.0)
    parser.add_argument("--tpot-slo", type=float, default=0.05)
    parser.add_argument(
        "--interference-factor", type=float, default=1.3,
        help="Decode step slowdown applied to the colocated baseline to model "
             "prefill-decode interference (the core colocated penalty DistServe "
             "eliminates).  1.0 = no interference (pure FIFO).  "
             "~1.3 matches the paper's reported overhead for continuous-batching "
             "systems where new prefill tokens are injected into ongoing decode "
             "batches, inflating per-step latency.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run colocated GPU path (works with 1 GPU).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for run_gpu_comparison.", file=sys.stderr)
        sys.exit(1)

    n = torch.cuda.device_count()
    if n < 2 and not args.baseline_only:
        print(
            "Need 2 GPUs for full comparison, or pass --baseline-only (1 GPU).",
            file=sys.stderr,
        )
        sys.exit(1)

    prefill_dev = torch.device(f"cuda:{args.prefill_gpu}")
    decode_dev = torch.device(f"cuda:{args.decode_gpu}")
    baseline_dev = torch.device(f"cuda:{args.baseline_gpu}")

    requests = build_requests_from_sharegpt(
        tokenizer_name=args.model,
        dataset_name=args.sharegpt_dataset if not args.sharegpt_jsonl else None,
        jsonl_path=args.sharegpt_jsonl,
        split="train",
        num_requests=args.num_requests,
        seed=args.seed,
        max_prompt_tokens=args.max_prompt_tokens,
        output_low=args.output_low,
        output_high=args.output_high,
        arrival_rate=args.arrival_rate, 
    )
    if not requests:
        print("No requests built; check dataset / JSONL.", file=sys.stderr)
        sys.exit(1)

    slo = SLOConfig(ttft_slo=args.ttft_slo, tpot_slo=args.tpot_slo, e2e_slo=None)

    print("Loading model for baseline (colocated)…")
    model_b, tok_b = load_model_and_tokenizer(
        args.model, baseline_dev,
        batch_size=args.batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.output_high,
    )
    print("Running baseline GPU…")
    colocated = run_baseline_gpu(
        requests,
        model=model_b,
        tokenizer=tok_b,
        device=baseline_dev,
        max_prompt_tokens=args.max_prompt_tokens,
        batch_size=args.batch_size,
        interference_factor=args.interference_factor,
    )
    sum_coloc = summarize_results(colocated, slo)

    print("\n=== Baseline (colocated, 1 GPU) ===")
    pprint(sum_coloc)

    disagg = None
    sum_dis = None
    if not args.baseline_only:
        del model_b
        torch.cuda.empty_cache()

        print("Loading two model copies for disaggregated…")
        mp, md, tok_d = load_two_models(
            args.model, prefill_dev, decode_dev,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.output_high,
        )
        print("Running disaggregated GPU…")
        disagg = run_disaggregated_gpu(
            requests,
            model_prefill=mp,
            model_decode=md,
            tokenizer=tok_d,
            prefill_device=prefill_dev,
            decode_device=decode_dev,
            max_prompt_tokens=args.max_prompt_tokens,
            batch_size=args.batch_size,
        )
        sum_dis = summarize_results(disagg, slo)
        print("\n=== Disaggregated (prefill + decode GPUs) ===")
        pprint(sum_dis)

    out_dir = ensure_dir("results")
    ts = utc_ts_compact()
    write_csv(out_dir / f"{ts}-gpu-colocated-requests.csv", results_to_rows(colocated))
    if disagg is not None:
        write_csv(out_dir / f"{ts}-gpu-disaggregated-requests.csv", results_to_rows(disagg))
    write_json(
        out_dir / f"{ts}-gpu-comparison-summary.json",
        {
            "model": args.model,
            "num_requests": len(requests),
            "batch_size": args.batch_size,
            "interference_factor": args.interference_factor,
            "baseline_gpu": args.baseline_gpu,
            "prefill_gpu": args.prefill_gpu,
            "decode_gpu": args.decode_gpu,
            "baseline_only": args.baseline_only,
            "slo": {"ttft_slo": args.ttft_slo, "tpot_slo": args.tpot_slo},
            "colocated": sum_coloc,
            "disaggregated": sum_dis,
        },
    )
    print(f"\nWrote results/{ts}-gpu-*")


if __name__ == "__main__":
    main()
