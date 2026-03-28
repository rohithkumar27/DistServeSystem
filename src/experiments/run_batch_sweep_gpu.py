"""
GPU batch-size sweep: colocated vs disaggregated at batch_size = [1, 2, 4, 8].

Uses real HuggingFace forward passes on CUDA. Measures how batching affects throughput
and latency for each design:
  - Colocated: single GPU, prefill+decode share resources
  - Disaggregated: prefill on GPU A, KV transfer, decode on GPU B

Requires at least 1 GPU for colocated sweep; 2 GPUs for disaggregated.
Use --baseline-only on a 1-GPU node.

Output: results/milestone/gpu_batch_sweep.csv + gpu_batch_sweep_meta.json
"""
from __future__ import annotations

import argparse
import sys

import torch

from src.core.metrics import SLOConfig, summarize_results
from src.experiments.io_utils import ensure_dir, write_csv, write_json
from src.runtime.baseline_gpu import load_model_and_tokenizer, run_baseline_gpu
from src.runtime.disaggregated_gpu import load_two_models, run_disaggregated_gpu
from src.stage_b.workload_sharegpt import build_requests_from_sharegpt


def _row(design: str, batch_size: int, s: dict) -> dict:
    return {
        "design":           design,
        "batch_size":       batch_size,
        "goodput":          s["goodput"],
        "mean_ttft":        s["mean_ttft"],
        "p95_ttft":         s["p95_ttft"],
        "p99_ttft":         s["p99_ttft"],
        "mean_e2e":         s["mean_e2e_latency"],
        "p99_e2e":          s["p99_e2e_latency"],
        "throughput_req_s": s["throughput_req_per_s"],
        "throughput_tok_s": s["throughput_tok_per_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU batch-size sweep: colocated vs disaggregated")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--num-requests", type=int, default=32,
                        help="Number of requests per run (keep small for fast sweeps)")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--output-low", type=int, default=8)
    parser.add_argument("--output-high", type=int, default=32)
    parser.add_argument("--arrival-rate", type=float, default=0.0,
                        help="Arrival rate for workload (0 = all arrive at t=0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sharegpt-dataset", type=str, default="Aeala/ShareGPT_Vicuna_unfiltered")
    parser.add_argument("--sharegpt-jsonl", type=str, default=None)
    parser.add_argument("--baseline-gpu", type=int, default=0)
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                        help="Comma-separated batch sizes to sweep")
    parser.add_argument("--ttft-slo", type=float, default=2.0)
    parser.add_argument("--tpot-slo", type=float, default=0.05)
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run colocated sweep (use on 1-GPU nodes)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This script requires a GPU.", file=sys.stderr)
        sys.exit(1)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    slo = SLOConfig(ttft_slo=args.ttft_slo, tpot_slo=args.tpot_slo)

    prefill_dev  = torch.device(f"cuda:{args.prefill_gpu}")
    decode_dev   = torch.device(f"cuda:{args.decode_gpu}")
    baseline_dev = torch.device(f"cuda:{args.baseline_gpu}")

    print(f"Building {args.num_requests} ShareGPT requests...")
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
        print("No requests built; check dataset/JSONL path.", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []

    # --- Colocated sweep (1 GPU) ---
    print(f"\nColocated sweep (batch_sizes={batch_sizes}) on cuda:{args.baseline_gpu}")
    model_b, tok_b = load_model_and_tokenizer(args.model, baseline_dev)

    for bs in batch_sizes:
        print(f"  batch_size={bs} ...", end=" ", flush=True)
        res = run_baseline_gpu(
            requests,
            model=model_b,
            tokenizer=tok_b,
            device=baseline_dev,
            max_prompt_tokens=args.max_prompt_tokens,
            batch_size=bs,
        )
        s = summarize_results(res, slo)
        rows.append(_row("colocated", bs, s))
        print(
            f"goodput={s['goodput']:.3f}  mean_ttft={s['mean_ttft']:.3f}s  "
            f"throughput={s['throughput_req_per_s']:.2f} req/s"
        )

    del model_b
    torch.cuda.empty_cache()

    # --- Disaggregated sweep (2 GPUs) ---
    if not args.baseline_only:
        if torch.cuda.device_count() < 2:
            print(
                "\nOnly 1 GPU visible — skipping disaggregated sweep. "
                "Rerun with --baseline-only to suppress this warning.",
                file=sys.stderr,
            )
        else:
            print(
                f"\nDisaggregated sweep (batch_sizes={batch_sizes}) "
                f"prefill=cuda:{args.prefill_gpu} decode=cuda:{args.decode_gpu}"
            )
            mp, md, tok_d = load_two_models(args.model, prefill_dev, decode_dev)

            for bs in batch_sizes:
                print(f"  batch_size={bs} ...", end=" ", flush=True)
                res = run_disaggregated_gpu(
                    requests,
                    model_prefill=mp,
                    model_decode=md,
                    tokenizer=tok_d,
                    prefill_device=prefill_dev,
                    decode_device=decode_dev,
                    max_prompt_tokens=args.max_prompt_tokens,
                    batch_size=bs,
                )
                s = summarize_results(res, slo)
                rows.append(_row("disaggregated", bs, s))
                print(
                    f"goodput={s['goodput']:.3f}  mean_ttft={s['mean_ttft']:.3f}s  "
                    f"throughput={s['throughput_req_per_s']:.2f} req/s"
                )

    out = ensure_dir("results/milestone")
    write_csv(out / "gpu_batch_sweep.csv", rows)
    write_json(out / "gpu_batch_sweep_meta.json", {
        "experiment":      "gpu_batch_sweep",
        "description":     "Throughput and latency vs batch_size for colocated and disaggregated GPU runtimes",
        "model":           args.model,
        "num_requests":    args.num_requests,
        "batch_sizes":     batch_sizes,
        "baseline_gpu":    args.baseline_gpu,
        "prefill_gpu":     args.prefill_gpu,
        "decode_gpu":      args.decode_gpu,
        "baseline_only":   args.baseline_only,
        "slo":             {"ttft_slo": args.ttft_slo, "tpot_slo": args.tpot_slo},
        "dataset":         args.sharegpt_jsonl or args.sharegpt_dataset,
        "max_prompt_tokens": args.max_prompt_tokens,
        "output_range":    [args.output_low, args.output_high],
    })
    print(f"\nWrote results/milestone/gpu_batch_sweep.csv + gpu_batch_sweep_meta.json")


if __name__ == "__main__":
    main()
