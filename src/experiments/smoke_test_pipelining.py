"""
Smoke test — verify that the pipelined disaggregated runner actually overlaps
prefill (GPU A) with decode (GPU B) across batches.

Runs a small fixed workload (12 requests, batch_size=2, burst arrivals) through
both colocated and disaggregated runners, then prints:

  - throughput (req/s, tok/s) for each design
  - end-to-end wall time (first arrival → last finish)
  - the speedup ratio (disagg / coloc on throughput)
  - a PASS/FAIL verdict

Expected behaviour after pipelining is enabled:
  disaggregated throughput > colocated throughput          → PASS
  disaggregated throughput within 5% of colocated          → INCONCLUSIVE
  disaggregated throughput materially lower than colocated → FAIL

Usage:
  python -m src.experiments.smoke_test_pipelining
  python -m src.experiments.smoke_test_pipelining --num-requests 16 --batch-size 4
  python -m src.experiments.smoke_test_pipelining --baseline-only   # single-GPU sanity only
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from src.core.metrics import SLOConfig, summarize_results
from src.runtime.baseline_gpu import load_model_and_tokenizer, run_baseline_gpu
from src.runtime.disaggregated_gpu import load_two_models, run_disaggregated_gpu
from src.stage_b.workload_sharegpt import build_requests_from_sharegpt


def _time(fn):
    t0 = time.time()
    out = fn()
    return time.time() - t0, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for disaggregated pipelining")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--num-requests", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Small batch size so several batches run (that's where overlap helps)")
    parser.add_argument("--output-low",  type=int, default=32)
    parser.add_argument("--output-high", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--arrival-rate", type=float, default=0.0,
                        help="0 = burst; keeps the test fast and shows pipelining clearly")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sharegpt-dataset", type=str,
                        default="Aeala/ShareGPT_Vicuna_unfiltered")
    parser.add_argument("--sharegpt-jsonl", type=str, default=None)
    parser.add_argument("--baseline-gpu", type=int, default=0)
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu",  type=int, default=1)
    parser.add_argument("--ttft-slo", type=float, default=2.0)
    parser.add_argument("--tpot-slo", type=float, default=0.05)
    parser.add_argument("--interference-factor", type=float, default=1.0,
                        help="Decode slowdown for colocated (1.0 = pure pipelining test, "
                             "1.3 = full DistServe interference model)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Skip disaggregated; sanity-check colocated only")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-batch timing breakdown (prefill/transfer/decode)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for smoke test.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("  Pipelining smoke test")
    print("=" * 70)
    print(f"  model:        {args.model}")
    print(f"  num_requests: {args.num_requests}")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  output range: [{args.output_low}, {args.output_high}]")
    print(f"  arrival:      {args.arrival_rate} req/s (0 = burst)")
    print("")

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
        print("ERROR: no requests built.", file=sys.stderr)
        sys.exit(1)

    print(f"Built {len(requests)} requests; prompt tokens "
          f"min/mean/max = {min(r.prompt_tokens for r in requests)}/"
          f"{sum(r.prompt_tokens for r in requests)/len(requests):.0f}/"
          f"{max(r.prompt_tokens for r in requests)}")
    print(f"Output tokens mean = {sum(r.output_tokens for r in requests)/len(requests):.1f}")
    print("")

    slo = SLOConfig(ttft_slo=args.ttft_slo, tpot_slo=args.tpot_slo)

    # ---- Colocated ----
    baseline_dev = torch.device(f"cuda:{args.baseline_gpu}")
    print(f"Loading colocated model on cuda:{args.baseline_gpu} ...")
    model_b, tok_b = load_model_and_tokenizer(
        args.model, baseline_dev,
        batch_size=args.batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.output_high,
    )
    print(f"Running colocated (batch_size={args.batch_size}) ...")
    wall_coloc, results_coloc = _time(lambda: run_baseline_gpu(
        requests, model=model_b, tokenizer=tok_b, device=baseline_dev,
        max_prompt_tokens=args.max_prompt_tokens, batch_size=args.batch_size,
        interference_factor=args.interference_factor,
        verbose=args.verbose,
    ))
    sum_coloc = summarize_results(results_coloc, slo)
    del model_b
    torch.cuda.empty_cache()

    # ---- Disaggregated ----
    sum_dis = None
    wall_disagg = 0.0
    if not args.baseline_only:
        if torch.cuda.device_count() < 2:
            print("Only 1 GPU visible — can't run disaggregated half of the test.")
            print("Pass --baseline-only or run on a 2-GPU node.")
        else:
            prefill_dev = torch.device(f"cuda:{args.prefill_gpu}")
            decode_dev  = torch.device(f"cuda:{args.decode_gpu}")
            print(f"Loading two model copies (cuda:{args.prefill_gpu} prefill, "
                  f"cuda:{args.decode_gpu} decode) ...")
            mp, md, tok_d = load_two_models(
                args.model, prefill_dev, decode_dev,
                batch_size=args.batch_size,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.output_high,
            )
            print(f"Running disaggregated (batch_size={args.batch_size}) ...")
            wall_disagg, results_disagg = _time(lambda: run_disaggregated_gpu(
                requests, model_prefill=mp, model_decode=md, tokenizer=tok_d,
                prefill_device=prefill_dev, decode_device=decode_dev,
                max_prompt_tokens=args.max_prompt_tokens, batch_size=args.batch_size,
                verbose=args.verbose,
            ))
            sum_dis = summarize_results(results_disagg, slo)

    # ---- Report ----
    print("")
    print("-" * 70)
    hdr = f"{'metric':<25} {'colocated':>14} {'disaggregated':>16} {'delta':>12}"
    print(hdr)
    print("-" * len(hdr))

    def row(name, c, d, fmt=".3f", unit=""):
        if d is None:
            print(f"{name:<25} {c:>14{fmt}}{unit} {'--':>15} {'--':>12}")
            return
        delta = d - c
        pct = (d / c - 1.0) * 100.0 if c != 0 else float("nan")
        print(f"{name:<25} {c:>14{fmt}}{unit} {d:>15{fmt}}{unit} "
              f"{delta:>+9.3f} ({pct:+5.1f}%)")

    row("count",                 sum_coloc["count"],                     sum_dis["count"]                     if sum_dis else None, ".0f")
    row("mean_ttft (s)",         sum_coloc["mean_ttft"],                 sum_dis["mean_ttft"]                 if sum_dis else None)
    row("p95_ttft (s)",          sum_coloc["p95_ttft"],                  sum_dis["p95_ttft"]                  if sum_dis else None)
    row("mean_tpot (s)",         sum_coloc["mean_tpot"],                 sum_dis["mean_tpot"]                 if sum_dis else None, ".4f")
    row("mean_e2e (s)",          sum_coloc["mean_e2e_latency"],          sum_dis["mean_e2e_latency"]          if sum_dis else None)
    row("goodput",               sum_coloc["goodput"],                   sum_dis["goodput"]                   if sum_dis else None)
    row("throughput (req/s)",    sum_coloc["throughput_req_per_s"],      sum_dis["throughput_req_per_s"]      if sum_dis else None, ".2f")
    row("throughput (tok/s)",    sum_coloc["throughput_tok_per_s"],      sum_dis["throughput_tok_per_s"]      if sum_dis else None, ".1f")
    row("wall time (s)",         wall_coloc,                             wall_disagg if sum_dis else None,    ".2f")
    print("-" * len(hdr))

    if sum_dis is None:
        print("\nColocated ran fine. Rerun on 2 GPUs to exercise the disaggregated path.")
        return

    thr_c = sum_coloc["throughput_req_per_s"]
    thr_d = sum_dis["throughput_req_per_s"]
    speedup = thr_d / thr_c if thr_c > 0 else float("nan")

    print("")
    print(f"Throughput speedup (disagg / coloc): {speedup:.3f}x "
          f"({(speedup - 1) * 100:+.1f}%)")
    print("")

    if speedup >= 1.10:
        print("VERDICT: PASS — pipelining is producing a measurable throughput gain.")
    elif speedup >= 0.95:
        print("VERDICT: INCONCLUSIVE — gain within timing noise. "
              "Try a larger workload or decode-heavier output range.")
    else:
        print("VERDICT: FAIL — disaggregated is slower than colocated. "
              "Pipelining is not taking effect. Check disaggregated_gpu.py.")
    print("")


if __name__ == "__main__":
    main()
