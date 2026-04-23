"""
3D workload sweep on real GPUs: batch_size x num_requests x arrival_rate,
for both colocated and disaggregated designs.

For each (design, num_requests, arrival_rate, batch_size) combination this
script:
  - Builds a fresh ShareGPT workload with the requested (num_requests, arrival_rate)
  - Runs inference and computes the full SLO summary
  - Writes a per-request CSV (one file per condition)
  - Appends a per-condition row to a master CSV
  - Appends a JSON line to conditions.jsonl with the full metric dict
  - Prints a progress line for live monitoring

Model copies are loaded ONCE per design and reused across every condition of
that design, so the cost of this sweep is dominated by inference, not model I/O.

Output tree (under results/milestone/workload_sweep/):
  summary.csv                   master table (one row per condition)
  conditions.jsonl              full metric dicts, append-only
  meta.json                     run configuration
  run.log                       captured stdout
  requests/<cond_key>.csv       per-request detail per condition
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.core.metrics import SLOConfig, results_to_rows, summarize_results
from src.experiments.io_utils import ensure_dir, write_csv, write_json
from src.runtime.baseline_gpu import load_model_and_tokenizer, run_baseline_gpu
from src.runtime.disaggregated_gpu import load_two_models, run_disaggregated_gpu
from src.stage_b.workload_sharegpt import build_requests_from_sharegpt


# ---------- helpers ----------------------------------------------------------


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _cond_key(design: str, num_requests: int, arrival_rate: float, batch_size: int) -> str:
    return f"{design}_n{num_requests}_r{arrival_rate:g}_bs{batch_size}"


def _summary_row(
    design: str,
    batch_size: int,
    num_requests: int,
    arrival_rate: float,
    s: dict,
    elapsed_s: float,
    prompt_tokens_stats: dict,
) -> dict:
    return {
        "design":           design,
        "batch_size":       batch_size,
        "num_requests":     num_requests,
        "arrival_rate":     arrival_rate,
        "count":            s["count"],
        "goodput":          s["goodput"],
        "mean_ttft":        s["mean_ttft"],
        "p50_ttft":         s["p50_ttft"],
        "p95_ttft":         s["p95_ttft"],
        "p99_ttft":         s["p99_ttft"],
        "mean_tpot":        s["mean_tpot"],
        "p95_tpot":         s["p95_tpot"],
        "mean_e2e":         s["mean_e2e_latency"],
        "p95_e2e":          s["p95_e2e_latency"],
        "p99_e2e":          s["p99_e2e_latency"],
        "throughput_req_s": s["throughput_req_per_s"],
        "throughput_tok_s": s["throughput_tok_per_s"],
        "elapsed_s":        elapsed_s,
        "prompt_tokens_mean": prompt_tokens_stats["mean"],
        "prompt_tokens_min":  prompt_tokens_stats["min"],
        "prompt_tokens_max":  prompt_tokens_stats["max"],
    }


def _prompt_stats(requests) -> dict:
    toks = [r.prompt_tokens for r in requests]
    return {"mean": sum(toks) / len(toks), "min": min(toks), "max": max(toks)}


# ---------- logging -----------------------------------------------------------


class TeeLogger:
    """Write to stdout AND to a file. Used so the printed log is preserved."""

    def __init__(self, path: Path) -> None:
        self.stream = open(path, "a", encoding="utf-8", buffering=1)

    def log(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.stream.write(msg + "\n")

    def close(self) -> None:
        self.stream.close()


# ---------- main --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU sweep across batch_size x num_requests x arrival_rate"
    )
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--output-low", type=int, default=8)
    parser.add_argument("--output-high", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sharegpt-dataset", type=str,
                        default="Aeala/ShareGPT_Vicuna_unfiltered")
    parser.add_argument("--sharegpt-jsonl", type=str, default=None)
    parser.add_argument("--baseline-gpu", type=int, default=0)
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                        help="Comma-separated batch sizes to sweep")
    parser.add_argument("--num-requests", type=str, default="16,32,64",
                        help="Comma-separated num_requests to sweep")
    parser.add_argument("--arrival-rates", type=str, default="0.0,1.0,2.0,4.0",
                        help="Comma-separated arrival rates (req/s). "
                             "0 = all at t=0 (burst).")
    parser.add_argument("--ttft-slo", type=float, default=2.0)
    parser.add_argument("--tpot-slo", type=float, default=0.05)
    parser.add_argument("--interference-factor", type=float, default=1.3,
                        help="Decode-step slowdown multiplier for the colocated baseline, "
                             "modelling prefill-decode GPU contention (1.0 = no penalty, "
                             "1.3 = DistServe paper's reported ~30%% overhead). "
                             "Disaggregated is unaffected — it runs prefill and decode on "
                             "separate GPUs so there is no contention.")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run colocated sweep (use on 1-GPU nodes)")
    parser.add_argument("--out-dir", type=str,
                        default="results/milestone/workload_sweep",
                        help="Output directory for all sweep artifacts")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This script requires a GPU.", file=sys.stderr)
        sys.exit(1)

    batch_sizes = _parse_csv_ints(args.batch_sizes)
    num_requests_list = _parse_csv_ints(args.num_requests)
    arrival_rates = _parse_csv_floats(args.arrival_rates)

    slo = SLOConfig(ttft_slo=args.ttft_slo, tpot_slo=args.tpot_slo)

    out_dir = ensure_dir(args.out_dir)
    requests_dir = ensure_dir(out_dir / "requests")
    summary_csv_path = out_dir / "summary.csv"
    jsonl_path = out_dir / "conditions.jsonl"
    meta_path = out_dir / "meta.json"
    log_path = out_dir / "run.log"

    logger = TeeLogger(log_path)
    log = logger.log

    started_at = datetime.now(timezone.utc).isoformat()
    total_conditions = (
        len(batch_sizes) * len(num_requests_list) * len(arrival_rates)
        * (1 if args.baseline_only else 2)
    )

    log("=" * 72)
    log("GPU Workload Sweep — batch_size x num_requests x arrival_rate")
    log("=" * 72)
    log(f"Started (UTC):    {started_at}")
    log(f"Model:            {args.model}")
    log(f"Batch sizes:      {batch_sizes}")
    log(f"Num requests:     {num_requests_list}")
    log(f"Arrival rates:    {arrival_rates}")
    log(f"Output token rng: [{args.output_low}, {args.output_high}]")
    log(f"Max prompt toks:  {args.max_prompt_tokens}")
    log(f"SLO:              TTFT<={args.ttft_slo}s TPOT<={args.tpot_slo}s")
    log(f"Interference:     {args.interference_factor}x  (colocated decode slowdown)")
    log(f"Seed:             {args.seed}")
    log(f"Baseline GPU:     cuda:{args.baseline_gpu}")
    log(f"Prefill/decode:   cuda:{args.prefill_gpu} / cuda:{args.decode_gpu}")
    log(f"Baseline only:    {args.baseline_only}")
    log(f"Total conditions: {total_conditions}")
    log(f"Output dir:       {out_dir}")
    log("")

    # Persist run metadata upfront (overwritten at end with final stats).
    write_json(meta_path, {
        "experiment":      "gpu_workload_sweep",
        "started_utc":     started_at,
        "model":           args.model,
        "batch_sizes":     batch_sizes,
        "num_requests":    num_requests_list,
        "arrival_rates":   arrival_rates,
        "output_range":    [args.output_low, args.output_high],
        "max_prompt_tokens": args.max_prompt_tokens,
        "slo":             {"ttft_slo": args.ttft_slo, "tpot_slo": args.tpot_slo},
        "seed":            args.seed,
        "baseline_gpu":    args.baseline_gpu,
        "prefill_gpu":     args.prefill_gpu,
        "decode_gpu":      args.decode_gpu,
        "baseline_only":   args.baseline_only,
        "dataset":         args.sharegpt_jsonl or args.sharegpt_dataset,
        "total_conditions": total_conditions,
    })

    # Fresh log files (only clear if they exist — conditions.jsonl is append-only)
    jsonl_path.write_text("", encoding="utf-8")

    rows: list[dict] = []
    jsonl_fh = jsonl_path.open("a", encoding="utf-8", buffering=1)
    cond_idx = 0
    sweep_t0 = time.time()

    # Precompute workloads once per (num_requests, arrival_rate) pair so both
    # designs and all batch sizes see identical inputs.
    workload_cache: dict[tuple[int, float], list] = {}

    def get_workload(num_requests: int, arrival_rate: float):
        key = (num_requests, arrival_rate)
        if key not in workload_cache:
            reqs = build_requests_from_sharegpt(
                tokenizer_name=args.model,
                dataset_name=args.sharegpt_dataset if not args.sharegpt_jsonl else None,
                jsonl_path=args.sharegpt_jsonl,
                split="train",
                num_requests=num_requests,
                seed=args.seed,
                max_prompt_tokens=args.max_prompt_tokens,
                output_low=args.output_low,
                output_high=args.output_high,
                arrival_rate=arrival_rate,
            )
            if not reqs:
                raise RuntimeError(
                    f"No requests built for num_requests={num_requests}, "
                    f"arrival_rate={arrival_rate}"
                )
            workload_cache[key] = reqs
            pt = _prompt_stats(reqs)
            log(
                f"  workload n={num_requests} r={arrival_rate:g} built: "
                f"{len(reqs)} reqs, prompt_tokens mean/min/max "
                f"{pt['mean']:.0f}/{pt['min']}/{pt['max']}"
            )
        return workload_cache[key]

    def run_condition(
        design: str, requests, bs: int, num_requests: int, arrival_rate: float, runner_fn
    ) -> None:
        nonlocal cond_idx
        cond_idx += 1
        key = _cond_key(design, num_requests, arrival_rate, bs)
        t0 = time.time()
        log(
            f"[{cond_idx}/{total_conditions}] {design:<13} "
            f"n={num_requests:<4} r={arrival_rate:<4g} bs={bs:<3} ... ",
        )
        result = runner_fn(requests, bs)
        elapsed = time.time() - t0
        s = summarize_results(result, slo)
        pt = _prompt_stats(requests)
        row = _summary_row(design, bs, num_requests, arrival_rate, s, elapsed, pt)
        rows.append(row)

        # Per-request CSV: one file per condition
        write_csv(requests_dir / f"{key}.csv", results_to_rows(result))

        # Append full condition dict to JSONL (one line per condition)
        jsonl_fh.write(json.dumps({
            "design":       design,
            "batch_size":   bs,
            "num_requests": num_requests,
            "arrival_rate": arrival_rate,
            "elapsed_s":    elapsed,
            "prompt_tokens_stats": pt,
            "summary":      s,
        }) + "\n")

        # Incremental CSV so partial progress survives crashes
        write_csv(summary_csv_path, rows)

        log(
            f"    -> goodput={s['goodput']:.3f}  mean_ttft={s['mean_ttft']:.3f}s  "
            f"p95_ttft={s['p95_ttft']:.3f}s  thr={s['throughput_req_per_s']:.2f} req/s  "
            f"tok/s={s['throughput_tok_per_s']:.1f}  elapsed={elapsed:.1f}s"
        )

    # ---------- Colocated phase ----------
    baseline_dev = torch.device(f"cuda:{args.baseline_gpu}")
    log(f"\nLoading colocated model on cuda:{args.baseline_gpu} ...")
    model_b, tok_b = load_model_and_tokenizer(args.model, baseline_dev)
    log("\n--- Colocated phase ---")
    for n in num_requests_list:
        for r in arrival_rates:
            reqs = get_workload(n, r)
            for bs in batch_sizes:
                run_condition(
                    "colocated", reqs, bs, n, r,
                    runner_fn=lambda rs, bs: run_baseline_gpu(
                        rs,
                        model=model_b,
                        tokenizer=tok_b,
                        device=baseline_dev,
                        max_prompt_tokens=args.max_prompt_tokens,
                        batch_size=bs,
                        interference_factor=args.interference_factor,
                    ),
                )
    del model_b
    torch.cuda.empty_cache()

    # ---------- Disaggregated phase ----------
    if not args.baseline_only:
        if torch.cuda.device_count() < 2:
            log(
                "\nOnly 1 GPU visible — skipping disaggregated phase. "
                "Pass --baseline-only to suppress this warning."
            )
        else:
            prefill_dev = torch.device(f"cuda:{args.prefill_gpu}")
            decode_dev  = torch.device(f"cuda:{args.decode_gpu}")
            log(f"\nLoading two model copies (prefill cuda:{args.prefill_gpu}, "
                f"decode cuda:{args.decode_gpu}) ...")
            mp, md, tok_d = load_two_models(args.model, prefill_dev, decode_dev)
            log("\n--- Disaggregated phase ---")
            for n in num_requests_list:
                for r in arrival_rates:
                    reqs = get_workload(n, r)
                    for bs in batch_sizes:
                        run_condition(
                            "disaggregated", reqs, bs, n, r,
                            runner_fn=lambda rs, bs: run_disaggregated_gpu(
                                rs,
                                model_prefill=mp,
                                model_decode=md,
                                tokenizer=tok_d,
                                prefill_device=prefill_dev,
                                decode_device=decode_dev,
                                max_prompt_tokens=args.max_prompt_tokens,
                                batch_size=bs,
                            ),
                        )

    jsonl_fh.close()
    write_csv(summary_csv_path, rows)

    total_elapsed = time.time() - sweep_t0
    finished_at = datetime.now(timezone.utc).isoformat()

    # Final meta with run stats
    meta = json.loads(meta_path.read_text())
    meta["finished_utc"] = finished_at
    meta["total_elapsed_s"] = total_elapsed
    meta["conditions_completed"] = len(rows)
    write_json(meta_path, meta)

    log("")
    log("=" * 72)
    log(f"Completed {len(rows)}/{total_conditions} conditions "
        f"in {total_elapsed:.1f}s")
    log(f"summary.csv:    {summary_csv_path}")
    log(f"conditions.jsonl: {jsonl_path}")
    log(f"per-request CSVs: {requests_dir}/")
    log(f"meta.json:      {meta_path}")
    log(f"log:            {log_path}")
    log("=" * 72)
    logger.close()


if __name__ == "__main__":
    main()
