"""
Workload-mix sweep: interactive fraction vs goodput / TTFT.

Varies the fraction of short, latency-sensitive ("interactive") requests vs long-prompt
requests at a fixed arrival rate. This isolates *why* disaggregation helps:
  - At frac=0.0 (all long-prompt): no short requests to protect, gains are modest.
  - At frac=0.5–0.75 (mixed): long prefills block short requests → colocated suffers most.
  - At frac=1.0 (all interactive): interference disappears, advantage shrinks.

Output: results/milestone/workload_mix.csv + workload_mix_meta.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict

from src.core.metrics import SLOConfig, summarize_results
from src.experiments.io_utils import ensure_dir, write_csv, write_json
from src.simulator.baseline import run_colocated
from src.simulator.config import BatchingConfig
from src.simulator.disaggregated import run_disaggregated
from src.simulator.timing import TimingModel
from src.simulator.workload import WorkloadConfig, generate_requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Workload-mix sweep: interactive fraction")
    parser.add_argument("--timing-json", type=str, default=None)
    parser.add_argument("--num-requests", type=int, default=500)
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    parser.add_argument("--ttft-slo", type=float, default=0.8)
    parser.add_argument("--tpot-slo", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interactive-fracs", type=str, default="0.0,0.25,0.5,0.75,1.0",
                        help="Comma-separated interactive fractions to sweep")
    args = parser.parse_args()

    if args.timing_json:
        from src.stage_b.fitted_timing import load_fitted_timing
        timing, _ = load_fitted_timing(args.timing_json)
    else:
        timing = TimingModel(colocated_interference=1.35)

    slo = SLOConfig(ttft_slo=args.ttft_slo, tpot_slo=args.tpot_slo)
    prefill_batch = BatchingConfig(max_batch_size=8, batch_wait_s=0.01)
    decode_batch = BatchingConfig(max_batch_size=16, batch_wait_s=0.0)

    interactive_fracs = [float(x) for x in args.interactive_fracs.split(",")]
    rows: list[dict] = []

    print(
        f"Workload-mix sweep: {len(interactive_fracs)} fracs, "
        f"arrival_rate={args.arrival_rate}, {args.num_requests} requests each"
    )
    for frac in interactive_fracs:
        print(f"  interactive_frac={frac:.2f} ...", end=" ", flush=True)
        workload = WorkloadConfig(
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            arrival_process="bursty",
            use_mixture=True,
            interactive_frac=frac,
            interactive_prompt_range=(32, 256),
            interactive_output_range=(16, 64),
            long_prompt_range=(1024, 4096),
            long_output_range=(16, 128),
            seed=args.seed,
        )
        reqs = generate_requests(workload)
        coloc = summarize_results(
            run_colocated(reqs, timing, prefill_batching=prefill_batch, decode_batching=decode_batch),
            slo,
        )
        disagg = summarize_results(
            run_disaggregated(reqs, timing, prefill_batching=prefill_batch, decode_batching=decode_batch),
            slo,
        )
        gain = (disagg["goodput"] - coloc["goodput"]) / max(coloc["goodput"], 1e-9) * 100
        rows.append({
            "interactive_frac":     frac,
            "coloc_goodput":        coloc["goodput"],
            "disagg_goodput":       disagg["goodput"],
            "goodput_gain_pct":     gain,
            "coloc_mean_ttft":      coloc["mean_ttft"],
            "disagg_mean_ttft":     disagg["mean_ttft"],
            "ttft_reduction_pct":   (coloc["mean_ttft"] - disagg["mean_ttft"]) / max(coloc["mean_ttft"], 1e-9) * 100,
            "coloc_p95_ttft":       coloc["p95_ttft"],
            "disagg_p95_ttft":      disagg["p95_ttft"],
            "coloc_p99_ttft":       coloc["p99_ttft"],
            "disagg_p99_ttft":      disagg["p99_ttft"],
            "coloc_p99_e2e":        coloc["p99_e2e_latency"],
            "disagg_p99_e2e":       disagg["p99_e2e_latency"],
        })
        print(
            f"goodput coloc={coloc['goodput']:.3f}  disagg={disagg['goodput']:.3f}  "
            f"gain={gain:+.1f}%  p95_ttft disagg={disagg['p95_ttft']:.3f}s"
        )

    out = ensure_dir("results/milestone")
    write_csv(out / "workload_mix.csv", rows)
    write_json(out / "workload_mix_meta.json", {
        "experiment":        "workload_mix",
        "description":       "Goodput vs interactive fraction at fixed arrival rate",
        "interactive_fracs": interactive_fracs,
        "arrival_rate":      args.arrival_rate,
        "num_requests":      args.num_requests,
        "seed":              args.seed,
        "slo":               asdict(slo),
        "timing":            asdict(timing),
        "timing_source":     args.timing_json or "synthetic_defaults",
        "batching":          {"prefill": asdict(prefill_batch), "decode": asdict(decode_batch)},
        "prompt_ranges": {
            "interactive_prompt": [32, 256],
            "interactive_output": [16, 64],
            "long_prompt":        [1024, 4096],
            "long_output":        [16, 128],
        },
    })
    print(f"\nWrote results/milestone/workload_mix.csv + workload_mix_meta.json")


if __name__ == "__main__":
    main()
