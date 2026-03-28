"""
Load-sweep experiment: arrival rate vs goodput / latency for colocated vs disaggregated.

This is the primary throughput-latency tradeoff curve. It shows:
  - At low load both designs handle all requests within SLO.
  - As load increases, colocated goodput collapses first (prefill-decode interference).
  - Disaggregated saturates later and maintains better tail latency throughout.

Fixed workload mix (75% interactive, 25% long-prompt) and timing model; only
arrival rate varies.

Output: results/milestone/load_sweep.csv + load_sweep_meta.json
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
    parser = argparse.ArgumentParser(description="Arrival-rate sweep: goodput vs load")
    parser.add_argument("--timing-json", type=str, default=None,
                        help="Fitted timing JSON from Stage B (else use synthetic defaults)")
    parser.add_argument("--num-requests", type=int, default=500)
    parser.add_argument("--ttft-slo", type=float, default=0.8)
    parser.add_argument("--tpot-slo", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rates", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0",
                        help="Comma-separated list of arrival rates to sweep (req/s)")
    args = parser.parse_args()

    if args.timing_json:
        from src.stage_b.fitted_timing import load_fitted_timing
        timing, _ = load_fitted_timing(args.timing_json)
    else:
        timing = TimingModel(colocated_interference=1.35)

    slo = SLOConfig(ttft_slo=args.ttft_slo, tpot_slo=args.tpot_slo)
    prefill_batch = BatchingConfig(max_batch_size=8, batch_wait_s=0.01)
    decode_batch = BatchingConfig(max_batch_size=16, batch_wait_s=0.0)

    arrival_rates = [float(x) for x in args.arrival_rates.split(",")]
    rows: list[dict] = []

    print(f"Load sweep: {len(arrival_rates)} arrival rates, {args.num_requests} requests each")
    for rate in arrival_rates:
        print(f"  {rate:.1f} req/s ...", end=" ", flush=True)
        workload = WorkloadConfig(
            num_requests=args.num_requests,
            arrival_rate=rate,
            arrival_process="bursty",
            use_mixture=True,
            interactive_frac=0.75,
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
        rows.append({
            "arrival_rate":       rate,
            "coloc_goodput":      coloc["goodput"],
            "disagg_goodput":     disagg["goodput"],
            "goodput_gain_pct":   (disagg["goodput"] - coloc["goodput"]) / max(coloc["goodput"], 1e-9) * 100,
            "coloc_mean_ttft":    coloc["mean_ttft"],
            "disagg_mean_ttft":   disagg["mean_ttft"],
            "coloc_p95_ttft":     coloc["p95_ttft"],
            "disagg_p95_ttft":    disagg["p95_ttft"],
            "coloc_p99_ttft":     coloc["p99_ttft"],
            "disagg_p99_ttft":    disagg["p99_ttft"],
            "coloc_p99_e2e":      coloc["p99_e2e_latency"],
            "disagg_p99_e2e":     disagg["p99_e2e_latency"],
            "coloc_throughput":   coloc["throughput_req_per_s"],
            "disagg_throughput":  disagg["throughput_req_per_s"],
        })
        print(
            f"goodput coloc={coloc['goodput']:.3f}  disagg={disagg['goodput']:.3f}  "
            f"p95_ttft coloc={coloc['p95_ttft']:.3f}s  disagg={disagg['p95_ttft']:.3f}s"
        )

    out = ensure_dir("results/milestone")
    write_csv(out / "load_sweep.csv", rows)
    write_json(out / "load_sweep_meta.json", {
        "experiment":    "load_sweep",
        "description":   "Goodput and latency vs arrival rate for colocated vs disaggregated",
        "arrival_rates": arrival_rates,
        "num_requests":  args.num_requests,
        "seed":          args.seed,
        "slo":           asdict(slo),
        "timing":        asdict(timing),
        "timing_source": args.timing_json or "synthetic_defaults",
        "batching":      {"prefill": asdict(prefill_batch), "decode": asdict(decode_batch)},
        "workload": {
            "arrival_process":        "bursty",
            "use_mixture":            True,
            "interactive_frac":       0.75,
            "interactive_prompt_range": [32, 256],
            "interactive_output_range": [16, 64],
            "long_prompt_range":      [1024, 4096],
            "long_output_range":      [16, 128],
        },
    })
    print(f"\nWrote results/milestone/load_sweep.csv + load_sweep_meta.json")


if __name__ == "__main__":
    main()
