from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from pprint import pprint

from src.core.metrics import SLOConfig, summarize_results
from src.experiments.io_utils import ensure_dir, utc_ts_compact, write_csv, write_json
from src.simulator.baseline import run_colocated
from src.simulator.config import BatchingConfig
from src.simulator.disaggregated import run_disaggregated
from src.simulator.timing import TimingModel
from src.simulator.workload import WorkloadConfig, generate_requests


def main() -> None:
    parser = argparse.ArgumentParser(description="KV transfer + capacity split ablations")
    parser.add_argument(
        "--timing-json",
        type=str,
        default=None,
        help="Optional fitted_timing.json from Stage B (else synthetic TimingModel)",
    )
    args = parser.parse_args()
    workload = WorkloadConfig(
        num_requests=600,
        arrival_rate=2.2,
        arrival_process="bursty",
        use_mixture=True,
        interactive_frac=0.75,
        interactive_prompt_range=(32, 256),
        interactive_output_range=(16, 64),
        long_prompt_range=(1024, 4096),
        long_output_range=(16, 128),
        seed=7,
    )
    slo = SLOConfig(ttft_slo=0.8, tpot_slo=0.03, e2e_slo=None)
    prefill_batch = BatchingConfig(max_batch_size=8, batch_wait_s=0.01)
    decode_batch = BatchingConfig(max_batch_size=16, batch_wait_s=0.0)

    if args.timing_json:
        from src.stage_b.fitted_timing import load_fitted_timing

        base_timing, _ = load_fitted_timing(args.timing_json)
    else:
        base_timing = TimingModel(colocated_interference=1.35)
    requests = generate_requests(workload)

    out_dir = ensure_dir("results")
    ts = utc_ts_compact()

    rows: list[dict] = []

    # Ablation 1: KV handoff overhead sweep.
    for transfer_per_prompt_token in [0.0, 5e-6, 1.5e-5, 3e-5, 6e-5]:
        timing = replace(base_timing, transfer_per_prompt_token=transfer_per_prompt_token)
        coloc = summarize_results(
            run_colocated(requests, timing, prefill_batching=prefill_batch, decode_batching=decode_batch),
            slo,
        )
        disagg = summarize_results(
            run_disaggregated(requests, timing, prefill_batching=prefill_batch, decode_batching=decode_batch),
            slo,
        )
        rows.append(
            {
                "ablation": "kv_transfer",
                "transfer_per_prompt_token": transfer_per_prompt_token,
                "colocated_goodput": coloc["goodput"],
                "disaggregated_goodput": disagg["goodput"],
                "colocated_p95_ttft": coloc["p95_ttft"],
                "disaggregated_p95_ttft": disagg["p95_ttft"],
                "colocated_p99_e2e": coloc["p99_e2e_latency"],
                "disaggregated_p99_e2e": disagg["p99_e2e_latency"],
            }
        )

    # Ablation 2: Capacity split (prefill vs decode multipliers).
    # Higher multiplier => less capacity for that phase.
    for pre_mult, dec_mult in [(0.8, 1.2), (1.0, 1.0), (1.2, 0.8), (1.4, 0.7)]:
        timing = replace(
            base_timing, prefill_capacity_multiplier=pre_mult, decode_capacity_multiplier=dec_mult
        )
        coloc = summarize_results(
            run_colocated(requests, timing, prefill_batching=prefill_batch, decode_batching=decode_batch),
            slo,
        )
        disagg = summarize_results(
            run_disaggregated(requests, timing, prefill_batching=prefill_batch, decode_batching=decode_batch),
            slo,
        )
        rows.append(
            {
                "ablation": "capacity_split",
                "prefill_capacity_multiplier": pre_mult,
                "decode_capacity_multiplier": dec_mult,
                "colocated_goodput": coloc["goodput"],
                "disaggregated_goodput": disagg["goodput"],
                "colocated_p95_ttft": coloc["p95_ttft"],
                "disaggregated_p95_ttft": disagg["p95_ttft"],
                "colocated_p99_e2e": coloc["p99_e2e_latency"],
                "disaggregated_p99_e2e": disagg["p99_e2e_latency"],
            }
        )

    write_csv(out_dir / f"{ts}-ablations.csv", rows)
    write_json(
        out_dir / f"{ts}-ablations-meta.json",
        {
            "workload": workload,
            "slo": slo,
            "timing_base": base_timing,
            "batching": {"prefill": prefill_batch, "decode": decode_batch},
            "outputs": {"table_csv": str(Path(out_dir / f"{ts}-ablations.csv"))},
        },
    )

    print("Wrote ablations:")
    pprint(str(out_dir / f"{ts}-ablations.csv"))


if __name__ == "__main__":
    main()

