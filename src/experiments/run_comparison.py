from __future__ import annotations

import argparse
from dataclasses import asdict
from pprint import pprint

from src.core.metrics import SLOConfig, results_to_rows, summarize_results
from src.experiments.io_utils import ensure_dir, utc_ts_compact, write_csv, write_json
from src.simulator.baseline import run_colocated
from src.simulator.config import BatchingConfig
from src.simulator.disaggregated import run_disaggregated
from src.simulator.timing import TimingModel
from src.simulator.workload import WorkloadConfig, generate_requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vs disaggregated comparison")
    parser.add_argument(
        "--workload",
        choices=("synthetic", "sharegpt"),
        default="synthetic",
        help="synthetic generator or ShareGPT-derived token counts",
    )
    parser.add_argument(
        "--timing-json",
        type=str,
        default=None,
        help="Path to fitted_timing.json from Stage B (profile_sharegpt.py). "
        "If omitted, uses synthetic TimingModel defaults.",
    )
    parser.add_argument("--ttft-slo", type=float, default=0.8)
    parser.add_argument("--tpot-slo", type=float, default=0.03)
    parser.add_argument("--num-requests", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sharegpt-dataset", type=str, default="anon8231489123/ShareGPT_Vicuna_unfiltered")
    parser.add_argument("--sharegpt-jsonl", type=str, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-low", type=int, default=16)
    parser.add_argument("--output-high", type=int, default=64)
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    args = parser.parse_args()

    slo = SLOConfig(ttft_slo=args.ttft_slo, tpot_slo=args.tpot_slo, e2e_slo=None)

    if args.timing_json:
        from src.stage_b.fitted_timing import load_fitted_timing

        timing, timing_meta = load_fitted_timing(args.timing_json)
    else:
        timing = TimingModel(
            colocated_interference=1.35,
            prefill_capacity_multiplier=1.0,
            decode_capacity_multiplier=1.0,
        )
        timing_meta = None

    prefill_batch = BatchingConfig(max_batch_size=8, batch_wait_s=0.01)
    decode_batch = BatchingConfig(max_batch_size=16, batch_wait_s=0.0)

    if args.workload == "synthetic":
        workload = WorkloadConfig(
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            arrival_process="bursty",
            use_mixture=True,
            interactive_frac=0.75,
            interactive_prompt_range=(32, 256),
            interactive_output_range=(16, 64),
            long_prompt_range=(1024, 4096),
            long_output_range=(16, 128),
            seed=args.seed,
        )
        requests = generate_requests(workload)
    else:
        from src.stage_b.workload_sharegpt import build_requests_from_sharegpt

        workload = {
            "kind": "sharegpt",
            "dataset": args.sharegpt_dataset,
            "jsonl": args.sharegpt_jsonl,
            "num_requests": args.num_requests,
            "seed": args.seed,
            "tokenizer": args.model,
        }
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

    colocated_results = run_colocated(
        requests, timing, prefill_batching=prefill_batch, decode_batching=decode_batch
    )
    disaggregated_results = run_disaggregated(
        requests, timing, prefill_batching=prefill_batch, decode_batching=decode_batch
    )

    print("Baseline colocated summary")
    colocated_summary = summarize_results(colocated_results, slo)
    pprint(colocated_summary)
    print()
    print("Disaggregated summary")
    disagg_summary = summarize_results(disaggregated_results, slo)
    pprint(disagg_summary)

    out_dir = ensure_dir("results")
    ts = utc_ts_compact()
    write_csv(out_dir / f"{ts}-colocated-requests.csv", results_to_rows(colocated_results))
    write_csv(out_dir / f"{ts}-disaggregated-requests.csv", results_to_rows(disaggregated_results))
    workload_payload = workload if isinstance(workload, dict) else asdict(workload)
    write_json(
        out_dir / f"{ts}-summary.json",
        {
            "workload": workload_payload,
            "slo": asdict(slo),
            "timing": asdict(timing),
            "timing_meta": timing_meta,
            "timing_source": args.timing_json,
            "batching": {"prefill": asdict(prefill_batch), "decode": asdict(decode_batch)},
            "colocated": colocated_summary,
            "disaggregated": disagg_summary,
        },
    )


if __name__ == "__main__":
    main()
