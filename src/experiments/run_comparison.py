from __future__ import annotations

from pprint import pprint

from src.core.metrics import SLOConfig, summarize_results
from src.simulator.baseline import run_colocated
from src.simulator.disaggregated import run_disaggregated
from src.simulator.timing import TimingModel
from src.simulator.workload import WorkloadConfig, generate_requests


def main() -> None:
    workload = WorkloadConfig(
        num_requests=250,
        arrival_rate=2.2,
        prompt_low=128,
        prompt_high=1024,
        output_low=32,
        output_high=96,
        seed=42,
    )
    slo = SLOConfig(ttft_slo=0.35, tpot_slo=0.012)
    timing = TimingModel()

    requests = generate_requests(workload)
    colocated_results = run_colocated(requests, timing)
    disaggregated_results = run_disaggregated(requests, timing)

    print("Baseline colocated summary")
    pprint(summarize_results(colocated_results, slo))
    print()
    print("Disaggregated summary")
    pprint(summarize_results(disaggregated_results, slo))


if __name__ == "__main__":
    main()
