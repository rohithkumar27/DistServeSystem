from src.core.metrics import SLOConfig, summarize_results
from src.simulator.baseline import run_colocated
from src.simulator.disaggregated import run_disaggregated
from src.simulator.timing import TimingModel
from src.simulator.workload import WorkloadConfig, generate_requests


def test_disaggregated_improves_ttft_or_tpot_summary() -> None:
    requests = generate_requests(WorkloadConfig(num_requests=50, arrival_rate=1.5, seed=1))
    timing = TimingModel()
    slo = SLOConfig(ttft_slo=0.35, tpot_slo=0.012)

    colocated = summarize_results(run_colocated(requests, timing), slo)
    disaggregated = summarize_results(run_disaggregated(requests, timing), slo)

    assert disaggregated["mean_ttft"] <= colocated["mean_ttft"]
