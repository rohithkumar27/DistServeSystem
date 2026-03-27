from src.core.metrics import SLOConfig, summarize_results
from src.simulator.baseline import run_colocated
from src.simulator.config import BatchingConfig
from src.simulator.disaggregated import run_disaggregated
from src.simulator.timing import TimingModel
from src.simulator.workload import WorkloadConfig, generate_requests


def test_disaggregated_improves_ttft_or_tpot_summary() -> None:
    requests = generate_requests(
        WorkloadConfig(
            num_requests=120,
            arrival_rate=2.0,
            arrival_process="bursty",
            use_mixture=True,
            seed=1,
        )
    )
    timing = TimingModel()
    slo = SLOConfig(ttft_slo=0.8, tpot_slo=0.03)
    prefill_batch = BatchingConfig(max_batch_size=8, batch_wait_s=0.01)
    decode_batch = BatchingConfig(max_batch_size=16, batch_wait_s=0.0)

    colocated = summarize_results(
        run_colocated(requests, timing, prefill_batching=prefill_batch, decode_batching=decode_batch),
        slo,
    )
    disaggregated = summarize_results(
        run_disaggregated(requests, timing, prefill_batching=prefill_batch, decode_batching=decode_batch),
        slo,
    )

    # Not all parameterizations guarantee strict improvement, but in the default
    # interference setting we should not regress on mean TTFT.
    assert disaggregated["mean_ttft"] <= colocated["mean_ttft"]

    # Smoke-check required metric keys for milestone reporting.
    for key in [
        "p95_ttft",
        "p99_ttft",
        "p95_e2e_latency",
        "p99_e2e_latency",
        "goodput",
        "throughput_req_per_s",
        "throughput_tok_per_s",
    ]:
        assert key in colocated
        assert key in disaggregated
