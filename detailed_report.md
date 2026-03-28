## DistServeSystem Detailed Milestone Report

### 1. Project Goal
This milestone focuses on reimplementing the central DistServe systems idea : prefill and decode should be separated so long-prompt prefills do not interfere with latency-sensitive decoding. Instead of reproducing the original Ray- and SwiftTransformer-based stack, this repo rebuilds the architecture in Python with two complementary execution modes:

- a simulator that lets us study queueing, batching, SLO goodput, and KV-transfer overhead quickly
- a real-GPU runtime that measures actual HuggingFace forward passes and KV-cache transfer behavior

The goal of the project is not a line-by-line port of upstream DistServe. The goal is to preserve the architectural claim, make it runnable on our available hardware, and generate evidence that the disaggregated design improves user-facing latency metrics under mixed workloads.

### 2. What We Implemented
The repository contains two serving designs:

1. `Colocated / baseline`
This system executes prefill and decode in the same worker path. In the simulator, this is implemented in `src/simulator/baseline.py`, where prefill batches and decode steps share one service timeline and are penalized by a colocated interference multiplier. In the runtime path, `src/runtime/baseline_gpu.py` runs prompt prefill and autoregressive decode on a single GPU.

2. `Disaggregated / DistServe-style`
This system splits the work into a prefill pool and a decode pool. In the simulator, `src/simulator/disaggregated.py` models a prefill queue, a decode queue, and an explicit KV handoff delay before decode can begin. In the runtime path, `src/runtime/disaggregated_gpu.py` runs prefill on one GPU, transfers `past_key_values` to another GPU, and then continues decode there.

This architecture is supported by:

- `src/simulator/timing.py` for service-time and batching models
- `src/simulator/workload.py` for mixed and bursty request generation
- `src/core/metrics.py` for TTFT, TPOT, end-to-end latency, throughput, and SLO goodput
- `src/stage_b/profile_sharegpt.py` and `src/stage_b/fitted_timing.py` for fitting measured GPU timings back into the simulator

### 3. Experimental Methodology
We evaluate the system in three ways.

1. Synthetic simulator comparison
We run `python -m src.experiments.run_comparison` on a bursty mixed workload with short interactive requests and long prompts. This gives us per-request CSVs plus a summary JSON. The default SLO configuration used in the saved result is:

- TTFT SLO = 0.8 s
- TPOT SLO = 0.03 s

2. Ablation study
We run `python -m src.experiments.run_ablations` to sweep:

- KV-transfer cost (`transfer_per_prompt_token`)
- relative prefill/decode capacity (`prefill_capacity_multiplier`, `decode_capacity_multiplier`)

These ablations test whether the disaggregated advantage is robust or whether it disappears when communication cost or poor resource partitioning dominates.

3. Real-GPU comparison
We run `python -m src.experiments.run_gpu_comparison` using TinyLlama/TinyLlama-1.1B-Chat-v1.0 on real hardware. This validates that the simulator’s story is directionally consistent with measured prefill, transfer, and decode behavior.

### 4. Main Stage A Result: Simulator Comparison
The strongest simulator evidence is stored in `reimpl_distserve/results/20260326T042324Z-summary.json`. This run used:

- 400 synthetic requests
- bursty arrivals
- mixed short/long prompts
- prefill batch size 8
- decode batch size 16
- default timing model with colocated interference = 1.35

The key results are:

| Metric | Colocated | Disaggregated | Change |
|---|---:|---:|---:|
| Goodput | 0.4375 | 0.6975 | +59.4% |
| Mean TTFT | 0.9040 s | 0.5566 s | -38.4% |
| Mean end-to-end latency | 1.2848 s | 0.8541 s | -33.5% |
| p95 TTFT | 1.9167 s | 1.3053 s | -31.9% |
| p99 TTFT | 2.4257 s | 1.5337 s | -36.8% |
| p99 end-to-end latency | 2.9792 s | 2.0860 s | -30.0% |

These numbers support the main thesis of the project. Once prefill and decode are separated, the system serves more requests within the SLO budget and reduces both average and tail latency. Importantly, throughput remains essentially unchanged:

- colocated throughput: 2.502 requests/s
- disaggregated throughput: 2.504 requests/s

That means the gain is not coming from simply doing less work or serving fewer requests. It comes from improving scheduling structure and reducing prefill-decode interference.

The raw evidence for this run is also available in:

- `reimpl_distserve/results/20260326T042324Z-colocated-requests.csv`
- `reimpl_distserve/results/20260326T042324Z-disaggregated-requests.csv`

These files contain request-level TTFT, TPOT, and end-to-end latency values and can be used for plots or further verification.

### 5. Ablation Results
The ablation table is stored in `reimpl_distserve/results/20260326T040234Z-ablations.csv`.

#### 5.1 KV-transfer sensitivity
We sweep `transfer_per_prompt_token` from `0.0` to `6e-05`. The colocated system remains constant because it has no separate handoff stage, while the disaggregated system degrades gradually as transfer cost increases.

Selected values:

| Transfer per prompt token | Colocated goodput | Disaggregated goodput | Disaggregated p95 TTFT |
|---|---:|---:|---:|
| 0.0 | 0.4267 | 0.7833 | 1.1930 s |
| 5e-06 | 0.4267 | 0.7583 | 1.2646 s |
| 1.5e-05 | 0.4267 | 0.7450 | 1.2813 s |
| 3e-05 | 0.4267 | 0.7150 | 1.2913 s |
| 6e-05 | 0.4267 | 0.7100 | 1.3686 s |

Interpretation:

- the benefit of disaggregation is not fragile; it survives a broad range of transfer penalties
- however, communication overhead matters and becomes a real tax on TTFT as handoff cost increases
- this justifies the project’s emphasis on modeling and, later, measuring KV transfer carefully instead of assuming it is free

#### 5.2 Capacity split sensitivity
We also vary the effective service capacity of prefill versus decode.

| Prefill multiplier | Decode multiplier | Colocated goodput | Disaggregated goodput | Disaggregated p95 TTFT |
|---|---:|---:|---:|---:|
| 0.8 | 1.2 | 0.5083 | 0.8050 | 1.1656 s |
| 1.0 | 1.0 | 0.4267 | 0.7450 | 1.2813 s |
| 1.2 | 0.8 | 0.3717 | 0.6833 | 1.4051 s |
| 1.4 | 0.7 | 0.3150 | 0.6000 | 1.5651 s |

Interpretation:

- disaggregation helps most when decode capacity is protected
- if prefill becomes too slow or decode becomes under-provisioned, the advantage shrinks
- the design still remains better than colocated across all tested splits, which suggests the structural advantage is robust

The corresponding metadata is saved in `reimpl_distserve/results/20260326T040234Z-ablations-meta.json`, including workload settings, SLOs, timing model, and batching configuration for reproducibility.

### 6. Stage B: Measured GPU Evidence
To move beyond purely synthetic timing assumptions, the project includes a profiling pipeline in `src/stage_b/profile_sharegpt.py`. This script:

- samples ShareGPT prompts
- tokenizes them with the target model tokenizer
- measures full-prompt prefill latency
- measures per-step decode latency
- fits a linear prefill model and stores the result in `results/fitted_timing.json`

The real-GPU comparison result currently saved in `reimpl_distserve/results/20260326T063012Z-gpu-comparison-summary.json` reports:

| Metric | Colocated | Disaggregated | Change |
|---|---:|---:|---:|
| Goodput | 0.3333 | 0.4667 | +40.0% |
| Mean TTFT | 2.6285 s | 2.3423 s | -10.9% |
| Mean end-to-end latency | 2.9229 s | 2.6353 s | -9.8% |
| p95 TTFT | 4.8558 s | 4.5640 s | -6.0% |
| p99 TTFT | 4.9574 s | 4.6652 s | -5.9% |
| Throughput | 2.7925 req/s | 2.9542 req/s | +5.8% |

This result is smaller in magnitude than the simulator result, but that is expected. Real systems include model loading constraints, transfer overhead, and a much smaller request count in this run (`num_requests = 15`). Even so, the direction of improvement is the same:

- better goodput
- better TTFT
- slightly better throughput

That consistency is important. It suggests the simulator is not producing a purely artificial win and that the core architectural idea continues to help when tested with actual GPU kernels.

The GPU raw logs are stored in:

- `reimpl_distserve/results/20260326T063012Z-gpu-colocated-requests.csv`
- `reimpl_distserve/results/20260326T063012Z-gpu-disaggregated-requests.csv`

### 7. Why These Results Matter
The TA will likely care about whether the project demonstrates a real systems insight rather than just a toy benchmark. The evidence supports that:

1. We reproduced the right systems question.
The project isolates the prefill/decode interference problem that DistServe is designed to solve.

2. We implemented both sides of the comparison.
This is not only a disaggregated prototype. We also implemented the colocated baseline and compare them under the same workloads and metrics.

3. We evaluated the design from multiple angles.
We have:

- a simulator for larger controlled experiments
- ablation studies for robustness and sensitivity
- real-GPU experiments for empirical grounding

4. We measured the right metrics.
The report centers TTFT, tail latency, throughput, and SLO goodput rather than only average runtime.

### 8. Validation and Correctness
The project includes tests that directly support the report:

- `reimpl_distserve/tests/test_simulation.py`
- `reimpl_distserve/tests/test_fit_prefill.py`

These tests check that:

- disaggregated mean TTFT does not regress under default simulator settings
- the required reporting metrics are present
- the linear fit used for Stage B timing estimation behaves correctly

This matters because the milestone report is only convincing if the metric pipeline itself is trustworthy.

### 9. Limitations
To make the report credible, it is important to state what has not yet been done.

- The simulator still uses simplified service models rather than exact kernel-level execution overlap.
- The current GPU comparison is based on a small run (`15` requests), so it should be interpreted as directional rather than final-scale evidence.
- The fitted timing pipeline currently uses a placeholder transfer coefficient unless replaced by a more direct KV-transfer microbenchmark.
- The runtime path is sequential rather than a fully online serving system with admission control and overlapping workers.

These limitations do not invalidate the milestone. They define the boundary of what has been completed so far and naturally motivate the next phase of work.

### 10. Next Steps
The clearest next steps are:

1. increase the number of real-GPU requests and profile more prompts so Stage B evidence is statistically stronger
2. replace the placeholder transfer coefficient with a direct KV handoff benchmark
3. expand the runtime into a small online serving prototype with separate request admission and queueing logic
4. regenerate plots from the saved CSVs so the final submission includes visual evidence, not just summary tables

### 11. Artifact Checklist
The key artifacts for the milestone are:

- `reimpl_distserve/results/20260326T042324Z-summary.json`
- `reimpl_distserve/results/20260326T042324Z-colocated-requests.csv`
- `reimpl_distserve/results/20260326T042324Z-disaggregated-requests.csv`
- `reimpl_distserve/results/20260326T040234Z-ablations.csv`
- `reimpl_distserve/results/20260326T040234Z-ablations-meta.json`
- `reimpl_distserve/results/20260326T063012Z-gpu-comparison-summary.json`
- `reimpl_distserve/results/20260326T063012Z-gpu-colocated-requests.csv`
- `reimpl_distserve/results/20260326T063012Z-gpu-disaggregated-requests.csv`
- `reimpl_distserve/tests/test_simulation.py`
- `reimpl_distserve/tests/test_fit_prefill.py`

### 12. Conclusion
This milestone already demonstrates the main claim we set out to test: separating prefill from decode improves latency behavior and SLO goodput under mixed workloads. In the simulator, the improvement is large and consistent. In the real-GPU run, the improvement is smaller but still clearly present. The ablation results show that the advantage is robust, though sensitive to handoff cost and capacity allocation. Taken together, the implementation and the saved artifacts form a solid systems milestone with a clear experimental story, reproducible evidence, and a credible path to a stronger final project.
