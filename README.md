# Reimplementation Scaffold

This folder is a clean starting point for a master's-level reimplementation of the main DistServe idea:

- baseline: colocated prefill and decode
- proposed: disaggregated prefill and decode
- metrics: TTFT, TPOT, throughput, SLO attainment

The goal is to help you build and test the architecture in your own way without depending on the full production backend.

## Recommended first steps

1. Create a virtual environment.
2. Install the requirements from `requirements.txt`.
3. Run the comparison script:

```bash
python -m src.experiments.run_comparison
```

## Structure

- `src/core`: request and metrics primitives
- `src/simulator`: workload generation and simple simulators
- `src/experiments`: runnable experiments
- `tests`: starter tests

## Current scope

This scaffold does not run a real LLM yet. It simulates service behavior so you can compare:

- colocated execution
- disaggregated execution

Once this is stable, you can add:

- real model-backed timing
- more realistic batching
- better scheduling
- queueing policies
- plotting scripts
