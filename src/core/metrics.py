from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.core.request import RequestResult


@dataclass(slots=True)
class SLOConfig:
    ttft_slo: float
    tpot_slo: float


def results_to_frame(results: Iterable[RequestResult]) -> pd.DataFrame:
    rows = [
        {
            "request_id": result.request_id,
            "arrival_time": result.arrival_time,
            "first_token_time": result.first_token_time,
            "finish_time": result.finish_time,
            "output_tokens": result.output_tokens,
            "ttft": result.ttft,
            "tpot": result.tpot,
        }
        for result in results
    ]
    return pd.DataFrame(rows)


def summarize_results(results: Iterable[RequestResult], slo: SLOConfig) -> dict:
    frame = results_to_frame(results)
    if frame.empty:
        return {
            "count": 0,
            "mean_ttft": 0.0,
            "mean_tpot": 0.0,
            "p90_ttft": 0.0,
            "p90_tpot": 0.0,
            "slo_attainment": 0.0,
            "throughput_req_per_s": 0.0,
        }

    elapsed = max(frame["finish_time"].max() - frame["arrival_time"].min(), 1e-9)
    within_slo = (frame["ttft"] <= slo.ttft_slo) & (frame["tpot"] <= slo.tpot_slo)

    return {
        "count": int(len(frame)),
        "mean_ttft": float(frame["ttft"].mean()),
        "mean_tpot": float(frame["tpot"].mean()),
        "p90_ttft": float(frame["ttft"].quantile(0.9)),
        "p90_tpot": float(frame["tpot"].quantile(0.9)),
        "slo_attainment": float(within_slo.mean()),
        "throughput_req_per_s": float(len(frame) / elapsed),
    }
