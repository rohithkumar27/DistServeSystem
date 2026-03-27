from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from src.core.request import RequestResult


@dataclass(slots=True)
class SLOConfig:
    ttft_slo: float
    tpot_slo: float
    e2e_slo: float | None = None


def _quantiles(values: list[float], qs: Sequence[float]) -> Mapping[str, float]:
    if not values:
        return {f"p{int(q * 100)}": 0.0 for q in qs}
    vs = sorted(values)
    n = len(vs)

    def pick(q: float) -> float:
        if n == 1:
            return float(vs[0])
        # Linear interpolation between closest ranks (like numpy default).
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return float(vs[lo] * (1 - frac) + vs[hi] * frac)

    return {f"p{int(q * 100)}": pick(q) for q in qs}


def results_to_rows(results: Iterable[RequestResult]) -> list[dict]:
    return [
        {
            "request_id": r.request_id,
            "arrival_time": r.arrival_time,
            "first_token_time": r.first_token_time,
            "finish_time": r.finish_time,
            "output_tokens": r.output_tokens,
            "ttft": r.ttft,
            "tpot": r.tpot,
            "e2e_latency": r.finish_time - r.arrival_time,
        }
        for r in results
    ]


def summarize_results(results: Iterable[RequestResult], slo: SLOConfig) -> dict:
    rows = results_to_rows(results)
    if not rows:
        return {
            "count": 0,
            "mean_ttft": 0.0,
            "mean_tpot": 0.0,
            "mean_e2e_latency": 0.0,
            "p50_ttft": 0.0,
            "p90_ttft": 0.0,
            "p95_ttft": 0.0,
            "p99_ttft": 0.0,
            "p50_tpot": 0.0,
            "p90_tpot": 0.0,
            "p95_tpot": 0.0,
            "p99_tpot": 0.0,
            "p50_e2e_latency": 0.0,
            "p90_e2e_latency": 0.0,
            "p95_e2e_latency": 0.0,
            "p99_e2e_latency": 0.0,
            "goodput": 0.0,
            "throughput_req_per_s": 0.0,
            "throughput_tok_per_s": 0.0,
        }

    arrivals = [float(r["arrival_time"]) for r in rows]
    finishes = [float(r["finish_time"]) for r in rows]
    ttfts = [float(r["ttft"]) for r in rows]
    tpots = [float(r["tpot"]) for r in rows]
    e2es = [float(r["e2e_latency"]) for r in rows]
    out_toks = [int(r["output_tokens"]) for r in rows]

    elapsed = max(max(finishes) - min(arrivals), 1e-9)

    within_slo_flags: list[bool] = []
    for ttft, tpot, e2e in zip(ttfts, tpots, e2es, strict=True):
        ok = (ttft <= slo.ttft_slo) and (tpot <= slo.tpot_slo)
        if slo.e2e_slo is not None:
            ok = ok and (e2e <= slo.e2e_slo)
        within_slo_flags.append(ok)

    goodput = sum(within_slo_flags) / len(within_slo_flags)

    q_ttft = _quantiles(ttfts, [0.5, 0.9, 0.95, 0.99])
    q_tpot = _quantiles(tpots, [0.5, 0.9, 0.95, 0.99])
    q_e2e = _quantiles(e2es, [0.5, 0.9, 0.95, 0.99])
    total_out_tokens = float(sum(out_toks))

    return {
        "count": int(len(rows)),
        "mean_ttft": float(sum(ttfts) / len(ttfts)),
        "mean_tpot": float(sum(tpots) / len(tpots)),
        "mean_e2e_latency": float(sum(e2es) / len(e2es)),
        "p50_ttft": q_ttft["p50"],
        "p90_ttft": q_ttft["p90"],
        "p95_ttft": q_ttft["p95"],
        "p99_ttft": q_ttft["p99"],
        "p50_tpot": q_tpot["p50"],
        "p90_tpot": q_tpot["p90"],
        "p95_tpot": q_tpot["p95"],
        "p99_tpot": q_tpot["p99"],
        "p50_e2e_latency": q_e2e["p50"],
        "p90_e2e_latency": q_e2e["p90"],
        "p95_e2e_latency": q_e2e["p95"],
        "p99_e2e_latency": q_e2e["p99"],
        "goodput": float(goodput),
        "throughput_req_per_s": float(len(rows) / elapsed),
        "throughput_tok_per_s": float(total_out_tokens / elapsed),
    }
