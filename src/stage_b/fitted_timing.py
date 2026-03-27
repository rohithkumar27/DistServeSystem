from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from src.simulator.timing import TimingModel


def timing_model_to_dict(t: TimingModel) -> dict[str, Any]:
    return asdict(t)


def timing_model_from_dict(d: dict[str, Any]) -> TimingModel:
    allowed = {f.name for f in fields(TimingModel)}
    return TimingModel(**{k: d[k] for k in allowed if k in d})


def save_fitted_timing(path: str | Path, timing: TimingModel, meta: dict[str, Any]) -> None:
    payload = {"meta": meta, "timing": timing_model_to_dict(timing)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_fitted_timing(path: str | Path) -> tuple[TimingModel, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    timing = timing_model_from_dict(payload["timing"])
    meta = payload.get("meta", {})
    return timing, meta


def fit_prefill_linear(prompt_tokens: list[int], prefill_s: list[float]) -> tuple[float, float]:
    """Least squares: prefill_s ≈ a + b * prompt_tokens."""
    n = len(prompt_tokens)
    if n < 2:
        a = prefill_s[0] if prefill_s else 0.02
        b = 0.0002
        return a, b
    sx = sum(prompt_tokens)
    sy = sum(prefill_s)
    sxx = sum(t * t for t in prompt_tokens)
    sxy = sum(prompt_tokens[i] * prefill_s[i] for i in range(n))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-18:
        b = 0.0
        a = sy / n
    else:
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
    return max(a, 0.0), max(b, 0.0)
