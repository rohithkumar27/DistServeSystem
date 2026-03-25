from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.core.request import Request


@dataclass(slots=True)
class WorkloadConfig:
    num_requests: int = 200
    arrival_rate: float = 2.0
    prompt_low: int = 128
    prompt_high: int = 1024
    output_low: int = 32
    output_high: int = 128
    seed: int = 7


def generate_requests(config: WorkloadConfig) -> List[Request]:
    rng = np.random.default_rng(config.seed)
    gaps = rng.exponential(1.0 / config.arrival_rate, size=config.num_requests)
    arrivals = np.cumsum(gaps)
    prompt_tokens = rng.integers(config.prompt_low, config.prompt_high + 1, size=config.num_requests)
    output_tokens = rng.integers(config.output_low, config.output_high + 1, size=config.num_requests)

    return [
        Request(
            request_id=i,
            arrival_time=float(arrivals[i]),
            prompt_tokens=int(prompt_tokens[i]),
            output_tokens=int(output_tokens[i]),
        )
        for i in range(config.num_requests)
    ]
