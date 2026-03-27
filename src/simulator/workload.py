from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal
import random

from src.core.request import Request


@dataclass(slots=True)
class WorkloadConfig:
    num_requests: int = 200
    arrival_rate: float = 2.0  # requests / second (used for Poisson gaps)
    arrival_process: Literal["poisson", "bursty"] = "poisson"

    # Simple uniform workload (kept for backwards compatibility)
    prompt_low: int = 128
    prompt_high: int = 1024
    output_low: int = 32
    output_high: int = 128

    # Mixed workload knobs (DistServe-motivating mix)
    use_mixture: bool = False
    interactive_frac: float = 0.7
    interactive_prompt_range: tuple[int, int] = (32, 256)
    interactive_output_range: tuple[int, int] = (16, 64)
    long_prompt_range: tuple[int, int] = (1024, 4096)
    long_output_range: tuple[int, int] = (16, 128)

    # Bursty arrival knobs (on/off-ish)
    burst_prob: float = 0.25
    burst_rate_multiplier: float = 8.0
    seed: int = 7


def generate_requests(config: WorkloadConfig) -> List[Request]:
    rng = random.Random(config.seed)

    def exp_gap(rate: float) -> float:
        return rng.expovariate(rate) if rate > 0 else 0.0

    arrivals: list[float] = []
    t = 0.0
    for _ in range(config.num_requests):
        if config.arrival_process == "poisson":
            gap = exp_gap(config.arrival_rate)
        elif config.arrival_process == "bursty":
            is_burst = rng.random() < config.burst_prob
            rate = config.arrival_rate * (config.burst_rate_multiplier if is_burst else 1.0)
            gap = exp_gap(rate)
        else:
            raise ValueError(f"Unknown arrival_process={config.arrival_process!r}")
        t += gap
        arrivals.append(t)

    prompt_tokens: list[int] = []
    output_tokens: list[int] = []

    if not config.use_mixture:
        for _ in range(config.num_requests):
            prompt_tokens.append(rng.randint(config.prompt_low, config.prompt_high))
            output_tokens.append(rng.randint(config.output_low, config.output_high))
    else:
        ip_lo, ip_hi = config.interactive_prompt_range
        io_lo, io_hi = config.interactive_output_range
        lp_lo, lp_hi = config.long_prompt_range
        lo_lo, lo_hi = config.long_output_range

        for _ in range(config.num_requests):
            interactive = rng.random() < config.interactive_frac
            if interactive:
                prompt_tokens.append(rng.randint(ip_lo, ip_hi))
                output_tokens.append(rng.randint(io_lo, io_hi))
            else:
                prompt_tokens.append(rng.randint(lp_lo, lp_hi))
                output_tokens.append(rng.randint(lo_lo, lo_hi))

    return [
        Request(
            request_id=i,
            arrival_time=float(arrivals[i]),
            prompt_tokens=int(prompt_tokens[i]),
            output_tokens=int(output_tokens[i]),
        )
        for i in range(config.num_requests)
    ]
