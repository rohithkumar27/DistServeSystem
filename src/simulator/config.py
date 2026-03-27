from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BatchingConfig:
    # Maximum number of requests to process together.
    max_batch_size: int = 8
    # How long the worker is willing to wait (after it becomes idle) to
    # collect more requests that have already arrived.
    batch_wait_s: float = 0.0

