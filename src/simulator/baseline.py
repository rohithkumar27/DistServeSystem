from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List

from src.core.request import Request, RequestResult
from src.simulator.config import BatchingConfig
from src.simulator.timing import TimingModel


def _take_batch(
    queue: Deque[Request], *, now: float, cfg: BatchingConfig
) -> List[Request]:
    if not queue:
        return []

    batch: List[Request] = []
    while queue and queue[0].arrival_time <= now and len(batch) < cfg.max_batch_size:
        batch.append(queue.popleft())

    if batch:
        return batch

    # Optionally wait a bit to form a batch (bounded by next arrival).
    if cfg.batch_wait_s <= 0:
        return []

    next_arrival = queue[0].arrival_time
    if next_arrival <= now + cfg.batch_wait_s:
        now = next_arrival
        while queue and queue[0].arrival_time <= now and len(batch) < cfg.max_batch_size:
            batch.append(queue.popleft())
    return batch


def run_colocated(
    requests: Iterable[Request],
    timing: TimingModel,
    *,
    prefill_batching: BatchingConfig | None = None,
    decode_batching: BatchingConfig | None = None,
) -> List[RequestResult]:
    """
    Monolithic baseline: a single worker pool executes prefill then decode for each request.

    This simulator is intentionally simple:
    - Prefill is batched over queued requests.
    - Decode is modeled as one "first-token step" + remaining tokens at a per-step rate.
    - Prefill/decode share capacity; we model interference via `timing.colocated_interference`.
    """
    prefill_cfg = prefill_batching or BatchingConfig()
    decode_cfg = decode_batching or BatchingConfig()

    requests_list = list(requests)
    pending: Deque[Request] = deque(sorted(requests_list, key=lambda r: r.arrival_time))
    now = 0.0
    results: List[RequestResult] = []

    while pending:
        now = max(now, pending[0].arrival_time)
        batch = _take_batch(pending, now=now, cfg=prefill_cfg)
        if not batch:
            # No arrivals yet (can happen if batch_wait_s == 0 and we advanced oddly)
            now = max(now, pending[0].arrival_time)
            continue

        # Prefill batch (interference inflated).
        batch_prefill_time = max(
            timing.prefill_time(r.prompt_tokens, batch_size=len(batch)) for r in batch
        ) * timing.colocated_interference
        prefill_done = now + batch_prefill_time

        # Decode batch: first token step occurs after prefill.
        first_step = timing.decode_step_time(batch_size=min(len(batch), decode_cfg.max_batch_size))
        first_step *= timing.colocated_interference

        # Everyone gets first token at the end of first step.
        first_token_time = prefill_done + first_step

        # Remaining tokens: use per-request remaining count, but same step time (simple model).
        for r in batch:
            remaining = max(r.output_tokens - 1, 0)
            finish_time = first_token_time + remaining * first_step
            results.append(
                RequestResult(
                    request_id=r.request_id,
                    arrival_time=r.arrival_time,
                    first_token_time=first_token_time,
                    finish_time=finish_time,
                    output_tokens=r.output_tokens,
                )
            )

        # Single server: it is busy until the slowest request completes in this decode batch.
        now = max(res.finish_time for res in results[-len(batch) :])

    return results
