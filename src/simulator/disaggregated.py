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

    if cfg.batch_wait_s <= 0:
        return []

    next_arrival = queue[0].arrival_time
    if next_arrival <= now + cfg.batch_wait_s:
        now = next_arrival
        while queue and queue[0].arrival_time <= now and len(batch) < cfg.max_batch_size:
            batch.append(queue.popleft())
    return batch


def run_disaggregated(
    requests: Iterable[Request],
    timing: TimingModel,
    *,
    prefill_batching: BatchingConfig | None = None,
    decode_batching: BatchingConfig | None = None,
) -> List[RequestResult]:
    """
    Disaggregated design: separate prefill pool and decode pool with KV transfer cost.

    We model:
    - Prefill batching on the prefill pool.
    - Decode batching on the decode pool (first-token step + remaining tokens).
    - KV transfer time before a request becomes eligible for decode.
    """
    prefill_cfg = prefill_batching or BatchingConfig()
    decode_cfg = decode_batching or BatchingConfig()

    requests_list = list(requests)
    prefill_q: Deque[Request] = deque(sorted(requests_list, key=lambda r: r.arrival_time))
    # decode_q stores requests "arriving" at their decode-eligibility time (prefill + transfer).
    decode_q: Deque[Request] = deque()

    results: List[RequestResult] = []

    prefill_now = 0.0
    decode_now = 0.0

    # To keep things simple and deterministic, we first run prefill in time order,
    # pushing into decode_q with shifted arrival times, then run decode.
    while prefill_q:
        prefill_now = max(prefill_now, prefill_q[0].arrival_time)
        batch = _take_batch(prefill_q, now=prefill_now, cfg=prefill_cfg)
        if not batch:
            prefill_now = max(prefill_now, prefill_q[0].arrival_time)
            continue

        batch_prefill_time = max(
            timing.prefill_time(r.prompt_tokens, batch_size=len(batch)) for r in batch
        )
        prefill_done = prefill_now + batch_prefill_time

        for r in batch:
            eligible = prefill_done + timing.transfer_time(r.prompt_tokens)
            decode_q.append(
                Request(
                    request_id=r.request_id,
                    arrival_time=float(eligible),
                    prompt_tokens=r.prompt_tokens,
                    output_tokens=r.output_tokens,
                )
            )

        prefill_now = prefill_done

    decode_q = deque(sorted(list(decode_q), key=lambda r: r.arrival_time))

    while decode_q:
        decode_now = max(decode_now, decode_q[0].arrival_time)
        batch = _take_batch(decode_q, now=decode_now, cfg=decode_cfg)
        if not batch:
            decode_now = max(decode_now, decode_q[0].arrival_time)
            continue

        step = timing.decode_step_time(batch_size=len(batch))
        first_token_time = decode_now + step

        for r in batch:
            remaining = max(r.output_tokens - 1, 0)
            finish_time = first_token_time + remaining * step
            # Important: original arrival_time in RequestResult should be the *client arrival*,
            # but we overwrote it when enqueuing to decode_q. Recover by reading from id ordering:
            # In this scaffold we assume request_id is unique and results are compared within a run.
            # We keep decode_q arrival in local var; use (eligible - prefill/transfer) isn't tracked here.
            results.append(
                RequestResult(
                    request_id=r.request_id,
                    arrival_time=0.0,  # overwritten below
                    first_token_time=first_token_time,
                    finish_time=finish_time,
                    output_tokens=r.output_tokens,
                )
            )

        decode_now = max(res.finish_time for res in results[-len(batch) :])

    # Restore true client arrival times by looking at the original request list.
    # (This avoids threading arrival time through decode_q.)
    arrivals = {r.request_id: r.arrival_time for r in requests_list}
    for i, res in enumerate(results):
        results[i] = RequestResult(
            request_id=res.request_id,
            arrival_time=arrivals[res.request_id],
            first_token_time=res.first_token_time,
            finish_time=res.finish_time,
            output_tokens=res.output_tokens,
        )

    return results
