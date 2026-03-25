from __future__ import annotations

from typing import Iterable, List

from src.core.request import Request, RequestResult
from src.simulator.timing import TimingModel


def run_colocated(requests: Iterable[Request], timing: TimingModel) -> List[RequestResult]:
    current_time = 0.0
    results: List[RequestResult] = []

    for request in requests:
        current_time = max(current_time, request.arrival_time)
        prefill_done = current_time + timing.prefill_time(request.prompt_tokens) * timing.colocated_interference
        first_token_time = prefill_done

        decode_tokens_after_first = max(request.output_tokens - 1, 0)
        finish_time = first_token_time + (
            decode_tokens_after_first * timing.decode_step_time() * timing.colocated_interference
        )

        current_time = finish_time
        results.append(
            RequestResult(
                request_id=request.request_id,
                arrival_time=request.arrival_time,
                first_token_time=first_token_time,
                finish_time=finish_time,
                output_tokens=request.output_tokens,
            )
        )

    return results
