from __future__ import annotations

from typing import Iterable, List

from src.core.request import Request, RequestResult
from src.simulator.timing import TimingModel


def run_disaggregated(requests: Iterable[Request], timing: TimingModel) -> List[RequestResult]:
    prefill_available_time = 0.0
    decode_available_time = 0.0
    results: List[RequestResult] = []

    for request in requests:
        prefill_start = max(prefill_available_time, request.arrival_time)
        prefill_done = prefill_start + timing.prefill_time(request.prompt_tokens)
        first_token_time = prefill_done
        prefill_available_time = prefill_done

        transfer_done = prefill_done + timing.transfer_time(request.prompt_tokens)
        decode_start = max(decode_available_time, transfer_done)

        decode_tokens_after_first = max(request.output_tokens - 1, 0)
        finish_time = decode_start + decode_tokens_after_first * timing.decode_step_time()
        decode_available_time = finish_time

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
