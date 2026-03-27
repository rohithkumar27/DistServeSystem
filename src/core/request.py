from dataclasses import dataclass


@dataclass(slots=True)
class Request:
    request_id: int
    arrival_time: float
    prompt_tokens: int
    output_tokens: int
    # Set for GPU runs (ShareGPT / real prompts); simulator-only workloads may omit.
    prompt_text: str | None = None


@dataclass(slots=True)
class RequestResult:
    request_id: int
    arrival_time: float
    first_token_time: float
    finish_time: float
    output_tokens: int

    @property
    def ttft(self) -> float:
        return self.first_token_time - self.arrival_time

    @property
    def tpot(self) -> float:
        decode_tokens = max(self.output_tokens - 1, 1)
        return (self.finish_time - self.first_token_time) / decode_tokens
