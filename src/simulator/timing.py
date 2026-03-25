from dataclasses import dataclass


@dataclass(slots=True)
class TimingModel:
    prefill_base: float = 0.020
    prefill_per_token: float = 0.00020
    decode_base: float = 0.004
    decode_per_token: float = 0.0025
    transfer_per_prompt_token: float = 0.000015
    colocated_interference: float = 1.35

    def prefill_time(self, prompt_tokens: int) -> float:
        return self.prefill_base + self.prefill_per_token * prompt_tokens

    def decode_step_time(self) -> float:
        return self.decode_base + self.decode_per_token

    def transfer_time(self, prompt_tokens: int) -> float:
        return self.transfer_per_prompt_token * prompt_tokens
