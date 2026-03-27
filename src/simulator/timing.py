from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TimingModel:
    prefill_base: float = 0.020
    prefill_per_token: float = 0.00020
    decode_base: float = 0.004
    decode_per_token: float = 0.0025
    transfer_per_prompt_token: float = 0.000015
    colocated_interference: float = 1.35

    # Capacity split knobs (model GPU allocation effects).
    # Higher multiplier => slower service (less capacity).
    prefill_capacity_multiplier: float = 1.0
    decode_capacity_multiplier: float = 1.0

    # Simple batching speedups (bigger batch => better throughput).
    # This is intentionally lightweight; Stage B can replace these with measured curves.
    prefill_batch_alpha: float = 0.20
    decode_batch_alpha: float = 0.10

    def prefill_time(self, prompt_tokens: int, batch_size: int = 1) -> float:
        batch = max(int(batch_size), 1)
        speedup = batch ** self.prefill_batch_alpha
        base = self.prefill_base + self.prefill_per_token * prompt_tokens
        return (base / speedup) * self.prefill_capacity_multiplier

    def decode_step_time(self, batch_size: int = 1) -> float:
        batch = max(int(batch_size), 1)
        speedup = batch ** self.decode_batch_alpha
        base = self.decode_base + self.decode_per_token
        return (base / speedup) * self.decode_capacity_multiplier

    def transfer_time(self, prompt_tokens: int) -> float:
        return self.transfer_per_prompt_token * prompt_tokens
