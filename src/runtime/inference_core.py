"""
Shared causal-LM timing: prefill forward + greedy decode steps with KV cache.

Aligned with DistServe’s separation of **prefill** (full prompt → KV) vs **decode**
(autoregressive steps), without SwiftTransformer — HuggingFace `forward` only.
"""
from __future__ import annotations

import time
from typing import Any

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _move_all_tensors(obj: Any, device: torch.device) -> Any:
    """Recursively move any tensors inside nested containers to `device`."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_all_tensors(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_all_tensors(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_all_tensors(v, device) for k, v in obj.items()}
    return obj


def _extract_kv_rows_from_past(past: Any) -> list[tuple[Any, ...]]:
    """
    Build a list of per-layer tuples suitable for `DynamicCache(ddp_cache_data=...)`.

    HF returns either a `DynamicCache` (has `.layers` with `.keys`/`.values`) or a legacy
    tuple of (key, value) per layer.
    """
    if past is None:
        raise ValueError("past_key_values is None")

    if hasattr(past, "layers") and len(getattr(past, "layers")) > 0:
        rows: list[tuple[Any, ...]] = []
        for layer in past.layers:
            if not getattr(layer, "is_initialized", False):
                raise RuntimeError("Cache layer not initialized after prefill; cannot transfer KV.")
            k = layer.keys
            v = layer.values
            sw = getattr(layer, "_sliding_window_tensor", None)
            if sw is not None:
                rows.append((k, v, sw))
            else:
                rows.append((k, v))
        return rows

    if isinstance(past, tuple):
        return list(past)

    raise TypeError(f"Unsupported past_key_values type for KV transfer: {type(past)}")


def _move_kv_row(row: tuple[Any, ...], device: torch.device) -> tuple[Any, ...]:
    out: list[Any] = []
    for x in row:
        if isinstance(x, torch.Tensor):
            out.append(x.to(device, non_blocking=False))
        else:
            out.append(x)
    return tuple(out)


def prepare_past_for_decode(past: Any, device: torch.device) -> Any:
    """
    Move KV to the decode GPU and return a `DynamicCache` that Llama can `.update()` on that device.

    Important: `DynamicCache` objects are **not** walked by recursive tensor moves; we must read
    `layer.keys` / `layer.values` and rebuild `DynamicCache(ddp_cache_data=...)` on the target device.
    """
    rows = _extract_kv_rows_from_past(past)
    moved_rows = [_move_kv_row(r, device) for r in rows]

    from transformers.cache_utils import DynamicCache

    return DynamicCache(ddp_cache_data=moved_rows, config=None)


def time_transfer(
    past: Any, target: torch.device, source_device: torch.device | None = None
) -> tuple[float, Any]:
    """Wall time to copy KV to decode device; returns (seconds, past on target)."""
    if source_device is not None and source_device.type == "cuda":
        torch.cuda.synchronize(source_device)
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    moved = prepare_past_for_decode(past, target)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    return time.perf_counter() - t0, moved


def timed_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> tuple[float, Any, torch.Tensor]:
    """
    One prefill forward over full prompt. Returns (seconds, past_key_values, next_token_ids [B,1]).
    """
    model.eval()
    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True)
        _sync(device)
        prefill_s = time.perf_counter() - t0
    past = out.past_key_values
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    return prefill_s, past, next_token


def timed_decode_steps(
    model: torch.nn.Module,
    past: Any,
    first_token: torch.Tensor,
    num_tokens: int,
    device: torch.device,
) -> tuple[list[float], float]:
    """
    Greedy decode for `num_tokens` steps. Returns (per_step_seconds, total_decode_seconds).
    """
    model.eval()
    next_token = first_token
    p = past
    step_times: list[float] = []
    with torch.no_grad():
        for _ in range(num_tokens):
            _sync(device)
            t0 = time.perf_counter()
            out = model(input_ids=next_token, past_key_values=p, use_cache=True)
            _sync(device)
            step_times.append(time.perf_counter() - t0)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            p = out.past_key_values
    total = sum(step_times)
    return step_times, total


def tokenize_prompt(
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_prompt_tokens: int,
) -> torch.Tensor:
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    return enc["input_ids"].to(device)
