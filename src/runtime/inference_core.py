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
            out.append(x.to(device, non_blocking=True))
        else:
            out.append(x)
    return tuple(out)


def prepare_past_for_decode(past: Any, device: torch.device) -> Any:
    """
    Move KV to the decode GPU and return a `DynamicCache` ready for Llama's `.update()`.

    We bypass DynamicCache.__init__ / DynamicLayer.update() entirely because the
    constructor calls DynamicLayer.lazy_initialization(), which sets
      self.keys = torch.tensor([])   # shape [0], 1-D
    and then does torch.cat([1D_empty, 4D_kv], dim=-2) — an incompatible-rank cat
    that hits a slow PyTorch fallback (~90 ms per tensor × 44 layers ≈ 4 s per batch).
    Direct tensor assignment avoids this path and makes transfer time ~10 ms.

    Packed transfer (fast path): instead of 44 separate .to(device) calls (22 layers ×
    K+V), we torch.stack all K tensors and all V tensors into two packed tensors and
    move each in a single .to() call. This reduces cudaMalloc calls from 44 → 2,
    cutting transfer time from ~2 s to ~5 ms on PCIe (cudaMalloc overhead dominated
    the time, not bandwidth).
    """
    from transformers.cache_utils import DynamicCache, DynamicLayer

    rows = _extract_kv_rows_from_past(past)

    if all(len(row) == 2 for row in rows):
        # Fast path: pack all layers into two tensors → 2 cudaMallocs total.
        keys_packed = torch.stack([row[0] for row in rows], dim=0)  # [L, B, kv_h, seq, head]
        vals_packed = torch.stack([row[1] for row in rows], dim=0)
        keys_on_dev = keys_packed.to(device, non_blocking=True)
        vals_on_dev = vals_packed.to(device, non_blocking=True)
        new_layers = []
        for i in range(len(rows)):
            layer = DynamicLayer()
            layer.dtype = keys_on_dev.dtype
            layer.device = device
            layer.keys = keys_on_dev[i]
            layer.values = vals_on_dev[i]
            layer.is_initialized = True
            new_layers.append(layer)
    else:
        # Fallback for sliding-window or other non-standard cache variants.
        new_layers = []
        for row in rows:
            k = row[0].to(device, non_blocking=True)
            v = row[1].to(device, non_blocking=True)
            layer = DynamicLayer()
            layer.dtype = k.dtype
            layer.device = device
            layer.keys = k
            layer.values = v
            layer.is_initialized = True
            if len(row) > 2:
                layer._sliding_window_tensor = row[2].to(device, non_blocking=True)
            new_layers.append(layer)

    # Construct DynamicCache without calling __init__ to skip DynamicLayer.update().
    cache = object.__new__(DynamicCache)
    cache.layers = new_layers
    cache.layer_class_to_replicate = None
    cache.offloading = False
    return cache


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


def tokenize_batch(
    tokenizer,
    prompts: list[str],
    device: torch.device,
    max_prompt_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize a list of prompts with left-padding so the last real token aligns at
    position -1 across the batch. Returns (input_ids [B,L], attention_mask [B,L]).
    """
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    tokenizer.padding_side = orig_side
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def timed_prefill_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> tuple[float, Any, torch.Tensor]:
    """
    Batched prefill forward. Returns (seconds, past_key_values, next_token_ids [B,1]).
    """
    model.eval()
    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        _sync(device)
        prefill_s = time.perf_counter() - t0
    past = out.past_key_values
    next_tokens = out.logits[:, -1:, :].argmax(dim=-1)  # [B, 1]
    return prefill_s, past, next_tokens


def timed_decode_steps_batch(
    model: torch.nn.Module,
    past: Any,
    first_tokens: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[float], float]:
    """
    Greedy decode for `max_new_tokens` steps over a batch.
    All requests in the batch run together each step; per-request finish times
    are derived by the caller using step_times[:output_tokens].
    Returns (per_step_seconds, total_decode_seconds).
    """
    model.eval()
    next_tokens = first_tokens  # [B, 1]
    p = past
    step_times: list[float] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            _sync(device)
            t0 = time.perf_counter()
            out = model(input_ids=next_tokens, past_key_values=p, use_cache=True)
            _sync(device)
            step_times.append(time.perf_counter() - t0)
            next_tokens = out.logits[:, -1:, :].argmax(dim=-1)
            p = out.past_key_values
    return step_times, sum(step_times)
