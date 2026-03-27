from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(slots=True)
class ShareGPTSample:
    """One conversation turn used as the prompt for profiling."""

    text: str
    source_id: int


def _conversation_to_prompt(conv: list[dict[str, Any]]) -> str | None:
    """Build a single user prompt string from ShareGPT-style turns."""
    if not conv:
        return None
    parts: list[str] = []
    for turn in conv:
        role = (turn.get("from") or turn.get("role") or "").lower()
        value = turn.get("value") or turn.get("content") or ""
        if not value:
            continue
        if role in ("human", "user", "system"):
            parts.append(value.strip())
        # Stop after first user message for a stable "prompt" (like chat input)
        if role in ("human", "user"):
            break
    if not parts:
        # Fallback: first non-empty value
        for turn in conv:
            v = turn.get("value") or turn.get("content")
            if v:
                return str(v).strip()
        return None
    return "\n".join(parts)


def iter_sharegpt_from_hf(
    *,
    dataset_name: str = "Aeala/ShareGPT_Vicuna_unfiltered",
    split: str = "train",
    max_samples: int = 200,
    trust_remote_code: bool = True,
) -> Iterator[ShareGPTSample]:
    """Stream prompts from a HuggingFace ShareGPT-style dataset."""
    from datasets import load_dataset
    from datasets.exceptions import DataFilesNotFoundError

    # Cap rows pulled from hub to avoid huge downloads.
    cap = max(max_samples * 10, 50)
    hf_split = split if "[" in split else f"{split}[:{cap}]"
    # Some older ShareGPT repos are just raw JSON without a dataset script and can fail with
    # DataFilesNotFoundError. Try a small set of common mirrors/reuploads automatically.
    candidates = [dataset_name]
    if dataset_name != "Aeala/ShareGPT_Vicuna_unfiltered":
        candidates.append("Aeala/ShareGPT_Vicuna_unfiltered")
    if dataset_name != "anon8231489123/ShareGPT_Vicuna_unfiltered":
        candidates.append("anon8231489123/ShareGPT_Vicuna_unfiltered")

    last_err: Exception | None = None
    ds = None
    for name in candidates:
        try:
            ds = load_dataset(name, split=hf_split, trust_remote_code=trust_remote_code)
            dataset_name = name
            break
        except DataFilesNotFoundError as e:
            last_err = e
        except Exception as e:
            last_err = e

    if ds is None:
        raise RuntimeError(
            f"Failed to load ShareGPT dataset from HF. Tried: {candidates}. \n"
            f"Last error: {last_err}. \n"
            "Fix: pass a local ShareGPT JSONL via --sharegpt-jsonl (run_gpu_comparison) or --jsonl (profile_sharegpt)."
        ) from last_err
    n = 0
    scanned = 0
    max_scan = max_samples * 200
    for i, row in enumerate(ds):
        scanned += 1
        if scanned > max_scan:
            break
        if n >= max_samples:
            break
        conv = row.get("conversations")
        if conv is None:
            continue
        prompt = _conversation_to_prompt(list(conv))
        if not prompt or len(prompt) < 8:
            continue
        yield ShareGPTSample(text=prompt, source_id=i)
        n += 1


def iter_sharegpt_jsonl(path: str | Path, max_samples: int = 200) -> Iterator[ShareGPTSample]:
    """Load ShareGPT-style JSONL: each line has `conversations` array."""
    p = Path(path)
    n = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            if n >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations")
            if not conv:
                continue
            prompt = _conversation_to_prompt(conv)
            if not prompt:
                continue
            yield ShareGPTSample(text=prompt, source_id=n)
            n += 1
