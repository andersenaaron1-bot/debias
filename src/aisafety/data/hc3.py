"""HC3 dataset helpers for local use or Hugging Face download."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def first_nonempty_answer(lst):
    if not lst:
        return None
    for x in lst:
        if isinstance(x, str) and x.strip():
            return x.strip()
    return None


def load_hc3_descriptions(ds_split) -> pd.DataFrame:
    """Convert a datasets Split into a description-level DataFrame."""
    records = []
    for ex in ds_split:
        q = ex["question"]
        h = first_nonempty_answer(ex.get("human_answers", []))
        a = first_nonempty_answer(ex.get("chatgpt_answers", []))
        if not (q and h and a):
            continue

        hc3_src = ex.get("source", "unknown")
        records.append(
            {
                "item_type": "hc3",
                "title": q,
                "text": h,
                "source": "human",
                "hc3_source": hc3_src,
                "y": 0,
            }
        )
        records.append(
            {
                "item_type": "hc3",
                "title": q,
                "text": a,
                "source": "llm",
                "hc3_source": hc3_src,
                "y": 1,
            }
        )
    return pd.DataFrame(records)


def load_hc3_dataset(subset: str = "all", cache_dir: str | Path | None = None):
    """Load HC3 via huggingface datasets"""
    from datasets import load_dataset

    return load_dataset("Hello-SimpleAI/HC3", subset, cache_dir=str(cache_dir) if cache_dir else None)
