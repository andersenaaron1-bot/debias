"""JSONL utilities + lightweight schema validation for reward pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _require_str(val, *, name: str) -> str:
    if not isinstance(val, str):
        raise TypeError(f"{name} must be a string, got {type(val).__name__}")
    if not val.strip():
        raise ValueError(f"{name} must be non-empty")
    return val


def validate_style_group(row: dict) -> None:
    if not isinstance(row, dict):
        raise TypeError("style group row must be a dict")
    _require_str(row.get("group_id"), name="group_id")
    _require_str(row.get("style_axis"), name="style_axis")
    _require_str(row.get("source_dataset"), name="source_dataset")
    _require_str(row.get("domain"), name="domain")
    variants = row.get("variants")
    if not isinstance(variants, list) or not variants:
        raise TypeError("variants must be a non-empty list of strings")
    cleaned = [v for v in variants if isinstance(v, str) and v.strip()]
    if len(cleaned) < 2:
        raise ValueError("variants must contain at least 2 non-empty strings")


def validate_pref_pair(row: dict) -> None:
    if not isinstance(row, dict):
        raise TypeError("preference pair row must be a dict")
    _require_str(row.get("pair_id"), name="pair_id")
    _require_str(row.get("source_dataset"), name="source_dataset")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not (isinstance(chosen, str) and chosen.strip()):
        raise ValueError("chosen must be a non-empty string")
    if not (isinstance(rejected, str) and rejected.strip()):
        raise ValueError("rejected must be a non-empty string")

