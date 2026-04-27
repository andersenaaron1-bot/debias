"""Shared IO helpers for D4 mechanistic tracing workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha1_hex(text: str) -> str:
    """Return a stable SHA1 hex digest for text identifiers."""

    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""

    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def read_jsonl(path: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Read JSONL rows, keeping only object rows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
                if max_rows is not None and len(rows) >= int(max_rows):
                    break
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object with deterministic key ordering."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def resolve_path(base_dir: Path, value: str | Path | None) -> Path | None:
    """Resolve a path-like value relative to a workspace root."""

    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def file_status(base_dir: Path, value: str | Path | None) -> dict[str, Any]:
    """Return path-resolution and existence metadata for a manifest path."""

    path = resolve_path(base_dir, value)
    return {
        "path": None if path is None else str(path),
        "exists": False if path is None else path.exists(),
        "is_file": False if path is None else path.is_file(),
    }

