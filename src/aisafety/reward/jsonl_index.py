"""Random-access sampling from JSONL files via byte offsets."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonlOffsets:
    path: Path
    offsets: list[int]

    def __len__(self) -> int:  # pragma: no cover
        return len(self.offsets)

    def read_at(self, offset: int) -> dict[str, Any]:
        with self.path.open("rb") as f:
            f.seek(int(offset))
            line = f.readline()
        if not line:
            raise EOFError(f"Offset {offset} beyond EOF for {self.path}")
        return json.loads(line.decode("utf-8"))


def build_offsets(path: Path) -> JsonlOffsets:
    offsets: list[int] = []
    off = 0
    with path.open("rb") as f:
        for line in f:
            offsets.append(off)
            off += len(line)
    if not offsets:
        raise ValueError(f"No JSONL rows found in {path}")
    return JsonlOffsets(path=path, offsets=offsets)


def build_offsets_by_key(path: Path, *, key: str) -> dict[str, JsonlOffsets]:
    by_key: dict[str, list[int]] = defaultdict(list)
    off = 0
    with path.open("rb") as f:
        for line in f:
            row = json.loads(line.decode("utf-8"))
            kval = str(row.get(key) or "").strip()
            if kval:
                by_key[kval].append(off)
            off += len(line)
    out: dict[str, JsonlOffsets] = {}
    for kval, offsets in by_key.items():
        out[kval] = JsonlOffsets(path=path, offsets=offsets)
    if not out:
        raise ValueError(f"No rows with key={key!r} found in {path}")
    return out

