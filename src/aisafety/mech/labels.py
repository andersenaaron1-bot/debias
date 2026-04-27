"""Dataset sampling, atom-label, and content-anchor split helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from aisafety.mech.d4_io import sha1_hex


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list, deduplicating in input order."""

    out: list[int] = []
    seen: set[int] = set()
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        item = int(part)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def parse_str_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated string list."""

    if value is None:
        return None
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    return out or None


def select_hidden_layers(num_layers: int, *, stride: int, tail_layers: int) -> list[int]:
    """Select residual hidden-state indices for a coarse-to-late layer sweep."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    selected = {1, int(num_layers)}
    step = max(1, int(stride))
    selected.update(range(step, int(num_layers) + 1, step))
    tail = max(0, int(tail_layers))
    if tail:
        selected.update(range(max(1, int(num_layers) - tail + 1), int(num_layers) + 1))
    return sorted(int(x) for x in selected if 1 <= int(x) <= int(num_layers))


def sample_atom_probe_rows(
    rows: list[dict[str, Any]],
    *,
    max_train_per_item_type: int,
    max_val_per_item_type: int,
    max_test_per_item_type: int,
) -> pd.DataFrame:
    """Deterministically cap atom-probe rows by split and item type."""

    limits = {
        "train": int(max_train_per_item_type),
        "val": int(max_val_per_item_type),
        "test": int(max_test_per_item_type),
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("split") or ""), str(row.get("item_type") or ""))
        grouped.setdefault(key, []).append(row)

    kept: list[dict[str, Any]] = []
    for (split, _item_type), group_rows in grouped.items():
        limit = limits.get(split, 0)
        ordered = sorted(
            group_rows,
            key=lambda row: sha1_hex(str(row.get("example_id") or row.get("text") or "")),
        )
        if limit > 0:
            ordered = ordered[:limit]
        kept.extend(ordered)
    return pd.DataFrame(kept)


def build_atom_label_frame(df: pd.DataFrame, *, atoms: list[str], q: float) -> pd.DataFrame:
    """Add high/low quantile atom labels within each item type."""

    out = df.copy()
    for atom in atoms:
        label_col = f"{atom}__label"
        out[label_col] = -1
        for _item_type, group in out.groupby("item_type"):
            train_scores = group.loc[group["split"] == "train", "atom_scores"].map(
                lambda scores: float((scores or {}).get(atom, 0.0))
            )
            if train_scores.empty:
                continue
            lo = float(train_scores.quantile(1.0 - float(q)))
            hi = float(train_scores.quantile(float(q)))
            group_scores = group["atom_scores"].map(lambda scores: float((scores or {}).get(atom, 0.0)))
            pos_idx = group.index[group_scores >= hi]
            neg_idx = group.index[group_scores <= lo]
            out.loc[pos_idx, label_col] = 1
            out.loc[neg_idx, label_col] = 0
        out[f"{atom}__score"] = out["atom_scores"].map(lambda scores: float((scores or {}).get(atom, 0.0)))
    return out


def raw_content_pair_id(row: dict[str, Any]) -> str:
    """Return a real pair id, ignoring common placeholder ids."""

    pair_id = str(row.get("pair_id") or "").strip()
    if pair_id and pair_id.lower() not in {"nan", "none", "null"}:
        return pair_id
    return ""


def content_pair_id(row: dict[str, Any], *, index: int, id_counts: Counter[str]) -> str:
    """Return a unique deterministic id for a content-anchor pair."""

    pair_id = raw_content_pair_id(row)
    if pair_id and id_counts[pair_id] == 1:
        return pair_id
    parts = [
        pair_id,
        str(row.get("source_dataset") or ""),
        str(row.get("domain") or ""),
        str(row.get("prompt") or ""),
        str(row.get("chosen_text") or row.get("chosen") or ""),
        str(row.get("rejected_text") or row.get("rejected") or ""),
        str(index),
    ]
    return f"synthetic:{sha1_hex(chr(31).join(parts))}"


def flatten_content_pairs(rows: list[dict[str, Any]], *, seed: int, max_pairs: int) -> pd.DataFrame:
    """Flatten chosen/rejected pairs after a deterministic pair-level split."""

    id_counts = Counter(raw_content_pair_id(row) for row in rows)
    indexed_rows = [(content_pair_id(row, index=i, id_counts=id_counts), row) for i, row in enumerate(rows)]
    ordered = sorted(indexed_rows, key=lambda item: sha1_hex(f"{int(seed)}:{item[0]}"))
    if max_pairs > 0:
        ordered = ordered[: int(max_pairs)]

    n_pairs = len(ordered)
    n_train = int(0.8 * n_pairs)
    n_val = int(0.1 * n_pairs)
    text_rows: list[dict[str, Any]] = []
    for pair_index, (pair_id, row) in enumerate(ordered):
        if pair_index < n_train:
            split = "train"
        elif pair_index < n_train + n_val:
            split = "val"
        else:
            split = "test"
        for label, key in ((1, "chosen_text"), (0, "rejected_text")):
            text_rows.append(
                {
                    "pair_id": pair_id,
                    "split": split,
                    "label": int(label),
                    "domain": row.get("domain"),
                    "source_dataset": row.get("source_dataset"),
                    "text": str(row.get(key) or ""),
                }
            )
    return pd.DataFrame(text_rows)


def split_counts(df: pd.DataFrame, *, split_col: str = "split") -> dict[str, int]:
    """Return sorted split counts for a dataframe."""

    if df.empty or split_col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[split_col].value_counts().sort_index().items()}


def has_nonzero_train_val_test(counts: dict[str, int]) -> bool:
    """Check whether split counts satisfy the D4 recovery gate."""

    return all(int(counts.get(split, 0)) > 0 for split in ("train", "val", "test"))

