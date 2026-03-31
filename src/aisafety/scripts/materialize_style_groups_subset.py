"""Materialize a filtered style-group subset for reward-invariance experiments.

This is intended to derive small, experiment-specific slices from an existing
style-group export without rebuilding the full rewrite suite.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from aisafety.config import DATA_DIR, DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DATA_DIR / "derived" / "style_groups")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--include-axes", type=str, required=True, help="Comma-separated source axes to keep.")
    p.add_argument(
        "--rename-axes",
        type=str,
        default="",
        help="Comma-separated rename rules like paws_surface=paraphrase_surface",
    )
    p.add_argument("--max-groups-per-axis", type=int, default=0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _parse_rename_map(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in _parse_csv(text):
        if "=" not in part:
            raise ValueError(f"Invalid rename spec {part!r}; expected old=new")
        old, new = part.split("=", 1)
        out[old.strip()] = new.strip()
    return out


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _rename_axis(row: dict[str, Any], rename_map: dict[str, str]) -> dict[str, Any]:
    out = dict(row)
    axis = str(out.get("style_axis") or "")
    out["style_axis"] = rename_map.get(axis, axis)
    return out


def _limit_rows(
    rows: list[dict[str, Any]],
    *,
    max_groups_per_axis: int,
    seed: int,
) -> list[dict[str, Any]]:
    if int(max_groups_per_axis) <= 0:
        return rows

    by_axis_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        axis = str(row.get("style_axis") or "")
        group_id = str(row.get("group_id") or "")
        if axis and group_id:
            by_axis_group[axis][group_id].append(row)

    kept_group_ids: set[tuple[str, str]] = set()
    for axis, groups in by_axis_group.items():
        ordered = sorted(groups, key=lambda gid: _sha1_hex(f"{seed}:{axis}:{gid}"))
        keep = ordered[: int(max_groups_per_axis)]
        kept_group_ids.update((axis, gid) for gid in keep)

    return [
        row
        for row in rows
        if (str(row.get("style_axis") or ""), str(row.get("group_id") or "")) in kept_group_ids
    ]


def _summarize_rows(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    split_counts: dict[str, dict[str, int]] = {}
    group_counts: dict[str, dict[str, int]] = {}
    for split, rows in sorted(rows_by_split.items()):
        axis_counter: Counter[str] = Counter()
        groups_by_axis: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            axis = str(row.get("style_axis") or "")
            group_id = str(row.get("group_id") or "")
            axis_counter[axis] += 1
            groups_by_axis[axis].add(group_id)
        split_counts[split] = dict(sorted(axis_counter.items()))
        group_counts[split] = {axis: len(groups) for axis, groups in sorted(groups_by_axis.items())}
    return {"row_counts_by_split_axis": split_counts, "group_counts_by_split_axis": group_counts}


def main() -> None:
    args = parse_args()
    include_axes = set(_parse_csv(str(args.include_axes)))
    rename_map = _parse_rename_map(str(args.rename_axes))
    if not include_axes:
        raise ValueError("--include-axes must not be empty")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        in_path = Path(args.input_dir) / f"style_groups_{split}.jsonl"
        rows = _load_jsonl(in_path)
        rows = [row for row in rows if str(row.get("style_axis") or "") in include_axes]
        rows = [_rename_axis(row, rename_map) for row in rows]
        rows = _limit_rows(rows, max_groups_per_axis=int(args.max_groups_per_axis), seed=int(args.seed))
        out_path = Path(args.out_dir) / f"style_groups_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows_by_split[split] = rows

    summary = {
        "input_dir": str(args.input_dir),
        "out_dir": str(args.out_dir),
        "include_axes": sorted(include_axes),
        "rename_axes": rename_map,
        "max_groups_per_axis": int(args.max_groups_per_axis),
        "seed": int(args.seed),
        **_summarize_rows(rows_by_split),
    }
    summary_path = (
        Path(args.summary_json)
        if args.summary_json is not None
        else Path(args.out_dir) / "summary.json"
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote subset style groups to {args.out_dir}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
