"""Materialize train/val/test files from a scored cue corpus.

This writes both:
  1. raw split files containing every row for a given split
  2. balanced split files with one human row and one LLM row per group

The balanced view is the default cue-head training target because it prevents
multi-variant local generations or multi-generator H-LLMC2 rows from swamping
the human side.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from aisafety.config import DATA_DIR, DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=DATA_DIR / "derived" / "cue_discovery" / "splits")
    p.add_argument(
        "--balanced-out-dir",
        type=Path,
        default=DATA_DIR / "derived" / "cue_discovery" / "balanced_splits",
    )
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--balanced-llm-per-group", type=int, default=1)
    return p.parse_args()


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


def _write_split_rows(rows: list[dict[str, Any]], out_dir: Path, *, prefix: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: dict[str, str] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if str(row.get("split")) == split]
        out_path = out_dir / f"{prefix}_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_paths[split] = str(out_path)
    return out_paths


def _choose_rows_for_group(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    llm_per_group: int,
) -> list[dict[str, Any]]:
    humans = [row for row in rows if str(row.get("source")) == "human"]
    llms = [row for row in rows if str(row.get("source")) == "llm"]
    if not humans or not llms:
        return []

    def _row_key(row: dict[str, Any], source_tag: str) -> str:
        return _sha1_hex(
            f"{seed}:{row.get('group_id')}:{source_tag}:{row.get('generator')}:{row.get('example_id')}"
        )

    chosen: list[dict[str, Any]] = []
    chosen.append(sorted(humans, key=lambda row: _row_key(row, "human"))[0])
    chosen.extend(sorted(llms, key=lambda row: _row_key(row, "llm"))[: max(1, int(llm_per_group))])
    return chosen


def _balanced_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    llm_per_group: int,
) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get("group_id") or "")
        if group_id:
            by_group[group_id].append(row)

    kept: list[dict[str, Any]] = []
    for group_id in sorted(by_group):
        kept.extend(_choose_rows_for_group(by_group[group_id], seed=seed, llm_per_group=llm_per_group))
    return kept


def _count_by(rows: list[dict[str, Any]], key_name: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        bucket = str(row.get(key_name) or "")
        source = str(row.get("source") or "")
        buckets[bucket][source] += 1
    for bucket, counts in sorted(buckets.items()):
        out[bucket] = dict(sorted(counts.items()))
    return out


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(Path(args.input_jsonl))
    if not rows:
        raise ValueError(f"No rows found in {args.input_jsonl}")

    split_paths = _write_split_rows(rows, Path(args.out_dir), prefix="corpus_scored")
    balanced = _balanced_rows(rows, seed=int(args.seed), llm_per_group=int(args.balanced_llm_per_group))
    balanced_paths = _write_split_rows(
        balanced,
        Path(args.balanced_out_dir),
        prefix="corpus_scored_balanced",
    )

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "seed": int(args.seed),
        "balanced_llm_per_group": int(args.balanced_llm_per_group),
        "n_rows": int(len(rows)),
        "n_balanced_rows": int(len(balanced)),
        "split_paths": split_paths,
        "balanced_split_paths": balanced_paths,
        "by_dataset_source": _count_by(rows, "dataset"),
        "balanced_by_dataset_source": _count_by(balanced, "dataset"),
        "by_generator_source": _count_by(rows, "generator"),
        "balanced_by_generator_source": _count_by(balanced, "generator"),
    }

    summary_json = (
        Path(args.summary_json)
        if args.summary_json is not None
        else Path(args.balanced_out_dir).parent / "split_summary.json"
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote split files to {args.out_dir}")
    print(f"Wrote balanced split files to {args.balanced_out_dir}")
    print(f"Wrote split summary to {summary_json}")


if __name__ == "__main__":
    main()
