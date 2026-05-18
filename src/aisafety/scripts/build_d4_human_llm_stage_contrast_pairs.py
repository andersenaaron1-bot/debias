"""Build order-swapped human-vs-LLM BT rows for training-stage contrasts."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

from aisafety.config import DATA_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text, token_count
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.scripts.build_d4_human_llm_alignment_pairs import _csv_set
from aisafety.scripts.run_d4_surface_counterfactual_audit import _cap_rows


DEFAULT_PAIR_JSONL = DATA_DIR / "derived" / "d4_human_llm_alignment_pairs_v1" / "pairs.jsonl"
DEFAULT_OUT_DIR = DATA_DIR / "derived" / "d4_human_llm_stage_contrast_pairs_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-datasets", type=str, default="")
    parser.add_argument("--exclude-datasets", type=str, default="")
    parser.add_argument("--include-subsets", type=str, default="")
    parser.add_argument("--include-item-types", type=str, default="")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--include-order-swaps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, limit: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = rows[: int(limit)]
    fields: list[str] = []
    for row in preview:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(preview)


def _prompt(row: dict[str, Any]) -> str:
    for key in ("prompt", "question", "title"):
        value = flat_text(str(row.get(key) or ""))
        if value:
            return value
    return "Compare the two responses."


def _filtered_pairs(
    rows: list[dict[str, Any]],
    *,
    include_datasets: set[str],
    exclude_datasets: set[str],
    include_subsets: set[str],
    include_item_types: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        source_dataset = str(row.get("source_dataset") or "")
        subset = str(row.get("subset") or "")
        item_type = str(row.get("item_type") or "")
        if include_datasets and source_dataset not in include_datasets:
            continue
        if source_dataset in exclude_datasets:
            continue
        if include_subsets and subset not in include_subsets:
            continue
        if include_item_types and item_type not in include_item_types:
            continue
        if not flat_text(str(row.get("human_text") or "")):
            continue
        if not flat_text(str(row.get("llm_text") or "")):
            continue
        out.append(row)
    return out


def _bt_row(row: dict[str, Any], *, order: str) -> dict[str, Any]:
    human_text = flat_text(str(row.get("human_text") or ""))
    llm_text = flat_text(str(row.get("llm_text") or ""))
    if order == "llm_first":
        option_a = llm_text
        option_b = human_text
        llm_option = "A"
        human_option = "B"
    elif order == "human_first":
        option_a = human_text
        option_b = llm_text
        llm_option = "B"
        human_option = "A"
    else:
        raise ValueError(f"Unknown order: {order}")

    pair_id = str(row.get("pair_id") or sha1_hex(human_text + "\n" + llm_text))
    bt_pair_id = sha1_hex(
        "|".join(
            [
                pair_id,
                str(row.get("source_dataset") or ""),
                str(row.get("subset") or ""),
                order,
                sha1_hex(option_a),
                sha1_hex(option_b),
            ]
        )
    )
    human_tokens = token_count(human_text)
    llm_tokens = token_count(llm_text)
    return {
        "bt_pair_id": bt_pair_id,
        "pair_id": pair_id,
        "source_dataset": str(row.get("source_dataset") or ""),
        "subset": str(row.get("subset") or ""),
        "split": str(row.get("split") or ""),
        "item_type": str(row.get("item_type") or ""),
        "bundle_creation_role": str(row.get("bundle_creation_role") or ""),
        "group_id": str(row.get("group_id") or ""),
        "title": str(row.get("title") or ""),
        "question": str(row.get("question") or ""),
        "prompt": _prompt(row),
        "llm_generator": str(row.get("llm_generator") or ""),
        "presentation_order": order,
        "llm_option": llm_option,
        "human_option": human_option,
        "option_a_text": option_a,
        "option_b_text": option_b,
        "human_text": human_text,
        "llm_text": llm_text,
        "option_a_sha1": sha1_hex(option_a),
        "option_b_sha1": sha1_hex(option_b),
        "human_sha1": sha1_hex(human_text),
        "llm_sha1": sha1_hex(llm_text),
        "option_a_tokens": token_count(option_a),
        "option_b_tokens": token_count(option_b),
        "human_tokens": int(human_tokens),
        "llm_tokens": int(llm_tokens),
        "token_delta_llm_minus_human": int(llm_tokens - human_tokens),
        "length_ratio_llm_over_human": float(llm_tokens / max(float(human_tokens), 1.0)),
    }


def build_hllm_bt_rows(
    pair_rows: list[dict[str, Any]],
    *,
    include_order_swaps: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pair_rows:
        orders = ["llm_first", "human_first"] if include_order_swaps else ["llm_first"]
        for order in orders:
            rows.append(_bt_row(row, order=order))

    counts_source = Counter(str(row["source_dataset"]) for row in rows)
    counts_source_subset = Counter(f"{row['source_dataset']}::{row['subset']}" for row in rows)
    counts_order = Counter(str(row["presentation_order"]) for row in rows)
    counts_item_type = Counter(str(row["item_type"]) for row in rows)
    summary = {
        "n_bt_pairs": int(len(rows)),
        "n_source_pairs_used": int(len({row["pair_id"] for row in rows})),
        "include_order_swaps": bool(include_order_swaps),
        "counts_by_source_dataset": dict(sorted(counts_source.items())),
        "counts_by_source_dataset_subset": dict(sorted(counts_source_subset.items())),
        "counts_by_presentation_order": dict(sorted(counts_order.items())),
        "counts_by_item_type": dict(sorted(counts_item_type.items())),
    }
    return rows, summary


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    pair_path = _resolve(workspace_root, args.pair_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)

    source_rows = _filtered_pairs(
        read_jsonl(pair_path),
        include_datasets=_csv_set(str(args.include_datasets)),
        exclude_datasets=_csv_set(str(args.exclude_datasets)),
        include_subsets=_csv_set(str(args.include_subsets)),
        include_item_types=_csv_set(str(args.include_item_types)),
    )
    source_rows = _cap_rows(source_rows, max_rows=int(args.max_pairs), seed=int(args.seed))
    if not source_rows:
        raise ValueError(f"No human-vs-LLM rows selected from {pair_path}")

    bt_rows, summary = build_hllm_bt_rows(source_rows, include_order_swaps=bool(args.include_order_swaps))
    if not bt_rows:
        raise ValueError("No human-vs-LLM BT rows were emitted.")

    out_dir.mkdir(parents=True, exist_ok=True)
    bt_path = out_dir / "bt_pairs.jsonl"
    preview_path = out_dir / "bt_pairs_preview.csv"
    summary_path = out_dir / "summary.json"
    _write_jsonl(bt_path, bt_rows)
    _write_csv(preview_path, bt_rows)
    write_json(
        summary_path,
        {
            "stage": "D4-human-LLM-stage-contrast-pair-build",
            "pair_jsonl": str(pair_path),
            "out_dir": str(out_dir),
            "bt_pairs_jsonl": str(bt_path),
            "preview_csv": str(preview_path),
            "seed": int(args.seed),
            "include_datasets": sorted(_csv_set(str(args.include_datasets))),
            "exclude_datasets": sorted(_csv_set(str(args.exclude_datasets))),
            "include_subsets": sorted(_csv_set(str(args.include_subsets))),
            "include_item_types": sorted(_csv_set(str(args.include_item_types))),
            "max_pairs": int(args.max_pairs),
            **summary,
        },
    )
    print(f"bt_pairs={bt_path}")
    print(f"summary={summary_path}")
    print(f"n_bt_pairs={len(bt_rows)}")
    print(f"n_source_pairs={summary['n_source_pairs_used']}")


if __name__ == "__main__":
    main()
