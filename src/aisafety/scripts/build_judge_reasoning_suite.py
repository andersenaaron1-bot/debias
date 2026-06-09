"""Build a multi-domain canonical comparison suite from dataset specifications."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_json, read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.scripts.build_judge_reasoning_pairs import build_comparisons


DEFAULT_CONFIG = Path("configs") / "datasets" / "judge_reasoning_suite_v1.json"
DEFAULT_OUT_DIR = Path("data") / "derived" / "judge_reasoning_suite_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root used for relative input_jsonl paths; defaults to workspace-root.",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument(
        "--include-datasets",
        default="",
        help="Optional comma-separated dataset_id allowlist.",
    )
    parser.add_argument("--max-pairs-per-dataset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: str | Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _cap_pairs(
    rows: list[dict[str, Any]],
    *,
    max_pairs: int,
    seed: int,
    dataset_id: str,
) -> list[dict[str, Any]]:
    if int(max_pairs) <= 0:
        return rows
    pair_ids = sorted(
        {str(row["pair_id"]) for row in rows},
        key=lambda value: sha1_hex(f"{seed}:{dataset_id}:{value}"),
    )[: int(max_pairs)]
    keep = set(pair_ids)
    return [row for row in rows if str(row["pair_id"]) in keep]


def build_suite(
    config: dict[str, Any],
    *,
    workspace_root: Path,
    input_root: Path | None = None,
    skip_missing: bool,
    include_datasets: set[str] | None = None,
    max_pairs_per_dataset: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    specs = config.get("datasets") or []
    if not isinstance(specs, list) or not specs:
        raise ValueError("Suite config requires a nonempty datasets list.")
    included = {
        str(value).strip()
        for value in (include_datasets or set())
        if str(value).strip()
    }
    available = {
        str(spec.get("dataset_id") or "").strip()
        for spec in specs
        if isinstance(spec, dict)
    }
    unknown = sorted(included - available)
    if unknown:
        raise ValueError(f"Unknown --include-datasets values: {unknown}")

    for spec in specs:
        if not isinstance(spec, dict):
            raise TypeError("Each dataset specification must be an object.")
        dataset_id = str(spec.get("dataset_id") or "").strip()
        input_value = str(spec.get("input_jsonl") or "").strip()
        if not dataset_id or not input_value:
            raise ValueError("Each dataset requires dataset_id and input_jsonl.")
        if included and dataset_id not in included:
            continue
        input_path = _resolve(input_root or workspace_root, input_value)
        if not input_path.is_file():
            if not skip_missing and bool(spec.get("required", True)):
                raise FileNotFoundError(input_path)
            dataset_rows.append(
                {
                    "dataset_id": dataset_id,
                    "input_jsonl": str(input_path),
                    "status": "missing",
                    "n_input_rows": 0,
                    "n_pairs": 0,
                    "n_comparisons": 0,
                }
            )
            continue

        source = read_jsonl(input_path)
        static_metadata = dict(spec.get("metadata") or {})
        for key in ("validity_type", "difficulty_tier", "analysis_split"):
            if spec.get(key) is not None:
                static_metadata[key] = spec.get(key)
        if static_metadata:
            source = [
                {
                    **row,
                    **{
                        key: value
                        for key, value in static_metadata.items()
                        if row.get(key) is None
                    },
                }
                for row in source
            ]
        emitted = build_comparisons(
            source,
            input_format=str(spec.get("input_format") or "auto"),
            source_label=dataset_id,
            task_type=str(spec.get("task_type") or "pairwise_judgment"),
            comparison_dimension=str(spec.get("comparison_dimension") or "overall_quality"),
            prompt_key=str(spec.get("prompt_key") or "prompt"),
            option_a_key=str(spec.get("option_a_key") or "option_a_text"),
            option_b_key=str(spec.get("option_b_key") or "option_b_text"),
            chosen_key=str(spec.get("chosen_key") or "chosen"),
            rejected_key=str(spec.get("rejected_key") or "rejected"),
            target_key=str(spec.get("target_key") or "target_option"),
            pair_id_key=str(spec.get("pair_id_key") or "pair_id"),
            condition_id_key=str(spec.get("condition_id_key") or "condition_id"),
            condition_label_key=str(spec.get("condition_label_key") or "condition_label"),
            dimension_key=str(spec.get("dimension_key") or "comparison_dimension"),
            item_text_key=str(spec.get("item_text_key") or "input"),
            item_label_key=str(spec.get("item_label_key") or "label"),
            binary_positive_label=str(spec.get("binary_positive_label") or "1"),
            binary_group_key=str(spec.get("binary_group_key") or ""),
            binary_target_kind=str(spec.get("binary_target_kind") or "objective"),
            binary_question=str(
                spec.get("binary_question")
                or "Which option better satisfies the stated comparison dimension?"
            ),
            include_order_swaps=bool(spec.get("include_order_swaps", True)),
            seed=int(seed),
        )
        cap = int(spec.get("max_pairs") or max_pairs_per_dataset)
        emitted = _cap_pairs(
            emitted,
            max_pairs=cap,
            seed=int(seed),
            dataset_id=dataset_id,
        )
        comparisons.extend(emitted)
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "input_jsonl": str(input_path),
                "status": "ok",
                "comparison_dimension": str(
                    spec.get("comparison_dimension") or "overall_quality"
                ),
                "task_type": str(spec.get("task_type") or "pairwise_judgment"),
                "n_input_rows": int(len(source)),
                "n_pairs": int(len({row["pair_id"] for row in emitted})),
                "n_comparisons": int(len(emitted)),
            }
        )
    return comparisons, dataset_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    config_path = _resolve(workspace_root, args.config)
    input_root = (
        _resolve(workspace_root, args.input_root)
        if args.input_root is not None
        else workspace_root
    )
    out_dir = _resolve(workspace_root, args.out_dir)
    config = read_json(config_path)
    include_datasets = {
        value.strip()
        for value in str(args.include_datasets).split(",")
        if value.strip()
    }
    comparisons, dataset_rows = build_suite(
        config,
        workspace_root=workspace_root,
        input_root=input_root,
        skip_missing=bool(args.skip_missing),
        include_datasets=include_datasets,
        max_pairs_per_dataset=int(args.max_pairs_per_dataset),
        seed=int(args.seed),
    )
    if not comparisons:
        raise ValueError("Suite emitted no comparisons.")

    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons_path = out_dir / "comparisons.jsonl"
    dataset_path = out_dir / "datasets.jsonl"
    _write_jsonl(comparisons_path, comparisons)
    _write_jsonl(dataset_path, dataset_rows)
    dimension_counts = Counter(str(row["comparison_dimension"]) for row in comparisons)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-reasoning-suite-build",
            "suite_id": str(config.get("suite_id") or config_path.stem),
            "config": str(config_path),
            "input_root": str(input_root),
            "out_dir": str(out_dir),
            "comparisons_jsonl": str(comparisons_path),
            "datasets_jsonl": str(dataset_path),
            "seed": int(args.seed),
            "skip_missing": bool(args.skip_missing),
            "include_datasets": sorted(include_datasets),
            "n_datasets": int(sum(row["status"] == "ok" for row in dataset_rows)),
            "n_pairs": int(len({row["pair_id"] for row in comparisons})),
            "n_comparisons": int(len(comparisons)),
            "counts_by_dimension": dict(sorted(dimension_counts.items())),
        },
    )
    print(f"comparisons={comparisons_path}")
    print(f"n_datasets={sum(row['status'] == 'ok' for row in dataset_rows)}")
    print(f"n_pairs={len({row['pair_id'] for row in comparisons})}")
    print(f"n_comparisons={len(comparisons)}")


if __name__ == "__main__":
    main()
