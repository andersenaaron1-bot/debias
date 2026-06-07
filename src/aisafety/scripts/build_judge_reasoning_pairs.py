"""Build canonical pairwise comparisons for judge-reasoning trajectory studies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.mech.judge_reasoning import (
    JudgeComparison,
    normalize_choice,
    opposite_choice,
    row_metadata,
)


DEFAULT_OUT_DIR = Path("data") / "derived" / "judge_reasoning_pairs_v1"
CANONICAL_KEYS = {
    "comparison_id",
    "pair_id",
    "source_dataset",
    "subset",
    "split",
    "task_type",
    "comparison_dimension",
    "prompt",
    "option_a_text",
    "option_b_text",
    "target_option",
    "target_kind",
    "presentation_order",
    "condition_id",
    "condition_label",
    "metadata",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--input-format",
        choices=["auto", "canonical", "bt", "preference", "binary_items", "generic"],
        default="auto",
    )
    parser.add_argument("--source-label", default="")
    parser.add_argument("--task-type", default="pairwise_judgment")
    parser.add_argument("--comparison-dimension", default="overall_quality")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--option-a-key", default="option_a_text")
    parser.add_argument("--option-b-key", default="option_b_text")
    parser.add_argument("--chosen-key", default="chosen")
    parser.add_argument("--rejected-key", default="rejected")
    parser.add_argument("--target-key", default="target_option")
    parser.add_argument("--pair-id-key", default="pair_id")
    parser.add_argument("--condition-id-key", default="condition_id")
    parser.add_argument("--condition-label-key", default="condition_label")
    parser.add_argument("--dimension-key", default="comparison_dimension")
    parser.add_argument("--item-text-key", default="input")
    parser.add_argument("--item-label-key", default="label")
    parser.add_argument("--binary-positive-label", default="1")
    parser.add_argument("--binary-group-key", default="")
    parser.add_argument(
        "--binary-target-kind",
        choices=["objective", "consensus", "preference"],
        default="objective",
    )
    parser.add_argument(
        "--binary-question",
        default="Which option better satisfies the stated comparison dimension?",
    )
    parser.add_argument(
        "--include-order-swaps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _detected_format(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "canonical"
    keys = set(rows[0])
    if {"option_a_text", "option_b_text"}.issubset(keys):
        return "canonical" if "comparison_id" in keys else "bt"
    if {"chosen", "rejected"}.issubset(keys):
        return "preference"
    return "generic"


def _base_id(row: dict[str, Any], *, pair_id_key: str, option_a: str, option_b: str) -> str:
    raw = str(row.get(pair_id_key) or row.get("pair_id") or "").strip()
    return raw or sha1_hex(option_a + "\n" + option_b)


def _comparison(
    *,
    row: dict[str, Any],
    source_label: str,
    pair_id: str,
    prompt: str,
    option_a: str,
    option_b: str,
    target_option: str,
    target_kind: str,
    presentation_order: str,
    task_type: str,
    comparison_dimension: str,
    condition_id_key: str,
    condition_label_key: str,
) -> JudgeComparison:
    comparison_id = sha1_hex(
        "|".join(
            [
                source_label,
                pair_id,
                presentation_order,
                sha1_hex(option_a),
                sha1_hex(option_b),
                str(row.get(condition_id_key) or ""),
            ]
        )
    )
    return JudgeComparison(
        comparison_id=comparison_id,
        pair_id=pair_id,
        source_dataset=str(row.get("source_dataset") or source_label),
        subset=str(row.get("subset") or ""),
        split=str(row.get("split") or ""),
        task_type=str(row.get("task_type") or task_type),
        comparison_dimension=str(row.get("comparison_dimension") or comparison_dimension),
        prompt=prompt,
        option_a_text=option_a,
        option_b_text=option_b,
        target_option=normalize_choice(target_option),
        target_kind=target_kind if normalize_choice(target_option) else "none",
        presentation_order=presentation_order,
        condition_id=str(row.get(condition_id_key) or row.get("counterfactual_id") or ""),
        condition_label=str(row.get(condition_label_key) or row.get("direction") or ""),
        metadata=row_metadata(row, exclude=CANONICAL_KEYS),
    )


def _swap(comparison: JudgeComparison) -> JudgeComparison:
    order_pairs = {
        "original": "swapped",
        "swapped": "original",
        "positive_first": "negative_first",
        "negative_first": "positive_first",
        "chosen_first": "rejected_first",
        "rejected_first": "chosen_first",
    }
    presentation_order = order_pairs.get(
        comparison.presentation_order,
        f"swapped:{comparison.presentation_order}" if comparison.presentation_order else "swapped",
    )
    comparison_id = sha1_hex(
        "|".join(
            [
                comparison.source_dataset,
                comparison.pair_id,
                presentation_order,
                sha1_hex(comparison.option_b_text),
                sha1_hex(comparison.option_a_text),
                comparison.condition_id,
            ]
        )
    )
    return JudgeComparison(
        comparison_id=comparison_id,
        pair_id=comparison.pair_id,
        source_dataset=comparison.source_dataset,
        subset=comparison.subset,
        split=comparison.split,
        task_type=comparison.task_type,
        comparison_dimension=comparison.comparison_dimension,
        prompt=comparison.prompt,
        option_a_text=comparison.option_b_text,
        option_b_text=comparison.option_a_text,
        target_option=opposite_choice(comparison.target_option),
        target_kind=comparison.target_kind,
        presentation_order=presentation_order,
        condition_id=comparison.condition_id,
        condition_label=comparison.condition_label,
        metadata=dict(comparison.metadata or {}),
    )


def _normalize_pair_rows(
    rows: list[dict[str, Any]],
    *,
    input_format: str,
    source_label: str,
    task_type: str,
    comparison_dimension: str,
    prompt_key: str,
    option_a_key: str,
    option_b_key: str,
    chosen_key: str,
    rejected_key: str,
    target_key: str,
    pair_id_key: str,
    condition_id_key: str,
    condition_label_key: str,
    dimension_key: str,
    include_order_swaps: bool,
) -> list[JudgeComparison]:
    out: list[JudgeComparison] = []
    for row in rows:
        if input_format == "preference":
            option_a = flat_text(str(row.get(chosen_key) or ""))
            option_b = flat_text(str(row.get(rejected_key) or ""))
            target = "A"
            target_kind = "preference"
        else:
            option_a = flat_text(
                str(row.get(option_a_key) or row.get("option_a_text") or "")
            )
            option_b = flat_text(
                str(row.get(option_b_key) or row.get("option_b_text") or "")
            )
            target = normalize_choice(row.get(target_key) or row.get("target_option"))
            target_kind = str(row.get("target_kind") or ("objective" if target else "none"))
        if not option_a or not option_b:
            continue
        prompt = flat_text(
            str(
                row.get(prompt_key)
                or row.get("prompt")
                or "Compare the two options."
            )
        )
        pair_id = _base_id(row, pair_id_key=pair_id_key, option_a=option_a, option_b=option_b)
        dimension = str(
            row.get(dimension_key)
            or row.get("comparison_dimension")
            or comparison_dimension
        )
        presentation_order = str(row.get("presentation_order") or "original")
        comparison = _comparison(
            row=row,
            source_label=source_label,
            pair_id=pair_id,
            prompt=prompt,
            option_a=option_a,
            option_b=option_b,
            target_option=target,
            target_kind=target_kind,
            presentation_order=presentation_order,
            task_type=task_type,
            comparison_dimension=dimension,
            condition_id_key=condition_id_key,
            condition_label_key=condition_label_key,
        )
        out.append(comparison)
        already_ordered = input_format in {"bt", "canonical"} and bool(
            row.get("presentation_order")
        )
        if include_order_swaps and not already_ordered:
            out.append(_swap(comparison))
    return out


def _binary_item_rows(
    rows: list[dict[str, Any]],
    *,
    source_label: str,
    task_type: str,
    comparison_dimension: str,
    item_text_key: str,
    item_label_key: str,
    positive_label: str,
    group_key: str,
    target_kind: str,
    question: str,
    include_order_swaps: bool,
    seed: int,
) -> list[JudgeComparison]:
    grouped: dict[str, dict[bool, list[dict[str, Any]]]] = defaultdict(
        lambda: {True: [], False: []}
    )
    for row in rows:
        text = flat_text(str(row.get(item_text_key) or ""))
        if not text:
            continue
        group = str(row.get(group_key) or "all") if group_key else "all"
        is_positive = str(row.get(item_label_key)) == str(positive_label)
        grouped[group][is_positive].append({**row, "_text": text})

    out: list[JudgeComparison] = []
    for group, labels in sorted(grouped.items()):
        positive = sorted(labels[True], key=lambda row: sha1_hex(f"{seed}:pos:{row['_text']}"))
        negative = sorted(labels[False], key=lambda row: sha1_hex(f"{seed}:neg:{row['_text']}"))
        for index, (pos, neg) in enumerate(zip(positive, negative)):
            pair_id = sha1_hex(f"{source_label}|{group}|{index}|{pos['_text']}|{neg['_text']}")
            merged = {
                "source_dataset": source_label,
                "subset": group,
                "binary_positive_metadata": row_metadata(pos, exclude={"_text"}),
                "binary_negative_metadata": row_metadata(neg, exclude={"_text"}),
            }
            for key in ("validity_type", "difficulty_tier", "analysis_split"):
                if pos.get(key) is not None and pos.get(key) == neg.get(key):
                    merged[key] = pos.get(key)
            comparison = _comparison(
                row=merged,
                source_label=source_label,
                pair_id=pair_id,
                prompt=question,
                option_a=pos["_text"],
                option_b=neg["_text"],
                target_option="A",
                target_kind=str(target_kind),
                presentation_order="positive_first",
                task_type=task_type,
                comparison_dimension=comparison_dimension,
                condition_id_key="condition_id",
                condition_label_key="condition_label",
            )
            out.append(comparison)
            if include_order_swaps:
                out.append(_swap(comparison))
    return out


def build_comparisons(
    rows: list[dict[str, Any]],
    *,
    input_format: str,
    source_label: str,
    task_type: str,
    comparison_dimension: str,
    prompt_key: str = "prompt",
    option_a_key: str = "option_a_text",
    option_b_key: str = "option_b_text",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    target_key: str = "target_option",
    pair_id_key: str = "pair_id",
    condition_id_key: str = "condition_id",
    condition_label_key: str = "condition_label",
    dimension_key: str = "comparison_dimension",
    item_text_key: str = "input",
    item_label_key: str = "label",
    binary_positive_label: str = "1",
    binary_group_key: str = "",
    binary_target_kind: str = "objective",
    binary_question: str = "Which option better satisfies the stated comparison dimension?",
    include_order_swaps: bool = True,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    detected = _detected_format(rows) if input_format == "auto" else input_format
    if detected == "binary_items":
        comparisons = _binary_item_rows(
            rows,
            source_label=source_label,
            task_type=task_type,
            comparison_dimension=comparison_dimension,
            item_text_key=item_text_key,
            item_label_key=item_label_key,
            positive_label=binary_positive_label,
            group_key=binary_group_key,
            target_kind=binary_target_kind,
            question=binary_question,
            include_order_swaps=include_order_swaps,
            seed=seed,
        )
    else:
        comparisons = _normalize_pair_rows(
            rows,
            input_format=detected,
            source_label=source_label,
            task_type=task_type,
            comparison_dimension=comparison_dimension,
            prompt_key=prompt_key,
            option_a_key=option_a_key,
            option_b_key=option_b_key,
            chosen_key=chosen_key,
            rejected_key=rejected_key,
            target_key=target_key,
            pair_id_key=pair_id_key,
            condition_id_key=condition_id_key,
            condition_label_key=condition_label_key,
            dimension_key=dimension_key,
            include_order_swaps=include_order_swaps,
        )
    return [comparison.as_dict() for comparison in comparisons]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    input_path = _resolve(workspace_root, args.input_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    source_label = str(args.source_label or input_path.parent.name or input_path.stem)
    rows = read_jsonl(input_path)
    comparisons = build_comparisons(
        rows,
        input_format=str(args.input_format),
        source_label=source_label,
        task_type=str(args.task_type),
        comparison_dimension=str(args.comparison_dimension),
        prompt_key=str(args.prompt_key),
        option_a_key=str(args.option_a_key),
        option_b_key=str(args.option_b_key),
        chosen_key=str(args.chosen_key),
        rejected_key=str(args.rejected_key),
        target_key=str(args.target_key),
        pair_id_key=str(args.pair_id_key),
        condition_id_key=str(args.condition_id_key),
        condition_label_key=str(args.condition_label_key),
        dimension_key=str(args.dimension_key),
        item_text_key=str(args.item_text_key),
        item_label_key=str(args.item_label_key),
        binary_positive_label=str(args.binary_positive_label),
        binary_group_key=str(args.binary_group_key),
        binary_target_kind=str(args.binary_target_kind),
        binary_question=str(args.binary_question),
        include_order_swaps=bool(args.include_order_swaps),
        seed=int(args.seed),
    )
    unique_pairs = sorted(
        {str(row["pair_id"]) for row in comparisons},
        key=lambda item: sha1_hex(f"{args.seed}:judge-reasoning:{item}"),
    )
    if int(args.max_pairs) > 0:
        unique_pairs = unique_pairs[: int(args.max_pairs)]
        keep = set(unique_pairs)
        comparisons = [row for row in comparisons if str(row["pair_id"]) in keep]
    if not comparisons:
        raise ValueError(f"No canonical comparisons emitted from {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "comparisons.jsonl"
    _write_jsonl(output_path, comparisons)
    counts = Counter(str(row["comparison_dimension"]) for row in comparisons)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-reasoning-pair-build",
            "input_jsonl": str(input_path),
            "input_format": str(args.input_format),
            "source_label": source_label,
            "task_type": str(args.task_type),
            "comparison_dimension": str(args.comparison_dimension),
            "include_order_swaps": bool(args.include_order_swaps),
            "seed": int(args.seed),
            "n_comparisons": int(len(comparisons)),
            "n_pairs": int(len({row["pair_id"] for row in comparisons})),
            "counts_by_dimension": dict(sorted(counts.items())),
            "comparisons_jsonl": str(output_path),
        },
    )
    print(f"comparisons={output_path}")
    print(f"n_comparisons={len(comparisons)}")
    print(f"n_pairs={len({row['pair_id'] for row in comparisons})}")


if __name__ == "__main__":
    main()
