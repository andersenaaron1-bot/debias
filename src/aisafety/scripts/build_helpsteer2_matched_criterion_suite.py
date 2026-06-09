"""Build a matched HelpSteer2 criterion-switching judgment suite."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
import os
from pathlib import Path
from typing import Any, Iterable

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text
from aisafety.mech.d4_io import resolve_path, sha1_hex, write_json
from aisafety.scripts.build_judge_deliberation_source_pack import (
    DEFAULT_WEIGHTS,
)
from aisafety.scripts.build_judge_reasoning_source_pack import (
    ATTRIBUTE_NAMES,
    _analysis_split,
    _load_hf,
)


DEFAULT_OUT_DIR = (
    Path("data") / "derived" / "helpsteer2_matched_criterion_suite_v1"
)
CORE_ATTRIBUTES = ("helpfulness", "correctness", "coherence")
CRITERIA = {
    "overall": (
        "Choose the better response overall. If neither response is clearly "
        "better because their strengths trade off, choose C."
    ),
    "correctness": (
        "Judge only factual and logical correctness. Choose C when the "
        "responses are tied on correctness."
    ),
    "helpfulness": (
        "Judge only how useful each response is for satisfying the request. "
        "Choose C when they are tied on helpfulness."
    ),
    "coherence": (
        "Judge only clarity and internal coherence. Choose C when they are "
        "tied on coherence."
    ),
    "weighted": (
        "Apply this fixed rubric: correctness 40%, helpfulness 30%, coherence "
        "20%, appropriate complexity 5%, and appropriate verbosity 5%. "
        "Choose C when the weighted scores are effectively tied."
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(os.environ.get("ARTROOT") or PROJECT_ROOT),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-id", default="nvidia/HelpSteer2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-pairs-per-stratum", type=int, default=8)
    parser.add_argument("--min-pairs-per-stratum", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--weighted-tie-epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(base: Path, path: Path) -> Path:
    resolved = resolve_path(base, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _attribute_vector(row: dict[str, Any]) -> tuple[float, ...] | None:
    values: list[float] = []
    for name in ATTRIBUTE_NAMES:
        try:
            values.append(float(row[name]))
        except (KeyError, TypeError, ValueError):
            return None
    return tuple(values)


def _weighted_score(attributes: tuple[float, ...]) -> float:
    return float(
        sum(
            DEFAULT_WEIGHTS[name] * value
            for name, value in zip(ATTRIBUTE_NAMES, attributes, strict=True)
        )
    )


def classify_pair(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    weighted_tie_epsilon: float,
) -> tuple[str, bool, float] | None:
    """Return stratum, whether to reverse orientation, and selection strength."""

    diffs = tuple(a - b for a, b in zip(left, right, strict=True))
    nonzero_indices = [
        index for index, value in enumerate(diffs) if abs(value) >= 0.5
    ]
    if (
        len(nonzero_indices) == 1
        and ATTRIBUTE_NAMES[nonzero_indices[0]] in CORE_ATTRIBUTES
    ):
        gap = diffs[nonzero_indices[0]]
        return "single_attribute", bool(gap < 0), abs(gap)

    weighted_gap = _weighted_score(left) - _weighted_score(right)
    if (
        max((abs(value) for value in diffs), default=0.0) <= 1.0
        and sum(abs(value) for value in diffs) <= 2.0
        and abs(weighted_gap) <= float(weighted_tie_epsilon)
    ):
        return "near_tie", False, -sum(abs(value) for value in diffs)

    if all(value >= 0 for value in diffs) and any(value > 0 for value in diffs):
        return "dominance", False, sum(diffs)
    if all(value <= 0 for value in diffs) and any(value < 0 for value in diffs):
        return "dominance", True, -sum(diffs)

    core_indices = [ATTRIBUTE_NAMES.index(name) for name in CORE_ATTRIBUTES]
    core_diffs = [diffs[index] for index in core_indices]
    if any(value >= 1.0 for value in core_diffs) and any(
        value <= -1.0 for value in core_diffs
    ):
        positive = sum(value for value in core_diffs if value > 0)
        negative = sum(-value for value in core_diffs if value < 0)
        return "tradeoff", False, min(positive, negative)
    return None


def _target_label(gap: float, *, epsilon: float = 0.0) -> str:
    if gap > float(epsilon):
        return "A"
    if gap < -float(epsilon):
        return "B"
    return "C"


def criterion_target(
    *,
    stratum: str,
    criterion_id: str,
    option_a_attributes: tuple[float, ...],
    option_b_attributes: tuple[float, ...],
    weighted_tie_epsilon: float,
) -> str:
    if criterion_id == "overall":
        return "A" if stratum in {"dominance", "single_attribute"} else "C"
    if criterion_id in ATTRIBUTE_NAMES:
        index = ATTRIBUTE_NAMES.index(criterion_id)
        return _target_label(
            option_a_attributes[index] - option_b_attributes[index]
        )
    if criterion_id == "weighted":
        return _target_label(
            _weighted_score(option_a_attributes)
            - _weighted_score(option_b_attributes),
            epsilon=weighted_tie_epsilon,
        )
    raise ValueError(f"Unknown criterion: {criterion_id}")


def build_matched_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    max_pairs_per_stratum: int,
    min_pairs_per_stratum: int,
    weighted_tie_epsilon: float,
    seed: int,
) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        prompt = flat_text(str(row.get("prompt") or ""))
        response = flat_text(str(row.get("response") or ""))
        attributes = _attribute_vector(row)
        if not prompt or not response or attributes is None:
            continue
        by_prompt[prompt].append(
            {
                "source_index": index,
                "response": response,
                "attributes": attributes,
            }
        )

    candidates: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for prompt, group in by_prompt.items():
        for left, right in itertools.combinations(group, 2):
            classified = classify_pair(
                left["attributes"],
                right["attributes"],
                weighted_tie_epsilon=weighted_tie_epsilon,
            )
            if classified is None:
                continue
            stratum, reverse, strength = classified
            option_a, option_b = (right, left) if reverse else (left, right)
            pair_id = sha1_hex(
                "|".join(
                    [
                        "helpsteer2-matched",
                        stratum,
                        prompt,
                        option_a["response"],
                        option_b["response"],
                    ]
                )
            )
            candidates[stratum].append(
                (
                    float(strength),
                    {
                        "pair_id": pair_id,
                        "origin_pair_id": pair_id,
                        "prompt": prompt,
                        "option_a_text": option_a["response"],
                        "option_b_text": option_b["response"],
                        "option_a_attributes": dict(
                            zip(ATTRIBUTE_NAMES, option_a["attributes"], strict=True)
                        ),
                        "option_b_attributes": dict(
                            zip(ATTRIBUTE_NAMES, option_b["attributes"], strict=True)
                        ),
                        "pair_stratum": stratum,
                        "analysis_split": _analysis_split(pair_id, seed=seed),
                        "source_indices": [
                            int(option_a["source_index"]),
                            int(option_b["source_index"]),
                        ],
                    },
                )
            )

    selected: list[dict[str, Any]] = []
    required = ("dominance", "single_attribute", "tradeoff", "near_tie")
    for stratum in required:
        used_prompts: set[str] = set()
        values = sorted(
            candidates.get(stratum, []),
            key=lambda item: (
                -item[0],
                sha1_hex(f"{seed}:{stratum}:{item[1]['pair_id']}"),
            ),
        )
        chosen: list[dict[str, Any]] = []
        for _strength, row in values:
            prompt = str(row["prompt"])
            if prompt in used_prompts:
                continue
            used_prompts.add(prompt)
            chosen.append(row)
            if len(chosen) >= int(max_pairs_per_stratum):
                break
        if len(chosen) < int(min_pairs_per_stratum):
            raise ValueError(
                f"HelpSteer2 stratum {stratum!r} emitted only {len(chosen)} "
                f"pairs; required {min_pairs_per_stratum}."
            )
        selected.extend(chosen)
    return selected


def _comparison_row(
    pair: dict[str, Any],
    *,
    criterion_id: str,
    presentation_order: str,
    weighted_tie_epsilon: float,
) -> dict[str, Any]:
    original_a = tuple(
        float(pair["option_a_attributes"][name]) for name in ATTRIBUTE_NAMES
    )
    original_b = tuple(
        float(pair["option_b_attributes"][name]) for name in ATTRIBUTE_NAMES
    )
    target = criterion_target(
        stratum=str(pair["pair_stratum"]),
        criterion_id=criterion_id,
        option_a_attributes=original_a,
        option_b_attributes=original_b,
        weighted_tie_epsilon=weighted_tie_epsilon,
    )
    swapped = presentation_order == "swapped"
    if swapped:
        target = {"A": "B", "B": "A", "C": "C"}[target]
        option_a_text = pair["option_b_text"]
        option_b_text = pair["option_a_text"]
        option_a_attributes = pair["option_b_attributes"]
        option_b_attributes = pair["option_a_attributes"]
    else:
        option_a_text = pair["option_a_text"]
        option_b_text = pair["option_b_text"]
        option_a_attributes = pair["option_a_attributes"]
        option_b_attributes = pair["option_b_attributes"]
    comparison_id = sha1_hex(
        f"{pair['pair_id']}|{criterion_id}|{presentation_order}"
    )
    source_dataset = f"helpsteer2_matched_{pair['pair_stratum']}"
    metadata = {
        "allow_tie": True,
        "criterion_id": criterion_id,
        "criterion_family": "helpsteer2_matched",
        "criterion_text": CRITERIA[criterion_id],
        "criterion_determinacy": (
            "explicit_attribute"
            if criterion_id != "overall"
            else (
                "ordered"
                if pair["pair_stratum"] in {"dominance", "single_attribute"}
                else "underdetermined"
            )
        ),
        "determinacy_level": str(pair["pair_stratum"]),
        "pair_stratum": str(pair["pair_stratum"]),
        "origin_pair_id": str(pair["pair_id"]),
        "analysis_split": str(pair["analysis_split"]),
        "option_a_attributes": option_a_attributes,
        "option_b_attributes": option_b_attributes,
        "weighted_rubric": DEFAULT_WEIGHTS,
    }
    return {
        "comparison_id": comparison_id,
        "pair_id": str(pair["pair_id"]),
        "origin_pair_id": str(pair["pair_id"]),
        "source_dataset": source_dataset,
        "subset": str(pair["pair_stratum"]),
        "split": "validation",
        "task_type": "matched_criterion_judgment",
        "comparison_dimension": (
            "overall_quality" if criterion_id == "overall" else criterion_id
        ),
        "prompt": str(pair["prompt"]),
        "option_a_text": str(option_a_text),
        "option_b_text": str(option_b_text),
        "target_option": target,
        "target_kind": "criterion_proxy",
        "presentation_order": presentation_order,
        "condition_id": criterion_id,
        "condition_label": criterion_id,
        "criterion_id": criterion_id,
        "criterion_family": "helpsteer2_matched",
        "criterion_text": CRITERIA[criterion_id],
        "criterion_determinacy": metadata["criterion_determinacy"],
        "determinacy_level": str(pair["pair_stratum"]),
        "pair_stratum": str(pair["pair_stratum"]),
        "analysis_split": str(pair["analysis_split"]),
        "allow_tie": True,
        "metadata": metadata,
    }


def build_comparisons(
    pairs: Iterable[dict[str, Any]],
    *,
    weighted_tie_epsilon: float,
) -> list[dict[str, Any]]:
    return [
        _comparison_row(
            pair,
            criterion_id=criterion_id,
            presentation_order=order,
            weighted_tie_epsilon=weighted_tie_epsilon,
        )
        for pair in pairs
        for criterion_id in CRITERIA
        for order in ("original", "swapped")
    ]


def shard_comparisons(
    pairs: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    *,
    num_shards: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    shards = [[] for _ in range(max(int(num_shards), 1))]
    pair_shards: dict[str, int] = {}
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        by_stratum[str(pair["pair_stratum"])].append(str(pair["pair_id"]))
    for stratum, pair_ids in sorted(by_stratum.items()):
        ordered = sorted(
            pair_ids,
            key=lambda value: sha1_hex(f"{seed}:shard:{stratum}:{value}"),
        )
        for index, pair_id in enumerate(ordered):
            pair_shards[pair_id] = index % len(shards)
    for row in comparisons:
        shards[pair_shards[str(row["pair_id"])]].append(row)
    return shards


def materialize(
    *,
    rows: Iterable[dict[str, Any]],
    out_dir: Path,
    max_pairs_per_stratum: int,
    min_pairs_per_stratum: int,
    num_shards: int,
    weighted_tie_epsilon: float,
    seed: int,
    source_description: str,
) -> dict[str, Any]:
    pairs = build_matched_pairs(
        rows,
        max_pairs_per_stratum=max_pairs_per_stratum,
        min_pairs_per_stratum=min_pairs_per_stratum,
        weighted_tie_epsilon=weighted_tie_epsilon,
        seed=seed,
    )
    comparisons = build_comparisons(
        pairs,
        weighted_tie_epsilon=weighted_tie_epsilon,
    )
    shards = shard_comparisons(
        pairs,
        comparisons,
        num_shards=num_shards,
        seed=seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "pairs.jsonl", pairs)
    _write_jsonl(out_dir / "comparisons.jsonl", comparisons)
    shard_paths: list[str] = []
    for index, shard in enumerate(shards):
        path = out_dir / f"comparisons_shard_{index}.jsonl"
        _write_jsonl(path, shard)
        shard_paths.append(str(path))
    manifest = {
        "stage": "helpsteer2-matched-criterion-suite",
        "source": source_description,
        "out_dir": str(out_dir),
        "criteria": CRITERIA,
        "weighted_rubric": DEFAULT_WEIGHTS,
        "weighted_tie_epsilon": float(weighted_tie_epsilon),
        "max_pairs_per_stratum": int(max_pairs_per_stratum),
        "min_pairs_per_stratum": int(min_pairs_per_stratum),
        "num_shards": int(num_shards),
        "seed": int(seed),
        "n_pairs": int(len(pairs)),
        "n_comparisons": int(len(comparisons)),
        "counts_by_stratum": dict(
            sorted(Counter(str(row["pair_stratum"]) for row in pairs).items())
        ),
        "counts_by_target": dict(
            sorted(Counter(str(row["target_option"]) for row in comparisons).items())
        ),
        "shards": [
            {"path": path, "n_comparisons": int(len(shard))}
            for path, shard in zip(shard_paths, shards, strict=True)
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    artifact_root = _resolve(workspace_root, args.artifact_root)
    cache_dir = (
        _resolve(artifact_root, args.cache_dir)
        if args.cache_dir is not None
        else None
    )
    out_dir = _resolve(artifact_root, args.out_dir)
    rows = _load_hf(
        str(args.dataset_id),
        None,
        split=str(args.split),
        cache_dir=cache_dir,
    )
    manifest = materialize(
        rows=rows,
        out_dir=out_dir,
        max_pairs_per_stratum=int(args.max_pairs_per_stratum),
        min_pairs_per_stratum=int(args.min_pairs_per_stratum),
        num_shards=int(args.num_shards),
        weighted_tie_epsilon=float(args.weighted_tie_epsilon),
        seed=int(args.seed),
        source_description=f"{args.dataset_id}:{args.split}",
    )
    print(f"out_dir={out_dir}")
    print(f"n_pairs={manifest['n_pairs']}")
    print(f"n_comparisons={manifest['n_comparisons']}")
    print(f"counts_by_stratum={manifest['counts_by_stratum']}")
    print(f"counts_by_target={manifest['counts_by_target']}")


if __name__ == "__main__":
    main()
