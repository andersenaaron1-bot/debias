"""Build held-out HelpSteer2 episodes for staged criterion-switch experiments."""

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
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.scripts.build_helpsteer2_matched_criterion_suite import CRITERIA
from aisafety.scripts.build_judge_reasoning_source_pack import (
    ATTRIBUTE_NAMES,
    _load_hf,
)


DEFAULT_OUT_DIR = (
    Path("data") / "derived" / "helpsteer2_criterion_switch_suite_v1"
)
PRIMARY_CRITERIA = ("correctness", "helpfulness", "coherence")
CONDITIONS = ("stable", "reminder", "switch", "placebo", "delayed")
TRANSITION_TYPES = ("choice_to_choice", "tie_to_choice", "same_target")


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
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--exclude-pairs-jsonl",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--max-pairs-per-transition", type=int, default=16)
    parser.add_argument("--min-pairs-per-transition", type=int, default=8)
    parser.add_argument("--min-choice-gap", type=float, default=1.0)
    parser.add_argument("--num-shards", type=int, default=2)
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
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
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


def _criterion_gap(
    left: tuple[float, ...],
    right: tuple[float, ...],
    criterion_id: str,
) -> float:
    index = ATTRIBUTE_NAMES.index(str(criterion_id))
    return float(left[index] - right[index])


def _target(gap: float) -> str:
    if gap > 0:
        return "A"
    if gap < 0:
        return "B"
    return "C"


def _semantic_swap(label: str) -> str:
    return {"A": "B", "B": "A", "C": "C", "": ""}.get(str(label), "")


def _pair_signature(prompt: str, left: str, right: str) -> str:
    responses = sorted([flat_text(str(left)), flat_text(str(right))])
    return sha1_hex(
        "|".join(["helpsteer2-content-pair", flat_text(str(prompt)), *responses])
    )


def _analysis_split(
    pair_id: str,
    *,
    transition_type: str,
    selected_index: int,
    seed: int,
) -> str:
    del pair_id, transition_type, seed
    bucket = int(selected_index) % 5
    if bucket <= 2:
        return "fit"
    if bucket == 3:
        return "selection"
    return "intervention"


def _canonical_pair(
    *,
    prompt: str,
    left: dict[str, Any],
    right: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    responses = sorted([str(left["response"]), str(right["response"])])
    pair_id = sha1_hex(
        "|".join(["helpsteer2-criterion-switch", prompt, *responses])
    )
    reverse = (
        int(sha1_hex(f"{seed}:switch-orientation:{pair_id}")[:8], 16) % 2
        == 1
    )
    option_a, option_b = (right, left) if reverse else (left, right)
    return option_a, option_b, pair_id


def _transition_candidates(
    targets: dict[str, str],
    gaps: dict[str, float],
    *,
    min_choice_gap: float,
) -> dict[str, list[tuple[str, str, float]]]:
    candidates: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for left, right in itertools.combinations(PRIMARY_CRITERIA, 2):
        left_target = targets[left]
        right_target = targets[right]
        left_gap = abs(float(gaps[left]))
        right_gap = abs(float(gaps[right]))
        if (
            left_target in {"A", "B"}
            and right_target in {"A", "B"}
            and left_target != right_target
            and min(left_gap, right_gap) >= float(min_choice_gap)
        ):
            candidates["choice_to_choice"].append(
                (left, right, min(left_gap, right_gap))
            )
        if (
            (left_target == "C") != (right_target == "C")
            and max(left_gap, right_gap) >= float(min_choice_gap)
        ):
            candidates["tie_to_choice"].append(
                (left, right, max(left_gap, right_gap))
            )
        if (
            left_target == right_target
            and left_target in {"A", "B"}
            and min(left_gap, right_gap) >= float(min_choice_gap)
        ):
            candidates["same_target"].append(
                (left, right, min(left_gap, right_gap))
            )
    return candidates


def build_switch_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    excluded_pair_signatures: set[str],
    max_pairs_per_transition: int,
    min_pairs_per_transition: int,
    min_choice_gap: float,
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
                "source_index": int(index),
                "response": response,
                "attributes": attributes,
            }
        )

    candidates: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(
        list
    )
    for prompt, group in by_prompt.items():
        for left, right in itertools.combinations(group, 2):
            if left["response"] == right["response"]:
                continue
            option_a, option_b, pair_id = _canonical_pair(
                prompt=prompt,
                left=left,
                right=right,
                seed=seed,
            )
            signature = _pair_signature(
                prompt,
                str(option_a["response"]),
                str(option_b["response"]),
            )
            if signature in excluded_pair_signatures:
                continue
            gaps = {
                criterion: _criterion_gap(
                    option_a["attributes"],
                    option_b["attributes"],
                    criterion,
                )
                for criterion in PRIMARY_CRITERIA
            }
            targets = {
                criterion: _target(gap)
                for criterion, gap in gaps.items()
            }
            transition_candidates = _transition_candidates(
                targets,
                gaps,
                min_choice_gap=min_choice_gap,
            )
            for transition_type, criterion_pairs in transition_candidates.items():
                criterion_pairs = sorted(
                    criterion_pairs,
                    key=lambda value: (
                        -value[2],
                        sha1_hex(
                            f"{seed}:{pair_id}:{transition_type}:"
                            f"{value[0]}:{value[1]}"
                        ),
                    ),
                )
                initial_criterion, updated_criterion, strength = criterion_pairs[0]
                if (
                    int(
                        sha1_hex(
                            f"{seed}:criterion-direction:{pair_id}:"
                            f"{transition_type}"
                        )[:8],
                        16,
                    )
                    % 2
                    == 1
                ):
                    initial_criterion, updated_criterion = (
                        updated_criterion,
                        initial_criterion,
                    )
                candidates[transition_type].append(
                    (
                        float(strength),
                        {
                            "pair_id": pair_id,
                            "origin_pair_id": pair_id,
                            "pair_signature": signature,
                            "prompt": prompt,
                            "option_a_text": option_a["response"],
                            "option_b_text": option_b["response"],
                            "option_a_attributes": dict(
                                zip(
                                    ATTRIBUTE_NAMES,
                                    option_a["attributes"],
                                    strict=True,
                                )
                            ),
                            "option_b_attributes": dict(
                                zip(
                                    ATTRIBUTE_NAMES,
                                    option_b["attributes"],
                                    strict=True,
                                )
                            ),
                            "criterion_gaps_a_minus_b": gaps,
                            "criterion_targets": targets,
                            "transition_type": transition_type,
                            "initial_criterion_id": initial_criterion,
                            "updated_criterion_id": updated_criterion,
                            "initial_target_semantic": targets[
                                initial_criterion
                            ],
                            "updated_target_semantic": targets[
                                updated_criterion
                            ],
                            "source_indices": [
                                int(option_a["source_index"]),
                                int(option_b["source_index"]),
                            ],
                        },
                    )
                )

    selected: list[dict[str, Any]] = []
    used_pairs: set[str] = set()
    used_prompts: set[str] = set()
    for transition_type in TRANSITION_TYPES:
        values = sorted(
            candidates.get(transition_type, []),
            key=lambda item: (
                -item[0],
                sha1_hex(
                    f"{seed}:select:{transition_type}:{item[1]['pair_id']}"
                ),
            ),
        )
        chosen: list[dict[str, Any]] = []
        for _strength, row in values:
            if (
                str(row["pair_id"]) in used_pairs
                or str(row["prompt"]) in used_prompts
            ):
                continue
            chosen.append(row)
            used_pairs.add(str(row["pair_id"]))
            used_prompts.add(str(row["prompt"]))
            if len(chosen) >= int(max_pairs_per_transition):
                break
        if len(chosen) < int(min_pairs_per_transition):
            raise ValueError(
                f"Transition {transition_type!r} emitted {len(chosen)} pairs; "
                f"required {min_pairs_per_transition}."
            )
        ordered = sorted(
            chosen,
            key=lambda row: sha1_hex(
                f"{seed}:split-order:{transition_type}:{row['pair_id']}"
            ),
        )
        for index, row in enumerate(ordered):
            row["analysis_split"] = _analysis_split(
                str(row["pair_id"]),
                transition_type=transition_type,
                selected_index=index,
                seed=seed,
            )
            selected.append(row)
    return selected


def _episode(
    pair: dict[str, Any],
    *,
    condition: str,
    presentation_order: str,
) -> dict[str, Any]:
    initial_criterion = str(pair["initial_criterion_id"])
    updated_criterion = str(pair["updated_criterion_id"])
    if condition in {"stable", "reminder", "placebo"}:
        phase1_criterion = initial_criterion
        phase2_criterion = initial_criterion
    elif condition == "switch":
        phase1_criterion = initial_criterion
        phase2_criterion = updated_criterion
    elif condition == "delayed":
        phase1_criterion = ""
        phase2_criterion = updated_criterion
    else:
        raise ValueError(f"Unknown condition: {condition}")

    phase1_semantic = (
        str(pair["criterion_targets"][phase1_criterion])
        if phase1_criterion
        else ""
    )
    phase2_semantic = str(pair["criterion_targets"][phase2_criterion])
    swapped = presentation_order == "swapped"
    option_a = (
        str(pair["option_b_text"])
        if swapped
        else str(pair["option_a_text"])
    )
    option_b = (
        str(pair["option_a_text"])
        if swapped
        else str(pair["option_b_text"])
    )
    phase1_target = (
        _semantic_swap(phase1_semantic) if swapped else phase1_semantic
    )
    phase2_target = (
        _semantic_swap(phase2_semantic) if swapped else phase2_semantic
    )
    episode_id = sha1_hex(
        f"{pair['pair_id']}|{condition}|{presentation_order}"
    )
    return {
        "episode_id": episode_id,
        "comparison_id": episode_id,
        "pair_id": str(pair["pair_id"]),
        "origin_pair_id": str(pair["pair_id"]),
        "source_dataset": "helpsteer2_criterion_switch",
        "subset": str(pair["transition_type"]),
        "split": str(pair.get("source_split") or ""),
        "task_type": "staged_criterion_switch",
        "comparison_dimension": "criterion_use",
        "prompt": str(pair["prompt"]),
        "option_a_text": option_a,
        "option_b_text": option_b,
        "presentation_order": presentation_order,
        "condition_id": condition,
        "condition_label": condition,
        "transition_type": str(pair["transition_type"]),
        "analysis_split": str(pair["analysis_split"]),
        "initial_criterion_id": initial_criterion,
        "updated_criterion_id": updated_criterion,
        "phase1_criterion_id": phase1_criterion,
        "phase2_criterion_id": phase2_criterion,
        "phase1_criterion_text": (
            CRITERIA[phase1_criterion] if phase1_criterion else ""
        ),
        "phase2_criterion_text": CRITERIA[phase2_criterion],
        "phase1_target_option": phase1_target,
        "phase2_target_option": phase2_target,
        "phase1_target_semantic": phase1_semantic,
        "phase2_target_semantic": phase2_semantic,
        "expected_target_change": bool(
            phase1_semantic and phase1_semantic != phase2_semantic
        ),
        "allow_tie": True,
        "metadata": {
            "criterion_targets": pair["criterion_targets"],
            "criterion_gaps_a_minus_b": pair[
                "criterion_gaps_a_minus_b"
            ],
            "option_a_attributes_canonical": pair["option_a_attributes"],
            "option_b_attributes_canonical": pair["option_b_attributes"],
            "analysis_split": str(pair["analysis_split"]),
            "transition_type": str(pair["transition_type"]),
            "allow_tie": True,
        },
    }


def build_episodes(pairs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _episode(
            pair,
            condition=condition,
            presentation_order=order,
        )
        for pair in pairs
        for condition in CONDITIONS
        for order in ("original", "swapped")
    ]


def shard_episodes(
    pairs: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    num_shards: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    shards = [[] for _ in range(max(int(num_shards), 1))]
    pair_shards: dict[str, int] = {}
    by_transition: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        by_transition[str(pair["transition_type"])].append(
            str(pair["pair_id"])
        )
    for transition_type, pair_ids in sorted(by_transition.items()):
        ordered = sorted(
            pair_ids,
            key=lambda value: sha1_hex(
                f"{seed}:switch-shard:{transition_type}:{value}"
            ),
        )
        for index, pair_id in enumerate(ordered):
            pair_shards[pair_id] = index % len(shards)
    for row in episodes:
        shards[pair_shards[str(row["pair_id"])]].append(row)
    return shards


def materialize(
    *,
    rows: Iterable[dict[str, Any]],
    excluded_pair_signatures: set[str],
    out_dir: Path,
    source_split: str,
    max_pairs_per_transition: int,
    min_pairs_per_transition: int,
    min_choice_gap: float,
    num_shards: int,
    seed: int,
    source_description: str,
) -> dict[str, Any]:
    pairs = build_switch_pairs(
        rows,
        excluded_pair_signatures=excluded_pair_signatures,
        max_pairs_per_transition=max_pairs_per_transition,
        min_pairs_per_transition=min_pairs_per_transition,
        min_choice_gap=min_choice_gap,
        seed=seed,
    )
    for pair in pairs:
        pair["source_split"] = str(source_split)
    episodes = build_episodes(pairs)
    shards = shard_episodes(
        pairs,
        episodes,
        num_shards=num_shards,
        seed=seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "pairs.jsonl", pairs)
    _write_jsonl(out_dir / "episodes.jsonl", episodes)
    shard_rows: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        path = out_dir / f"episodes_shard_{index}.jsonl"
        _write_jsonl(path, shard)
        shard_rows.append(
            {
                "path": str(path),
                "n_episodes": int(len(shard)),
                "n_pairs": int(
                    len({str(row["pair_id"]) for row in shard})
                ),
            }
        )
    manifest = {
        "stage": "helpsteer2-criterion-switch-suite",
        "source": source_description,
        "source_split": str(source_split),
        "out_dir": str(out_dir),
        "primary_criteria": list(PRIMARY_CRITERIA),
        "conditions": list(CONDITIONS),
        "transition_types": list(TRANSITION_TYPES),
        "max_pairs_per_transition": int(max_pairs_per_transition),
        "min_pairs_per_transition": int(min_pairs_per_transition),
        "min_choice_gap": float(min_choice_gap),
        "num_shards": int(num_shards),
        "seed": int(seed),
        "n_excluded_pair_signatures": int(len(excluded_pair_signatures)),
        "n_pairs": int(len(pairs)),
        "n_episodes": int(len(episodes)),
        "counts_by_transition": dict(
            sorted(
                Counter(
                    str(row["transition_type"]) for row in pairs
                ).items()
            )
        ),
        "counts_by_analysis_split": dict(
            sorted(
                Counter(
                    str(row["analysis_split"]) for row in pairs
                ).items()
            )
        ),
        "counts_by_criterion_pair": dict(
            sorted(
                Counter(
                    f"{row['initial_criterion_id']}->{row['updated_criterion_id']}"
                    for row in pairs
                ).items()
            )
        ),
        "shards": shard_rows,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    artifact_root = _resolve(workspace_root, args.artifact_root)
    out_dir = _resolve(artifact_root, args.out_dir)
    cache_dir = (
        _resolve(artifact_root, args.cache_dir)
        if args.cache_dir is not None
        else None
    )
    excluded_pair_signatures: set[str] = set()
    for raw_path in args.exclude_pairs_jsonl:
        path = _resolve(artifact_root, raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            prompt = str(row.get("prompt") or "")
            left = str(
                row.get("option_a_text")
                or row.get("response_a")
                or row.get("chosen")
                or ""
            )
            right = str(
                row.get("option_b_text")
                or row.get("response_b")
                or row.get("rejected")
                or ""
            )
            if prompt and left and right:
                excluded_pair_signatures.add(
                    _pair_signature(prompt, left, right)
                )
    rows = _load_hf(
        str(args.dataset_id),
        None,
        split=str(args.split),
        cache_dir=cache_dir,
    )
    manifest = materialize(
        rows=rows,
        excluded_pair_signatures=excluded_pair_signatures,
        out_dir=out_dir,
        source_split=str(args.split),
        max_pairs_per_transition=int(args.max_pairs_per_transition),
        min_pairs_per_transition=int(args.min_pairs_per_transition),
        min_choice_gap=float(args.min_choice_gap),
        num_shards=int(args.num_shards),
        seed=int(args.seed),
        source_description=f"{args.dataset_id}:{args.split}",
    )
    print(f"out_dir={out_dir}")
    print(f"n_pairs={manifest['n_pairs']}")
    print(f"n_episodes={manifest['n_episodes']}")
    print(f"counts_by_transition={manifest['counts_by_transition']}")
    print(f"counts_by_analysis_split={manifest['counts_by_analysis_split']}")
    print(f"counts_by_criterion_pair={manifest['counts_by_criterion_pair']}")


if __name__ == "__main__":
    main()
