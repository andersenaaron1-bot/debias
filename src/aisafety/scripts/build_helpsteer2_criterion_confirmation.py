"""Build the locked HelpSteer2 criterion-operationalization confirmation suite."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.scripts.build_helpsteer2_criterion_switch_suite import (
    PRIMARY_CRITERIA,
    _pair_signature,
    _semantic_swap,
    build_switch_pairs,
)
from aisafety.scripts.build_helpsteer2_matched_criterion_suite import CRITERIA
from aisafety.scripts.build_judge_reasoning_source_pack import _load_hf


DEFAULT_OUT_DIR = (
    Path("data") / "derived" / "helpsteer2_criterion_confirmation_v1"
)
TRANSITION_QUOTAS = {
    "choice_to_choice": 9,
    "tie_to_choice": 9,
    "same_target": 6,
}
MAIN_CONDITIONS = (
    "early_criterion",
    "late_criterion",
    "early_evidence",
    "late_evidence",
)
CEILING_CONDITION = "late_explicit_target"
ATTRIBUTE_LABELS = {
    "correctness": "correctness",
    "helpfulness": "helpfulness",
    "coherence": "clarity and internal coherence",
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
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--exclude-pairs-jsonl",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--candidate-pool-per-transition", type=int, default=24)
    parser.add_argument("--min-choice-gap", type=float, default=1.0)
    parser.add_argument("--main-branches", type=int, default=2)
    parser.add_argument("--ceiling-pairs-per-conflict-transition", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(base: Path, path: Path) -> Path:
    resolved = resolve_path(base, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in values:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(values)


def _strength(row: dict[str, Any]) -> float:
    gaps = row["criterion_gaps_a_minus_b"]
    return min(
        abs(float(gaps[str(row["initial_criterion_id"])])),
        abs(float(gaps[str(row["updated_criterion_id"])])),
    )


def _canonicalize_transition(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    if (
        str(value["transition_type"]) == "tie_to_choice"
        and str(value["initial_target_semantic"]) != "C"
    ):
        value["initial_criterion_id"], value["updated_criterion_id"] = (
            value["updated_criterion_id"],
            value["initial_criterion_id"],
        )
        value["initial_target_semantic"], value["updated_target_semantic"] = (
            value["updated_target_semantic"],
            value["initial_target_semantic"],
        )
    return value


def select_confirmation_pairs(
    candidates: Iterable[dict[str, Any]],
    *,
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    """Select a balanced, deterministic confirmation set without model outputs."""

    by_transition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        row = _canonicalize_transition(raw)
        by_transition[str(row["transition_type"])].append(row)

    selected: list[dict[str, Any]] = []
    for transition_type, quota in quotas.items():
        available = list(by_transition.get(transition_type, []))
        if len(available) < int(quota):
            raise ValueError(
                f"Transition {transition_type!r} has {len(available)} candidates; "
                f"required {quota}."
            )
        counts: Counter[str] = Counter()
        chosen: list[dict[str, Any]] = []
        while len(chosen) < int(quota):

            def rank(row: dict[str, Any]) -> tuple[Any, ...]:
                initial = str(row["initial_criterion_id"])
                updated = str(row["updated_criterion_id"])
                criterion_pair = "|".join(sorted([initial, updated]))
                target_transition = (
                    f"{row['initial_target_semantic']}->"
                    f"{row['updated_target_semantic']}"
                )
                return (
                    counts[f"initial:{initial}"],
                    counts[f"updated:{updated}"],
                    counts[f"pair:{criterion_pair}"],
                    counts[f"target:{target_transition}"],
                    -_strength(row),
                    sha1_hex(
                        f"{seed}:confirmation-select:{transition_type}:"
                        f"{row['pair_id']}"
                    ),
                )

            winner = min(available, key=rank)
            available.remove(winner)
            chosen.append(winner)
            initial = str(winner["initial_criterion_id"])
            updated = str(winner["updated_criterion_id"])
            criterion_pair = "|".join(sorted([initial, updated]))
            target_transition = (
                f"{winner['initial_target_semantic']}->"
                f"{winner['updated_target_semantic']}"
            )
            counts[f"initial:{initial}"] += 1
            counts[f"updated:{updated}"] += 1
            counts[f"pair:{criterion_pair}"] += 1
            counts[f"target:{target_transition}"] += 1

        for row in chosen:
            row["analysis_split"] = "confirmation"
            row["confirmation_locked"] = True
            selected.append(row)
    return selected


def _displayed_options(
    pair: dict[str, Any],
    presentation_order: str,
) -> tuple[str, str]:
    if str(presentation_order) == "swapped":
        return str(pair["option_b_text"]), str(pair["option_a_text"])
    return str(pair["option_a_text"]), str(pair["option_b_text"])


def _displayed_attributes(
    pair: dict[str, Any],
    presentation_order: str,
) -> tuple[dict[str, float], dict[str, float]]:
    left = dict(pair["option_a_attributes"])
    right = dict(pair["option_b_attributes"])
    return (right, left) if str(presentation_order) == "swapped" else (left, right)


def _score(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.2f}"


def score_evidence_text(
    pair: dict[str, Any],
    *,
    criterion_id: str,
    presentation_order: str,
) -> str:
    option_a, option_b = _displayed_attributes(pair, presentation_order)
    label = ATTRIBUTE_LABELS[str(criterion_id)]
    return (
        "Independent criterion-specific annotation evidence is available. "
        "Scores range from 0 (low) to 4 (high); use the scores as evidence, "
        "not as a precomputed verdict.\n"
        f"Option A {label} score: {_score(option_a[criterion_id])}\n"
        f"Option B {label} score: {_score(option_b[criterion_id])}"
    )


def _confirmation_episode(
    pair: dict[str, Any],
    *,
    condition: str,
    presentation_order: str,
    main_branches: int,
    explicit_ceiling: bool,
) -> dict[str, Any]:
    updated = str(pair["updated_criterion_id"])
    initial = str(pair["initial_criterion_id"])
    early = condition in {"early_criterion", "early_evidence"}
    evidence = condition in {"early_evidence", "late_evidence"}
    explicit = condition == CEILING_CONDITION
    phase1_criterion = updated if early else initial
    phase2_criterion = updated
    phase1_semantic = str(pair["criterion_targets"][phase1_criterion])
    phase2_semantic = str(pair["criterion_targets"][phase2_criterion])
    swapped = str(presentation_order) == "swapped"
    option_a, option_b = _displayed_options(pair, presentation_order)
    phase1_target = _semantic_swap(phase1_semantic) if swapped else phase1_semantic
    phase2_target = _semantic_swap(phase2_semantic) if swapped else phase2_semantic
    phase1_evidence = (
        score_evidence_text(
            pair,
            criterion_id=updated,
            presentation_order=presentation_order,
        )
        if evidence and early
        else ""
    )
    phase2_evidence = (
        score_evidence_text(
            pair,
            criterion_id=updated,
            presentation_order=presentation_order,
        )
        if evidence and not early
        else ""
    )
    phase1_cache_group = (
        "early_evidence"
        if condition == "early_evidence"
        else "early_criterion"
        if condition == "early_criterion"
        else "late_commitment"
    )
    episode_id = sha1_hex(
        f"{pair['pair_id']}|confirmation|{condition}|{presentation_order}"
    )
    return {
        "episode_id": episode_id,
        "comparison_id": episode_id,
        "pair_id": str(pair["pair_id"]),
        "origin_pair_id": str(pair["pair_id"]),
        "source_dataset": "helpsteer2_criterion_confirmation",
        "subset": str(pair["transition_type"]),
        "split": str(pair.get("source_split") or ""),
        "task_type": "criterion_operationalization_confirmation",
        "comparison_dimension": "criterion_use",
        "prompt": str(pair["prompt"]),
        "option_a_text": option_a,
        "option_b_text": option_b,
        "presentation_order": str(presentation_order),
        "condition_id": condition,
        "condition_label": condition,
        "transition_type": str(pair["transition_type"]),
        "analysis_split": "confirmation",
        "information_timing": "early" if early else "late",
        "information_type": (
            "explicit_target"
            if explicit
            else "score_evidence"
            if evidence
            else "criterion_only"
        ),
        "initial_criterion_id": initial,
        "updated_criterion_id": updated,
        "phase1_criterion_id": phase1_criterion,
        "phase2_criterion_id": phase2_criterion,
        "phase1_criterion_text": CRITERIA[phase1_criterion],
        "phase2_criterion_text": CRITERIA[phase2_criterion],
        "phase1_evidence_text": phase1_evidence,
        "phase2_evidence_text": phase2_evidence,
        "phase2_explicit_target_option": phase2_target if explicit else "",
        "phase1_cache_group": phase1_cache_group,
        "branches_per_episode": 1 if explicit else int(main_branches),
        "phase1_target_option": phase1_target,
        "phase2_target_option": phase2_target,
        "phase1_target_semantic": phase1_semantic,
        "phase2_target_semantic": phase2_semantic,
        "expected_target_change": bool(phase1_semantic != phase2_semantic),
        "allow_tie": True,
        "explicit_ceiling_pair": bool(explicit_ceiling),
        "metadata": {
            "criterion_targets": pair["criterion_targets"],
            "criterion_gaps_a_minus_b": pair["criterion_gaps_a_minus_b"],
            "option_a_attributes_canonical": pair["option_a_attributes"],
            "option_b_attributes_canonical": pair["option_b_attributes"],
            "pair_signature": str(pair["pair_signature"]),
            "confirmation_locked": True,
        },
    }


def build_confirmation_episodes(
    pairs: Iterable[dict[str, Any]],
    *,
    main_branches: int,
    ceiling_pairs_per_conflict_transition: int,
    seed: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    values = list(pairs)
    ceiling_pair_ids: set[str] = set()
    for transition_type in ("choice_to_choice", "tie_to_choice"):
        eligible = [
            row for row in values if str(row["transition_type"]) == transition_type
        ]
        ordered = sorted(
            eligible,
            key=lambda row: sha1_hex(
                f"{seed}:explicit-ceiling:{transition_type}:{row['pair_id']}"
            ),
        )
        ceiling_pair_ids.update(
            str(row["pair_id"])
            for row in ordered[: int(ceiling_pairs_per_conflict_transition)]
        )

    episodes = [
        _confirmation_episode(
            pair,
            condition=condition,
            presentation_order=order,
            main_branches=main_branches,
            explicit_ceiling=str(pair["pair_id"]) in ceiling_pair_ids,
        )
        for pair in values
        for condition in MAIN_CONDITIONS
        for order in ("original", "swapped")
    ]
    episodes.extend(
        _confirmation_episode(
            pair,
            condition=CEILING_CONDITION,
            presentation_order=order,
            main_branches=main_branches,
            explicit_ceiling=True,
        )
        for pair in values
        if str(pair["pair_id"]) in ceiling_pair_ids
        for order in ("original", "swapped")
    )
    return episodes, ceiling_pair_ids


def _audit_prompt(
    pair: dict[str, Any],
    *,
    criterion_id: str,
    presentation_order: str,
) -> str:
    option_a, option_b = _displayed_options(pair, presentation_order)
    return (
        "Human audit of a criterion-specific comparison\n\n"
        "Judge only under the stated decision rule. Do not judge overall "
        "quality and do not infer the dataset annotation.\n\n"
        f"Decision rule ({criterion_id}):\n{CRITERIA[criterion_id]}\n\n"
        f"Context or question:\n{pair['prompt']}\n\n"
        f"Option A:\n{option_a}\n\n"
        f"Option B:\n{option_b}\n\n"
        "Record exactly one verdict in the response sheet:\n"
        "A = Option A is better under the rule\n"
        "B = Option B is better under the rule\n"
        "C = tied or underdetermined under the rule\n"
    )


def write_audit_bundle(
    out_dir: Path,
    pairs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    audit_dir = out_dir / "human_audit"
    prompt_dir = audit_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = []
    for pair in pairs:
        for role, criterion_id in (
            ("initial", str(pair["initial_criterion_id"])),
            ("updated", str(pair["updated_criterion_id"])),
        ):
            for presentation_order in ("original", "swapped"):
                specs.append(
                    {
                        "pair": pair,
                        "criterion_role": role,
                        "criterion_id": criterion_id,
                        "presentation_order": presentation_order,
                    }
                )
    specs.sort(
        key=lambda row: sha1_hex(
            f"audit-order|{row['pair']['pair_id']}|"
            f"{row['criterion_role']}|{row['criterion_id']}|"
            f"{row['presentation_order']}"
        )
    )
    items: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for spec in specs:
        pair = spec["pair"]
        role = str(spec["criterion_role"])
        criterion_id = str(spec["criterion_id"])
        presentation_order = str(spec["presentation_order"])
        semantic_target = str(pair["criterion_targets"][criterion_id])
        audit_id = sha1_hex(
            f"audit|{pair['pair_id']}|{role}|{criterion_id}|"
            f"{presentation_order}"
        )[:16]
        filename = f"{len(items) + 1:04d}_{audit_id}.txt"
        prompt_text = _audit_prompt(
            pair,
            criterion_id=criterion_id,
            presentation_order=presentation_order,
        )
        (prompt_dir / filename).write_text(prompt_text, encoding="utf-8")
        items.append(
            {
                "audit_id": audit_id,
                "prompt_file": filename,
                "pair_id": str(pair["pair_id"]),
                "transition_type": str(pair["transition_type"]),
                "criterion_role": role,
                "criterion_id": criterion_id,
                "presentation_order": presentation_order,
            }
        )
        references.append(
            {
                "audit_id": audit_id,
                "pair_id": str(pair["pair_id"]),
                "criterion_role": role,
                "criterion_id": criterion_id,
                "presentation_order": presentation_order,
                "proxy_target_semantic": semantic_target,
                "proxy_target_displayed": (
                    _semantic_swap(semantic_target)
                    if presentation_order == "swapped"
                    else semantic_target
                ),
            }
        )
    _write_jsonl(audit_dir / "audit_items.jsonl", items)
    _write_jsonl(audit_dir / "private_proxy_reference.jsonl", references)
    response_path = audit_dir / "audit_responses.csv"
    with response_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audit_id",
                "prompt_file",
                "verdict",
                "confidence_1_to_5",
                "reviewer",
                "notes",
            ],
        )
        writer.writeheader()
        for row in items:
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "prompt_file": row["prompt_file"],
                    "verdict": "",
                    "confidence_1_to_5": "",
                    "reviewer": "",
                    "notes": "",
                }
            )
    (audit_dir / "README.txt").write_text(
        "Judge the 96 files under prompts/ without opening "
        "private_proxy_reference.jsonl. Enter A, B, or C in the verdict "
        "column of audit_responses.csv. Confidence and notes are optional. "
        "Do not inspect model outputs before completing the audit.\n",
        encoding="utf-8",
    )
    return {
        "audit_dir": str(audit_dir),
        "prompt_dir": str(prompt_dir),
        "response_csv": str(response_path),
        "n_audit_prompts": int(len(items)),
    }


def _excluded_signatures(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
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
                excluded.add(_pair_signature(prompt, left, right))
    return excluded


def materialize(
    *,
    rows: Iterable[dict[str, Any]],
    excluded_pair_signatures: set[str],
    out_dir: Path,
    source_split: str,
    candidate_pool_per_transition: int,
    min_choice_gap: float,
    main_branches: int,
    ceiling_pairs_per_conflict_transition: int,
    seed: int,
    source_description: str,
) -> dict[str, Any]:
    pool = build_switch_pairs(
        rows,
        excluded_pair_signatures=excluded_pair_signatures,
        max_pairs_per_transition=int(candidate_pool_per_transition),
        min_pairs_per_transition=max(TRANSITION_QUOTAS.values()),
        min_choice_gap=float(min_choice_gap),
        seed=int(seed),
    )
    pairs = select_confirmation_pairs(
        pool,
        quotas=TRANSITION_QUOTAS,
        seed=int(seed),
    )
    for pair in pairs:
        pair["source_split"] = str(source_split)
    episodes, ceiling_pair_ids = build_confirmation_episodes(
        pairs,
        main_branches=int(main_branches),
        ceiling_pairs_per_conflict_transition=int(
            ceiling_pairs_per_conflict_transition
        ),
        seed=int(seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "candidate_pool.jsonl", pool)
    _write_jsonl(out_dir / "pairs.jsonl", pairs)
    _write_jsonl(out_dir / "episodes.jsonl", episodes)
    audit = write_audit_bundle(out_dir, pairs)
    planned_traces = sum(int(row["branches_per_episode"]) for row in episodes)
    freeze_hash = sha1_hex(
        "\n".join(
            sorted(
                f"{row['pair_id']}|{row['initial_criterion_id']}|"
                f"{row['updated_criterion_id']}|{row['transition_type']}"
                for row in pairs
            )
        )
    )
    manifest = {
        "stage": "helpsteer2-criterion-operationalization-confirmation",
        "source": source_description,
        "source_split": str(source_split),
        "out_dir": str(out_dir),
        "seed": int(seed),
        "confirmation_freeze_hash": freeze_hash,
        "transition_quotas": TRANSITION_QUOTAS,
        "main_conditions": list(MAIN_CONDITIONS),
        "ceiling_condition": CEILING_CONDITION,
        "main_branches": int(main_branches),
        "ceiling_branches": 1,
        "candidate_pool_per_transition": int(candidate_pool_per_transition),
        "min_choice_gap": float(min_choice_gap),
        "n_excluded_pair_signatures": int(len(excluded_pair_signatures)),
        "n_candidate_pairs": int(len(pool)),
        "n_pairs": int(len(pairs)),
        "n_episodes": int(len(episodes)),
        "n_planned_traces": int(planned_traces),
        "n_ceiling_pairs": int(len(ceiling_pair_ids)),
        "ceiling_pair_ids": sorted(ceiling_pair_ids),
        "counts_by_transition": dict(
            sorted(Counter(str(row["transition_type"]) for row in pairs).items())
        ),
        "counts_by_criterion_transition": dict(
            sorted(
                Counter(
                    f"{row['initial_criterion_id']}->{row['updated_criterion_id']}"
                    for row in pairs
                ).items()
            )
        ),
        "counts_by_target_transition": dict(
            sorted(
                Counter(
                    f"{row['initial_target_semantic']}->"
                    f"{row['updated_target_semantic']}"
                    for row in pairs
                ).items()
            )
        ),
        "candidate_pool_jsonl": str(out_dir / "candidate_pool.jsonl"),
        "pairs_jsonl": str(out_dir / "pairs.jsonl"),
        "episodes_jsonl": str(out_dir / "episodes.jsonl"),
        **audit,
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
    exclude_paths = [
        _resolve(artifact_root, path) for path in args.exclude_pairs_jsonl
    ]
    for path in exclude_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = _load_hf(
        str(args.dataset_id),
        None,
        split=str(args.split),
        cache_dir=cache_dir,
    )
    manifest = materialize(
        rows=rows,
        excluded_pair_signatures=_excluded_signatures(exclude_paths),
        out_dir=out_dir,
        source_split=str(args.split),
        candidate_pool_per_transition=int(args.candidate_pool_per_transition),
        min_choice_gap=float(args.min_choice_gap),
        main_branches=int(args.main_branches),
        ceiling_pairs_per_conflict_transition=int(
            args.ceiling_pairs_per_conflict_transition
        ),
        seed=int(args.seed),
        source_description=f"{args.dataset_id}:{args.split}",
    )
    print(f"out_dir={out_dir}")
    print(f"confirmation_freeze_hash={manifest['confirmation_freeze_hash']}")
    print(f"n_pairs={manifest['n_pairs']}")
    print(f"n_episodes={manifest['n_episodes']}")
    print(f"n_planned_traces={manifest['n_planned_traces']}")
    print(f"n_audit_prompts={manifest['n_audit_prompts']}")
    print(f"audit_prompt_dir={manifest['prompt_dir']}")


if __name__ == "__main__":
    main()
