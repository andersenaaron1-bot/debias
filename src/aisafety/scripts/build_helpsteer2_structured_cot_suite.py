"""Build a matched HelpSteer2 structured-CoT operationalization suite."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_json, read_jsonl, resolve_path, sha1_hex
from aisafety.mech.d4_io import write_json
from aisafety.scripts.build_helpsteer2_criterion_confirmation import (
    _displayed_options,
    score_evidence_text,
)
from aisafety.scripts.build_helpsteer2_criterion_switch_suite import (
    _semantic_swap,
)
from aisafety.scripts.build_helpsteer2_matched_criterion_suite import CRITERIA


MAIN_CONDITIONS = (
    "free_cot",
    "generic_scaffold",
    "criterion_scaffold",
    "score_evidence",
)
CEILING_CONDITION = "explicit_target"

CRITERION_SCAFFOLD = (
    "Use this four-step procedure.\n"
    "1. Translate the active decision rule into concrete, observable tests "
    "that can be applied to both options.\n"
    "2. Apply only those tests to Option A and identify the relevant textual "
    "evidence.\n"
    "3. Apply the same tests to Option B and identify the relevant textual "
    "evidence.\n"
    "4. Compare the two criterion-specific assessments and determine whether "
    "one option is better or the rule leaves them tied or underdetermined.\n"
    "Keep the steps explicit and do not substitute overall response quality "
    "for the active decision rule."
)

GENERIC_SCAFFOLD = (
    "Use this four-step procedure.\n"
    "1. Summarize the main approach taken by Option A without judging it.\n"
    "2. Summarize the main approach taken by Option B in the same level of "
    "detail.\n"
    "3. List the main similarities and differences in content, organization, "
    "and presentation.\n"
    "4. Check both summaries for omitted context and unsupported assumptions.\n"
    "Keep the steps explicit and balanced, but do not yet use these "
    "observations to state a verdict."
)

CONTINUATION = (
    "Continue from the prior analysis under the same active decision rule. "
    "Use the analysis already produced to form the final verdict without "
    "restarting from scratch."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-suite-dir", type=Path, required=True)
    parser.add_argument("--main-branches", type=int, default=2)
    parser.add_argument("--ceiling-branches", type=int, default=1)
    parser.add_argument("--no-explicit-target", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "data/derived/helpsteer2_structured_cot_suite_v1"
        ),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in values:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    return len(values)


def _displayed_target(semantic: str, presentation_order: str) -> str:
    return (
        _semantic_swap(str(semantic))
        if str(presentation_order) == "swapped"
        else str(semantic)
    )


def _condition_fields(
    pair: dict[str, Any],
    *,
    condition: str,
    presentation_order: str,
) -> dict[str, Any]:
    criterion_id = str(pair["updated_criterion_id"])
    target = str(pair["criterion_targets"][criterion_id])
    displayed_target = _displayed_target(target, presentation_order)
    if condition == "criterion_scaffold":
        instructions = CRITERION_SCAFFOLD
        evidence = ""
        structure_family = "criterion_operational"
        target_disclosure = "none"
    elif condition == "generic_scaffold":
        instructions = GENERIC_SCAFFOLD
        evidence = ""
        structure_family = "generic_matched"
        target_disclosure = "none"
    elif condition == "score_evidence":
        instructions = ""
        evidence = score_evidence_text(
            pair,
            criterion_id=criterion_id,
            presentation_order=presentation_order,
        )
        structure_family = "free"
        target_disclosure = "criterion_scores"
    elif condition == "explicit_target":
        instructions = ""
        evidence = (
            "An independent criterion-specific adjudication is supplied as a "
            "ceiling control. Under the stated decision rule, the evidence "
            f"implies Option {displayed_target}. Treat this implication as "
            "given while forming the verdict."
        )
        structure_family = "free"
        target_disclosure = "explicit_target"
    elif condition == "free_cot":
        instructions = ""
        evidence = ""
        structure_family = "free"
        target_disclosure = "none"
    else:
        raise ValueError(f"Unknown structured-CoT condition: {condition}")
    return {
        "phase1_reasoning_instructions": instructions,
        "phase1_evidence_text": evidence,
        "structure_family": structure_family,
        "target_disclosure": target_disclosure,
        "phase1_target_semantic": target,
        "phase2_target_semantic": target,
        "phase1_target_option": displayed_target,
        "phase2_target_option": displayed_target,
    }


def structured_cot_episode(
    pair: dict[str, Any],
    *,
    condition: str,
    presentation_order: str,
    branches: int,
) -> dict[str, Any]:
    criterion_id = str(pair["updated_criterion_id"])
    option_a, option_b = _displayed_options(pair, presentation_order)
    fields = _condition_fields(
        pair,
        condition=condition,
        presentation_order=presentation_order,
    )
    episode_id = sha1_hex(
        f"{pair['pair_id']}|structured-cot|{condition}|"
        f"{presentation_order}"
    )
    return {
        "episode_id": episode_id,
        "comparison_id": episode_id,
        "pair_id": str(pair["pair_id"]),
        "origin_pair_id": str(pair["pair_id"]),
        "source_dataset": "helpsteer2_structured_cot",
        "subset": str(pair["transition_type"]),
        "split": str(pair.get("source_split") or ""),
        "task_type": "criterion_operationalization_structured_cot",
        "comparison_dimension": "criterion_use",
        "prompt": str(pair["prompt"]),
        "option_a_text": option_a,
        "option_b_text": option_b,
        "presentation_order": str(presentation_order),
        "condition_id": condition,
        "condition_label": condition,
        "transition_type": str(pair["transition_type"]),
        "analysis_split": "structured_cot_followup",
        "information_timing": "early",
        "information_type": fields["target_disclosure"],
        "structure_family": fields["structure_family"],
        "target_disclosure": fields["target_disclosure"],
        "initial_criterion_id": str(pair["initial_criterion_id"]),
        "updated_criterion_id": criterion_id,
        "phase1_criterion_id": criterion_id,
        "phase2_criterion_id": criterion_id,
        "phase1_criterion_text": CRITERIA[criterion_id],
        "phase2_criterion_text": CRITERIA[criterion_id],
        "phase1_reasoning_instructions": fields[
            "phase1_reasoning_instructions"
        ],
        "phase1_evidence_text": fields["phase1_evidence_text"],
        "phase2_evidence_text": "",
        "phase2_explicit_target_option": "",
        "phase2_update_override": CONTINUATION,
        "phase1_cache_group": f"structured_cot:{condition}",
        "branches_per_episode": int(branches),
        "direct_criterion_ids": [criterion_id],
        "phase1_target_option": fields["phase1_target_option"],
        "phase2_target_option": fields["phase2_target_option"],
        "phase1_target_semantic": fields["phase1_target_semantic"],
        "phase2_target_semantic": fields["phase2_target_semantic"],
        "expected_target_change": False,
        "allow_tie": True,
        "metadata": {
            "criterion_targets": pair["criterion_targets"],
            "criterion_gaps_a_minus_b": pair[
                "criterion_gaps_a_minus_b"
            ],
            "option_a_attributes_canonical": pair["option_a_attributes"],
            "option_b_attributes_canonical": pair["option_b_attributes"],
            "pair_signature": str(pair["pair_signature"]),
            "confirmation_locked": True,
            "source_initial_criterion_id": str(
                pair["initial_criterion_id"]
            ),
        },
    }


def build_structured_cot_episodes(
    pairs: Iterable[dict[str, Any]],
    *,
    ceiling_pair_ids: set[str],
    main_branches: int,
    ceiling_branches: int,
    include_explicit_target: bool,
) -> list[dict[str, Any]]:
    values = list(pairs)
    episodes = [
        structured_cot_episode(
            pair,
            condition=condition,
            presentation_order=order,
            branches=int(main_branches),
        )
        for pair in values
        for condition in MAIN_CONDITIONS
        for order in ("original", "swapped")
    ]
    if include_explicit_target:
        episodes.extend(
            structured_cot_episode(
                pair,
                condition=CEILING_CONDITION,
                presentation_order=order,
                branches=int(ceiling_branches),
            )
            for pair in values
            if str(pair["pair_id"]) in ceiling_pair_ids
            for order in ("original", "swapped")
        )
    return episodes


def materialize(
    *,
    source_suite_dir: Path,
    out_dir: Path,
    main_branches: int,
    ceiling_branches: int,
    include_explicit_target: bool,
    seed: int,
) -> dict[str, Any]:
    source_manifest = read_json(source_suite_dir / "manifest.json")
    pairs = read_jsonl(source_suite_dir / "pairs.jsonl")
    if not pairs:
        raise ValueError(f"No source pairs found in {source_suite_dir}")
    ceiling_pair_ids = {
        str(value)
        for value in source_manifest.get("ceiling_pair_ids") or []
    }
    if include_explicit_target and not ceiling_pair_ids:
        raise ValueError(
            "Source suite does not record explicit-target ceiling pairs."
        )
    episodes = build_structured_cot_episodes(
        pairs,
        ceiling_pair_ids=ceiling_pair_ids,
        main_branches=int(main_branches),
        ceiling_branches=int(ceiling_branches),
        include_explicit_target=bool(include_explicit_target),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "pairs.jsonl", pairs)
    _write_jsonl(out_dir / "episodes.jsonl", episodes)
    planned_traces = sum(
        int(row["branches_per_episode"]) for row in episodes
    )
    direct_keys = {
        (
            str(row["pair_id"]),
            str(row["presentation_order"]),
            str(row["updated_criterion_id"]),
        )
        for row in episodes
    }
    manifest = {
        "stage": "helpsteer2-structured-cot-suite",
        "source_suite_dir": str(source_suite_dir),
        "source_confirmation_freeze_hash": source_manifest.get(
            "confirmation_freeze_hash"
        ),
        "out_dir": str(out_dir),
        "seed": int(seed),
        "main_conditions": list(MAIN_CONDITIONS),
        "ceiling_condition": (
            CEILING_CONDITION if include_explicit_target else ""
        ),
        "main_branches": int(main_branches),
        "ceiling_branches": int(ceiling_branches),
        "include_explicit_target": bool(include_explicit_target),
        "n_pairs": int(len(pairs)),
        "n_episodes": int(len(episodes)),
        "n_planned_traces": int(planned_traces),
        "n_planned_direct_rows": int(len(direct_keys)),
        "n_ceiling_pairs": int(len(ceiling_pair_ids)),
        "ceiling_pair_ids": sorted(ceiling_pair_ids),
        "counts_by_condition": dict(
            sorted(Counter(row["condition_id"] for row in episodes).items())
        ),
        "counts_by_transition": dict(
            sorted(Counter(row["transition_type"] for row in pairs).items())
        ),
        "criterion_scaffold": CRITERION_SCAFFOLD,
        "generic_scaffold": GENERIC_SCAFFOLD,
        "phase2_continuation": CONTINUATION,
        "pairs_jsonl": str(out_dir / "pairs.jsonl"),
        "episodes_jsonl": str(out_dir / "episodes.jsonl"),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    source_suite_dir = _resolve(workspace_root, args.source_suite_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    manifest = materialize(
        source_suite_dir=source_suite_dir,
        out_dir=out_dir,
        main_branches=int(args.main_branches),
        ceiling_branches=int(args.ceiling_branches),
        include_explicit_target=not bool(args.no_explicit_target),
        seed=int(args.seed),
    )
    print(f"out_dir={out_dir}")
    print(f"n_pairs={manifest['n_pairs']}")
    print(f"n_episodes={manifest['n_episodes']}")
    print(f"n_planned_traces={manifest['n_planned_traces']}")
    print(f"n_planned_direct_rows={manifest['n_planned_direct_rows']}")


if __name__ == "__main__":
    main()
