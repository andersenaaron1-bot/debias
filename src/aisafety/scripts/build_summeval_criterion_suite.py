"""Build a SummEval criterion-operationalization validation suite.

The emitted episode schema intentionally matches
``run_judge_criterion_switch_behavior`` so the same behavioral runner can be
used for HelpSteer2 and SummEval external validation.
"""

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
from aisafety.scripts.build_helpsteer2_criterion_switch_suite import (
    _semantic_swap,
)
from aisafety.scripts.build_judge_reasoning_source_pack import _load_hf


DEFAULT_OUT_DIR = Path("data") / "derived" / "summeval_criterion_suite_v1"
CRITERIA: dict[str, str] = {
    "coherence": (
        "Judge which summary is more coherent as a summary: it should be "
        "well organized, internally connected, and easy to follow as a whole."
    ),
    "consistency": (
        "Judge which summary is more factually consistent with the source "
        "document. Prefer the summary with fewer unsupported or contradictory "
        "claims."
    ),
    "fluency": (
        "Judge which summary is more fluent: it should be grammatical, "
        "natural, and readable sentence by sentence."
    ),
    "relevance": (
        "Judge which summary better captures the important information from "
        "the source document and avoids irrelevant or missing central points."
    ),
}
CRITERION_LABELS: dict[str, str] = {
    "coherence": "coherence",
    "consistency": "source consistency",
    "fluency": "fluency",
    "relevance": "relevance",
}
MAIN_CONDITIONS = (
    "free_cot",
    "generic_scaffold",
    "criterion_scaffold",
    "score_evidence",
)
CEILING_CONDITION = "explicit_target"
TRANSITION_TYPES = ("criterion_flip", "tie_to_choice", "same_target")
SUMMARY_KEYS = (
    "summary",
    "decoded",
    "candidate",
    "generated_summary",
    "machine_summary",
    "response",
    "system_output",
)
SOURCE_KEYS = (
    "source",
    "document",
    "article",
    "source_document",
    "input",
    "context",
)
DOC_ID_KEYS = (
    "doc_id",
    "document_id",
    "source_id",
    "filepath",
    "file_id",
    "topic_id",
    "id",
)

CRITERION_SCAFFOLD = (
    "Use this four-step procedure.\n"
    "1. Translate the active summary-evaluation criterion into concrete, "
    "observable tests that can be applied to both summaries.\n"
    "2. Apply only those tests to Option A and identify the relevant evidence "
    "from the source and the summary.\n"
    "3. Apply the same tests to Option B using the same standard.\n"
    "4. Compare the two criterion-specific assessments and determine whether "
    "one option is better or the active criterion leaves them tied or "
    "underdetermined.\n"
    "Keep the steps explicit and do not substitute overall summary quality for "
    "the active criterion."
)
GENERIC_SCAFFOLD = (
    "Use this four-step procedure.\n"
    "1. Summarize the main content of Option A without judging it.\n"
    "2. Summarize the main content of Option B in the same level of detail.\n"
    "3. List the main similarities and differences in content, wording, and "
    "organization.\n"
    "4. Check both summaries for omissions and unsupported assumptions.\n"
    "Keep the steps explicit and balanced, but do not yet state a verdict."
)
CONTINUATION = (
    "Continue from the prior analysis under the same active summary-evaluation "
    "criterion. Use the analysis already produced to form the final verdict "
    "without restarting from scratch."
)


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
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--dataset-id", default="mteb/summeval")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-pairs-per-transition", type=int, default=16)
    parser.add_argument("--min-pairs-per-transition", type=int, default=8)
    parser.add_argument("--min-choice-gap", type=float, default=0.45)
    parser.add_argument("--tie-threshold", type=float, default=0.15)
    parser.add_argument("--main-branches", type=int, default=2)
    parser.add_argument("--ceiling-branches", type=int, default=1)
    parser.add_argument("--no-explicit-target", action="store_true")
    parser.add_argument("--max-source-chars", type=int, default=3500)
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
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    return len(values)


def _first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and flat_text(value):
            return flat_text(value)
    return ""


def _references_text(value: Any) -> str:
    if isinstance(value, str):
        return flat_text(value)
    if isinstance(value, list):
        values = [flat_text(str(item)) for item in value if flat_text(str(item))]
        if values:
            return "\n\n".join(f"Reference {index + 1}: {text}" for index, text in enumerate(values[:4]))
    return ""


def _score_mapping(payload: Any) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not isinstance(payload, dict):
        return scores
    source = payload.get("scores") if isinstance(payload.get("scores"), dict) else payload
    for criterion in CRITERIA:
        try:
            scores[criterion] = float(source[criterion])
        except (KeyError, TypeError, ValueError):
            pass
    return scores


def _annotation_scores(row: dict[str, Any]) -> dict[str, float]:
    direct = _score_mapping(row)
    if len(direct) == len(CRITERIA):
        return direct
    buckets: dict[str, list[float]] = {criterion: [] for criterion in CRITERIA}
    for key in (
        "expert_annotations",
        "turker_annotations",
        "annotations",
        "human_annotations",
        "expert_scores",
        "scores",
    ):
        value = row.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            for criterion, score in _score_mapping(item).items():
                buckets[criterion].append(float(score))
    averaged = {
        criterion: float(sum(values) / len(values))
        for criterion, values in buckets.items()
        if values
    }
    return {**direct, **averaged}


def _doc_id(row: dict[str, Any], *, row_index: int, source_text: str) -> str:
    for key in DOC_ID_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return flat_text(str(value))
    if source_text:
        return sha1_hex(f"summeval-source:{source_text}")[:16]
    return f"row-{row_index}"


def _summary_items(row: dict[str, Any]) -> list[tuple[str, str, dict[str, float]]]:
    direct_summary = _first_text(row, SUMMARY_KEYS)
    direct_scores = _annotation_scores(row)
    if direct_summary and len(direct_scores) == len(CRITERIA):
        return [
            (
                flat_text(str(row.get("model_id") or row.get("system_id") or "")),
                direct_summary,
                direct_scores,
            )
        ]

    items: list[tuple[str, str, dict[str, float]]] = []
    for key in ("summaries", "machine_summaries", "system_outputs", "outputs"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        for index, item in enumerate(value):
            if isinstance(item, dict):
                summary = _first_text(item, SUMMARY_KEYS)
                scores = _annotation_scores(item)
                system = flat_text(
                    str(item.get("model_id") or item.get("system_id") or index)
                )
            else:
                summary = flat_text(str(item))
                scores = {}
                system = str(index)
            if summary and len(scores) == len(CRITERIA):
                items.append((system, summary, scores))
    return items


def normalize_summeval_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_source_chars: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        source = _first_text(row, SOURCE_KEYS)
        references = _references_text(
            row.get("references") or row.get("reference") or row.get("refs")
        )
        if not source and references:
            source = references
        if int(max_source_chars) > 0 and len(source) > int(max_source_chars):
            source = source[: int(max_source_chars)].rstrip() + " ..."
        doc_id = _doc_id(row, row_index=row_index, source_text=source)
        for summary_index, (system_id, summary, scores) in enumerate(
            _summary_items(row)
        ):
            normalized.append(
                {
                    "source_document_id": doc_id,
                    "source_text": source,
                    "references_text": references,
                    "summary_id": sha1_hex(
                        f"summeval-summary:{doc_id}:{system_id}:{summary}"
                    ),
                    "system_id": system_id or f"summary-{summary_index}",
                    "summary_text": summary,
                    "scores": scores,
                }
            )
    return normalized


def _target(gap: float, *, tie_threshold: float) -> str:
    if abs(float(gap)) <= float(tie_threshold):
        return "C"
    return "A" if float(gap) > 0 else "B"


def _criterion_gap(
    left: dict[str, Any],
    right: dict[str, Any],
    criterion_id: str,
) -> float:
    return float(left["scores"][criterion_id]) - float(right["scores"][criterion_id])


def _canonical_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    content_id = sha1_hex(
        "|".join(
            [
                "summeval-content-pair",
                str(left["source_document_id"]),
                *sorted([str(left["summary_id"]), str(right["summary_id"])]),
            ]
        )
    )
    reverse = int(sha1_hex(f"{seed}:summeval-orient:{content_id}")[:8], 16) % 2 == 1
    option_a, option_b = (right, left) if reverse else (left, right)
    return option_a, option_b, content_id


def _transition_candidates(
    targets: dict[str, str],
    gaps: dict[str, float],
    *,
    min_choice_gap: float,
) -> dict[str, list[tuple[str, str, float]]]:
    candidates: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for left, right in itertools.combinations(CRITERIA, 2):
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
            candidates["criterion_flip"].append(
                (left, right, min(left_gap, right_gap))
            )
        if (
            (left_target == "C") != (right_target == "C")
            and max(left_gap, right_gap) >= float(min_choice_gap)
        ):
            if left_target == "C":
                candidates["tie_to_choice"].append(
                    (left, right, max(left_gap, right_gap))
                )
            else:
                candidates["tie_to_choice"].append(
                    (right, left, max(left_gap, right_gap))
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


def _analysis_split(selected_index: int) -> str:
    bucket = int(selected_index) % 5
    if bucket <= 2:
        return "fit"
    if bucket == 3:
        return "selection"
    return "intervention"


def _score_strength(row: dict[str, Any]) -> float:
    gaps = row["criterion_gaps_a_minus_b"]
    return min(
        abs(float(gaps[str(row["initial_criterion_id"])])),
        abs(float(gaps[str(row["updated_criterion_id"])])),
    )


def build_summeval_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    max_pairs_per_transition: int,
    min_pairs_per_transition: int,
    min_choice_gap: float,
    tie_threshold: float,
    max_source_chars: int,
    seed: int,
) -> list[dict[str, Any]]:
    normalized = normalize_summeval_rows(
        rows,
        max_source_chars=int(max_source_chars),
    )
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        if not row["source_text"] or not row["summary_text"]:
            continue
        by_doc[str(row["source_document_id"])].append(row)

    candidates: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for doc_id, group in by_doc.items():
        for left, right in itertools.combinations(group, 2):
            if left["summary_text"] == right["summary_text"]:
                continue
            option_a, option_b, content_id = _canonical_pair(
                left,
                right,
                seed=int(seed),
            )
            gaps = {
                criterion: _criterion_gap(option_a, option_b, criterion)
                for criterion in CRITERIA
            }
            targets = {
                criterion: _target(gap, tie_threshold=float(tie_threshold))
                for criterion, gap in gaps.items()
            }
            for transition_type, criterion_pairs in _transition_candidates(
                targets,
                gaps,
                min_choice_gap=float(min_choice_gap),
            ).items():
                ordered = sorted(
                    criterion_pairs,
                    key=lambda value: (
                        -value[2],
                        sha1_hex(
                            f"{seed}:summeval-criteria:{content_id}:"
                            f"{transition_type}:{value[0]}:{value[1]}"
                        ),
                    ),
                )
                initial, updated, strength = ordered[0]
                if transition_type in {"criterion_flip", "same_target"}:
                    flip = (
                        int(
                            sha1_hex(
                                f"{seed}:summeval-direction:{content_id}:"
                                f"{transition_type}"
                            )[:8],
                            16,
                        )
                        % 2
                        == 1
                    )
                    if flip:
                        initial, updated = updated, initial
                pair_id = sha1_hex(
                    "|".join(
                        [
                            "summeval-criterion-suite",
                            content_id,
                            transition_type,
                            initial,
                            updated,
                        ]
                    )
                )
                candidates[transition_type].append(
                    (
                        float(strength),
                        {
                            "pair_id": pair_id,
                            "origin_pair_id": content_id,
                            "pair_signature": content_id,
                            "source_dataset": "summeval",
                            "source_document_id": doc_id,
                            "source_system_a": str(option_a["system_id"]),
                            "source_system_b": str(option_b["system_id"]),
                            "prompt": (
                                "Source document:\n"
                                f"{option_a['source_text']}"
                            ),
                            "option_a_text": str(option_a["summary_text"]),
                            "option_b_text": str(option_b["summary_text"]),
                            "option_a_attributes": dict(option_a["scores"]),
                            "option_b_attributes": dict(option_b["scores"]),
                            "criterion_gaps_a_minus_b": gaps,
                            "criterion_targets": targets,
                            "transition_type": transition_type,
                            "initial_criterion_id": initial,
                            "updated_criterion_id": updated,
                            "initial_target_semantic": targets[initial],
                            "updated_target_semantic": targets[updated],
                            "source_split": "summeval",
                            "source_summary_ids": [
                                str(option_a["summary_id"]),
                                str(option_b["summary_id"]),
                            ],
                        },
                    )
                )

    selected: list[dict[str, Any]] = []
    used_content: set[str] = set()
    for transition_type in TRANSITION_TYPES:
        available = sorted(
            candidates.get(transition_type, []),
            key=lambda item: (
                -item[0],
                sha1_hex(
                    f"{seed}:summeval-select:{transition_type}:"
                    f"{item[1]['pair_id']}"
                ),
            ),
        )
        chosen: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        while available and len(chosen) < int(max_pairs_per_transition):

            def rank(item: tuple[float, dict[str, Any]]) -> tuple[Any, ...]:
                strength, row = item
                updated = str(row["updated_criterion_id"])
                initial = str(row["initial_criterion_id"])
                content_seen = str(row["origin_pair_id"]) in used_content
                return (
                    content_seen,
                    counts[f"updated:{updated}"],
                    counts[f"initial:{initial}"],
                    counts[f"pair:{'|'.join(sorted([initial, updated]))}"],
                    -float(strength),
                    sha1_hex(f"{seed}:summeval-rank:{row['pair_id']}"),
                )

            index, (_, row) = min(
                enumerate(available),
                key=lambda indexed: rank(indexed[1]),
            )
            available.pop(index)
            row = dict(row)
            row["analysis_split"] = _analysis_split(len(chosen))
            row["summeval_locked"] = True
            chosen.append(row)
            used_content.add(str(row["origin_pair_id"]))
            counts[f"updated:{row['updated_criterion_id']}"] += 1
            counts[f"initial:{row['initial_criterion_id']}"] += 1
            counts[
                f"pair:{'|'.join(sorted([str(row['initial_criterion_id']), str(row['updated_criterion_id'])]))}"
            ] += 1
        if len(chosen) < int(min_pairs_per_transition):
            raise ValueError(
                f"Transition {transition_type!r} emitted {len(chosen)} pairs; "
                f"required {min_pairs_per_transition}. Lower thresholds or "
                "provide a larger SummEval source file."
            )
        selected.extend(chosen)
    return selected


def _displayed_options(
    pair: dict[str, Any],
    presentation_order: str,
) -> tuple[str, str]:
    if str(presentation_order) == "swapped":
        return str(pair["option_b_text"]), str(pair["option_a_text"])
    return str(pair["option_a_text"]), str(pair["option_b_text"])


def _displayed_target(semantic: str, presentation_order: str) -> str:
    return _semantic_swap(str(semantic)) if str(presentation_order) == "swapped" else str(semantic)


def _score(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.2f}"


def score_evidence_text(
    pair: dict[str, Any],
    *,
    criterion_id: str,
    presentation_order: str,
) -> str:
    if str(presentation_order) == "swapped":
        option_a = dict(pair["option_b_attributes"])
        option_b = dict(pair["option_a_attributes"])
    else:
        option_a = dict(pair["option_a_attributes"])
        option_b = dict(pair["option_b_attributes"])
    label = CRITERION_LABELS[str(criterion_id)]
    return (
        "Independent human summary-evaluation evidence is available. Higher "
        "scores are better; use the scores as criterion-specific evidence, "
        "not as a precomputed verdict.\n"
        f"Option A {label} score: {_score(option_a[criterion_id])}\n"
        f"Option B {label} score: {_score(option_b[criterion_id])}"
    )


def _condition_fields(
    pair: dict[str, Any],
    *,
    condition: str,
    presentation_order: str,
) -> dict[str, str]:
    criterion_id = str(pair["updated_criterion_id"])
    target = str(pair["criterion_targets"][criterion_id])
    displayed_target = _displayed_target(target, presentation_order)
    if condition == "criterion_scaffold":
        instructions = CRITERION_SCAFFOLD
        evidence = ""
        target_disclosure = "none"
        structure_family = "criterion_operational"
    elif condition == "generic_scaffold":
        instructions = GENERIC_SCAFFOLD
        evidence = ""
        target_disclosure = "none"
        structure_family = "generic_matched"
    elif condition == "score_evidence":
        instructions = ""
        evidence = score_evidence_text(
            pair,
            criterion_id=criterion_id,
            presentation_order=presentation_order,
        )
        target_disclosure = "criterion_scores"
        structure_family = "free"
    elif condition == "explicit_target":
        instructions = ""
        evidence = (
            "An independent human summary-evaluation adjudication is supplied "
            "as a ceiling control. Under the active criterion, the evidence "
            f"implies Option {displayed_target}. Treat this implication as "
            "given while forming the verdict."
        )
        target_disclosure = "explicit_target"
        structure_family = "free"
    elif condition == "free_cot":
        instructions = ""
        evidence = ""
        target_disclosure = "none"
        structure_family = "free"
    else:
        raise ValueError(f"Unknown SummEval condition: {condition}")
    return {
        "phase1_reasoning_instructions": instructions,
        "phase1_evidence_text": evidence,
        "target_disclosure": target_disclosure,
        "structure_family": structure_family,
        "phase1_target_option": displayed_target,
        "phase2_target_option": displayed_target,
        "phase1_target_semantic": target,
        "phase2_target_semantic": target,
    }


def summeval_episode(
    pair: dict[str, Any],
    *,
    condition: str,
    presentation_order: str,
    branches: int,
) -> dict[str, Any]:
    updated = str(pair["updated_criterion_id"])
    option_a, option_b = _displayed_options(pair, presentation_order)
    fields = _condition_fields(
        pair,
        condition=condition,
        presentation_order=presentation_order,
    )
    episode_id = sha1_hex(
        f"{pair['pair_id']}|summeval|{condition}|{presentation_order}"
    )
    return {
        "episode_id": episode_id,
        "comparison_id": episode_id,
        "pair_id": str(pair["pair_id"]),
        "origin_pair_id": str(pair["origin_pair_id"]),
        "source_dataset": "summeval_criterion_validation",
        "subset": str(pair["transition_type"]),
        "split": str(pair.get("source_split") or ""),
        "task_type": "summary_criterion_operationalization",
        "comparison_dimension": "summary_criterion_use",
        "prompt": str(pair["prompt"]),
        "option_a_text": option_a,
        "option_b_text": option_b,
        "presentation_order": str(presentation_order),
        "condition_id": condition,
        "condition_label": condition,
        "transition_type": str(pair["transition_type"]),
        "analysis_split": str(pair.get("analysis_split") or "summeval"),
        "information_timing": "early",
        "information_type": fields["target_disclosure"],
        "structure_family": fields["structure_family"],
        "target_disclosure": fields["target_disclosure"],
        "initial_criterion_id": str(pair["initial_criterion_id"]),
        "updated_criterion_id": updated,
        "phase1_criterion_id": updated,
        "phase2_criterion_id": updated,
        "phase1_criterion_text": CRITERIA[updated],
        "phase2_criterion_text": CRITERIA[updated],
        "phase1_reasoning_instructions": fields[
            "phase1_reasoning_instructions"
        ],
        "phase1_evidence_text": fields["phase1_evidence_text"],
        "phase2_evidence_text": "",
        "phase2_explicit_target_option": "",
        "phase2_update_override": CONTINUATION,
        "phase1_cache_group": f"summeval:{condition}:{updated}",
        "branches_per_episode": int(branches),
        "direct_criterion_ids": [updated],
        "phase1_target_option": fields["phase1_target_option"],
        "phase2_target_option": fields["phase2_target_option"],
        "phase1_target_semantic": fields["phase1_target_semantic"],
        "phase2_target_semantic": fields["phase2_target_semantic"],
        "expected_target_change": False,
        "allow_tie": True,
        "metadata": {
            "criterion_targets": pair["criterion_targets"],
            "criterion_gaps_a_minus_b": pair["criterion_gaps_a_minus_b"],
            "option_a_attributes_canonical": pair["option_a_attributes"],
            "option_b_attributes_canonical": pair["option_b_attributes"],
            "pair_signature": str(pair["pair_signature"]),
            "source_document_id": str(pair["source_document_id"]),
            "source_system_a": str(pair["source_system_a"]),
            "source_system_b": str(pair["source_system_b"]),
            "source_summary_ids": pair["source_summary_ids"],
        },
    }


def build_episodes(
    pairs: Iterable[dict[str, Any]],
    *,
    main_branches: int,
    ceiling_branches: int,
    include_explicit_target: bool,
) -> list[dict[str, Any]]:
    values = list(pairs)
    episodes = [
        summeval_episode(
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
            summeval_episode(
                pair,
                condition=CEILING_CONDITION,
                presentation_order=order,
                branches=int(ceiling_branches),
            )
            for pair in values
            for order in ("original", "swapped")
        )
    return episodes


def _load_rows(args: argparse.Namespace, *, workspace_root: Path) -> list[dict[str, Any]]:
    if args.input_jsonl is not None:
        path = _resolve(workspace_root, args.input_jsonl)
        return read_jsonl(path)
    cache_dir = (
        _resolve(Path(args.artifact_root).resolve(), args.cache_dir)
        if args.cache_dir is not None
        else None
    )
    config = str(args.dataset_config or "").strip() or None
    dataset = _load_hf(
        str(args.dataset_id),
        config,
        split=str(args.split),
        cache_dir=cache_dir,
    )
    return [dict(row) for row in dataset]


def materialize(
    *,
    rows: Iterable[dict[str, Any]],
    out_dir: Path,
    max_pairs_per_transition: int,
    min_pairs_per_transition: int,
    min_choice_gap: float,
    tie_threshold: float,
    main_branches: int,
    ceiling_branches: int,
    include_explicit_target: bool,
    max_source_chars: int,
    seed: int,
    source_description: dict[str, Any],
) -> dict[str, Any]:
    source_rows = list(rows)
    pairs = build_summeval_pairs(
        source_rows,
        max_pairs_per_transition=int(max_pairs_per_transition),
        min_pairs_per_transition=int(min_pairs_per_transition),
        min_choice_gap=float(min_choice_gap),
        tie_threshold=float(tie_threshold),
        max_source_chars=int(max_source_chars),
        seed=int(seed),
    )
    episodes = build_episodes(
        pairs,
        main_branches=int(main_branches),
        ceiling_branches=int(ceiling_branches),
        include_explicit_target=bool(include_explicit_target),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "pairs.jsonl", pairs)
    _write_jsonl(out_dir / "episodes.jsonl", episodes)
    planned_traces = sum(int(row["branches_per_episode"]) for row in episodes)
    direct_keys = {
        (
            str(row["pair_id"]),
            str(row["presentation_order"]),
            str(row["updated_criterion_id"]),
        )
        for row in episodes
    }
    manifest = {
        "stage": "summeval-criterion-operationalization-suite",
        "source": source_description,
        "out_dir": str(out_dir),
        "seed": int(seed),
        "criteria": CRITERIA,
        "main_conditions": list(MAIN_CONDITIONS),
        "ceiling_condition": CEILING_CONDITION if include_explicit_target else "",
        "main_branches": int(main_branches),
        "ceiling_branches": int(ceiling_branches),
        "include_explicit_target": bool(include_explicit_target),
        "max_pairs_per_transition": int(max_pairs_per_transition),
        "min_pairs_per_transition": int(min_pairs_per_transition),
        "min_choice_gap": float(min_choice_gap),
        "tie_threshold": float(tie_threshold),
        "max_source_chars": int(max_source_chars),
        "n_source_rows": int(len(source_rows)),
        "n_pairs": int(len(pairs)),
        "n_episodes": int(len(episodes)),
        "n_planned_traces": int(planned_traces),
        "n_planned_direct_rows": int(len(direct_keys)),
        "counts_by_condition": dict(
            sorted(Counter(row["condition_id"] for row in episodes).items())
        ),
        "counts_by_transition": dict(
            sorted(Counter(row["transition_type"] for row in pairs).items())
        ),
        "counts_by_updated_criterion": dict(
            sorted(Counter(row["updated_criterion_id"] for row in pairs).items())
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
    out_dir = _resolve(workspace_root, args.out_dir)
    rows = _load_rows(args, workspace_root=workspace_root)
    source_description = {
        "input_jsonl": str(args.input_jsonl or ""),
        "dataset_id": str(args.dataset_id),
        "dataset_config": str(args.dataset_config or ""),
        "split": str(args.split),
    }
    manifest = materialize(
        rows=rows,
        out_dir=out_dir,
        max_pairs_per_transition=int(args.max_pairs_per_transition),
        min_pairs_per_transition=int(args.min_pairs_per_transition),
        min_choice_gap=float(args.min_choice_gap),
        tie_threshold=float(args.tie_threshold),
        main_branches=int(args.main_branches),
        ceiling_branches=int(args.ceiling_branches),
        include_explicit_target=not bool(args.no_explicit_target),
        max_source_chars=int(args.max_source_chars),
        seed=int(args.seed),
        source_description=source_description,
    )
    print(f"out_dir={out_dir}")
    print(f"n_source_rows={manifest['n_source_rows']}")
    print(f"n_pairs={manifest['n_pairs']}")
    print(f"n_episodes={manifest['n_episodes']}")
    print(f"n_planned_traces={manifest['n_planned_traces']}")
    print(f"n_planned_direct_rows={manifest['n_planned_direct_rows']}")


if __name__ == "__main__":
    main()
