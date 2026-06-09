"""Build the controlled task progression for deliberation experiments."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.scripts.build_judge_reasoning_source_pack import (
    ATTRIBUTE_NAMES,
    _analysis_split,
    _cap,
    _load_hf,
    _pair_row,
)


DEFAULT_BASE_SOURCE_DIR = Path("data") / "derived" / "judge_reasoning_sources_v1"
DEFAULT_OUT_DIR = Path("data") / "derived" / "judge_deliberation_sources_v1"
DEFAULT_WEIGHTS = {
    "helpfulness": 0.30,
    "correctness": 0.40,
    "coherence": 0.20,
    "complexity": 0.05,
    "verbosity": 0.05,
}
CRITERIA = {
    "overall": {
        "dimension": "overall_quality",
        "text": (
            "Choose the better response overall. Balance substantive correctness, "
            "helpfulness, coherence, and appropriate detail."
        ),
    },
    "correctness": {
        "dimension": "correctness",
        "text": (
            "Choose the response that is more factually and logically correct. "
            "Ignore style unless it changes the meaning."
        ),
    },
    "helpfulness": {
        "dimension": "helpfulness",
        "text": (
            "Choose the response that is more useful for satisfying the user's "
            "request. Treat correctness as a necessary part of helpfulness."
        ),
    },
    "coherence": {
        "dimension": "coherence",
        "text": (
            "Choose the response that is clearer, more internally coherent, and "
            "easier to follow."
        ),
    },
    "weighted": {
        "dimension": "weighted_quality",
        "text": (
            "Apply this fixed rubric: correctness 40%, helpfulness 30%, coherence "
            "20%, appropriate complexity 5%, and appropriate verbosity 5%. Choose "
            "the response with the higher weighted score."
        ),
    },
}
OPTION_RE = re.compile(r"(?m)^\s*\(([A-Z])\)\s*(.+?)\s*$")
GSM_FINAL_RE = re.compile(r"(?m)(####\s*)([^\n]+)\s*$")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


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
    parser.add_argument("--base-source-dir", type=Path, default=DEFAULT_BASE_SOURCE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-pairs-per-dataset", type=int, default=200)
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


def _replace_last_match(text: str, match: re.Match[str], replacement: str) -> str:
    return text[: match.start()] + replacement + text[match.end() :]


def perturb_answer(value: str) -> str:
    """Return a deterministic nearby but different numeric/symbolic answer."""

    text = str(value or "").strip()
    if not text:
        return ""
    fraction = re.fullmatch(r"\s*(-?\d+)\s*/\s*(\d+)\s*", text.replace(",", ""))
    if fraction:
        numerator = int(fraction.group(1))
        denominator = int(fraction.group(2))
        return f"{numerator + 1}/{denominator}"
    numeric = text.replace(",", "")
    try:
        number = Decimal(numeric)
    except InvalidOperation:
        matches = list(NUMBER_RE.finditer(text))
        if matches:
            match = matches[-1]
            replacement = perturb_answer(match.group(0))
            return _replace_last_match(text, match, replacement)
        return f"{text} + 1"
    changed = number + Decimal(1)
    if number == number.to_integral_value():
        return str(int(changed))
    return format(changed.normalize(), "f")


def _replace_last_boxed(solution: str) -> tuple[str, str, str] | None:
    marker = r"\boxed{"
    start = solution.rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(solution)):
        char = solution[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                answer = solution[content_start:index]
                wrong = perturb_answer(answer)
                if not wrong or wrong == answer:
                    return None
                corrupted = solution[:content_start] + wrong + solution[index:]
                return corrupted, answer, wrong
    return None


def build_gsm8k_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = flat_text(str(row.get("question") or ""))
        answer = str(row.get("answer") or "").strip()
        match = GSM_FINAL_RE.search(answer)
        if not question or match is None:
            continue
        correct_final = match.group(2).strip()
        wrong_final = perturb_answer(correct_final)
        if not wrong_final or wrong_final == correct_final:
            continue
        corrupted = _replace_last_match(
            answer,
            match,
            f"{match.group(1)}{wrong_final}",
        )
        pair_id = sha1_hex(f"gsm8k:{index}:{question}:{correct_final}")
        out.append(
            _pair_row(
                pair_id=pair_id,
                source_dataset="gsm8k_verification",
                subset="main",
                split="test",
                prompt=question,
                option_a=answer,
                option_b=corrupted,
                target_option="A",
                target_kind="objective",
                comparison_dimension="solution_correctness",
                task_type="formal_verification",
                validity_type="objective",
                difficulty_tier="formal",
                seed=seed,
                metadata={
                    "criterion_id": "exact_correctness",
                    "criterion_text": (
                        "Choose the proposed solution whose final answer follows "
                        "correctly from its reasoning."
                    ),
                    "criterion_determinacy": "exact",
                    "determinacy_level": "formal_verifiable",
                    "source_example_id": str(row.get("id") or index),
                    "correct_final_answer": correct_final,
                    "corrupted_final_answer": wrong_final,
                    "corruption_type": "final_answer_numeric_perturbation",
                },
            )
        )
    return out


def build_math500_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        problem = flat_text(str(row.get("problem") or row.get("question") or ""))
        solution = str(row.get("solution") or row.get("answer") or "").strip()
        replacement = _replace_last_boxed(solution)
        if not problem or replacement is None:
            continue
        corrupted, correct_final, wrong_final = replacement
        pair_id = sha1_hex(f"math500:{index}:{problem}:{correct_final}")
        out.append(
            _pair_row(
                pair_id=pair_id,
                source_dataset="math500_verification",
                subset=str(row.get("subject") or "all"),
                split="test",
                prompt=problem,
                option_a=solution,
                option_b=corrupted,
                target_option="A",
                target_kind="objective",
                comparison_dimension="solution_correctness",
                task_type="formal_verification",
                validity_type="objective",
                difficulty_tier=str(row.get("level") or "formal"),
                seed=seed,
                metadata={
                    "criterion_id": "exact_correctness",
                    "criterion_text": (
                        "Choose the proposed mathematical solution whose final "
                        "answer is correct and supported by the derivation."
                    ),
                    "criterion_determinacy": "exact",
                    "determinacy_level": "formal_verifiable",
                    "source_example_id": str(row.get("unique_id") or index),
                    "correct_final_answer": correct_final,
                    "corrupted_final_answer": wrong_final,
                    "corruption_type": "boxed_final_answer_perturbation",
                },
            )
        )
    return out


def build_bbh_logical_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    subset: str,
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt = str(row.get("input") or row.get("question") or "").strip()
        target = str(row.get("target") or row.get("answer") or "").strip().upper()
        target = target.strip("() ")
        choices = [(label, flat_text(text)) for label, text in OPTION_RE.findall(prompt)]
        by_label = {label: text for label, text in choices if text}
        if not prompt or target not in by_label or len(by_label) < 2:
            continue
        correct = by_label[target]
        distractors = [
            (label, text)
            for label, text in choices
            if label != target and text and text != correct
        ]
        if not distractors:
            continue
        wrong_label, wrong = min(
            distractors,
            key=lambda item: (
                abs(len(item[1].split()) - len(correct.split())),
                sha1_hex(f"{seed}:bbh:{subset}:{item[0]}:{item[1]}"),
            ),
        )
        pair_id = sha1_hex(f"bbh:{subset}:{index}:{prompt}:{target}")
        out.append(
            _pair_row(
                pair_id=pair_id,
                source_dataset="bbh_logical_deduction",
                subset=subset,
                split="test",
                prompt=prompt,
                option_a=f"({target}) {correct}",
                option_b=f"({wrong_label}) {wrong}",
                target_option="A",
                target_kind="objective",
                comparison_dimension="logical_correctness",
                task_type="formal_verification",
                validity_type="objective",
                difficulty_tier="logical",
                seed=seed,
                metadata={
                    "criterion_id": "exact_correctness",
                    "criterion_text": (
                        "Choose the answer that follows from all stated logical "
                        "constraints."
                    ),
                    "criterion_determinacy": "exact",
                    "determinacy_level": "logical_verifiable",
                    "source_example_id": str(row.get("id") or index),
                    "correct_choice_label": target,
                    "distractor_choice_label": wrong_label,
                },
            )
        )
    return out


def _attribute_target(
    row: dict[str, Any],
    *,
    criterion_id: str,
) -> str:
    left = row.get("option_a_attributes")
    right = row.get("option_b_attributes")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return ""
    if criterion_id in ATTRIBUTE_NAMES:
        left_score = float(left.get(criterion_id, 0.0))
        right_score = float(right.get(criterion_id, 0.0))
    elif criterion_id == "weighted":
        left_score = sum(
            DEFAULT_WEIGHTS[name] * float(left.get(name, 0.0))
            for name in ATTRIBUTE_NAMES
        )
        right_score = sum(
            DEFAULT_WEIGHTS[name] * float(right.get(name, 0.0))
            for name in ATTRIBUTE_NAMES
        )
    else:
        return ""
    if left_score > right_score:
        return "A"
    if right_score > left_score:
        return "B"
    return ""


def build_criterion_variants(
    rows: Iterable[dict[str, Any]],
    *,
    source_prefix: str,
    criteria: Iterable[str],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    outputs = {criterion: [] for criterion in criteria}
    for row in rows:
        origin_pair_id = str(row.get("pair_id") or "")
        if not origin_pair_id:
            continue
        for criterion_id in outputs:
            spec = CRITERIA[criterion_id]
            target = _attribute_target(row, criterion_id=criterion_id)
            if criterion_id == "overall":
                target = str(row.get("target_option") or "")
            variant = dict(row)
            variant.update(
                {
                    "pair_id": origin_pair_id,
                    "origin_pair_id": origin_pair_id,
                    "source_dataset": f"{source_prefix}_{criterion_id}",
                    "comparison_dimension": spec["dimension"],
                    "criterion_id": criterion_id,
                    "criterion_family": source_prefix,
                    "criterion_text": spec["text"],
                    "condition_id": criterion_id,
                    "condition_label": criterion_id,
                    "target_option": target,
                    "target_kind": (
                        "criterion_proxy"
                        if target and criterion_id != "overall"
                        else str(row.get("target_kind") or "none")
                    ),
                    "criterion_determinacy": (
                        "explicit_proxy"
                        if criterion_id != "overall"
                        else "underspecified"
                    ),
                    "determinacy_level": (
                        "criterion_resolved_tradeoff"
                        if criterion_id != "overall"
                        else (
                            "plural_tradeoff"
                            if source_prefix.startswith("helpsteer2")
                            else "ecological_open"
                        )
                    ),
                    "analysis_split": str(
                        row.get("analysis_split")
                        or _analysis_split(origin_pair_id, seed=seed)
                    ),
                }
            )
            outputs[criterion_id].append(variant)
    return outputs


def _read_required(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing base source file: {path}. Build judge_reasoning_sources_v1 first."
        )
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Base source file is empty: {path}")
    return rows


def _annotate_base_rows(
    rows: Iterable[dict[str, Any]],
    *,
    criterion_id: str,
    criterion_text: str,
    criterion_determinacy: str,
    determinacy_level: str,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "criterion_id": str(row.get("criterion_id") or criterion_id),
            "criterion_text": str(row.get("criterion_text") or criterion_text),
            "criterion_determinacy": str(
                row.get("criterion_determinacy") or criterion_determinacy
            ),
            "determinacy_level": str(
                row.get("determinacy_level") or determinacy_level
            ),
        }
        for row in rows
    ]


def _load_bbh(cache_dir: Path | None) -> list[tuple[str, Iterable[dict[str, Any]]]]:
    dataset_ids = ("lukaemon/bbh", "maveriq/bigbenchhard")
    subsets = (
        "logical_deduction_three_objects",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
    )
    errors: list[str] = []
    for dataset_id in dataset_ids:
        loaded: list[tuple[str, Iterable[dict[str, Any]]]] = []
        try:
            for subset in subsets:
                loaded.append(
                    (
                        subset,
                        _load_hf(
                            dataset_id,
                            subset,
                            split="test",
                            cache_dir=cache_dir,
                        ),
                    )
                )
            return loaded
        except Exception as exc:  # pragma: no cover - remote dataset variants
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("Could not load a BBH dataset: " + " | ".join(errors))


def materialize(
    *,
    base_source_dir: Path,
    cache_dir: Path | None,
    out_dir: Path,
    max_pairs_per_dataset: int,
    seed: int,
) -> dict[str, Any]:
    datasets: dict[str, list[dict[str, Any]]] = {
        "arc_challenge": _annotate_base_rows(
            _read_required(base_source_dir / "arc_challenge.jsonl"),
            criterion_id="factual_correctness",
            criterion_text="Choose the answer that is factually correct.",
            criterion_determinacy="explicit_gold",
            determinacy_level="factual_explicit",
        ),
        "truthfulqa_binary": _annotate_base_rows(
            _read_required(base_source_dir / "truthfulqa_binary.jsonl"),
            criterion_id="truthfulness",
            criterion_text=(
                "Choose the answer that is truthful rather than a common "
                "misconception."
            ),
            criterion_determinacy="explicit_gold",
            determinacy_level="factual_misconception",
        ),
        "ethics_commonsense_hard": _annotate_base_rows(
            _read_required(base_source_dir / "ethics_commonsense_hard.jsonl"),
            criterion_id="moral_acceptability",
            criterion_text=(
                "Choose the action that better matches ordinary judgments of "
                "moral acceptability."
            ),
            criterion_determinacy="consensus",
            determinacy_level="consensus_normative",
        ),
        "helpsteer2_dominated": _annotate_base_rows(
            _read_required(base_source_dir / "helpsteer2_dominated.jsonl"),
            criterion_id="overall",
            criterion_text=CRITERIA["overall"]["text"],
            criterion_determinacy="ordered_proxy",
            determinacy_level="dominated_preference",
        ),
    }
    tradeoff = _read_required(base_source_dir / "helpsteer2_tradeoff.jsonl")
    d4 = _read_required(base_source_dir / "d4_human_llm.jsonl")

    datasets["gsm8k_verification"] = build_gsm8k_pairs(
        _load_hf("openai/gsm8k", "main", split="test", cache_dir=cache_dir),
        seed=seed,
    )
    datasets["math500_verification"] = build_math500_pairs(
        _load_hf("HuggingFaceH4/MATH-500", None, split="test", cache_dir=cache_dir),
        seed=seed,
    )
    bbh_rows: list[dict[str, Any]] = []
    for subset, rows in _load_bbh(cache_dir):
        bbh_rows.extend(build_bbh_logical_pairs(rows, subset=subset, seed=seed))
    datasets["bbh_logical_deduction"] = bbh_rows

    datasets.update(
        {
            f"helpsteer2_tradeoff_{criterion}": rows
            for criterion, rows in build_criterion_variants(
                tradeoff,
                source_prefix="helpsteer2_tradeoff",
                criteria=CRITERIA,
                seed=seed,
            ).items()
        }
    )
    datasets.update(
        {
            f"d4_human_llm_{criterion}": rows
            for criterion, rows in build_criterion_variants(
                d4,
                source_prefix="d4_human_llm",
                criteria=CRITERIA,
                seed=seed,
            ).items()
        }
    )

    empty = sorted(name for name, rows in datasets.items() if not rows)
    if empty:
        raise ValueError(f"Deliberation source pack produced empty datasets: {empty}")

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for name, rows in datasets.items():
        capped = _cap(
            rows,
            max_pairs=int(max_pairs_per_dataset),
            seed=seed,
            salt=f"deliberation:{name}",
        )
        path = out_dir / f"{name}.jsonl"
        count = _write_jsonl(path, capped)
        outputs[name] = {
            "path": str(path),
            "n_available": int(len(rows)),
            "n_written": int(count),
            "analysis_split_counts": dict(
                sorted(Counter(str(row["analysis_split"]) for row in capped).items())
            ),
        }

    manifest = {
        "stage": "judge-deliberation-source-pack",
        "base_source_dir": str(base_source_dir),
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "out_dir": str(out_dir),
        "max_pairs_per_dataset": int(max_pairs_per_dataset),
        "seed": int(seed),
        "criteria": CRITERIA,
        "weighted_rubric": DEFAULT_WEIGHTS,
        "datasets": outputs,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    artifact_root = _resolve(workspace_root, args.artifact_root)
    base_source_dir = _resolve(artifact_root, args.base_source_dir)
    cache_dir = (
        _resolve(artifact_root, args.cache_dir)
        if args.cache_dir is not None
        else None
    )
    out_dir = _resolve(artifact_root, args.out_dir)
    manifest = materialize(
        base_source_dir=base_source_dir,
        cache_dir=cache_dir,
        out_dir=out_dir,
        max_pairs_per_dataset=int(args.max_pairs_per_dataset),
        seed=int(args.seed),
    )
    print(f"out_dir={manifest['out_dir']}")
    for name, summary in manifest["datasets"].items():
        print(f"{name}: available={summary['n_available']} written={summary['n_written']}")


if __name__ == "__main__":
    main()
