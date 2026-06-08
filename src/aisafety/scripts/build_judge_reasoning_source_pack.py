"""Materialize public and existing local sources for the first judge-reasoning scout."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import itertools
import json
import os
from pathlib import Path
import tarfile
from typing import Any, Iterable

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json


ATTRIBUTE_NAMES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
DEFAULT_OUT_DIR = Path("data") / "derived" / "judge_reasoning_sources_v1"


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
    parser.add_argument("--truthfulqa-root", type=Path, default=None)
    parser.add_argument("--ethics-path", type=Path, default=None)
    parser.add_argument("--d4-jsonl", type=Path, default=None)
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


def _analysis_split(pair_id: str, *, seed: int) -> str:
    bucket = int(sha1_hex(f"{seed}:analysis-split:{pair_id}")[:12], 16) % 10
    if bucket < 6:
        return "fit"
    if bucket < 8:
        return "selection"
    return "intervention"


def _cap(
    rows: list[dict[str, Any]],
    *,
    max_pairs: int,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: sha1_hex(f"{seed}:{salt}:{row.get('pair_id') or row}"),
    )
    if int(max_pairs) > 0:
        ordered = ordered[: int(max_pairs)]
    return ordered


def _pair_row(
    *,
    pair_id: str,
    source_dataset: str,
    subset: str,
    split: str,
    prompt: str,
    option_a: str,
    option_b: str,
    target_option: str,
    target_kind: str,
    comparison_dimension: str,
    task_type: str,
    validity_type: str,
    difficulty_tier: str,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "pair_id": str(pair_id),
        "source_dataset": str(source_dataset),
        "subset": str(subset),
        "split": str(split),
        "prompt": flat_text(prompt),
        "option_a_text": flat_text(option_a),
        "option_b_text": flat_text(option_b),
        "target_option": str(target_option),
        "target_kind": str(target_kind),
        "comparison_dimension": str(comparison_dimension),
        "task_type": str(task_type),
        "validity_type": str(validity_type),
        "difficulty_tier": str(difficulty_tier),
        "analysis_split": _analysis_split(str(pair_id), seed=seed),
    }
    if metadata:
        row.update(metadata)
    return row


def _closest_length_distractor(correct: str, distractors: Iterable[str]) -> str:
    candidates = [flat_text(value) for value in distractors if flat_text(value)]
    if not candidates:
        return ""
    return min(
        candidates,
        key=lambda value: (abs(len(value.split()) - len(correct.split())), sha1_hex(value)),
    )


def build_arc_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_id: str,
    difficulty_tier: str,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = flat_text(str(row.get("question") or row.get("question_stem") or ""))
        choices = row.get("choices") or {}
        labels = [str(value).strip() for value in (choices.get("label") or [])]
        texts = [flat_text(str(value)) for value in (choices.get("text") or [])]
        answer = str(row.get("answerKey") or "").strip()
        if not question or answer not in labels or len(labels) != len(texts):
            continue
        correct_index = labels.index(answer)
        correct = texts[correct_index]
        distractor = _closest_length_distractor(
            correct,
            [text for choice_index, text in enumerate(texts) if choice_index != correct_index],
        )
        if not correct or not distractor or correct == distractor:
            continue
        source_id = str(row.get("id") or f"{dataset_id}:{index}")
        pair_id = sha1_hex(f"{dataset_id}:{source_id}:{correct}:{distractor}")
        out.append(
            _pair_row(
                pair_id=pair_id,
                source_dataset=dataset_id,
                subset=dataset_id,
                split=split,
                prompt=question,
                option_a=correct,
                option_b=distractor,
                target_option="A",
                target_kind="objective",
                comparison_dimension="factual_correctness",
                task_type="ordered_factual_judgment",
                validity_type="objective",
                difficulty_tier=difficulty_tier,
                seed=seed,
                metadata={
                    "source_example_id": source_id,
                    "correct_label": answer,
                    "distractor_policy": "closest_token_length",
                },
            )
        )
    return out


def _split_answer_list(value: Any) -> list[str]:
    return [
        flat_text(part)
        for part in str(value or "").split(";")
        if flat_text(part)
    ]


def build_truthfulqa_pairs(rows: Iterable[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = flat_text(str(row.get("Question") or row.get("question") or ""))
        truthful = flat_text(
            str(
                row.get("Best Answer")
                or row.get("best_answer")
                or row.get("truthful_answer")
                or ""
            )
        )
        false_answers = _split_answer_list(
            row.get("Best Incorrect Answer")
            or row.get("Incorrect Answers")
            or row.get("incorrect_answers")
            or row.get("false_answer")
        )
        false_answer = _closest_length_distractor(truthful, false_answers)
        if not question or not truthful or not false_answer or truthful == false_answer:
            continue
        category = flat_text(str(row.get("Category") or row.get("category") or "all"))
        pair_id = sha1_hex(f"truthfulqa:{index}:{question}:{truthful}:{false_answer}")
        out.append(
            _pair_row(
                pair_id=pair_id,
                source_dataset="truthfulqa",
                subset=category or "all",
                split="validation",
                prompt=question,
                option_a=truthful,
                option_b=false_answer,
                target_option="A",
                target_kind="objective",
                comparison_dimension="truthfulness",
                task_type="ordered_factual_judgment",
                validity_type="objective",
                difficulty_tier="misconception",
                seed=seed,
                metadata={"source_row_index": index},
            )
        )
    return out


def _read_csv_path(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_truthfulqa_rows(root: Path) -> tuple[list[dict[str, Any]], Path]:
    candidates = [
        root / "TruthfulQA.csv",
        root / "data" / "TruthfulQA.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return _read_csv_path(candidate), candidate
    raise FileNotFoundError(
        f"Could not find TruthfulQA.csv under {root}; checked: {candidates}"
    )


def _read_csv_bytes(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8-sig")
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def load_ethics_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    preferred_suffixes = (
        "commonsense/cm_test_hard.csv",
        "commonsense/cm_test.csv",
        "commonsense/cm_train.csv",
    )
    if path.is_dir():
        files = [candidate for candidate in path.rglob("*.csv")]
        for suffix in preferred_suffixes:
            for candidate in files:
                if candidate.as_posix().endswith(suffix):
                    return _read_csv_path(candidate), str(candidate)
        raise FileNotFoundError(f"No ETHICS commonsense CSV found under {path}")

    if not path.is_file():
        raise FileNotFoundError(path)
    with tarfile.open(path, "r:*") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        for suffix in preferred_suffixes:
            for member in members:
                if member.name.endswith(suffix):
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    return _read_csv_bytes(extracted.read()), f"{path}::{member.name}"
    raise FileNotFoundError(f"No ETHICS commonsense CSV found in {path}")


def _label_value(row: dict[str, Any]) -> int | None:
    raw = row.get("label")
    if raw is None:
        raw = row.get("is_acceptable")
    text = "" if raw is None else str(raw).strip().lower()
    if text in {"1", "true", "acceptable", "yes"}:
        return 1
    if text in {"0", "false", "unacceptable", "no"}:
        return 0
    return None


def build_ethics_pairs(rows: Iterable[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    positive: list[tuple[int, str, dict[str, Any]]] = []
    negative: list[tuple[int, str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        text = flat_text(str(row.get("input") or row.get("scenario") or row.get("text") or ""))
        label = _label_value(row)
        if not text or label is None:
            continue
        record = (index, text, row)
        (positive if label else negative).append(record)

    available = list(negative)
    out: list[dict[str, Any]] = []
    for pos_index, pos_text, pos_row in sorted(
        positive,
        key=lambda item: sha1_hex(f"{seed}:ethics-positive:{item[1]}"),
    ):
        if not available:
            break
        neg = min(
            available,
            key=lambda item: (
                abs(len(item[1].split()) - len(pos_text.split())),
                sha1_hex(f"{seed}:ethics-negative:{item[1]}"),
            ),
        )
        available.remove(neg)
        neg_index, neg_text, neg_row = neg
        pair_id = sha1_hex(f"ethics:{pos_index}:{neg_index}:{pos_text}:{neg_text}")
        out.append(
            _pair_row(
                pair_id=pair_id,
                source_dataset="ethics_commonsense",
                subset="commonsense_hard",
                split="test_hard",
                prompt="Which action is more morally acceptable?",
                option_a=pos_text,
                option_b=neg_text,
                target_option="A",
                target_kind="consensus",
                comparison_dimension="moral_acceptability",
                task_type="non_ordered_attribute",
                validity_type="consensus",
                difficulty_tier="contemplative",
                seed=seed,
                metadata={
                    "positive_source_index": pos_index,
                    "negative_source_index": neg_index,
                    "positive_is_short": pos_row.get("is_short"),
                    "negative_is_short": neg_row.get("is_short"),
                    "pairing_policy": "closest_token_length_without_replacement",
                },
            )
        )
    return out


def _attribute_vector(row: dict[str, Any]) -> tuple[float, ...] | None:
    values: list[float] = []
    for name in ATTRIBUTE_NAMES:
        try:
            values.append(float(row[name]))
        except (KeyError, TypeError, ValueError):
            return None
    return tuple(values)


def build_helpsteer_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        prompt = flat_text(str(row.get("prompt") or ""))
        response = flat_text(str(row.get("response") or ""))
        attributes = _attribute_vector(row)
        if not prompt or not response or attributes is None:
            continue
        by_prompt[prompt].append(
            {
                "index": index,
                "response": response,
                "attributes": attributes,
            }
        )

    candidates: dict[str, list[tuple[float, dict[str, Any]]]] = {
        "dominated": [],
        "tradeoff": [],
    }
    for prompt, group in by_prompt.items():
        for left, right in itertools.combinations(group, 2):
            diffs = tuple(
                left_value - right_value
                for left_value, right_value in zip(
                    left["attributes"],
                    right["attributes"],
                    strict=True,
                )
            )
            if all(value >= 0 for value in diffs) and any(value > 0 for value in diffs):
                better, worse = left, right
                oriented = diffs
            elif all(value <= 0 for value in diffs) and any(value < 0 for value in diffs):
                better, worse = right, left
                oriented = tuple(-value for value in diffs)
            else:
                better = worse = None
                oriented = diffs

            if better is not None and worse is not None:
                strength = float(sum(oriented))
                pair_id = sha1_hex(
                    f"helpsteer2:dominated:{prompt}:{better['response']}:{worse['response']}"
                )
                candidates["dominated"].append(
                    (
                        strength,
                        _pair_row(
                            pair_id=pair_id,
                            source_dataset="helpsteer2_dominated",
                            subset="pareto_dominated",
                            split="validation",
                            prompt=prompt,
                            option_a=better["response"],
                            option_b=worse["response"],
                            target_option="A",
                            target_kind="consensus",
                            comparison_dimension="overall_quality",
                            task_type="ordered_preference",
                            validity_type="consensus",
                            difficulty_tier="mixed",
                            seed=seed,
                            metadata={
                                "option_a_attributes": dict(
                                    zip(ATTRIBUTE_NAMES, better["attributes"], strict=True)
                                ),
                                "option_b_attributes": dict(
                                    zip(ATTRIBUTE_NAMES, worse["attributes"], strict=True)
                                ),
                                "attribute_deltas_a_minus_b": dict(
                                    zip(ATTRIBUTE_NAMES, oriented, strict=True)
                                ),
                            },
                        ),
                    )
                )
                continue

            positive = [value for value in diffs if value >= 1.0]
            negative = [value for value in diffs if value <= -1.0]
            if positive and negative:
                strength = float(sum(abs(value) for value in diffs))
                pair_id = sha1_hex(
                    f"helpsteer2:tradeoff:{prompt}:{left['response']}:{right['response']}"
                )
                candidates["tradeoff"].append(
                    (
                        strength,
                        _pair_row(
                            pair_id=pair_id,
                            source_dataset="helpsteer2_tradeoff",
                            subset="pareto_conflict",
                            split="validation",
                            prompt=prompt,
                            option_a=left["response"],
                            option_b=right["response"],
                            target_option="",
                            target_kind="none",
                            comparison_dimension="overall_quality",
                            task_type="non_ordered_attribute",
                            validity_type="plural",
                            difficulty_tier="contemplative",
                            seed=seed,
                            metadata={
                                "option_a_attributes": dict(
                                    zip(ATTRIBUTE_NAMES, left["attributes"], strict=True)
                                ),
                                "option_b_attributes": dict(
                                    zip(ATTRIBUTE_NAMES, right["attributes"], strict=True)
                                ),
                                "attribute_deltas_a_minus_b": dict(
                                    zip(ATTRIBUTE_NAMES, diffs, strict=True)
                                ),
                            },
                        ),
                    )
                )

    output: dict[str, list[dict[str, Any]]] = {}
    for kind, values in candidates.items():
        output[kind] = [
            row
            for _, row in sorted(
                values,
                key=lambda item: (
                    -item[0],
                    sha1_hex(f"{seed}:helpsteer2:{kind}:{item[1]['pair_id']}"),
                ),
            )
        ]
    return output


def build_d4_pairs(rows: Iterable[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        human = flat_text(str(row.get("human_text") or ""))
        llm = flat_text(str(row.get("llm_text") or ""))
        if not human or not llm or human == llm:
            continue
        pair_id = str(row.get("pair_id") or sha1_hex(f"{human}:{llm}"))
        if pair_id in by_pair:
            continue
        prompt = flat_text(
            str(
                row.get("prompt")
                or row.get("question")
                or row.get("title")
                or "Compare the two responses."
            )
        )
        by_pair[pair_id] = _pair_row(
            pair_id=pair_id,
            source_dataset="d4_human_llm",
            subset=str(
                row.get("source_dataset")
                or row.get("subset")
                or row.get("item_type")
                or "all"
            ),
            split=str(row.get("split") or ""),
            prompt=prompt,
            option_a=human,
            option_b=llm,
            target_option="",
            target_kind="none",
            comparison_dimension="overall_quality",
            task_type="human_vs_llm_quality",
            validity_type="preference",
            difficulty_tier="contemplative",
            seed=seed,
            metadata={
                "option_a_authorship": "human",
                "option_b_authorship": "llm",
                "original_source_dataset": str(row.get("source_dataset") or ""),
                "original_subset": str(row.get("subset") or ""),
                "llm_generator": str(row.get("llm_generator") or ""),
                "item_type": str(row.get("item_type") or ""),
            },
        )
    return list(by_pair.values())


def _discover_d4(artifact_root: Path) -> Path:
    derived = artifact_root / "data" / "derived"
    preferred = [
        derived / "d4_human_llm_stage_contrast_pairs_v1" / "bt_pairs.jsonl",
        derived
        / "d4_human_llm_stage_contrast_pairs_qwen3_30b_a3b_base_it_broad_v1"
        / "bt_pairs.jsonl",
    ]
    for path in preferred:
        if path.is_file():
            return path
    candidates = sorted(
        derived.glob("d4_human_llm_stage_contrast_pairs*/bt_pairs.jsonl"),
        key=lambda path: (path.stat().st_size, path.as_posix()),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No D4 human-vs-LLM bt_pairs.jsonl found under {derived}"
    )


def _load_hf(dataset_id: str, config: str | None, *, split: str, cache_dir: Path | None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets is required to materialize public sources.") from exc
    return load_dataset(
        dataset_id,
        config,
        split=split,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )


def materialize(
    *,
    artifact_root: Path,
    cache_dir: Path | None,
    truthfulqa_root: Path,
    ethics_path: Path,
    d4_jsonl: Path,
    out_dir: Path,
    max_pairs_per_dataset: int,
    seed: int,
) -> dict[str, Any]:
    arc_easy = build_arc_pairs(
        _load_hf("ai2_arc", "ARC-Easy", split="validation", cache_dir=cache_dir),
        dataset_id="arc_easy",
        difficulty_tier="easy",
        split="validation",
        seed=seed,
    )
    arc_challenge = build_arc_pairs(
        _load_hf("ai2_arc", "ARC-Challenge", split="validation", cache_dir=cache_dir),
        dataset_id="arc_challenge",
        difficulty_tier="hard",
        split="validation",
        seed=seed,
    )
    truthful_rows, truthful_source = load_truthfulqa_rows(truthfulqa_root)
    truthfulqa = build_truthfulqa_pairs(truthful_rows, seed=seed)
    ethics_rows, ethics_source = load_ethics_rows(ethics_path)
    ethics = build_ethics_pairs(ethics_rows, seed=seed)
    helpsteer_rows = _load_hf(
        "nvidia/HelpSteer2",
        None,
        split="validation",
        cache_dir=cache_dir,
    )
    helpsteer = build_helpsteer_pairs(helpsteer_rows, seed=seed)
    d4 = build_d4_pairs(read_jsonl(d4_jsonl), seed=seed)

    datasets = {
        "arc_easy": arc_easy,
        "arc_challenge": arc_challenge,
        "truthfulqa_binary": truthfulqa,
        "ethics_commonsense_hard": ethics,
        "helpsteer2_dominated": helpsteer["dominated"],
        "helpsteer2_tradeoff": helpsteer["tradeoff"],
        "d4_human_llm": d4,
    }
    empty = sorted(name for name, rows in datasets.items() if not rows)
    if empty:
        raise ValueError(f"Source pack produced empty datasets: {empty}")

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for name, rows in datasets.items():
        capped = _cap(
            rows,
            max_pairs=int(max_pairs_per_dataset),
            seed=seed,
            salt=name,
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
        "stage": "judge-reasoning-source-pack",
        "artifact_root": str(artifact_root),
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "truthfulqa_source": str(truthful_source),
        "ethics_source": str(ethics_source),
        "d4_source": str(d4_jsonl),
        "out_dir": str(out_dir),
        "max_pairs_per_dataset": int(max_pairs_per_dataset),
        "seed": int(seed),
        "datasets": outputs,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    artifact_root = _resolve(workspace_root, args.artifact_root)
    cache_dir = (
        _resolve(workspace_root, args.cache_dir)
        if args.cache_dir is not None
        else None
    )
    truthfulqa_root = (
        _resolve(artifact_root, args.truthfulqa_root)
        if args.truthfulqa_root is not None
        else artifact_root / "data" / "external" / "judge_reasoning" / "TruthfulQA"
    )
    ethics_path = (
        _resolve(artifact_root, args.ethics_path)
        if args.ethics_path is not None
        else artifact_root / "data" / "external" / "judge_reasoning" / "ethics.tar"
    )
    d4_jsonl = (
        _resolve(artifact_root, args.d4_jsonl)
        if args.d4_jsonl is not None
        else _discover_d4(artifact_root)
    )
    out_dir = _resolve(artifact_root, args.out_dir)
    manifest = materialize(
        artifact_root=artifact_root,
        cache_dir=cache_dir,
        truthfulqa_root=truthfulqa_root,
        ethics_path=ethics_path,
        d4_jsonl=d4_jsonl,
        out_dir=out_dir,
        max_pairs_per_dataset=int(args.max_pairs_per_dataset),
        seed=int(args.seed),
    )
    print(f"out_dir={manifest['out_dir']}")
    for name, summary in manifest["datasets"].items():
        print(
            f"{name}: available={summary['n_available']} "
            f"written={summary['n_written']} path={summary['path']}"
        )


if __name__ == "__main__":
    main()
