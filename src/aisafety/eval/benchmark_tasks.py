"""Reward-benchmark task adapters for candidate-scoring evaluation.

This module normalizes multiple-choice / candidate-selection datasets into a
common schema that can be scored by the scalar reward model used in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from aisafety.config import DEFAULT_SEED


@dataclass(frozen=True)
class RewardBenchmarkExample:
    benchmark: str
    example_id: str
    prompt: str
    responses: tuple[str, ...]
    choice_labels: tuple[str, ...]
    choice_texts: tuple[str, ...]
    correct_idx: int
    group: str | None = None


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_id: str
    config_name: str | None
    default_split: str
    description: str
    normalizer: Callable[[dict[str, Any], int], RewardBenchmarkExample]


@dataclass(frozen=True)
class RunSpec:
    name: str
    adapter_dir: Path | None
    value_head: Path


def parse_run_spec(spec: str) -> RunSpec:
    """Parse `name=ADAPTER_DIR::VALUE_HEAD`."""
    raw = str(spec or "").strip()
    if not raw or "=" not in raw:
        raise ValueError("Run spec must be formatted as name=ADAPTER_DIR::VALUE_HEAD")
    name, payload = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing run name in spec: {spec!r}")
    if "::" not in payload:
        raise ValueError(f"Run spec must contain '::' between adapter_dir and value_head: {spec!r}")
    adapter_raw, value_raw = payload.split("::", 1)
    value_raw = value_raw.strip()
    if not value_raw:
        raise ValueError(f"Missing value head path in spec: {spec!r}")
    adapter_raw = adapter_raw.strip()
    adapter = None if not adapter_raw else Path(adapter_raw)
    return RunSpec(name=name, adapter_dir=adapter, value_head=Path(value_raw))


def available_benchmarks() -> list[str]:
    return sorted(BENCHMARKS)


def benchmark_descriptions() -> list[tuple[str, str]]:
    return [(name, BENCHMARKS[name].description) for name in available_benchmarks()]


def load_benchmark_examples(
    benchmark: str,
    *,
    split: str | None = None,
    cache_dir: Path | None = None,
    max_examples: int | None = None,
    seed: int = DEFAULT_SEED,
) -> list[RewardBenchmarkExample]:
    """Load and normalize one benchmark split via `datasets`."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets is required for reward benchmarks.") from exc

    bench = str(benchmark).strip()
    if bench not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark: {bench!r}. Available: {available_benchmarks()}")
    spec = BENCHMARKS[bench]

    ds = load_dataset(
        spec.dataset_id,
        spec.config_name,
        split=str(split or spec.default_split),
        cache_dir=None if cache_dir is None else str(cache_dir),
        # PIQA and some older benchmark loaders ship small dataset scripts on HF.
        # The eval job is non-interactive, so opt in explicitly for this fixed,
        # known benchmark suite rather than blocking on a trust prompt.
        trust_remote_code=True,
    )

    if max_examples is not None and int(max_examples) > 0:
        ds = ds.shuffle(seed=int(seed))
        n = min(int(max_examples), len(ds))
        ds = ds.select(range(n))

    out: list[RewardBenchmarkExample] = []
    for i, row in enumerate(ds):
        out.append(spec.normalizer(dict(row), i))
    return out


def compute_mcq_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize one run on one benchmark from per-example records."""
    if not records:
        return {
            "n_examples": 0,
            "accuracy": float("nan"),
            "mean_correct_margin": float("nan"),
            "mean_score_spread": float("nan"),
            "mean_gold_rank": float("nan"),
            "mrr": float("nan"),
            "mean_num_choices": float("nan"),
        }

    correct = np.asarray([bool(r["is_correct"]) for r in records], dtype=np.float32)
    margins = np.asarray([float(r["correct_margin"]) for r in records], dtype=np.float32)
    spreads = np.asarray([float(r["score_spread"]) for r in records], dtype=np.float32)
    gold_ranks = np.asarray([float(r["gold_rank"]) for r in records], dtype=np.float32)
    num_choices = np.asarray([float(r["num_choices"]) for r in records], dtype=np.float32)

    return {
        "n_examples": int(len(records)),
        "accuracy": float(correct.mean()),
        "mean_correct_margin": float(margins.mean()),
        "mean_score_spread": float(spreads.mean()),
        "mean_gold_rank": float(gold_ranks.mean()),
        "mrr": float((1.0 / gold_ranks).mean()),
        "mean_num_choices": float(num_choices.mean()),
    }


def make_mcq_record(
    example: RewardBenchmarkExample,
    *,
    scores: list[float],
    run_name: str,
) -> dict[str, Any]:
    if len(scores) != len(example.responses):
        raise ValueError("scores length must match example.responses length")

    arr = np.asarray(scores, dtype=np.float32)
    pred_idx = int(np.argmax(arr))
    gold_idx = int(example.correct_idx)
    other = np.delete(arr, gold_idx)
    best_other = float(other.max()) if len(other) else float("nan")
    gold_score = float(arr[gold_idx])
    # Rank is 1 + number of options with strictly higher score.
    gold_rank = 1 + int(np.sum(arr > gold_score))

    return {
        "run": str(run_name),
        "benchmark": str(example.benchmark),
        "group": None if example.group is None else str(example.group),
        "example_id": str(example.example_id),
        "gold_idx": gold_idx,
        "gold_label": str(example.choice_labels[gold_idx]),
        "gold_text": str(example.choice_texts[gold_idx]),
        "pred_idx": pred_idx,
        "pred_label": str(example.choice_labels[pred_idx]),
        "pred_text": str(example.choice_texts[pred_idx]),
        "is_correct": bool(pred_idx == gold_idx),
        "gold_score": gold_score,
        "pred_score": float(arr[pred_idx]),
        "correct_margin": gold_score - best_other,
        "score_spread": float(arr.max() - arr.min()) if len(arr) else float("nan"),
        "gold_rank": int(gold_rank),
        "num_choices": int(len(arr)),
        "choice_labels": list(example.choice_labels),
        "choice_texts": list(example.choice_texts),
        "scores": [float(x) for x in arr.tolist()],
    }


def _nonempty(text: Any) -> str:
    return str(text or "").strip()


def _join_blocks(parts: list[str]) -> str:
    return "\n\n".join([p.strip() for p in parts if str(p).strip()])


def _make_mcq_prompt(*, stem: str, choices: list[tuple[str, str]], instruction: str) -> str:
    lines = [stem.strip(), "Options:"]
    lines.extend([f"{label}. {text}" for label, text in choices])
    if instruction.strip():
        lines.append("")
        lines.append(instruction.strip())
    return "\n".join(lines).strip()


def _example_id(benchmark: str, row: dict[str, Any], idx: int) -> str:
    for key in ("id", "ind", "idx", "qID", "question_id"):
        val = row.get(key)
        if val is None:
            continue
        sval = str(val).strip()
        if sval:
            return sval
    return f"{benchmark}:{idx}"


def _normalize_arc(row: dict[str, Any], idx: int, *, benchmark: str) -> RewardBenchmarkExample:
    question = _nonempty(row.get("question") or row.get("question_stem"))
    choices_raw = row.get("choices") or {}
    labels = [str(x).strip() for x in (choices_raw.get("label") or [])]
    texts = [_nonempty(x) for x in (choices_raw.get("text") or [])]
    if not question or not labels or len(labels) != len(texts):
        raise ValueError(f"Invalid ARC row at index {idx}")
    answer = _nonempty(row.get("answerKey"))
    if answer not in labels:
        raise ValueError(f"ARC answer {answer!r} not found in labels {labels!r}")
    prompt = _make_mcq_prompt(
        stem=f"Question:\n{question}",
        choices=list(zip(labels, texts, strict=True)),
        instruction="Select the single best answer.",
    )
    return RewardBenchmarkExample(
        benchmark=benchmark,
        example_id=_example_id(benchmark, row, idx),
        prompt=prompt,
        responses=tuple(texts),
        choice_labels=tuple(labels),
        choice_texts=tuple(texts),
        correct_idx=int(labels.index(answer)),
        group=_nonempty(row.get("subject")) or None,
    )


def _normalize_piqa(row: dict[str, Any], idx: int) -> RewardBenchmarkExample:
    goal = _nonempty(row.get("goal"))
    sol1 = _nonempty(row.get("sol1"))
    sol2 = _nonempty(row.get("sol2"))
    label = row.get("label")
    if not goal or not sol1 or not sol2 or label is None:
        raise ValueError(f"Invalid PIQA row at index {idx}")
    prompt = _make_mcq_prompt(
        stem=f"Task:\n{goal}",
        choices=[("A", sol1), ("B", sol2)],
        instruction="Select the solution that best accomplishes the task.",
    )
    return RewardBenchmarkExample(
        benchmark="piqa",
        example_id=_example_id("piqa", row, idx),
        prompt=prompt,
        responses=(sol1, sol2),
        choice_labels=("A", "B"),
        choice_texts=(sol1, sol2),
        correct_idx=int(label),
    )


def _normalize_winogrande(row: dict[str, Any], idx: int) -> RewardBenchmarkExample:
    sentence = _nonempty(row.get("sentence"))
    option1 = _nonempty(row.get("option1"))
    option2 = _nonempty(row.get("option2"))
    answer = _nonempty(row.get("answer"))
    if not sentence or not option1 or not option2 or answer not in {"1", "2"}:
        raise ValueError(f"Invalid Winogrande row at index {idx}")
    filled1 = sentence.replace("_", option1)
    filled2 = sentence.replace("_", option2)
    prompt = _make_mcq_prompt(
        stem=f"Fill in the blank with the most sensible option.\n\nSentence:\n{sentence}",
        choices=[("A", option1), ("B", option2)],
        instruction="Select the option that best completes the sentence.",
    )
    return RewardBenchmarkExample(
        benchmark="winogrande",
        example_id=_example_id("winogrande", row, idx),
        prompt=prompt,
        responses=(filled1, filled2),
        choice_labels=("A", "B"),
        choice_texts=(option1, option2),
        correct_idx=int(answer) - 1,
    )


def _normalize_hellaswag(row: dict[str, Any], idx: int) -> RewardBenchmarkExample:
    ctx_a = _nonempty(row.get("ctx_a"))
    ctx_b = _nonempty(row.get("ctx_b"))
    endings = tuple(_nonempty(x) for x in (row.get("endings") or []))
    label_raw = row.get("label")
    if not ctx_a or not endings:
        raise ValueError(f"Invalid HellaSwag row at index {idx}")
    if label_raw is None:
        raise ValueError(f"Missing HellaSwag label at index {idx}")
    label = int(label_raw)
    context = _join_blocks(["Context:", " ".join(x for x in [ctx_a, ctx_b] if x).strip()])
    labels = tuple(chr(ord("A") + i) for i in range(len(endings)))
    prompt = _make_mcq_prompt(
        stem=f"Choose the most plausible continuation.\n\n{context}",
        choices=list(zip(labels, endings, strict=True)),
        instruction="Select the continuation that best fits the context.",
    )
    return RewardBenchmarkExample(
        benchmark="hellaswag",
        example_id=_example_id("hellaswag", row, idx),
        prompt=prompt,
        responses=endings,
        choice_labels=labels,
        choice_texts=endings,
        correct_idx=label,
        group=_nonempty(row.get("activity_label")) or None,
    )


def _normalize_social_iqa(row: dict[str, Any], idx: int) -> RewardBenchmarkExample:
    context = _nonempty(row.get("context"))
    question = _nonempty(row.get("question"))
    a = _nonempty(row.get("answerA"))
    b = _nonempty(row.get("answerB"))
    c = _nonempty(row.get("answerC"))
    label = _nonempty(row.get("label"))
    if not context or not question or not a or not b or not c or label not in {"1", "2", "3"}:
        raise ValueError(f"Invalid SocialIQA row at index {idx}")
    choices = [("A", a), ("B", b), ("C", c)]
    prompt = _make_mcq_prompt(
        stem=_join_blocks(["Context:", context, f"Question:\n{question}"]),
        choices=choices,
        instruction="Select the best answer.",
    )
    choice_texts = tuple(x[1] for x in choices)
    choice_labels = tuple(x[0] for x in choices)
    return RewardBenchmarkExample(
        benchmark="social_iqa",
        example_id=_example_id("social_iqa", row, idx),
        prompt=prompt,
        responses=choice_texts,
        choice_labels=choice_labels,
        choice_texts=choice_texts,
        correct_idx=int(label) - 1,
    )


def _normalize_boolq(row: dict[str, Any], idx: int) -> RewardBenchmarkExample:
    passage = _nonempty(row.get("passage"))
    question = _nonempty(row.get("question"))
    answer = row.get("answer")
    if not passage or not question or not isinstance(answer, bool):
        raise ValueError(f"Invalid BoolQ row at index {idx}")
    prompt = _make_mcq_prompt(
        stem=_join_blocks(["Passage:", passage, f"Question:\n{question}"]),
        choices=[("A", "yes"), ("B", "no")],
        instruction="Select yes or no.",
    )
    return RewardBenchmarkExample(
        benchmark="boolq",
        example_id=_example_id("boolq", row, idx),
        prompt=prompt,
        responses=("yes", "no"),
        choice_labels=("A", "B"),
        choice_texts=("yes", "no"),
        correct_idx=0 if bool(answer) else 1,
    )


def _normalize_mmlu(row: dict[str, Any], idx: int) -> RewardBenchmarkExample:
    question = _nonempty(row.get("question"))
    choices_raw = row.get("choices") or []
    choices = tuple(_nonempty(x) for x in choices_raw)
    answer_raw = row.get("answer")
    if not question or not choices or answer_raw is None:
        raise ValueError(f"Invalid MMLU row at index {idx}")
    labels = tuple(chr(ord("A") + i) for i in range(len(choices)))
    if isinstance(answer_raw, (int, np.integer)):
        correct_idx = int(answer_raw)
    else:
        answer = _nonempty(answer_raw).upper()
        if answer in labels:
            correct_idx = int(labels.index(answer))
        elif answer.isdigit():
            correct_idx = int(answer)
        else:
            raise ValueError(f"Unsupported MMLU answer value: {answer_raw!r}")
    if correct_idx < 0 or correct_idx >= len(choices):
        raise ValueError(f"MMLU answer index out of range: {correct_idx}")
    prompt = _make_mcq_prompt(
        stem=f"Question:\n{question}",
        choices=list(zip(labels, choices, strict=True)),
        instruction="Select the single best answer.",
    )
    return RewardBenchmarkExample(
        benchmark="mmlu",
        example_id=_example_id("mmlu", row, idx),
        prompt=prompt,
        responses=choices,
        choice_labels=labels,
        choice_texts=choices,
        correct_idx=correct_idx,
        group=_nonempty(row.get("subject")) or None,
    )


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "arc_easy": BenchmarkSpec(
        name="arc_easy",
        dataset_id="ai2_arc",
        config_name="ARC-Easy",
        default_split="validation",
        description="Elementary science QA with fixed answer choices.",
        normalizer=lambda row, idx: _normalize_arc(row, idx, benchmark="arc_easy"),
    ),
    "arc_challenge": BenchmarkSpec(
        name="arc_challenge",
        dataset_id="ai2_arc",
        config_name="ARC-Challenge",
        default_split="validation",
        description="Harder science QA with fixed answer choices.",
        normalizer=lambda row, idx: _normalize_arc(row, idx, benchmark="arc_challenge"),
    ),
    "boolq": BenchmarkSpec(
        name="boolq",
        dataset_id="google/boolq",
        config_name=None,
        default_split="validation",
        description="Passage-grounded yes/no QA.",
        normalizer=_normalize_boolq,
    ),
    "hellaswag": BenchmarkSpec(
        name="hellaswag",
        dataset_id="hellaswag",
        config_name=None,
        default_split="validation",
        description="Common-sense continuation selection.",
        normalizer=_normalize_hellaswag,
    ),
    "mmlu": BenchmarkSpec(
        name="mmlu",
        dataset_id="cais/mmlu",
        config_name="all",
        default_split="validation",
        description="Broad academic multiple-choice knowledge benchmark.",
        normalizer=_normalize_mmlu,
    ),
    "piqa": BenchmarkSpec(
        name="piqa",
        dataset_id="piqa",
        config_name=None,
        default_split="validation",
        description="Physical commonsense solution selection.",
        normalizer=_normalize_piqa,
    ),
    "social_iqa": BenchmarkSpec(
        name="social_iqa",
        dataset_id="social_i_qa",
        config_name=None,
        default_split="validation",
        description="Social commonsense QA with three-way choices.",
        normalizer=_normalize_social_iqa,
    ),
    "winogrande": BenchmarkSpec(
        name="winogrande",
        dataset_id="winogrande",
        config_name="winogrande_xl",
        default_split="validation",
        description="Pronoun resolution / coreference-style choice task.",
        normalizer=_normalize_winogrande,
    ),
}


__all__ = [
    "BENCHMARKS",
    "BenchmarkSpec",
    "RewardBenchmarkExample",
    "RunSpec",
    "available_benchmarks",
    "benchmark_descriptions",
    "compute_mcq_metrics",
    "load_benchmark_examples",
    "make_mcq_record",
    "parse_run_spec",
]
