"""Measure structured-CoT adherence and its association with criterion use."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, write_json
from aisafety.scripts.analyze_judge_criterion_confirmation import (
    _bootstrap,
    _checkpoint_frame,
    _seed_offset,
)


SCAFFOLD_CONDITIONS = (
    "free_cot",
    "generic_scaffold",
    "criterion_scaffold",
)
CRITERION_MARKERS = {
    "coherence": (
        "coherence",
        "coherent",
        "clarity",
        "clear",
        "internally consistent",
        "organization",
    ),
    "correctness": (
        "correct",
        "incorrect",
        "factual",
        "factually",
        "logical",
        "logic",
        "error",
    ),
    "helpfulness": (
        "helpful",
        "helpfulness",
        "useful",
        "satisfy",
        "request",
        "actionable",
        "relevant",
    ),
    "overall": (
        "overall",
        "better response",
        "quality",
        "strength",
        "tradeoff",
        "trade-off",
    ),
    "weighted": (
        "weighted",
        "40%",
        "30%",
        "20%",
        "complexity",
        "verbosity",
    ),
}
OPERATIONAL_MARKERS = (
    "criterion",
    "test",
    "assess",
    "evaluate",
    "standard",
    "requirement",
    "check whether",
    "specifically",
)
COMPARISON_MARKERS = (
    "compare",
    "compared",
    "better",
    "worse",
    "stronger",
    "weaker",
    "tie",
    "tied",
    "equal",
    "underdetermined",
    "neither",
)
GENERIC_SUMMARY_MARKERS = (
    "summary",
    "summarize",
    "approach",
    "main point",
)
GENERIC_COMPARISON_MARKERS = (
    "similar",
    "similarity",
    "difference",
    "differ",
    "organization",
    "presentation",
)
GENERIC_CHECK_MARKERS = (
    "omitted",
    "omission",
    "unsupported",
    "assumption",
    "context",
)
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "before",
    "being",
    "between",
    "both",
    "could",
    "does",
    "each",
    "from",
    "have",
    "into",
    "more",
    "most",
    "option",
    "other",
    "response",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "using",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}
ADHERENCE_METRICS = (
    "criterion_procedure_score",
    "explicit_step_fraction",
    "content_compliant",
    "format_compliant",
    "strict_compliant",
    "generic_content_compliant",
    "phase1_response_words",
    "phase1_generated_tokens",
    "phase1_budget_saturated",
)
OUTCOME_METRICS = (
    "forced_target_adoption",
    "order_consistent_target_adoption",
    "order_consistent_rate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase1-budget", type=int, default=128)
    parser.add_argument("--endpoint-budget", type=int, default=384)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--audit-sample", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/judge_structured_cot_adherence_v1"
        ),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9'-]{3,}", text.lower())
        if token not in STOPWORDS
    }


def _grounding_overlap(
    response: str,
    option_text: str,
    other_option_text: str,
) -> int:
    option_tokens = _tokens(option_text)
    distinctive = option_tokens - _tokens(other_option_text)
    candidates = distinctive if len(distinctive) >= 2 else option_tokens
    return len(_tokens(response) & candidates)


def _step_present(text: str, step: int) -> bool:
    pattern = (
        rf"(?:^|\n)\s*(?:\*\*|__)?(?:step\s*)?{step}"
        rf"(?:\s*[:.)\]-]|\s+)"
    )
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def score_adherence(row: dict[str, Any]) -> dict[str, Any]:
    response = str(row.get("phase1_response_text") or "")
    lower = response.lower()
    criterion_id = str(row.get("phase1_criterion_id") or "")
    criterion_markers = CRITERION_MARKERS.get(criterion_id, ())
    steps = [_step_present(response, step) for step in range(1, 5)]
    option_a = bool(
        re.search(r"\b(?:option|response|answer)\s*a\b", lower)
    )
    option_b = bool(
        re.search(r"\b(?:option|response|answer)\s*b\b", lower)
    )
    overlap_a = _grounding_overlap(
        response,
        str(row.get("option_a_text") or ""),
        str(row.get("option_b_text") or ""),
    )
    overlap_b = _grounding_overlap(
        response,
        str(row.get("option_b_text") or ""),
        str(row.get("option_a_text") or ""),
    )
    criterion_reference = _contains_any(lower, criterion_markers)
    operational_language = _contains_any(lower, OPERATIONAL_MARKERS)
    comparison = _contains_any(lower, COMPARISON_MARKERS)
    no_premature_final = not bool(
        re.search(r"\bfinal\s*:\s*[abc]\b", lower)
    )
    option_a_grounded = option_a and overlap_a >= 1
    option_b_grounded = option_b and overlap_b >= 1
    components = (
        criterion_reference,
        operational_language,
        option_a,
        option_b,
        option_a_grounded,
        option_b_grounded,
        comparison,
        no_premature_final,
    )
    content_compliant = all(components)
    format_compliant = sum(steps) >= 3
    generic_content_compliant = all(
        (
            option_a,
            option_b,
            _contains_any(lower, GENERIC_SUMMARY_MARKERS),
            _contains_any(lower, GENERIC_COMPARISON_MARKERS),
            _contains_any(lower, GENERIC_CHECK_MARKERS),
            no_premature_final,
        )
    )
    return {
        "phase1_response_chars": len(response),
        "phase1_response_words": len(response.split()),
        "step_1_present": steps[0],
        "step_2_present": steps[1],
        "step_3_present": steps[2],
        "step_4_present": steps[3],
        "explicit_step_count": int(sum(steps)),
        "explicit_step_fraction": float(sum(steps) / 4.0),
        "criterion_reference": criterion_reference,
        "operational_language": operational_language,
        "option_a_application": option_a,
        "option_b_application": option_b,
        "option_a_overlap_tokens": overlap_a,
        "option_b_overlap_tokens": overlap_b,
        "option_a_grounded": option_a_grounded,
        "option_b_grounded": option_b_grounded,
        "criterion_comparison": comparison,
        "no_premature_final": no_premature_final,
        "criterion_procedure_score": float(np.mean(components)),
        "content_compliant": content_compliant,
        "format_compliant": format_compliant,
        "strict_compliant": bool(content_compliant and format_compliant),
        "generic_content_compliant": generic_content_compliant,
    }


def adherence_rows(
    traces: list[dict[str, Any]],
    *,
    phase1_budget: int,
    endpoint_budget: int,
) -> pd.DataFrame:
    selected = [
        row
        for row in traces
        if str(row.get("condition_id") or "") in SCAFFOLD_CONDITIONS
    ]
    if not selected:
        return pd.DataFrame()
    trace_frame = pd.DataFrame(selected)
    scored = pd.DataFrame([score_adherence(row) for row in selected])
    base_columns = [
        "trace_id",
        "pair_id",
        "condition_id",
        "transition_type",
        "presentation_order",
        "branch_index",
        "phase1_criterion_id",
        "phase1_target_semantic",
        "phase2_target_semantic",
        "option_a_text",
        "option_b_text",
        "phase1_response_text",
        "phase1_generated_tokens",
    ]
    output = pd.concat(
        [
            trace_frame[base_columns].reset_index(drop=True),
            scored.reset_index(drop=True),
        ],
        axis=1,
    )
    checkpoints = _checkpoint_frame(selected)
    endpoint = checkpoints[
        checkpoints["stage"].eq("phase2")
        & checkpoints["budget_tokens"].eq(int(endpoint_budget))
    ][
        [
            "trace_id",
            "forced_choice_semantic",
            "forced_choice_confidence",
        ]
    ]
    output = output.merge(endpoint, on="trace_id", how="left")
    output["phase1_budget_saturated"] = (
        pd.to_numeric(output["phase1_generated_tokens"], errors="coerce")
        >= int(phase1_budget)
    )
    output["forced_target_adoption"] = output[
        "forced_choice_semantic"
    ].eq(output["phase2_target_semantic"])
    return output


def branch_pair_rows(rows: pd.DataFrame) -> pd.DataFrame:
    values: list[dict[str, Any]] = []
    for keys, group in rows.groupby(
        ["pair_id", "condition_id", "transition_type", "branch_index"],
        sort=True,
    ):
        by_order = group.set_index("presentation_order")
        if not {"original", "swapped"}.issubset(by_order.index):
            continue
        original = str(by_order.loc["original", "forced_choice_semantic"])
        swapped = str(by_order.loc["swapped", "forced_choice_semantic"])
        target = str(by_order.loc["original", "phase2_target_semantic"])
        consistent = bool(original and original == swapped)
        values.append(
            {
                "pair_id": keys[0],
                "condition_id": keys[1],
                "transition_type": keys[2],
                "branch_index": int(keys[3]),
                "mean_criterion_procedure_score": float(
                    group["criterion_procedure_score"].mean()
                ),
                "min_criterion_procedure_score": float(
                    group["criterion_procedure_score"].min()
                ),
                "both_orders_content_compliant": float(
                    group["content_compliant"].astype(bool).all()
                ),
                "both_orders_strict_compliant": float(
                    group["strict_compliant"].astype(bool).all()
                ),
                "forced_target_adoption": float(
                    group["forced_target_adoption"].mean()
                ),
                "order_consistent_rate": float(consistent),
                "order_consistent_target_adoption": float(
                    consistent and original == target
                ),
            }
        )
    return pd.DataFrame(values)


def _condition_summary(
    rows: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    pair_means = (
        rows.groupby(
            ["pair_id", "condition_id", "transition_type"],
            sort=True,
        )
        .agg(
            **{
                metric: (metric, "mean")
                for metric in ADHERENCE_METRICS
            }
        )
        .reset_index()
    )
    output: list[dict[str, Any]] = []
    for condition, group in pair_means.groupby("condition_id", sort=True):
        for metric in ADHERENCE_METRICS:
            mean, low, high = _bootstrap(
                group[metric].to_numpy(dtype=float),
                n_bootstrap=int(bootstrap),
                seed=int(seed) + _seed_offset(condition, metric),
            )
            output.append(
                {
                    "condition_id": condition,
                    "metric": metric,
                    "n_pairs": int(group["pair_id"].nunique()),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(output)


def _paired_condition_effects(
    rows: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    pair_means = (
        rows.groupby(["pair_id", "condition_id"], sort=True)[
            list(ADHERENCE_METRICS)
        ]
        .mean()
        .reset_index()
    )
    contrasts = {
        "criterion_scaffold_minus_free": (
            "criterion_scaffold",
            "free_cot",
        ),
        "criterion_scaffold_minus_generic": (
            "criterion_scaffold",
            "generic_scaffold",
        ),
    }
    output: list[dict[str, Any]] = []
    for contrast, (left, right) in contrasts.items():
        selected = pair_means[
            pair_means["condition_id"].isin([left, right])
        ]
        for metric in ADHERENCE_METRICS:
            wide = selected.pivot(
                index="pair_id",
                columns="condition_id",
                values=metric,
            )
            if left not in wide or right not in wide:
                continue
            effects = (wide[left] - wide[right]).dropna()
            mean, low, high = _bootstrap(
                effects.to_numpy(dtype=float),
                n_bootstrap=int(bootstrap),
                seed=int(seed) + _seed_offset(contrast, metric),
            )
            output.append(
                {
                    "contrast": contrast,
                    "left_condition": left,
                    "right_condition": right,
                    "metric": metric,
                    "n_pairs": int(len(effects)),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(output)


def _bootstrap_within_pair_slope(
    frame: pd.DataFrame,
    *,
    predictor: str,
    outcome: str,
    bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    clean = frame[["pair_id", predictor, outcome]].dropna().copy()
    contributions: list[tuple[float, float]] = []
    for _, group in clean.groupby("pair_id", sort=True):
        x = group[predictor].to_numpy(dtype=float)
        y = group[outcome].to_numpy(dtype=float)
        x_residual = x - x.mean()
        y_residual = y - y.mean()
        contributions.append(
            (
                float(np.sum(x_residual * y_residual)),
                float(np.sum(np.square(x_residual))),
            )
        )
    if not contributions:
        return np.nan, np.nan, np.nan
    values = np.asarray(contributions, dtype=float)
    denominator = float(values[:, 1].sum())
    if denominator <= 0:
        return np.nan, np.nan, np.nan
    estimate = float(values[:, 0].sum() / denominator)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        len(values),
        size=(max(int(bootstrap), 1), len(values)),
    )
    numerators = values[indices, 0].sum(axis=1)
    denominators = values[indices, 1].sum(axis=1)
    samples = np.divide(
        numerators,
        denominators,
        out=np.full_like(numerators, np.nan),
        where=denominators > 0,
    )
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return estimate, np.nan, np.nan
    return (
        estimate,
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _bootstrap_group_difference(
    frame: pd.DataFrame,
    *,
    predictor: str,
    outcome: str,
    bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    clean = frame[["pair_id", predictor, outcome]].dropna().copy()
    contributions: list[tuple[float, float, float, float]] = []
    for _, group in clean.groupby("pair_id", sort=True):
        high = group[group[predictor].astype(bool)][outcome]
        low = group[~group[predictor].astype(bool)][outcome]
        contributions.append(
            (
                float(high.sum()),
                float(len(high)),
                float(low.sum()),
                float(len(low)),
            )
        )
    if not contributions:
        return np.nan, np.nan, np.nan
    values = np.asarray(contributions, dtype=float)

    def difference(sample: np.ndarray) -> np.ndarray:
        high_count = sample[..., 1].sum(axis=-1)
        low_count = sample[..., 3].sum(axis=-1)
        high_mean = np.divide(
            sample[..., 0].sum(axis=-1),
            high_count,
            out=np.full_like(high_count, np.nan),
            where=high_count > 0,
        )
        low_mean = np.divide(
            sample[..., 2].sum(axis=-1),
            low_count,
            out=np.full_like(low_count, np.nan),
            where=low_count > 0,
        )
        return high_mean - low_mean

    estimate = float(difference(values[np.newaxis, ...])[0])
    if not np.isfinite(estimate):
        return estimate, np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        len(values),
        size=(max(int(bootstrap), 1), len(values)),
    )
    samples = difference(values[indices])
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return estimate, np.nan, np.nan
    return (
        estimate,
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _outcome_associations(
    pairs: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    selected = pairs[
        pairs["condition_id"].eq("criterion_scaffold")
    ].copy()
    output: list[dict[str, Any]] = []
    specifications = (
        ("within_pair_score_slope", "mean_criterion_procedure_score"),
        (
            "both_orders_content_compliant_minus_other",
            "both_orders_content_compliant",
        ),
        (
            "both_orders_strict_compliant_minus_other",
            "both_orders_strict_compliant",
        ),
    )
    for association, predictor in specifications:
        for outcome in OUTCOME_METRICS:
            kwargs = {
                "frame": selected,
                "predictor": predictor,
                "outcome": outcome,
                "bootstrap": int(bootstrap),
                "seed": int(seed)
                + _seed_offset(association, predictor, outcome),
            }
            if association == "within_pair_score_slope":
                estimate, low, high = _bootstrap_within_pair_slope(
                    **kwargs
                )
            else:
                estimate, low, high = _bootstrap_group_difference(
                    **kwargs
                )
            output.append(
                {
                    "association": association,
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_pairs": int(selected["pair_id"].nunique()),
                    "n_pair_branches": int(len(selected)),
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(output)


def _outcome_by_compliance(
    pairs: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    selected = pairs[
        pairs["condition_id"].eq("criterion_scaffold")
    ].copy()
    output: list[dict[str, Any]] = []
    for definition in (
        "both_orders_content_compliant",
        "both_orders_strict_compliant",
    ):
        for compliant, group in selected.groupby(definition, sort=True):
            for outcome in OUTCOME_METRICS:
                pair_means = group.groupby("pair_id")[outcome].mean()
                mean, low, high = _bootstrap(
                    pair_means.to_numpy(dtype=float),
                    n_bootstrap=int(bootstrap),
                    seed=int(seed)
                    + _seed_offset(definition, compliant, outcome),
                )
                output.append(
                    {
                        "compliance_definition": definition,
                        "compliant": bool(compliant),
                        "outcome": outcome,
                        "n_pairs": int(group["pair_id"].nunique()),
                        "n_pair_branches": int(len(group)),
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(output)


def _audit_sample(
    rows: pd.DataFrame,
    *,
    size: int,
    seed: int,
) -> pd.DataFrame:
    if size <= 0 or rows.empty:
        return pd.DataFrame()
    samples: list[pd.DataFrame] = []
    groups = list(
        rows.groupby(
            ["condition_id", "strict_compliant"],
            sort=True,
            dropna=False,
        )
    )
    per_group = max(int(np.ceil(size / max(len(groups), 1))), 1)
    for (condition, compliant), group in groups:
        samples.append(
            group.sample(
                n=min(per_group, len(group)),
                random_state=(
                    int(seed) + _seed_offset(condition, compliant)
                )
                % (2**32 - 1),
            )
        )
    sample = pd.concat(samples, ignore_index=True).head(int(size)).copy()
    sample["human_content_compliant"] = ""
    sample["human_format_compliant"] = ""
    sample["human_notes"] = ""
    columns = [
        "trace_id",
        "pair_id",
        "condition_id",
        "presentation_order",
        "branch_index",
        "phase1_criterion_id",
        "criterion_procedure_score",
        "content_compliant",
        "format_compliant",
        "strict_compliant",
        "generic_content_compliant",
        "phase1_response_text",
        "human_content_compliant",
        "human_format_compliant",
        "human_notes",
    ]
    return sample[columns]


def analyze(
    *,
    run_dir: Path,
    phase1_budget: int,
    endpoint_budget: int,
    bootstrap: int,
    audit_sample: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    traces = read_jsonl(run_dir / "switch_traces.jsonl")
    if not traces:
        raise ValueError(f"No structured-CoT traces found in {run_dir}")
    rows = adherence_rows(
        traces,
        phase1_budget=int(phase1_budget),
        endpoint_budget=int(endpoint_budget),
    )
    pair_rows = branch_pair_rows(rows)
    return {
        "trace_adherence_rows": rows,
        "branch_pair_adherence_rows": pair_rows,
        "adherence_summary": _condition_summary(
            rows,
            bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "adherence_condition_effects": _paired_condition_effects(
            rows,
            bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "criterion_scaffold_outcome_by_compliance": (
            _outcome_by_compliance(
                pair_rows,
                bootstrap=int(bootstrap),
                seed=int(seed),
            )
        ),
        "criterion_scaffold_adherence_associations": (
            _outcome_associations(
                pair_rows,
                bootstrap=int(bootstrap),
                seed=int(seed),
            )
        ),
        "adherence_audit_sample": _audit_sample(
            rows,
            size=int(audit_sample),
            seed=int(seed),
        ),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    run_dir = _resolve(workspace_root, args.run_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    outputs = analyze(
        run_dir=run_dir,
        phase1_budget=int(args.phase1_budget),
        endpoint_budget=int(args.endpoint_budget),
        bootstrap=int(args.bootstrap),
        audit_sample=int(args.audit_sample),
        seed=int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-structured-cot-adherence",
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "phase1_budget": int(args.phase1_budget),
            "endpoint_budget": int(args.endpoint_budget),
            "bootstrap": int(args.bootstrap),
            "audit_sample": int(args.audit_sample),
            "seed": int(args.seed),
            "scaffold_conditions": list(SCAFFOLD_CONDITIONS),
            "adherence_metrics": list(ADHERENCE_METRICS),
            "outcome_metrics": list(OUTCOME_METRICS),
            "outputs": paths,
        },
    )
    print("\n=== SCAFFOLD ADHERENCE SUMMARY ===")
    print(outputs["adherence_summary"].round(3).to_string(index=False))
    print("\n=== SCAFFOLD MANIPULATION CHECK ===")
    print(
        outputs["adherence_condition_effects"]
        .round(3)
        .to_string(index=False)
    )
    print("\n=== CRITERION ADHERENCE BY COMPLIANCE ===")
    print(
        outputs["criterion_scaffold_outcome_by_compliance"]
        .round(3)
        .to_string(index=False)
    )
    print("\n=== PAIR-CLUSTERED ADHERENCE ASSOCIATIONS ===")
    print(
        outputs["criterion_scaffold_adherence_associations"]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nout_dir={out_dir}")
    print(
        "audit_sample="
        f"{out_dir / 'adherence_audit_sample.csv'}"
    )


if __name__ == "__main__":
    main()
