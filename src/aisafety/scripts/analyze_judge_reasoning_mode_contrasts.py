"""Analyze direct-versus-deliberative judge behavior from completed trace rows."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text
from aisafety.mech.d4_io import read_json, read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.mech.judge_reasoning import normalize_choice
from aisafety.scripts.analyze_judge_reasoning_trajectories import (
    decision_dynamics_rows,
    summarize_decision_dynamics,
)


DEFAULT_TRACE_DIR = (
    Path("artifacts") / "mechanistic" / "judge_reasoning_trajectories_v1"
)
DEFAULT_OUT_DIR = (
    Path("artifacts") / "mechanistic" / "judge_reasoning_mode_contrasts_v1"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--direct-mode", default="direct")
    parser.add_argument("--deliberative-mode", default="thinking")
    parser.add_argument("--trajectory-analysis-dir", type=Path, default=None)
    parser.add_argument("--confidence-thresholds", default="0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--commitment-persistence", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _metadata_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _selected_text_hash(row: pd.Series) -> str:
    choice = normalize_choice(row.get("final_choice"))
    if choice == "A":
        text = flat_text(str(row.get("option_a_text") or ""))
    elif choice == "B":
        text = flat_text(str(row.get("option_b_text") or ""))
    else:
        return ""
    return sha1_hex(text) if text else ""


def _mode_budget(manifest: dict[str, Any], mode: str) -> int:
    if mode == "direct":
        return int(manifest.get("max_new_tokens_direct") or 0)
    return int(manifest.get("max_new_tokens_thinking") or 0)


def prepare_traces(
    rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        raise ValueError("Trace artifact contains no rows.")
    for column in (
        "validity_type",
        "difficulty_tier",
        "analysis_split",
        "original_source_dataset",
    ):
        if column not in frame.columns:
            metadata = frame.get("metadata", pd.Series([{}] * len(frame)))
            frame[column] = [_metadata_value(value, column) for value in metadata]
    frame["valid_choice"] = frame["valid_choice"].eq(True)
    frame["final_choice"] = frame["final_choice"].map(normalize_choice)
    frame["target_option"] = frame["target_option"].map(normalize_choice)
    frame["scored"] = frame["target_option"].isin(["A", "B"])
    frame["target_success"] = (
        frame["valid_choice"]
        & frame["scored"]
        & frame["final_choice"].eq(frame["target_option"])
    )
    frame["selected_text_hash"] = frame.apply(_selected_text_hash, axis=1)
    budgets = {
        mode: _mode_budget(manifest, mode)
        for mode in frame["reasoning_mode"].dropna().astype(str).unique()
    }
    frame["token_budget"] = frame["reasoning_mode"].map(budgets).fillna(0).astype(int)
    frame["budget_saturated"] = (
        frame["token_budget"].gt(0)
        & frame["generated_tokens"].astype(int).ge(frame["token_budget"])
    )
    frame["invalid_reason"] = np.where(
        frame["valid_choice"],
        "valid",
        np.where(frame["budget_saturated"], "budget_saturated", "no_verdict_before_eos"),
    )
    return frame


def _binary_entropy(n_a: int, n_b: int) -> float:
    total = int(n_a) + int(n_b)
    if total <= 0:
        return float("nan")
    values = [count / total for count in (n_a, n_b) if count > 0]
    return float(-sum(value * math.log2(value) for value in values))


def branch_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "reasoning_mode",
        "source_dataset",
        "pair_id",
        "comparison_id",
        "presentation_order",
        "target_option",
        "target_kind",
        "scored",
    ]
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        valid = group[group["valid_choice"]]
        counts = Counter(valid["final_choice"].astype(str))
        n_a = int(counts.get("A", 0))
        n_b = int(counts.get("B", 0))
        majority = ""
        if n_a > n_b:
            majority = "A"
        elif n_b > n_a:
            majority = "B"
        selected_hash = ""
        if majority:
            hashes = valid.loc[
                valid["final_choice"].eq(majority),
                "selected_text_hash",
            ]
            if not hashes.empty:
                selected_hash = str(hashes.mode().iloc[0])
        n_valid = int(len(valid))
        majority_count = max(n_a, n_b)
        rows.append(
            {
                "reasoning_mode": keys[0],
                "source_dataset": keys[1],
                "pair_id": keys[2],
                "comparison_id": keys[3],
                "presentation_order": keys[4],
                "target_option": keys[5],
                "target_kind": keys[6],
                "scored": bool(keys[7]),
                "n_branches": int(len(group)),
                "n_valid_branches": n_valid,
                "valid_rate": float(group["valid_choice"].mean()),
                "budget_saturation_rate": float(group["budget_saturated"].mean()),
                "mean_generated_tokens": float(group["generated_tokens"].mean()),
                "n_choice_a": n_a,
                "n_choice_b": n_b,
                "majority_choice": majority,
                "majority_selected_text_hash": selected_hash,
                "branch_agreement": (
                    float(majority_count / n_valid) if n_valid else float("nan")
                ),
                "branch_entropy": _binary_entropy(n_a, n_b),
                "majority_target_selected": (
                    float(majority == keys[5])
                    if bool(keys[7]) and majority
                    else float("nan")
                ),
                "trace_target_success_rate": (
                    float(group["target_success"].mean())
                    if bool(keys[7])
                    else float("nan")
                ),
                "valid_target_success_rate": (
                    float(valid["target_success"].mean())
                    if bool(keys[7]) and n_valid
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def order_invariance_rows(branches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in branches.groupby(
        ["reasoning_mode", "source_dataset", "pair_id"],
        sort=True,
        dropna=False,
    ):
        valid_majorities = group[
            group["majority_selected_text_hash"].fillna("").astype(str).ne("")
        ]
        hashes = valid_majorities["majority_selected_text_hash"].astype(str).tolist()
        rows.append(
            {
                "reasoning_mode": keys[0],
                "source_dataset": keys[1],
                "pair_id": keys[2],
                "n_orders": int(group["comparison_id"].nunique()),
                "n_orders_with_majority": int(len(valid_majorities)),
                "order_consistent_majority": (
                    float(len(set(hashes)) == 1) if len(hashes) >= 2 else float("nan")
                ),
                "mean_branch_agreement": float(group["branch_agreement"].mean()),
                "mean_branch_entropy": float(group["branch_entropy"].mean()),
                "mean_valid_rate": float(group["valid_rate"].mean()),
                "mean_budget_saturation_rate": float(
                    group["budget_saturation_rate"].mean()
                ),
                "mean_trace_target_success_rate": float(
                    group["trace_target_success_rate"].mean()
                )
                if group["scored"].any()
                else float("nan"),
                "mean_valid_target_success_rate": float(
                    group["valid_target_success_rate"].mean()
                )
                if group["scored"].any()
                else float("nan"),
                "mean_majority_target_selected": float(
                    group["majority_target_selected"].mean()
                )
                if group["scored"].any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def matched_trace_rows(
    frame: pd.DataFrame,
    *,
    direct_mode: str,
    deliberative_mode: str,
) -> pd.DataFrame:
    keys = ["source_dataset", "pair_id", "comparison_id", "branch_index"]
    columns = keys + [
        "valid_choice",
        "budget_saturated",
        "generated_tokens",
        "scored",
        "target_success",
        "selected_text_hash",
        "final_choice",
    ]
    direct = frame[frame["reasoning_mode"].eq(direct_mode)][columns].copy()
    deliberative = frame[frame["reasoning_mode"].eq(deliberative_mode)][columns].copy()
    merged = direct.merge(
        deliberative,
        on=keys,
        suffixes=("_direct", "_deliberative"),
        validate="one_to_one",
    )
    merged["both_valid"] = (
        merged["valid_choice_direct"] & merged["valid_choice_deliberative"]
    )
    merged["valid_to_invalid"] = (
        merged["valid_choice_direct"] & ~merged["valid_choice_deliberative"]
    )
    merged["invalid_to_valid"] = (
        ~merged["valid_choice_direct"] & merged["valid_choice_deliberative"]
    )
    merged["semantic_choice_changed"] = (
        merged["both_valid"]
        & merged["selected_text_hash_direct"].ne(
            merged["selected_text_hash_deliberative"]
        )
    )
    merged["wrong_to_correct"] = (
        merged["scored_direct"]
        & merged["both_valid"]
        & ~merged["target_success_direct"]
        & merged["target_success_deliberative"]
    )
    merged["correct_to_wrong"] = (
        merged["scored_direct"]
        & merged["both_valid"]
        & merged["target_success_direct"]
        & ~merged["target_success_deliberative"]
    )
    merged["stable_correct"] = (
        merged["scored_direct"]
        & merged["both_valid"]
        & merged["target_success_direct"]
        & merged["target_success_deliberative"]
    )
    merged["stable_wrong"] = (
        merged["scored_direct"]
        & merged["both_valid"]
        & ~merged["target_success_direct"]
        & ~merged["target_success_deliberative"]
    )
    merged["generated_token_delta"] = (
        merged["generated_tokens_deliberative"] - merged["generated_tokens_direct"]
    )
    return merged


def matched_comparison_rows(
    branches: pd.DataFrame,
    *,
    direct_mode: str,
    deliberative_mode: str,
) -> pd.DataFrame:
    keys = ["source_dataset", "pair_id", "comparison_id"]
    columns = keys + [
        "scored",
        "majority_choice",
        "majority_selected_text_hash",
        "majority_target_selected",
        "valid_rate",
        "budget_saturation_rate",
        "branch_agreement",
        "branch_entropy",
    ]
    direct = branches[branches["reasoning_mode"].eq(direct_mode)][columns].copy()
    deliberative = branches[
        branches["reasoning_mode"].eq(deliberative_mode)
    ][columns].copy()
    merged = direct.merge(
        deliberative,
        on=keys,
        suffixes=("_direct", "_deliberative"),
        validate="one_to_one",
    )
    direct_valid = merged["majority_choice_direct"].fillna("").astype(str).ne("")
    deliberative_valid = (
        merged["majority_choice_deliberative"].fillna("").astype(str).ne("")
    )
    merged["both_majorities_valid"] = direct_valid & deliberative_valid
    merged["majority_valid_to_invalid"] = direct_valid & ~deliberative_valid
    merged["majority_invalid_to_valid"] = ~direct_valid & deliberative_valid
    merged["majority_semantic_choice_changed"] = (
        merged["both_majorities_valid"]
        & merged["majority_selected_text_hash_direct"].ne(
            merged["majority_selected_text_hash_deliberative"]
        )
    )
    direct_correct = merged["majority_target_selected_direct"].eq(1.0)
    deliberative_correct = merged["majority_target_selected_deliberative"].eq(1.0)
    scored = merged["scored_direct"].eq(True)
    merged["majority_wrong_to_correct"] = (
        scored & merged["both_majorities_valid"] & ~direct_correct & deliberative_correct
    )
    merged["majority_correct_to_wrong"] = (
        scored & merged["both_majorities_valid"] & direct_correct & ~deliberative_correct
    )
    merged["majority_stable_correct"] = (
        scored & merged["both_majorities_valid"] & direct_correct & deliberative_correct
    )
    merged["majority_stable_wrong"] = (
        scored & merged["both_majorities_valid"] & ~direct_correct & ~deliberative_correct
    )
    return merged


def pair_mode_effects(
    pair_modes: pd.DataFrame,
    *,
    direct_mode: str,
    deliberative_mode: str,
) -> pd.DataFrame:
    index = ["source_dataset", "pair_id"]
    value_columns = [
        "mean_valid_rate",
        "mean_budget_saturation_rate",
        "mean_branch_agreement",
        "mean_branch_entropy",
        "order_consistent_majority",
        "mean_trace_target_success_rate",
        "mean_valid_target_success_rate",
        "mean_majority_target_selected",
    ]
    wide = pair_modes.pivot(index=index, columns="reasoning_mode", values=value_columns)
    rows: list[dict[str, Any]] = []
    for source_dataset, pair_id in wide.index:
        row: dict[str, Any] = {
            "source_dataset": source_dataset,
            "pair_id": pair_id,
        }
        for metric in value_columns:
            direct = (
                wide.loc[(source_dataset, pair_id), (metric, direct_mode)]
                if (metric, direct_mode) in wide.columns
                else float("nan")
            )
            deliberative = (
                wide.loc[(source_dataset, pair_id), (metric, deliberative_mode)]
                if (metric, deliberative_mode) in wide.columns
                else float("nan")
            )
            row[f"{direct_mode}_{metric}"] = direct
            row[f"{deliberative_mode}_{metric}"] = deliberative
            row[f"delta_{metric}"] = deliberative - direct
        direct_agreement = row[f"{direct_mode}_mean_branch_agreement"]
        if pd.isna(direct_agreement):
            difficulty = "missing"
        elif float(direct_agreement) >= 0.999:
            difficulty = "unanimous"
        elif float(direct_agreement) >= 0.75:
            difficulty = "low_disagreement"
        else:
            difficulty = "high_disagreement"
        row["direct_disagreement_tier"] = difficulty
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def effect_summary(effects: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        column
        for column in effects.columns
        if column.startswith("delta_")
    ]
    rows: list[dict[str, Any]] = []
    for source, group in effects.groupby("source_dataset", sort=True):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "source_dataset": source,
                    "metric": metric,
                    "n_pairs": int(len(values)),
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "median": float(values.median()) if len(values) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def difficulty_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in effects.groupby(
        ["source_dataset", "direct_disagreement_tier"],
        sort=True,
        dropna=False,
    ):
        rows.append(
            {
                "source_dataset": keys[0],
                "direct_disagreement_tier": keys[1],
                "n_pairs": int(group["pair_id"].nunique()),
                "thinking_valid_rate": _summary_mean(
                    group,
                    "thinking_mean_valid_rate",
                ),
                "thinking_budget_saturation_rate": _summary_mean(
                    group,
                    "thinking_mean_budget_saturation_rate",
                ),
                "delta_target_success": _summary_mean(
                    group,
                    "delta_mean_trace_target_success_rate",
                ),
                "delta_branch_agreement": _summary_mean(
                    group,
                    "delta_mean_branch_agreement",
                ),
                "delta_branch_entropy": _summary_mean(
                    group,
                    "delta_mean_branch_entropy",
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_effects(
    effects: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    metrics = [
        column
        for column in effects.columns
        if column.startswith("delta_")
    ]
    rows: list[dict[str, Any]] = []
    groups = [("all", effects)]
    groups.extend(
        (str(source), group)
        for source, group in effects.groupby("source_dataset", sort=True)
    )
    for source, group in groups:
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if not len(values):
                continue
            boot = np.empty((int(n_bootstrap),), dtype=float)
            for index in range(int(n_bootstrap)):
                sample = rng.choice(values, size=len(values), replace=True)
                boot[index] = float(np.mean(sample))
            rows.append(
                {
                    "source_dataset": source,
                    "metric": metric,
                    "n_pairs": int(len(values)),
                    "mean": float(np.mean(values)),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "bootstrap_samples": int(n_bootstrap),
                }
            )
    return pd.DataFrame(rows)


def trace_outcome_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["reasoning_mode", "source_dataset"],
        sort=True,
        dropna=False,
    ):
        valid = group[group["valid_choice"]]
        scored = group[group["scored"]]
        valid_scored = scored[scored["valid_choice"]]
        rows.append(
            {
                "reasoning_mode": keys[0],
                "source_dataset": keys[1],
                "n_pairs": int(group["pair_id"].nunique()),
                "n_comparisons": int(group["comparison_id"].nunique()),
                "n_traces": int(len(group)),
                "valid_rate": float(group["valid_choice"].mean()),
                "budget_saturation_rate": float(group["budget_saturated"].mean()),
                "invalid_budget_saturation_share": float(
                    group.loc[~group["valid_choice"], "budget_saturated"].mean()
                )
                if (~group["valid_choice"]).any()
                else float("nan"),
                "mean_generated_tokens": float(group["generated_tokens"].mean()),
                "unconditional_target_success_rate": float(
                    scored["target_success"].mean()
                )
                if len(scored)
                else float("nan"),
                "valid_target_success_rate": float(
                    valid_scored["target_success"].mean()
                )
                if len(valid_scored)
                else float("nan"),
                "choice_a_rate_valid": float(valid["final_choice"].eq("A").mean())
                if len(valid)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def matched_summary(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, group in matched.groupby("source_dataset", sort=True):
        scored = group[group["scored_direct"]]
        both_valid_scored = scored[scored["both_valid"]]
        rows.append(
            {
                "source_dataset": source,
                "n_pairs": int(group["pair_id"].nunique()),
                "n_matched_traces": int(len(group)),
                "both_valid_rate": float(group["both_valid"].mean()),
                "valid_to_invalid_rate": float(group["valid_to_invalid"].mean()),
                "invalid_to_valid_rate": float(group["invalid_to_valid"].mean()),
                "semantic_choice_change_rate_both_valid": float(
                    group.loc[group["both_valid"], "semantic_choice_changed"].mean()
                )
                if group["both_valid"].any()
                else float("nan"),
                "wrong_to_correct_rate_both_valid_scored": float(
                    both_valid_scored["wrong_to_correct"].mean()
                )
                if len(both_valid_scored)
                else float("nan"),
                "correct_to_wrong_rate_both_valid_scored": float(
                    both_valid_scored["correct_to_wrong"].mean()
                )
                if len(both_valid_scored)
                else float("nan"),
                "net_correction_rate_both_valid_scored": float(
                    (
                        both_valid_scored["wrong_to_correct"].astype(int)
                        - both_valid_scored["correct_to_wrong"].astype(int)
                    ).mean()
                )
                if len(both_valid_scored)
                else float("nan"),
                "mean_generated_token_delta": float(
                    group["generated_token_delta"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def matched_comparison_summary(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, group in matched.groupby("source_dataset", sort=True):
        scored = group[group["scored_direct"].eq(True)]
        both_valid_scored = scored[scored["both_majorities_valid"]]
        rows.append(
            {
                "source_dataset": source,
                "n_pairs": int(group["pair_id"].nunique()),
                "n_ordered_comparisons": int(len(group)),
                "both_majorities_valid_rate": float(
                    group["both_majorities_valid"].mean()
                ),
                "majority_valid_to_invalid_rate": float(
                    group["majority_valid_to_invalid"].mean()
                ),
                "majority_invalid_to_valid_rate": float(
                    group["majority_invalid_to_valid"].mean()
                ),
                "majority_choice_change_rate": float(
                    group.loc[
                        group["both_majorities_valid"],
                        "majority_semantic_choice_changed",
                    ].mean()
                )
                if group["both_majorities_valid"].any()
                else float("nan"),
                "majority_wrong_to_correct_rate": float(
                    both_valid_scored["majority_wrong_to_correct"].mean()
                )
                if len(both_valid_scored)
                else float("nan"),
                "majority_correct_to_wrong_rate": float(
                    both_valid_scored["majority_correct_to_wrong"].mean()
                )
                if len(both_valid_scored)
                else float("nan"),
                "majority_net_correction_rate": float(
                    (
                        both_valid_scored["majority_wrong_to_correct"].astype(int)
                        - both_valid_scored["majority_correct_to_wrong"].astype(int)
                    ).mean()
                )
                if len(both_valid_scored)
                else float("nan"),
                "delta_branch_agreement": float(
                    (
                        group["branch_agreement_deliberative"]
                        - group["branch_agreement_direct"]
                    ).mean()
                ),
                "delta_branch_entropy": float(
                    (
                        group["branch_entropy_deliberative"]
                        - group["branch_entropy_direct"]
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def commitment_threshold_sweep(
    predictions: pd.DataFrame,
    *,
    thresholds: list[float],
    persistence: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for threshold in thresholds:
        dynamics = decision_dynamics_rows(
            predictions,
            choice_confidence_threshold=float(threshold),
            target_confidence_threshold=float(threshold),
            persistence=int(persistence),
        )
        summary = summarize_decision_dynamics(dynamics)
        if summary.empty:
            continue
        summary.insert(0, "confidence_threshold", float(threshold))
        summary.insert(1, "commitment_persistence", int(persistence))
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def invalidity_examples(frame: pd.DataFrame, *, limit_per_source: int = 10) -> pd.DataFrame:
    invalid = frame[~frame["valid_choice"]].copy()
    if invalid.empty:
        return invalid
    invalid["response_tail"] = invalid["response_text"].fillna("").astype(str).str[-500:]
    invalid = invalid.sort_values(
        ["source_dataset", "budget_saturated", "generated_tokens"],
        ascending=[True, False, False],
    )
    columns = [
        "reasoning_mode",
        "source_dataset",
        "pair_id",
        "comparison_id",
        "branch_index",
        "generated_tokens",
        "token_budget",
        "budget_saturated",
        "invalid_reason",
        "response_tail",
    ]
    return (
        invalid.groupby("source_dataset", sort=True, group_keys=False)
        .head(int(limit_per_source))[columns]
        .reset_index(drop=True)
    )


def analyze_mode_contrasts(
    rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    direct_mode: str,
    deliberative_mode: str,
    bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    frame = prepare_traces(rows, manifest=manifest)
    available = set(frame["reasoning_mode"].astype(str))
    missing = sorted({direct_mode, deliberative_mode} - available)
    if missing:
        raise ValueError(f"Trace artifact lacks required reasoning modes: {missing}")
    branches = branch_summaries(frame)
    pair_modes = order_invariance_rows(branches)
    matched_traces = matched_trace_rows(
        frame,
        direct_mode=direct_mode,
        deliberative_mode=deliberative_mode,
    )
    matched_comparisons = matched_comparison_rows(
        branches,
        direct_mode=direct_mode,
        deliberative_mode=deliberative_mode,
    )
    effects = pair_mode_effects(
        pair_modes,
        direct_mode=direct_mode,
        deliberative_mode=deliberative_mode,
    )
    return {
        "trace_outcomes": trace_outcome_summary(frame),
        "invalidity_counts": (
            frame.groupby(
                ["reasoning_mode", "source_dataset", "invalid_reason"],
                sort=True,
            )
            .size()
            .rename("n_traces")
            .reset_index()
        ),
        "invalidity_examples": invalidity_examples(frame),
        "branch_summaries": branches,
        "order_invariance": pair_modes,
        "branch_index_matched_transitions": matched_traces,
        "branch_index_matched_summary": matched_summary(matched_traces),
        "matched_comparison_transitions": matched_comparisons,
        "matched_comparison_summary": matched_comparison_summary(
            matched_comparisons
        ),
        "pair_mode_effects": effects,
        "mode_effect_summary": effect_summary(effects),
        "difficulty_conditioned_summary": difficulty_summary(effects),
        "bootstrap_mode_effects": bootstrap_effects(
            effects,
            n_bootstrap=int(bootstrap),
            seed=int(seed),
        ),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    trace_dir = _resolve(workspace_root, args.trace_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    manifest = read_json(trace_dir / "manifest.json")
    rows = read_jsonl(trace_dir / "traces.jsonl")
    outputs = analyze_mode_contrasts(
        rows,
        manifest=manifest,
        direct_mode=str(args.direct_mode),
        deliberative_mode=str(args.deliberative_mode),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    if args.trajectory_analysis_dir is not None:
        analysis_dir = _resolve(workspace_root, args.trajectory_analysis_dir)
        predictions_path = analysis_dir / "probe_oof_predictions.csv"
        if not predictions_path.is_file():
            raise FileNotFoundError(predictions_path)
        thresholds = [
            float(value.strip())
            for value in str(args.confidence_thresholds).split(",")
            if value.strip()
        ]
        outputs["commitment_threshold_sweep"] = commitment_threshold_sweep(
            pd.read_csv(predictions_path),
            thresholds=thresholds,
            persistence=int(args.commitment_persistence),
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = str(path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-reasoning-mode-contrast-analysis",
            "trace_dir": str(trace_dir),
            "out_dir": str(out_dir),
            "direct_mode": str(args.direct_mode),
            "deliberative_mode": str(args.deliberative_mode),
            "trajectory_analysis_dir": (
                None
                if args.trajectory_analysis_dir is None
                else str(_resolve(workspace_root, args.trajectory_analysis_dir))
            ),
            "confidence_thresholds": str(args.confidence_thresholds),
            "commitment_persistence": int(args.commitment_persistence),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "n_traces": int(len(rows)),
            "n_pairs": int(len({str(row.get("pair_id") or "") for row in rows})),
            "outputs": output_paths,
        },
    )
    print(f"out_dir={out_dir}")
    print(f"n_traces={len(rows)}")
    print(f"n_pairs={len({str(row.get('pair_id') or '') for row in rows})}")
    print("\n=== Trace outcomes ===")
    print(outputs["trace_outcomes"].to_string(index=False))
    print("\n=== Majority direct-to-deliberative transitions ===")
    print(outputs["matched_comparison_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
