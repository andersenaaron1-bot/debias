"""Analyze matched HelpSteer2 criterion-switching budget runs."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.scripts.analyze_judge_reasoning_budget_sweep import (
    analyze_budget_sweep,
    grouped_bootstrap_summary,
)


DEFAULT_OUT_DIR = (
    Path("artifacts") / "mechanistic" / "helpsteer2_matched_criterion_analysis_v1"
)
TIE_HASH = sha1_hex("__TIE__")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Budget sweep directory; repeat for multiple disjoint shards.",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def enrich_order_rows(orders: pd.DataFrame) -> pd.DataFrame:
    frame = orders.copy()
    frame["pair_stratum"] = (
        frame["source_dataset"]
        .astype(str)
        .str.replace("helpsteer2_matched_", "", regex=False)
    )
    frame["predicted_tie"] = frame[
        "order_majority_selected_text_hash"
    ].astype(str).eq(TIE_HASH)
    frame["target_tie"] = frame["target_selected_text_hash"].astype(str).eq(
        TIE_HASH
    )
    frame["has_order_consistent_verdict"] = (
        frame["order_consistent_majority"].eq(1.0)
        & frame["order_majority_selected_text_hash"].fillna("").astype(str).ne("")
    )
    frame["robust_target_success"] = (
        frame["has_order_consistent_verdict"]
        & frame["order_majority_selected_text_hash"].astype(str).eq(
            frame["target_selected_text_hash"].astype(str)
        )
    ).astype(float)
    frame["justified_tie_success"] = np.where(
        frame["target_tie"],
        frame["robust_target_success"],
        np.nan,
    )
    frame["unjustified_tie"] = np.where(
        ~frame["target_tie"],
        (
            frame["has_order_consistent_verdict"] & frame["predicted_tie"]
        ).astype(float),
        np.nan,
    )
    return frame


def matched_summary(orders: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        "reasoning_mode",
        "pair_stratum",
        "criterion_id",
        "budget_tokens",
    ]
    for keys, group in orders.groupby(groups, sort=True, dropna=False):
        predicted_ties = group[
            group["has_order_consistent_verdict"] & group["predicted_tie"]
        ]
        target_ties = group[group["target_tie"]]
        rows.append(
            {
                **dict(zip(groups, keys, strict=True)),
                "n_pairs": int(group["pair_id"].nunique()),
                "robust_criterion_accuracy": float(
                    group["robust_target_success"].mean()
                ),
                "order_consistent_rate": float(
                    group["order_consistent_majority"].mean()
                ),
                "mean_branch_agreement": float(
                    group["mean_branch_agreement"].mean()
                ),
                "mean_branch_entropy": float(
                    group["mean_branch_entropy"].mean()
                ),
                "natural_valid_rate": float(
                    group["mean_natural_valid_rate"].mean()
                ),
                "mean_forced_confidence": float(
                    group["mean_forced_confidence"].mean()
                ),
                "predicted_tie_rate": float(group["predicted_tie"].mean()),
                "target_tie_rate": float(group["target_tie"].mean()),
                "tie_recall": (
                    float(target_ties["robust_target_success"].mean())
                    if len(target_ties)
                    else float("nan")
                ),
                "tie_precision": (
                    float(predicted_ties["target_tie"].mean())
                    if len(predicted_ties)
                    else float("nan")
                ),
                "unjustified_tie_rate": float(
                    group["unjustified_tie"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_budget_effects(orders: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "source_dataset",
        "pair_stratum",
        "criterion_id",
        "criterion_determinacy",
        "determinacy_level",
        "origin_pair_id",
        "pair_id",
    ]
    metrics = [
        "robust_target_success",
        "order_consistent_majority",
        "mean_branch_agreement",
        "mean_branch_entropy",
        "mean_natural_valid_rate",
        "mean_forced_confidence",
        "justified_tie_success",
        "unjustified_tie",
    ]
    direct = orders[orders["reasoning_mode"].eq("direct")][keys + metrics].copy()
    direct = direct.rename(
        columns={metric: f"direct_{metric}" for metric in metrics}
    )
    thinking = orders[orders["reasoning_mode"].eq("thinking")].copy()
    merged = thinking.merge(direct, on=keys, how="inner")
    for metric in metrics:
        merged[f"delta_{metric}"] = (
            merged[metric] - merged[f"direct_{metric}"]
        )
    return merged


def criterion_switch_rows(orders: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["reasoning_mode", "pair_stratum", "pair_id", "budget_tokens"]
    for keys, group in orders.groupby(groups, sort=True, dropna=False):
        by_criterion = {
            str(row["criterion_id"]): row
            for _, row in group.iterrows()
        }
        for left_name, right_name in itertools.combinations(
            sorted(by_criterion),
            2,
        ):
            left = by_criterion[left_name]
            right = by_criterion[right_name]
            expected_switch = (
                str(left["target_selected_text_hash"])
                != str(right["target_selected_text_hash"])
            )
            both_consistent = bool(
                left["has_order_consistent_verdict"]
                and right["has_order_consistent_verdict"]
            )
            observed_switch = (
                bool(
                    str(left["order_majority_selected_text_hash"])
                    != str(right["order_majority_selected_text_hash"])
                )
                if both_consistent
                else np.nan
            )
            rows.append(
                {
                    **dict(zip(groups, keys, strict=True)),
                    "left_criterion": left_name,
                    "right_criterion": right_name,
                    "criterion_pair": f"{left_name}__vs__{right_name}",
                    "expected_switch": bool(expected_switch),
                    "both_order_consistent": bool(both_consistent),
                    "observed_switch": observed_switch,
                    "both_criteria_correct": float(
                        left["robust_target_success"]
                        * right["robust_target_success"]
                    ),
                    "switch_compliance": (
                        float(bool(observed_switch) == bool(expected_switch))
                        if both_consistent
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def criterion_switch_summary(rows: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "reasoning_mode",
        "pair_stratum",
        "budget_tokens",
        "criterion_pair",
        "expected_switch",
    ]
    return (
        rows.groupby(groups, sort=True, dropna=False)
        .agg(
            n_pairs=("pair_id", "nunique"),
            both_order_consistent_rate=("both_order_consistent", "mean"),
            observed_switch_rate=("observed_switch", "mean"),
            switch_compliance_rate=("switch_compliance", "mean"),
            both_criteria_correct_rate=("both_criteria_correct", "mean"),
        )
        .reset_index()
    )


def analyze(
    score_rows: list[dict[str, Any]],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    generic = analyze_budget_sweep(
        score_rows,
        bootstrap=bootstrap,
        seed=seed,
    )
    orders = enrich_order_rows(generic["order_budget_rows"])
    effects = paired_budget_effects(orders)
    switches = criterion_switch_rows(orders)
    effect_metrics = [
        column for column in effects.columns if column.startswith("delta_")
    ]
    return {
        **generic,
        "matched_order_rows": orders,
        "matched_summary": matched_summary(orders),
        "matched_pair_budget_effects": effects,
        "matched_budget_effect_summary": grouped_bootstrap_summary(
            effects,
            group_columns=[
                "pair_stratum",
                "criterion_id",
                "budget_tokens",
            ],
            metrics=effect_metrics,
            n_bootstrap=bootstrap,
            seed=seed + 11,
        ),
        "criterion_switch_rows": switches,
        "criterion_switch_summary": criterion_switch_summary(switches),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    run_dirs = [_resolve(workspace_root, path) for path in args.run_dir]
    out_dir = _resolve(workspace_root, args.out_dir)
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        scores_path = run_dir / "budget_scores.jsonl"
        if not scores_path.is_file():
            raise FileNotFoundError(scores_path)
        rows.extend(read_jsonl(scores_path))
    outputs = analyze(
        rows,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
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
            "stage": "helpsteer2-matched-criterion-analysis",
            "run_dirs": [str(path) for path in run_dirs],
            "out_dir": str(out_dir),
            "n_score_rows": int(len(rows)),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "outputs": output_paths,
        },
    )

    summary = outputs["matched_summary"]
    display = summary[
        (
            summary["reasoning_mode"].eq("direct")
            & summary["budget_tokens"].eq(0)
        )
        | (
            summary["reasoning_mode"].eq("thinking")
            & summary["budget_tokens"].eq(
                summary.loc[
                    summary["reasoning_mode"].eq("thinking"),
                    "budget_tokens",
                ].max()
            )
        )
    ]
    print(f"out_dir={out_dir}")
    print("\n=== Matched criterion outcomes ===")
    print(
        display[
            [
                "reasoning_mode",
                "budget_tokens",
                "pair_stratum",
                "criterion_id",
                "n_pairs",
                "robust_criterion_accuracy",
                "order_consistent_rate",
                "predicted_tie_rate",
                "tie_recall",
                "unjustified_tie_rate",
                "mean_branch_agreement",
            ]
        ].to_string(index=False)
    )
    print("\n=== Criterion switching ===")
    switch_summary = outputs["criterion_switch_summary"]
    max_budget = switch_summary.loc[
        switch_summary["reasoning_mode"].eq("thinking"),
        "budget_tokens",
    ].max()
    print(
        switch_summary[
            (
                switch_summary["reasoning_mode"].eq("thinking")
                & switch_summary["budget_tokens"].eq(max_budget)
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
