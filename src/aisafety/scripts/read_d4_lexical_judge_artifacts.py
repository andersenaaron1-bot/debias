"""Print a compact readout from a D4 lexical judge-artifact audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-targets", type=int, default=12)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--target-regex", default=r"minus|stage_delta|interaction")
    return parser.parse_args()


def _read_csv(root: Path, filename: str) -> pd.DataFrame:
    path = root / filename
    if not path.is_file() or path.stat().st_size <= 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _filter_targets(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    if df.empty or not pattern:
        return df
    return df[df["target_name"].astype(str).str.contains(pattern, regex=True, flags=re.IGNORECASE)]


def _top_per_target(
    df: pd.DataFrame,
    *,
    signal_col: str,
    top_k: int,
    max_targets: int,
) -> pd.DataFrame:
    if df.empty:
        return df
    ranked_targets = (
        df.groupby("target_name", sort=False)[signal_col]
        .max()
        .sort_values(ascending=False)
        .head(max(int(max_targets), 1))
        .index
    )
    return (
        df[df["target_name"].isin(ranked_targets)]
        .sort_values(["target_name", signal_col], ascending=[True, False])
        .groupby("target_name", sort=False)
        .head(max(int(top_k), 1))
    )


def readout(
    root: Path,
    *,
    top_k: int,
    max_targets: int,
    min_support: int,
    target_regex: str,
) -> None:
    root = Path(root)
    hllm = _read_csv(root, "hllm_artifact_coefficients.csv")
    metrics = _read_csv(root, "hllm_heldout_metrics.csv")
    surface = _read_csv(root, "surface_edit_fragment_effects.csv")

    print("=== HLLM held-out lexical prediction ===")
    filtered_metrics = _filter_targets(metrics, target_regex)
    if filtered_metrics.empty:
        print("(empty)")
    else:
        metric_summary = (
            filtered_metrics.groupby("target_name", sort=True)
            .agg(
                n_folds=("fold", "size"),
                mean_r2=("r2", "mean"),
                mean_pearson=("pearson", "mean"),
                mean_mae=("mae", "mean"),
                baseline_mae=("baseline_mae", "mean"),
            )
            .reset_index()
            .sort_values("mean_pearson", ascending=False)
        )
        print(metric_summary.to_string(index=False))

    print("\n=== HLLM top word artifacts ===")
    filtered_hllm = _filter_targets(hllm, target_regex)
    if not filtered_hllm.empty:
        filtered_hllm = filtered_hllm[
            (filtered_hllm["artifact_kind"] == "word")
            & (filtered_hllm["support_pairs"] >= max(int(min_support), 1))
        ]
    filtered_hllm = _top_per_target(
        filtered_hllm,
        signal_col="abs_partial_corr",
        top_k=top_k,
        max_targets=max_targets,
    )
    if filtered_hllm.empty:
        print("(empty)")
    else:
        cols = [
            "target_name",
            "artifact_name",
            "support_pairs",
            "mean_llm_minus_human_presence",
            "partial_corr",
            "partial_slope",
            "elastic_coef",
        ]
        print(filtered_hllm[cols].to_string(index=False))

    print("\n=== Surface top rewrite fragments ===")
    filtered_surface = _filter_targets(surface, target_regex)
    if not filtered_surface.empty:
        filtered_surface = filtered_surface[
            filtered_surface["support_counterfactuals"] >= max(int(min_support), 1)
        ]
    filtered_surface = _top_per_target(
        filtered_surface,
        signal_col="abs_artifact_minus_global_mean",
        top_k=top_k,
        max_targets=max_targets,
    )
    if filtered_surface.empty:
        print("(empty)")
    else:
        cols = [
            "target_name",
            "artifact_name",
            "support_counterfactuals",
            "artifact_minus_global_mean",
            "mean_stratum_adjusted_target_value",
        ]
        print(filtered_surface[cols].to_string(index=False))


def main() -> None:
    args = _parse_args()
    readout(
        args.input,
        top_k=args.top_k,
        max_targets=args.max_targets,
        min_support=args.min_support,
        target_regex=args.target_regex,
    )


if __name__ == "__main__":
    main()
