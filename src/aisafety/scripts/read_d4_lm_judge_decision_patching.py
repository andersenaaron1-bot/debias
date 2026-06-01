"""Print compact residual, suppression, and verified-component patching results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _print(df: pd.DataFrame, cols: list[str], *, empty: str = "(empty)") -> None:
    if df.empty:
        print(empty)
        return
    print(df[[col for col in cols if col in df.columns]].to_string(index=False))


def readout(root: Path, *, top_k: int) -> None:
    root = Path(root)
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name != "logs")
    for run_dir in run_dirs:
        print(f"\n######## {run_dir.name} ########")
        residual = _csv(run_dir / "residual_patch_summary.csv")
        suppression = _csv(run_dir / "subspace_suppression_summary.csv")
        shortlist = _csv(run_dir / "component_shortlist.csv")
        verified = _csv(run_dir / "component_verification_summary.csv")

        print("\n=== Best decision-state residual patches ===")
        if not residual.empty:
            residual = residual.copy()
            residual["aggregate_recovery"] = (
                (residual["mean_patched_margin"] - residual["mean_neutral_margin"])
                / (residual["mean_observed_margin"] - residual["mean_neutral_margin"])
            )
            residual = residual.sort_values(
                ["dataset", "basis_eval_split", "patch_type", "aggregate_recovery"],
                ascending=[True, True, True, False],
            )
            residual = residual.groupby(["dataset", "basis_eval_split", "patch_type"], sort=True).head(3)
        _print(
            residual,
            [
                "dataset",
                "basis_eval_split",
                "patch_type",
                "hidden_layer",
                "n_counterfactuals",
                "mean_observed_margin",
                "mean_neutral_margin",
                "mean_patched_margin",
                "aggregate_recovery",
            ],
        )

        print("\n=== Best low-rank suppression layers ===")
        if not suppression.empty:
            suppression = suppression.copy()
            suppression["aggregate_attenuation"] = (
                (suppression["mean_suppressed_margin"] - suppression["mean_observed_margin"])
                / (suppression["mean_neutral_margin"] - suppression["mean_observed_margin"])
            )
            suppression = suppression.sort_values(
                ["dataset", "basis_eval_split", "aggregate_attenuation"],
                ascending=[True, True, False],
            )
            suppression = suppression.groupby(["dataset", "basis_eval_split"], sort=True).head(4)
        _print(
            suppression,
            [
                "dataset",
                "basis_eval_split",
                "hidden_layer",
                "subspace_rank",
                "suppression_alpha",
                "n_counterfactuals",
                "mean_observed_margin",
                "mean_suppressed_margin",
                "aggregate_attenuation",
            ],
        )

        print("\n=== Gradient-attribution component shortlist ===")
        _print(
            shortlist.head(max(int(top_k), 1)),
            ["component_type", "hidden_layer", "component_index", "n_bt_rows", "mean_attribution", "mean_abs_attribution"],
        )

        print("\n=== Verified decision-position component patches ===")
        if not verified.empty:
            verified = verified.sort_values("aggregate_recovery", ascending=False).head(max(int(top_k), 1))
        _print(
            verified,
            [
                "component_type",
                "hidden_layer",
                "component_index",
                "n_counterfactuals",
                "mean_observed_margin",
                "mean_neutral_margin",
                "mean_patched_margin",
                "aggregate_recovery",
            ],
        )


def main() -> None:
    args = _parse_args()
    readout(args.input, top_k=int(args.top_k))


if __name__ == "__main__":
    main()
