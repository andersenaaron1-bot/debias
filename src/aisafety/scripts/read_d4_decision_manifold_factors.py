"""Print a compact readout for D4 decision-manifold factor analysis outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["pca", "sparse_pca", "nmf"], default="pca")
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n== {title} ==")
    if df.empty:
        print("(empty)")
    else:
        print(df.to_string(index=False))


def main() -> None:
    args = _parse_args()
    root = _resolve(Path(args.workspace_root).resolve(), args.factor_dir)
    summary = pd.read_csv(root / "component_summary.csv")
    loadings = pd.read_csv(root / "component_loadings.csv")
    top_units = pd.read_csv(root / "top_units.csv")
    method = str(args.method)

    method_summary = summary[summary["method"].astype(str) == method].head(args.components).copy()
    cols = [
        "component",
        "score_std",
        "score_abs_mean",
        "top_positive_feature",
        "top_negative_feature",
        "top_abs_feature",
    ]
    if "explained_variance_ratio" in method_summary.columns:
        cols.insert(1, "explained_variance_ratio")
    _print_table("component summary", method_summary[[c for c in cols if c in method_summary.columns]])

    for component in method_summary["component"].astype(str):
        comp_loadings = loadings[
            (loadings["method"].astype(str) == method)
            & (loadings["component"].astype(str) == component)
        ].copy()
        comp_loadings = comp_loadings.sort_values("abs_loading", ascending=False).head(args.top_k)
        _print_table(
            f"{component} top loadings",
            comp_loadings[["feature_name", "loading", "abs_loading"]].copy(),
        )
        comp_units = top_units[
            (top_units["method"].astype(str) == method)
            & (top_units["component"].astype(str) == component)
        ].copy()
        comp_units = comp_units.sort_values("abs_score", ascending=False).head(args.top_k)
        keep = [
            "unit_id",
            "score",
            "abs_score",
            "source_dataset",
            "item_type",
            "axis",
            "role",
        ]
        _print_table(f"{component} top units", comp_units[[c for c in keep if c in comp_units.columns]])


if __name__ == "__main__":
    main()
