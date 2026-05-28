"""Analyze latent factors in a D4 decision-manifold matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA, SparsePCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json


DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_decision_manifold_factors_v1"
META_COLUMNS = {
    "unit_id",
    "source_dataset",
    "subset",
    "item_type",
    "axis",
    "direction",
    "role",
    "template_label",
    "comparison_template",
    "scoring_mode",
    "model_id",
    "prompt_style",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifold-wide", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--min-feature-coverage", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--sparse-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _feature_columns(df: pd.DataFrame, *, min_coverage: float) -> list[str]:
    cols = []
    for col in df.columns:
        if col in META_COLUMNS:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        coverage = float(series.notna().mean())
        if coverage >= min_coverage and series.nunique(dropna=True) > 1:
            cols.append(col)
    if not cols:
        raise ValueError("No feature columns passed the coverage/variance filters.")
    return cols


def _component_names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{idx + 1}" for idx in range(n)]


def _loadings_frame(
    *,
    method: str,
    component_names: list[str],
    feature_cols: list[str],
    components: np.ndarray,
    explained: np.ndarray | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comp_idx, comp_name in enumerate(component_names):
        for feature, weight in zip(feature_cols, components[comp_idx]):
            rows.append(
                {
                    "method": method,
                    "component": comp_name,
                    "feature_name": feature,
                    "loading": float(weight),
                    "abs_loading": float(abs(weight)),
                    "explained_variance_ratio": None
                    if explained is None
                    else float(explained[comp_idx]),
                }
            )
    return pd.DataFrame(rows)


def _top_units(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    method: str,
    top_k: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    score_cols = [col for col in scores.columns if col != "unit_id"]
    for col in score_cols:
        frame = scores[["unit_id", col]].rename(columns={col: "score"}).copy()
        frame["abs_score"] = frame["score"].abs()
        top = frame.sort_values("abs_score", ascending=False).head(max(int(top_k), 0)).copy()
        top["method"] = method
        top["component"] = col
        rows.append(top.merge(metadata, on="unit_id", how="left"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _method_summary(
    *,
    method: str,
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component, group in scores.drop(columns=["unit_id"]).items():
        loading_group = loadings[loadings["component"] == component]
        rows.append(
            {
                "method": method,
                "component": component,
                "score_mean": float(group.mean()),
                "score_std": float(group.std()),
                "score_median": float(group.median()),
                "score_abs_mean": float(group.abs().mean()),
                "top_positive_feature": str(
                    loading_group.sort_values("loading", ascending=False)["feature_name"].iloc[0]
                ),
                "top_negative_feature": str(
                    loading_group.sort_values("loading", ascending=True)["feature_name"].iloc[0]
                ),
                "top_abs_feature": str(
                    loading_group.sort_values("abs_loading", ascending=False)["feature_name"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def _run_pca(x: np.ndarray, *, n_components: int, seed: int) -> tuple[PCA, np.ndarray]:
    model = PCA(n_components=n_components, random_state=seed)
    scores = model.fit_transform(x)
    return model, scores


def _run_sparse_pca(x: np.ndarray, *, n_components: int, alpha: float, seed: int) -> tuple[SparsePCA, np.ndarray]:
    model = SparsePCA(
        n_components=n_components,
        alpha=alpha,
        random_state=seed,
        n_jobs=-1,
    )
    scores = model.fit_transform(x)
    return model, scores


def _run_nmf(x: np.ndarray, *, n_components: int, seed: int) -> tuple[NMF, np.ndarray]:
    # Shift standardized features into nonnegative space for an exploratory NMF.
    x_nonnegative = x - np.nanmin(x, axis=0, keepdims=True)
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=seed,
        max_iter=1000,
    )
    scores = model.fit_transform(x_nonnegative)
    return model, scores


def analyze(
    df: pd.DataFrame,
    *,
    n_components: int,
    min_feature_coverage: float,
    top_k: int,
    sparse_alpha: float,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    feature_cols = _feature_columns(df, min_coverage=min_feature_coverage)
    n_components = min(int(n_components), len(feature_cols), len(df))
    if n_components < 1:
        raise ValueError("n_components must be positive after clipping.")

    metadata_cols = [col for col in df.columns if col in META_COLUMNS]
    metadata = df[metadata_cols].copy()
    if "unit_id" not in metadata.columns:
        raise ValueError("manifold_wide needs a unit_id column.")

    x_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    coverage = x_raw.notna().mean().rename("coverage").reset_index().rename(columns={"index": "feature_name"})
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    x = pipeline.fit_transform(x_raw)

    outputs: dict[str, pd.DataFrame] = {}
    summaries: list[pd.DataFrame] = []
    loadings: list[pd.DataFrame] = []
    top_units: list[pd.DataFrame] = []

    pca, pca_scores = _run_pca(x, n_components=n_components, seed=seed)
    pca_names = _component_names("pc", n_components)
    pca_score_df = pd.DataFrame(pca_scores, columns=pca_names)
    pca_score_df.insert(0, "unit_id", metadata["unit_id"].to_numpy())
    pca_loadings = _loadings_frame(
        method="pca",
        component_names=pca_names,
        feature_cols=feature_cols,
        components=pca.components_,
        explained=pca.explained_variance_ratio_,
    )
    loadings.append(pca_loadings)
    summaries.append(_method_summary(method="pca", scores=pca_score_df, loadings=pca_loadings))
    top_units.append(_top_units(pca_score_df, metadata, method="pca", top_k=top_k))
    outputs["pca_scores"] = pca_score_df

    sparse, sparse_scores = _run_sparse_pca(x, n_components=n_components, alpha=sparse_alpha, seed=seed)
    sparse_names = _component_names("spc", n_components)
    sparse_score_df = pd.DataFrame(sparse_scores, columns=sparse_names)
    sparse_score_df.insert(0, "unit_id", metadata["unit_id"].to_numpy())
    sparse_loadings = _loadings_frame(
        method="sparse_pca",
        component_names=sparse_names,
        feature_cols=feature_cols,
        components=sparse.components_,
    )
    loadings.append(sparse_loadings)
    summaries.append(_method_summary(method="sparse_pca", scores=sparse_score_df, loadings=sparse_loadings))
    top_units.append(_top_units(sparse_score_df, metadata, method="sparse_pca", top_k=top_k))
    outputs["sparse_pca_scores"] = sparse_score_df

    nmf, nmf_scores = _run_nmf(x, n_components=n_components, seed=seed)
    nmf_names = _component_names("nmf", n_components)
    nmf_score_df = pd.DataFrame(nmf_scores, columns=nmf_names)
    nmf_score_df.insert(0, "unit_id", metadata["unit_id"].to_numpy())
    nmf_loadings = _loadings_frame(
        method="nmf",
        component_names=nmf_names,
        feature_cols=feature_cols,
        components=nmf.components_,
    )
    loadings.append(nmf_loadings)
    summaries.append(_method_summary(method="nmf", scores=nmf_score_df, loadings=nmf_loadings))
    top_units.append(_top_units(nmf_score_df, metadata, method="nmf", top_k=top_k))
    outputs["nmf_scores"] = nmf_score_df

    outputs["component_loadings"] = pd.concat(loadings, ignore_index=True)
    outputs["component_summary"] = pd.concat(summaries, ignore_index=True)
    outputs["top_units"] = pd.concat(top_units, ignore_index=True)
    outputs["feature_coverage"] = coverage
    manifest = {
        "stage": "D4-decision-manifold-factor-analysis",
        "n_units": int(len(df)),
        "n_features_available": int(len([col for col in df.columns if col not in META_COLUMNS])),
        "n_features_used": int(len(feature_cols)),
        "n_components": int(n_components),
        "min_feature_coverage": float(min_feature_coverage),
        "top_k": int(top_k),
        "sparse_alpha": float(sparse_alpha),
        "seed": int(seed),
    }
    return outputs, manifest


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    manifold_wide = _resolve(workspace_root, args.manifold_wide)
    out_dir = _resolve(workspace_root, args.out_dir)
    df = pd.read_csv(manifold_wide)
    outputs, manifest = analyze(
        df,
        n_components=args.n_components,
        min_feature_coverage=args.min_feature_coverage,
        top_k=args.top_k,
        sparse_alpha=args.sparse_alpha,
        seed=args.seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[f"{name}_csv"] = str(path)
    manifest["manifold_wide"] = str(manifold_wide)
    manifest["out_dir"] = str(out_dir)
    manifest["outputs"] = output_paths | {"summary_json": str(out_dir / "summary.json")}
    write_json(out_dir / "summary.json", manifest)

    print(f"out_dir={out_dir}")
    print(f"n_units={manifest['n_units']}")
    print(f"n_features_used={manifest['n_features_used']}")
    print(f"component_summary={out_dir / 'component_summary.csv'}")
    print(f"component_loadings={out_dir / 'component_loadings.csv'}")
    print(f"top_units={out_dir / 'top_units.csv'}")


if __name__ == "__main__":
    main()
