"""Analyze latent factors in a D4 decision-manifold matrix."""

from __future__ import annotations

import argparse
import re
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
    parser.add_argument(
        "--min-unit-coverage",
        type=float,
        default=0.25,
        help="Drop units with less than this fraction of selected feature columns observed.",
    )
    parser.add_argument(
        "--include-feature-regex",
        action="append",
        default=[],
        help="Keep only feature columns matching at least one regex. May be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-feature-regex",
        action="append",
        default=[],
        help="Drop feature columns matching any regex. May be passed multiple times.",
    )
    parser.add_argument(
        "--stratify-by",
        action="append",
        default=[],
        help=(
            "Metadata columns for local-effect summaries, e.g. source_dataset,item_type "
            "or axis,role. May be comma-separated or passed multiple times."
        ),
    )
    parser.add_argument(
        "--center-within-strata",
        action="store_true",
        help="Before factorization, subtract each selected feature's stratum mean within --stratify-by groups.",
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--sparse-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _matches_any(value: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, value) for pattern in patterns)


def _feature_columns(
    df: pd.DataFrame,
    *,
    min_coverage: float,
    include_regex: list[str],
    exclude_regex: list[str],
) -> list[str]:
    cols = []
    for col in df.columns:
        if col in META_COLUMNS:
            continue
        if include_regex and not _matches_any(str(col), include_regex):
            continue
        if exclude_regex and _matches_any(str(col), exclude_regex):
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


def _parse_stratify_by(raw_items: list[str]) -> list[str]:
    cols: list[str] = []
    for raw in raw_items:
        for part in str(raw).split(","):
            col = part.strip()
            if col and col not in cols:
                cols.append(col)
    return cols


def _stratum_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot stratify by missing columns: {missing}")
    out = df[cols].copy()
    for col in cols:
        out[col] = out[col].fillna("").astype(str)
        out.loc[out[col] == "", col] = "<blank>"
    out["stratum_id"] = out[cols].agg("::".join, axis=1)
    return out


def _stratum_feature_summary(
    x_raw: pd.DataFrame,
    strata: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = x_raw.copy()
    work["stratum_id"] = strata["stratum_id"].to_numpy()
    for col in [c for c in strata.columns if c != "stratum_id"]:
        work[col] = strata[col].to_numpy()

    rows: list[dict[str, Any]] = []
    stratum_cols = [c for c in strata.columns if c != "stratum_id"]
    for stratum_id, group in work.groupby("stratum_id", sort=True):
        meta = {col: str(group[col].iloc[0]) for col in stratum_cols}
        n_units = int(len(group))
        for feature in feature_cols:
            values = pd.to_numeric(group[feature], errors="coerce")
            observed = values.dropna()
            if observed.empty:
                continue
            rows.append(
                {
                    "stratum_id": str(stratum_id),
                    **meta,
                    "feature_name": feature,
                    "n_units": n_units,
                    "n_observed": int(len(observed)),
                    "coverage": float(len(observed) / n_units) if n_units else 0.0,
                    "mean": float(observed.mean()),
                    "std": float(observed.std()) if len(observed) > 1 else 0.0,
                    "median": float(observed.median()),
                    "mean_abs": float(observed.abs().mean()),
                    "positive_rate": float((observed > 0).mean()),
                    "negative_rate": float((observed < 0).mean()),
                }
            )
    by_stratum = pd.DataFrame(rows)
    if by_stratum.empty:
        return by_stratum, pd.DataFrame()

    global_rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        global_values = pd.to_numeric(x_raw[feature], errors="coerce").dropna()
        local = by_stratum[by_stratum["feature_name"].astype(str) == feature].copy()
        if global_values.empty or local.empty:
            continue
        local_mean = pd.to_numeric(local["mean"], errors="coerce")
        global_mean = float(global_values.mean())
        mean_abs_local_mean = float(local_mean.abs().mean())
        max_abs_local_mean = float(local_mean.abs().max())
        denom = abs(global_mean)
        global_rows.append(
            {
                "feature_name": feature,
                "n_strata": int(len(local)),
                "n_observed": int(len(global_values)),
                "global_mean": global_mean,
                "global_mean_abs": float(global_values.abs().mean()),
                "mean_abs_stratum_mean": mean_abs_local_mean,
                "max_abs_stratum_mean": max_abs_local_mean,
                "positive_strata_rate": float((local_mean > 0).mean()),
                "negative_strata_rate": float((local_mean < 0).mean()),
                "cancellation_ratio": None if denom < 1e-12 else float(mean_abs_local_mean / denom),
            }
        )
    return by_stratum, pd.DataFrame(global_rows)


def _center_within_strata(x_raw: pd.DataFrame, strata: pd.DataFrame) -> pd.DataFrame:
    centered = x_raw.copy()
    stratum_id = strata["stratum_id"].astype(str)
    for col in centered.columns:
        values = pd.to_numeric(centered[col], errors="coerce")
        centered[col] = values - values.groupby(stratum_id).transform("mean")
    return centered


def _drop_constant_features(x_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep = []
    for col in x_raw.columns:
        if pd.to_numeric(x_raw[col], errors="coerce").nunique(dropna=True) > 1:
            keep.append(col)
    if not keep:
        raise ValueError("No nonconstant feature columns remain after preprocessing.")
    return x_raw[keep].copy(), keep


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
    min_unit_coverage: float,
    include_feature_regex: list[str],
    exclude_feature_regex: list[str],
    stratify_by: list[str],
    center_within_strata: bool,
    top_k: int,
    sparse_alpha: float,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    feature_cols = _feature_columns(
        df,
        min_coverage=min_feature_coverage,
        include_regex=include_feature_regex,
        exclude_regex=exclude_feature_regex,
    )
    unit_coverage = df[feature_cols].notna().mean(axis=1)
    keep_units = unit_coverage >= float(min_unit_coverage)
    if not keep_units.any():
        raise ValueError("No units passed the selected-feature unit coverage filter.")
    df = df.loc[keep_units].copy()
    n_components = min(int(n_components), len(feature_cols), len(df))
    if n_components < 1:
        raise ValueError("n_components must be positive after clipping.")

    metadata_cols = [col for col in df.columns if col in META_COLUMNS]
    metadata = df[metadata_cols].copy()
    if "unit_id" not in metadata.columns:
        raise ValueError("manifold_wide needs a unit_id column.")

    x_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    strata = _stratum_frame(df, stratify_by) if stratify_by else pd.DataFrame()
    stratum_feature_summary = pd.DataFrame()
    stratum_cancellation_summary = pd.DataFrame()
    if stratify_by:
        stratum_feature_summary, stratum_cancellation_summary = _stratum_feature_summary(
            x_raw,
            strata,
            feature_cols,
        )
        if center_within_strata:
            x_raw = _center_within_strata(x_raw, strata)
            x_raw, feature_cols = _drop_constant_features(x_raw)
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
    if not stratum_feature_summary.empty:
        outputs["stratum_feature_summary"] = stratum_feature_summary
    if not stratum_cancellation_summary.empty:
        outputs["stratum_cancellation_summary"] = stratum_cancellation_summary
    manifest = {
        "stage": "D4-decision-manifold-factor-analysis",
        "n_units": int(len(df)),
        "n_features_available": int(len([col for col in df.columns if col not in META_COLUMNS])),
        "n_features_used": int(len(feature_cols)),
        "n_components": int(n_components),
        "min_feature_coverage": float(min_feature_coverage),
        "min_unit_coverage": float(min_unit_coverage),
        "include_feature_regex": list(include_feature_regex),
        "exclude_feature_regex": list(exclude_feature_regex),
        "stratify_by": list(stratify_by),
        "center_within_strata": bool(center_within_strata),
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
        min_unit_coverage=args.min_unit_coverage,
        include_feature_regex=[str(item) for item in args.include_feature_regex],
        exclude_feature_regex=[str(item) for item in args.exclude_feature_regex],
        stratify_by=_parse_stratify_by([str(item) for item in args.stratify_by]),
        center_within_strata=bool(args.center_within_strata),
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
