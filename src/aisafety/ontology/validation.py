"""Statistical validation helpers for D2 atom co-occurrence and bundle support."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from aisafety.config import DEFAULT_SEED
from .atoms import ATOM_SPEC_NAMES, extract_atom_scores, get_atom_specs, get_bundle_priors


def score_records_with_atoms(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Attach atom scores to normalized text records."""

    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        text = str(record.get("text") or "")
        row["word_count"] = int(len(text.split()))
        row.update(extract_atom_scores(text))
        rows.append(row)
    return pd.DataFrame(rows)


def _zscore_by_group(df: pd.DataFrame, cols: list[str], *, group_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        zcol = f"z_{col}"
        out[zcol] = 0.0
        for _, grp in out.groupby(group_col):
            mean = float(grp[col].mean())
            std = float(grp[col].std(ddof=0))
            if std <= 1e-12:
                out.loc[grp.index, zcol] = 0.0
            else:
                out.loc[grp.index, zcol] = (grp[col] - mean) / std
    return out


def _bootstrap_ci(values: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    stats = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, values.size, size=values.size)
        stats.append(float(np.mean(values[idx])))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def compute_atom_summaries(
    df: pd.DataFrame,
    atom_cols: list[str] | None = None,
    *,
    n_bootstrap: int = 200,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Summarize atom prevalence, stability, and length sensitivity."""

    atom_cols = list(atom_cols or ATOM_SPEC_NAMES)
    rng = np.random.default_rng(int(seed))
    out: dict[str, Any] = {}
    for col in atom_cols:
        vals = df[col].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(vals > 0.0, n_bootstrap=n_bootstrap, rng=rng)
        if np.allclose(vals.std(), 0.0) or np.allclose(df["word_count"].to_numpy(dtype=float).std(), 0.0):
            length_corr = 0.0
        else:
            length_corr = spearmanr(vals, df["word_count"].to_numpy(dtype=float)).statistic
        by_item_type = {}
        for item_type, grp in df.groupby("item_type"):
            gvals = grp[col].to_numpy(dtype=float)
            by_item_type[str(item_type)] = {
                "mean": float(np.mean(gvals)),
                "std": float(np.std(gvals)),
                "prevalence_nonzero": float(np.mean(gvals > 0.0)),
            }
        out[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "prevalence_nonzero": float(np.mean(vals > 0.0)),
            "prevalence_ci_95_low": ci_low,
            "prevalence_ci_95_high": ci_high,
            "length_spearman": 0.0 if pd.isna(length_corr) else float(length_corr),
            "by_item_type": by_item_type,
        }
    return out


def _activation_matrix(df: pd.DataFrame, atom_cols: list[str], *, group_col: str, q: float = 0.8) -> pd.DataFrame:
    active = pd.DataFrame(index=df.index)
    for col in atom_cols:
        active[col] = 0
        for _, grp in df.groupby(group_col):
            threshold = float(grp[col].quantile(float(q)))
            active.loc[grp.index, col] = (grp[col] >= threshold).astype(int)
    return active


def compute_pairwise_cooccurrence(
    df: pd.DataFrame,
    atom_cols: list[str] | None = None,
    *,
    group_col: str = "item_type",
) -> pd.DataFrame:
    """Compute pairwise atom co-occurrence statistics."""

    atom_cols = list(atom_cols or ATOM_SPEC_NAMES)
    zcols = [f"z_{col}" for col in atom_cols]
    zdf = _zscore_by_group(df, atom_cols, group_col=group_col)
    pearson = zdf[zcols].corr(method="pearson")
    spearman = zdf[zcols].corr(method="spearman")
    active = _activation_matrix(df, atom_cols, group_col=group_col)
    rows: list[dict[str, Any]] = []
    for atom_a, atom_b in itertools.combinations(atom_cols, 2):
        a_active = active[atom_a].to_numpy(dtype=int)
        b_active = active[atom_b].to_numpy(dtype=int)
        union = int(np.sum((a_active + b_active) > 0))
        intersection = int(np.sum((a_active + b_active) == 2))
        rows.append(
            {
                "atom_a": atom_a,
                "atom_b": atom_b,
                "pearson_r": float(pearson.loc[f"z_{atom_a}", f"z_{atom_b}"]),
                "spearman_r": float(spearman.loc[f"z_{atom_a}", f"z_{atom_b}"]),
                "coactivation_jaccard": float(intersection / union) if union else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["pearson_r", "spearman_r"], ascending=False).reset_index(drop=True)


def _corr_from_df(df_z: pd.DataFrame, atoms: list[str]) -> pd.DataFrame:
    corr = df_z[[f"z_{atom}" for atom in atoms]].corr(method="pearson")
    corr.index = atoms
    corr.columns = atoms
    return corr.fillna(0.0)


def _mean_pairwise_corr(corr: pd.DataFrame, atoms: list[str]) -> float:
    if len(atoms) < 2:
        return 0.0
    vals = [float(corr.loc[a, b]) for a, b in itertools.combinations(atoms, 2)]
    return float(np.mean(vals)) if vals else 0.0


def _bundle_score(df_z: pd.DataFrame, atoms: list[str]) -> np.ndarray:
    if not atoms:
        return np.zeros(len(df_z), dtype=float)
    cols = [f"z_{atom}" for atom in atoms]
    return df_z[cols].to_numpy(dtype=float).mean(axis=1)


def _pca_var(df_z: pd.DataFrame, atoms: list[str]) -> float:
    if len(atoms) < 2:
        return 0.0
    x = df_z[[f"z_{atom}" for atom in atoms]].to_numpy(dtype=float)
    if np.allclose(x.std(axis=0), 0.0):
        return 0.0
    pca = PCA(n_components=1, random_state=DEFAULT_SEED)
    pca.fit(x)
    return float(pca.explained_variance_ratio_[0])


def _random_set_p_value(
    corr: pd.DataFrame,
    atoms: list[str],
    universe: list[str],
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(atoms) < 2:
        return 1.0, 0.0
    observed = _mean_pairwise_corr(corr, atoms)
    null_scores = []
    for _ in range(int(n_samples)):
        sampled = list(rng.choice(universe, size=len(atoms), replace=False))
        null_scores.append(_mean_pairwise_corr(corr, sampled))
    null_arr = np.asarray(null_scores, dtype=float)
    p_value = float(np.mean(null_arr >= observed)) if null_arr.size else 1.0
    return p_value, float(null_arr.mean()) if null_arr.size else 0.0


def _cluster_atoms(corr: pd.DataFrame, *, n_clusters: int) -> dict[str, int]:
    atoms = list(corr.index)
    if len(atoms) == 1:
        return {atoms[0]: 1}
    distance = 1.0 - corr.clip(-1.0, 1.0).to_numpy(dtype=float)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    labels = fcluster(linkage_matrix, t=max(2, int(n_clusters)), criterion="maxclust")
    return {atom: int(label) for atom, label in zip(atoms, labels)}


def _bootstrap_bundle_corr_ci(df_z: pd.DataFrame, atoms: list[str], *, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(atoms) < 2 or df_z.empty:
        return 0.0, 0.0
    stats = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, len(df_z), size=len(df_z))
        sampled = df_z.iloc[idx]
        corr = _corr_from_df(sampled, atoms)
        stats.append(_mean_pairwise_corr(corr, atoms))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def _bootstrap_co_cluster_matrix(
    df_z: pd.DataFrame,
    atoms: list[str],
    *,
    n_clusters: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    co_cluster = pd.DataFrame(0.0, index=atoms, columns=atoms)
    if not atoms:
        return co_cluster
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, len(df_z), size=len(df_z))
        sampled = df_z.iloc[idx]
        corr = _corr_from_df(sampled, atoms)
        labels = _cluster_atoms(corr, n_clusters=n_clusters)
        for a, b in itertools.combinations_with_replacement(atoms, 2):
            same = float(labels[a] == labels[b])
            co_cluster.loc[a, b] += same
            co_cluster.loc[b, a] += same
    return co_cluster / float(n_bootstrap)


def build_bundle_validation(
    df: pd.DataFrame,
    *,
    group_col: str = "item_type",
    source_col: str = "source",
    n_bootstrap: int = 200,
    n_null_samples: int = 500,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Validate theory-seeded bundles with co-occurrence statistics."""

    specs = get_atom_specs()
    bundle_priors = get_bundle_priors()
    validation_atoms = [atom for atom, spec in specs.items() if spec.validation_subset]
    df_z = _zscore_by_group(df, validation_atoms, group_col=group_col)
    corr = _corr_from_df(df_z, validation_atoms)
    active = _activation_matrix(df, validation_atoms, group_col=group_col)
    rng = np.random.default_rng(int(seed))

    bundles_for_clustering = [
        bundle for bundle, atoms in bundle_priors.items()
        if len([atom for atom in atoms if atom in validation_atoms]) >= 2
    ]
    n_clusters = max(2, len(bundles_for_clustering))
    cluster_labels = _cluster_atoms(corr, n_clusters=n_clusters)
    clusters: dict[int, list[str]] = {}
    for atom, label in cluster_labels.items():
        clusters.setdefault(int(label), []).append(atom)
    clusters = {label: sorted(atoms) for label, atoms in sorted(clusters.items())}
    co_cluster = _bootstrap_co_cluster_matrix(df_z, validation_atoms, n_clusters=n_clusters, n_bootstrap=n_bootstrap, rng=rng)

    bundle_rows: dict[str, Any] = {}
    for bundle, bundle_atoms_all in bundle_priors.items():
        bundle_atoms = [atom for atom in bundle_atoms_all if atom in validation_atoms]
        if not bundle_atoms:
            continue
        ci_low, ci_high = _bootstrap_bundle_corr_ci(df_z, bundle_atoms, n_bootstrap=n_bootstrap, rng=rng)
        p_value, null_mean = _random_set_p_value(corr, bundle_atoms, validation_atoms, n_samples=n_null_samples, rng=rng)
        mean_corr = _mean_pairwise_corr(corr, bundle_atoms)
        score = _bundle_score(df_z, bundle_atoms)
        source_auc = None
        if source_col in df_z.columns:
            y = (df_z[source_col] == "llm").astype(int).to_numpy()
            if len(np.unique(y)) == 2:
                try:
                    source_auc = float(roc_auc_score(y, score))
                except ValueError:
                    source_auc = None
        per_item_type = {}
        for item_type, grp in df_z.groupby(group_col):
            gcorr = _corr_from_df(grp, bundle_atoms)
            per_item_type[str(item_type)] = {
                "mean_pairwise_r": _mean_pairwise_corr(gcorr, bundle_atoms),
                "llm_minus_human_bundle_score": float(
                    grp.loc[grp[source_col] == "llm", [f"z_{atom}" for atom in bundle_atoms]].to_numpy(dtype=float).mean()
                    - grp.loc[grp[source_col] == "human", [f"z_{atom}" for atom in bundle_atoms]].to_numpy(dtype=float).mean()
                ) if source_col in grp.columns and {"human", "llm"}.issubset(set(grp[source_col])) else 0.0,
            }
        cluster_jaccards = []
        for cluster_id, cluster_atoms in clusters.items():
            left = set(bundle_atoms)
            right = set(cluster_atoms)
            cluster_jaccards.append((cluster_id, len(left & right) / float(len(left | right)) if left or right else 0.0))
        best_cluster_id, best_cluster_jaccard = max(cluster_jaccards, key=lambda x: x[1])
        pair_probs = [float(co_cluster.loc[a, b]) for a, b in itertools.combinations(bundle_atoms, 2)]
        union = []
        for a, b in itertools.combinations(bundle_atoms, 2):
            aa = active[a].to_numpy(dtype=int)
            bb = active[b].to_numpy(dtype=int)
            denom = int(np.sum((aa + bb) > 0))
            numer = int(np.sum((aa + bb) == 2))
            union.append(float(numer / denom) if denom else 0.0)
        status = "supported" if (mean_corr >= 0.12 and p_value <= 0.1 and best_cluster_jaccard >= 0.3) else "exploratory"
        bundle_rows[bundle] = {
            "member_atoms_all": bundle_atoms_all,
            "member_atoms_validation": bundle_atoms,
            "n_atoms_validation": int(len(bundle_atoms)),
            "observed_mean_pairwise_r": float(mean_corr),
            "bootstrap_ci_95_low": ci_low,
            "bootstrap_ci_95_high": ci_high,
            "null_mean_pairwise_r": float(null_mean),
            "empirical_p_value": float(p_value),
            "mean_coactivation_jaccard": float(np.mean(union)) if union else 0.0,
            "mean_co_clustering_probability": float(np.mean(pair_probs)) if pair_probs else 1.0,
            "first_pc_explained_variance": _pca_var(df_z, bundle_atoms),
            "source_auc_human_vs_llm": source_auc,
            "best_derived_cluster_id": int(best_cluster_id),
            "best_derived_cluster_jaccard": float(best_cluster_jaccard),
            "by_item_type": per_item_type,
            "status": status,
        }

    return {
        "validation_atoms": validation_atoms,
        "derived_clusters": clusters,
        "bundle_validation": bundle_rows,
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows
