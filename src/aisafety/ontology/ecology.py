"""Ecological validation helpers for D3 atom and bundle effects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from aisafety.config import DEFAULT_SEED
from aisafety.eval.debias import add_pairwise_debias_columns, pair_key_human_llm
from aisafety.ontology.validation import score_records_with_atoms


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ensure_choice_columns(df_trials: pd.DataFrame, *, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Ensure ``score_diff`` and ``choice`` columns exist for scored A/B trials."""

    required = {"A_text", "B_text", "A_source", "B_source", "item_type", "title"}
    missing = required - set(df_trials.columns)
    if missing:
        raise ValueError(f"Trials dataframe missing required columns: {sorted(missing)}")

    out = df_trials.copy()
    if "score_diff" not in out.columns:
        if {"score_A", "score_B"}.issubset(set(out.columns)):
            out["score_diff"] = pd.to_numeric(out["score_A"], errors="coerce") - pd.to_numeric(out["score_B"], errors="coerce")
        else:
            raise ValueError("Need either score_diff or both score_A and score_B columns.")

    if "choice" not in out.columns:
        diff = pd.to_numeric(out["score_diff"], errors="coerce").to_numpy(dtype=float)
        choice = np.where(diff > 0.0, "A", "B").astype(object)
        ties = np.isclose(diff, 0.0, equal_nan=False)
        if ties.any():
            keys = (
                out["item_type"].astype(str).fillna("")
                + "||"
                + out["title"].astype(str).fillna("")
                + "||"
                + out["A_text"].astype(str).fillna("")
                + "||"
                + out["B_text"].astype(str).fillna("")
            )
            tie_choices = np.asarray(
                ["A" if (int(_sha1_hex(f"{int(seed)}:{key}")[:8], 16) % 2 == 0) else "B" for key in keys],
                dtype=object,
            )
            choice[ties] = tie_choices[ties]
        out["choice"] = choice
    return out


def collapse_scored_trials_to_pairs(df_trials: pd.DataFrame, *, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Collapse balanced-order Laurito trials to one row per human/LLM pair."""

    scored = ensure_choice_columns(df_trials, seed=seed)
    debiased = add_pairwise_debias_columns(scored, logit_diff_col="score_diff", seed=seed)
    out = debiased.copy()
    out["pair_key"] = pair_key_human_llm(out).astype(str)
    out["human_text"] = np.where(out["A_source"].astype(str) == "human", out["A_text"], out["B_text"])
    out["llm_text"] = np.where(out["A_source"].astype(str) == "llm", out["A_text"], out["B_text"])
    out["chosen_source_raw"] = np.where(out["choice"].astype(str) == "A", out["A_source"], out["B_source"])
    grouped = (
        out.groupby("pair_key", as_index=False)
        .agg(
            item_type=("item_type", "first"),
            title=("title", "first"),
            human_text=("human_text", "first"),
            llm_text=("llm_text", "first"),
            preferred_source_debiased=("preferred_source_debiased", "first"),
            chosen_source_debiased=("chosen_source_debiased", "first"),
            llm_margin_pair=("llm_margin_pair", "mean"),
            n_order_rows=("pair_key", "size"),
            chosen_source_raw_mean=("chosen_source_raw", lambda s: float(np.mean(np.asarray(s) == "llm"))),
        )
    )
    grouped["y_llm_chosen"] = (grouped["chosen_source_debiased"].astype(str) == "llm").astype(int)
    return grouped


def build_pair_text_atom_scores(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Score unique human/LLM texts appearing in a pair-level dataframe."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in df_pairs.itertuples(index=False):
        for side, source in (("human_text", "human"), ("llm_text", "llm")):
            text = str(getattr(row, side) or "")
            text_id = _sha1_hex(text)
            if text_id in seen or not text.strip():
                continue
            seen.add(text_id)
            rows.append(
                {
                    "text_id": text_id,
                    "text": text,
                    "item_type": str(row.item_type),
                    "source": source,
                    "title": str(row.title),
                }
            )
    scored = score_records_with_atoms(rows)
    return scored


def load_bundle_members(bundle_validation_json: str | Path, *, min_bundle_atoms: int = 2) -> dict[str, list[str]]:
    """Load validated bundle members from a D2 bundle-validation JSON."""

    with Path(bundle_validation_json).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    bundle_rows = payload.get("bundle_validation", payload)
    out: dict[str, list[str]] = {}
    for bundle_name, bundle_payload in bundle_rows.items():
        atoms = [str(atom) for atom in bundle_payload.get("member_atoms_validation", []) if str(atom).strip()]
        if len(atoms) >= int(min_bundle_atoms):
            out[str(bundle_name)] = atoms
    return dict(sorted(out.items()))


def attach_atom_and_bundle_deltas(
    df_pairs: pd.DataFrame,
    *,
    text_atom_scores: pd.DataFrame,
    bundle_members: dict[str, list[str]],
) -> pd.DataFrame:
    """Attach pairwise ``llm - human`` atom and bundle deltas to collapsed trials."""

    atom_cols = [
        col
        for col in text_atom_scores.columns
        if col not in {"text_id", "text", "item_type", "source", "title", "word_count"}
    ]
    score_map = {
        str(row.text_id): {atom: float(getattr(row, atom)) for atom in atom_cols}
        for row in text_atom_scores.itertuples(index=False)
    }

    out = df_pairs.copy()
    human_ids = out["human_text"].astype(str).map(_sha1_hex)
    llm_ids = out["llm_text"].astype(str).map(_sha1_hex)

    for atom in atom_cols:
        out[f"atom__{atom}"] = [
            float(score_map.get(llm_id, {}).get(atom, 0.0)) - float(score_map.get(human_id, {}).get(atom, 0.0))
            for llm_id, human_id in zip(llm_ids, human_ids, strict=True)
        ]

    for bundle_name, atoms in bundle_members.items():
        cols = [f"atom__{atom}" for atom in atoms if f"atom__{atom}" in out.columns]
        if not cols:
            continue
        out[f"bundle__{bundle_name}"] = out[cols].mean(axis=1)

    return out


def _bootstrap_signed_effect(x: np.ndarray, y: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    vals = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, x.size, size=x.size)
        vals.append(_signed_effect(x[idx], y[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _standardize(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    if std <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - float(np.mean(x))) / std


def _signed_effect(x: np.ndarray, y: np.ndarray) -> float:
    z = _standardize(x)
    direction = np.where(y.astype(int) == 1, 1.0, -1.0)
    return float(np.mean(z * direction))


def _effect_metrics(
    x: np.ndarray,
    y: np.ndarray,
    margin: np.ndarray | None,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    if margin is not None:
        mask = mask & np.isfinite(margin)
    x = x[mask]
    y = y[mask].astype(int)
    margin = None if margin is None else margin[mask]
    if x.size == 0:
        return {
            "n_pairs": 0,
            "mean_llm_minus_human_delta": 0.0,
            "signed_effect_z": 0.0,
            "signed_effect_ci_95_low": 0.0,
            "signed_effect_ci_95_high": 0.0,
            "auc_llm_choice": None,
            "spearman_with_llm_margin": None,
            "mean_delta_when_llm_chosen": None,
            "mean_delta_when_human_chosen": None,
            "status": "insufficient",
        }
    auc = None
    if len(np.unique(y)) == 2:
        try:
            auc = float(roc_auc_score(y, x))
        except ValueError:
            auc = None
    rho = None
    if margin is not None and x.size >= 2 and not np.allclose(np.std(x), 0.0) and not np.allclose(np.std(margin), 0.0):
        rho = spearmanr(x, margin).statistic
        if pd.isna(rho):
            rho = None
        elif rho is not None:
            rho = float(rho)
    rng = np.random.default_rng(int(seed))
    signed = _signed_effect(x, y)
    ci_low, ci_high = _bootstrap_signed_effect(x, y, n_bootstrap=n_bootstrap, rng=rng)
    if ci_low > 0.0 or ci_high < 0.0:
        status = "relevant"
    else:
        status = "exploratory"
    return {
        "n_pairs": int(x.size),
        "mean_llm_minus_human_delta": float(np.mean(x)),
        "signed_effect_z": float(signed),
        "signed_effect_ci_95_low": float(ci_low),
        "signed_effect_ci_95_high": float(ci_high),
        "auc_llm_choice": auc,
        "spearman_with_llm_margin": rho,
        "mean_delta_when_llm_chosen": float(np.mean(x[y == 1])) if np.any(y == 1) else None,
        "mean_delta_when_human_chosen": float(np.mean(x[y == 0])) if np.any(y == 0) else None,
        "status": status,
    }


def build_ecological_effect_tables(
    df_pairs: pd.DataFrame,
    *,
    atom_prefix: str = "atom__",
    bundle_prefix: str = "bundle__",
    n_bootstrap: int = 500,
    seed: int = DEFAULT_SEED,
    bundle_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build D3 atom- and bundle-level ecological effect tables."""

    atom_cols = [col for col in df_pairs.columns if col.startswith(atom_prefix)]
    bundle_cols = [col for col in df_pairs.columns if col.startswith(bundle_prefix)]
    y = df_pairs["y_llm_chosen"].to_numpy(dtype=int)
    margin = df_pairs["llm_margin_pair"].to_numpy(dtype=float) if "llm_margin_pair" in df_pairs.columns else None

    def _build_table(cols: list[str], *, kind: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for col in cols:
            name = col[len(atom_prefix):] if kind == "atom" else col[len(bundle_prefix):]
            payload = _effect_metrics(
                df_pairs[col].to_numpy(dtype=float),
                y,
                margin,
                n_bootstrap=int(n_bootstrap),
                seed=int(seed),
            )
            by_item_type = {}
            for item_type, grp in df_pairs.groupby("item_type"):
                g_margin = grp["llm_margin_pair"].to_numpy(dtype=float) if "llm_margin_pair" in grp.columns else None
                by_item_type[str(item_type)] = _effect_metrics(
                    grp[col].to_numpy(dtype=float),
                    grp["y_llm_chosen"].to_numpy(dtype=int),
                    g_margin,
                    n_bootstrap=int(n_bootstrap),
                    seed=int(seed),
                )
            payload["by_item_type"] = by_item_type
            if kind == "bundle" and bundle_metadata is not None:
                meta = dict((bundle_metadata.get("bundle_validation", bundle_metadata) or {}).get(name, {}))
                if meta:
                    payload["d2_status"] = meta.get("status")
                    payload["d2_n_atoms_validation"] = meta.get("n_atoms_validation")
                    payload["d2_member_atoms_validation"] = meta.get("member_atoms_validation")
            out[name] = payload
        return out

    return _build_table(atom_cols, kind="atom"), _build_table(bundle_cols, kind="bundle")
