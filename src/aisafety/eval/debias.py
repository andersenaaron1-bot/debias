"""Utilities for debiasing paired human-vs-LLM A/B evaluations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def swap_ab_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every ``A_*`` column swapped with ``B_*``."""
    swapped = df.copy()
    for col_a in df.columns:
        if not col_a.startswith("A_"):
            continue
        col_b = "B_" + col_a[2:]
        if col_b not in df.columns:
            continue
        swapped[[col_a, col_b]] = swapped[[col_b, col_a]].to_numpy()
    return swapped


def compute_swap_debiased_logit_diff(logit_diff: np.ndarray, logit_diff_swapped: np.ndarray) -> np.ndarray:
    """Cancel additive A/B position bias using a swapped prompt pass."""
    return np.asarray(logit_diff) - np.asarray(logit_diff_swapped)


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        missing_s = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_s}")


def pair_key_human_llm(df: pd.DataFrame) -> pd.Series:
    """Hash key that groups swapped-order rows for the same human/LLM pair."""
    required = {"item_type", "title", "A_text", "B_text", "A_source", "B_source"}
    _require_columns(df, required)

    a_source = df["A_source"].astype(str)
    b_source = df["B_source"].astype(str)
    if not ((a_source.isin({"human", "llm"}) & b_source.isin({"human", "llm"})).all()):
        raise ValueError("pair_key_human_llm expects A_source/B_source to be in {'human','llm'} for all rows.")

    human_text = df["A_text"].where(a_source == "human", df["B_text"])
    llm_text = df["A_text"].where(a_source == "llm", df["B_text"])

    key_df = pd.DataFrame(
        {
            "item_type": df["item_type"].astype(str),
            "title": df["title"].astype(str),
            "human_text": human_text.astype(str),
            "llm_text": llm_text.astype(str),
        }
    )
    return pd.util.hash_pandas_object(key_df, index=False)


def add_pairwise_debias_columns(
    df: pd.DataFrame,
    *,
    logit_diff_col: str = "logit_diff",
    tie_break: str = "hash",
    seed: int = 0,
) -> pd.DataFrame:
    """Add debiased choice columns using swapped-order trial pairs."""
    required = {"A_source", "B_source", "A_text", "B_text", "item_type", "title", logit_diff_col}
    _require_columns(df, required)

    out = df.copy()
    logit_diff = pd.to_numeric(out[logit_diff_col], errors="coerce")
    a_source = out["A_source"].astype(str)
    a_is_llm = a_source == "llm"

    llm_margin = logit_diff.where(a_is_llm, -logit_diff)
    key = pair_key_human_llm(out)
    llm_margin_pair = llm_margin.groupby(key).transform("mean")

    preferred = np.where(llm_margin_pair > 0, "llm", "human").astype(object)
    ties = llm_margin_pair == 0
    if ties.any():
        if tie_break == "hash":
            pick_llm = ((key.astype("uint64") + np.uint64(seed)) % np.uint64(2)) == 0
            preferred[ties.to_numpy()] = np.where(pick_llm[ties].to_numpy(), "llm", "human")
        else:
            raise ValueError(f"Unknown tie_break: {tie_break}")

    pref_llm = preferred == "llm"
    choice_debiased = np.where(
        pref_llm,
        np.where(a_is_llm, "A", "B"),
        np.where(a_is_llm, "B", "A"),
    )
    chosen_source_debiased = np.where(choice_debiased == "A", out["A_source"], out["B_source"])

    out["llm_margin"] = llm_margin
    out["llm_margin_pair"] = llm_margin_pair
    out["preferred_source_debiased"] = preferred
    out["choice_debiased"] = choice_debiased
    out["chosen_source_debiased"] = chosen_source_debiased
    return out
