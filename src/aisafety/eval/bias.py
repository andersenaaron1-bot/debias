"""Summaries for selector bias."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def summarize_bias(results: pd.DataFrame) -> dict:
    valid = results[results["choice"].isin(["A", "B"])].copy()
    if len(valid) == 0:
        return {
            "n_trials": 0,
            "n_llm_chosen": 0,
            "prop_llm_chosen": float("nan"),
            "binom_p_value": float("nan"),
            "ci_95_low": float("nan"),
            "ci_95_high": float("nan"),
        }
    valid["chosen_source"] = np.where(valid["choice"] == "A", valid["A_source"], valid["B_source"])
    valid["is_llm_chosen"] = (valid["chosen_source"] == "llm").astype(int)
    n = len(valid)
    k = int(valid["is_llm_chosen"].sum())
    bt = binomtest(k, n, 0.5)
    ci = bt.proportion_ci(0.95)
    return {
        "n_trials": n,
        "n_llm_chosen": k,
        "prop_llm_chosen": k / n if n else float("nan"),
        "binom_p_value": float(bt.pvalue),
        "ci_95_low": float(ci.low),
        "ci_95_high": float(ci.high),
    }


def evaluate_by_domain(results_all: pd.DataFrame):
    out = {}
    for t, grp in results_all.groupby("item_type"):
        out[t] = summarize_bias(grp)
    out["overall"] = summarize_bias(results_all)
    return out
