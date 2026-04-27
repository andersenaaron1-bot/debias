"""Small metric helpers shared by mechanistic tracing stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Return ROC AUC or None when labels are degenerate."""

    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return Spearman correlation or None when either side is degenerate."""

    if len(x) == 0 or len(y) == 0:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    value = pd.Series(x).corr(pd.Series(y), method="spearman")
    return None if pd.isna(value) else float(value)

