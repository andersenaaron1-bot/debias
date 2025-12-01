"""Layer-wise linear probes and choice metrics."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def layerwise_logreg_auc(X, y, C: float = 1.0, n_splits: int = 5, seed: int = 0):
    L = X.shape[1]
    aucs = np.zeros(L, dtype=np.float32)
    coefs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for l in range(L):
        Xi = X[:, l, :]
        fold_aucs = []
        wsum = np.zeros(Xi.shape[1], dtype=np.float32)
        for tr, te in skf.split(Xi, y):
            clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs")
            clf.fit(Xi[tr], y[tr])
            fold_aucs.append(roc_auc_score(y[te], clf.predict_proba(Xi[te])[:, 1]))
            wsum += clf.coef_.ravel().astype(np.float32)
        aucs[l] = float(np.mean(fold_aucs))
        coefs.append(wsum / n_splits)
    return aucs, coefs


def build_choice_indices(df_desc_src, results_src):
    """Map each trial to the row index of its human and llm descriptions."""
    df_desc_src = df_desc_src.copy()
    df_desc_src["idx"] = np.arange(len(df_desc_src))
    idx_map = {(t, s): i for t, s, i in zip(df_desc_src["title"], df_desc_src["source"], df_desc_src["idx"])}

    hum_idx = []
    llm_idx = []
    y_choice = []

    for r in results_src.itertuples(index=False):
        i_hum = idx_map.get((r.title, "human"))
        i_llm = idx_map.get((r.title, "llm"))
        if i_hum is None or i_llm is None:
            continue

        hum_idx.append(i_hum)
        llm_idx.append(i_llm)

        is_llm_chosen = int(
            (r.choice == "A" and r.A_source == "llm") or (r.choice == "B" and r.B_source == "llm")
        )
        y_choice.append(is_llm_chosen)

    return np.array(hum_idx, dtype=int), np.array(llm_idx, dtype=int), np.array(y_choice, dtype=int)


def logreg_auc_single(X, y, C: float = 1.0, n_splits: int = 5, seed: int = 0, max_iter: int = 1000):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    wsum = np.zeros(X.shape[1], dtype=np.float32)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
        wsum += clf.coef_.ravel().astype(np.float32)
    w = wsum / n_splits
    return float(np.mean(aucs)), w


def layerwise_choice_auc(
    X,
    df_desc_src,
    results_src,
    C: float = 1.0,
    n_splits: int = 5,
    seed: int = 0,
    max_iter: int = 1000,
):
    L = X.shape[1]
    hum_idx, llm_idx, y_choice = build_choice_indices(df_desc_src, results_src)

    choice_aucs = np.zeros(L, dtype=np.float32)
    choice_coefs = []

    for l in range(L):
        Xi = X[llm_idx, l, :] - X[hum_idx, l, :]
        auc_l, w = logreg_auc_single(
            Xi, y_choice, C=C, n_splits=n_splits, seed=seed, max_iter=max_iter
        )
        choice_aucs[l] = auc_l
        choice_coefs.append(w.astype(np.float32))

    return choice_aucs, choice_coefs
