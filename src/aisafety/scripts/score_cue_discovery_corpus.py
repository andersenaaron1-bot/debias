"""Score a cue-discovery corpus with surface features and weak cue labels.

Example:
  python -m aisafety.scripts.score_cue_discovery_corpus ^
    --input-jsonl data\\derived\\cue_discovery\\corpus.jsonl ^
    --out-jsonl data\\derived\\cue_discovery\\corpus_scored.jsonl ^
    --summary-json data\\derived\\cue_discovery\\scored_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aisafety.config import DATA_DIR, DEFAULT_SEED
from aisafety.features.surface_cues import SURFACE_FEATURE_NAMES, extract_surface_features


UNIPOLAR_FAMILY_SPECS: dict[str, dict[str, float]] = {
    "academic_formality": {
        "academic_phrase_rate": 1.2,
        "nominalization_rate": 0.8,
        "long_word_ratio": 0.6,
        "contraction_rate": -0.8,
        "second_person_rate": -0.7,
        "promo_phrase_rate": -0.4,
    },
    "safety_corporate_tone": {
        "safety_phrase_rate": 1.2,
        "template_phrase_rate": 0.5,
        "second_person_rate": 0.3,
        "certainty_rate": 0.2,
    },
    "promotional_sales_tone": {
        "promo_phrase_rate": 1.3,
        "second_person_rate": 0.8,
        "exclamation_rate": 0.5,
        "certainty_rate": 0.3,
    },
    "narrative_packaging": {
        "narrative_phrase_rate": 1.3,
        "third_person_pronoun_rate": 0.6,
        "quote_rate": 0.3,
        "proper_case_rate": 0.2,
    },
    "template_boilerplate": {
        "template_phrase_rate": 1.3,
        "discourse_marker_rate": 0.5,
        "markdown_emphasis_rate": 0.4,
        "first_person_plural_rate": 0.3,
    },
}

BIPOLAR_FAMILY_SPECS: dict[str, dict[str, float]] = {
    "verbosity_compression": {
        "word_count": 1.1,
        "avg_sentence_len_words": 0.7,
        "comma_rate": 0.4,
        "bullet_line_ratio": -0.5,
        "newline_rate": -0.2,
    },
    "hedging_certainty": {
        "hedge_rate": 1.0,
        "certainty_rate": -1.0,
    },
}

BIPOLAR_LABELS = {
    "verbosity_compression": ("verbose", "compressed"),
    "hedging_certainty": ("hedged", "certain"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument(
        "--out-jsonl",
        type=Path,
        default=DATA_DIR / "derived" / "cue_discovery" / "corpus_scored.jsonl",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=DATA_DIR / "derived" / "cue_discovery" / "scored_summary.json",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--label-quantile", type=float, default=0.8)
    p.add_argument("--top-k-features", type=int, default=20)
    return p.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _zscore_by_item_type(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        zcol = f"z_{col}"
        out[zcol] = 0.0
        for _, grp in out.groupby("item_type"):
            mean = float(grp[col].mean())
            std = float(grp[col].std(ddof=0))
            if std <= 1e-12:
                out.loc[grp.index, zcol] = 0.0
            else:
                out.loc[grp.index, zcol] = (grp[col] - mean) / std
    return out


def _apply_family_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for family, spec in {**UNIPOLAR_FAMILY_SPECS, **BIPOLAR_FAMILY_SPECS}.items():
        score = np.zeros(len(out), dtype=float)
        for feat, weight in spec.items():
            score += float(weight) * out[f"z_{feat}"].to_numpy(dtype=float)
        out[f"{family}_score"] = score
    return out


def _quantile_thresholds(scores: pd.Series, *, q: float) -> tuple[float, float]:
    hi = float(scores.quantile(float(q)))
    lo = float(scores.quantile(1.0 - float(q)))
    return lo, hi


def _assign_weak_labels(df: pd.DataFrame, *, q: float) -> pd.DataFrame:
    out = df.copy()
    for family in UNIPOLAR_FAMILY_SPECS:
        label_col = f"{family}_label"
        id_col = f"{family}_label_id"
        out[label_col] = "neutral"
        out[id_col] = 0
        for _, grp in out.groupby("item_type"):
            _, hi = _quantile_thresholds(grp[f"{family}_score"], q=q)
            idx = grp.index[grp[f"{family}_score"] >= hi]
            out.loc[idx, label_col] = "high"
            out.loc[idx, id_col] = 1

    for family in BIPOLAR_FAMILY_SPECS:
        label_col = f"{family}_label"
        id_col = f"{family}_label_id"
        pos_label, neg_label = BIPOLAR_LABELS[family]
        out[label_col] = "neutral"
        out[id_col] = 0
        for _, grp in out.groupby("item_type"):
            lo, hi = _quantile_thresholds(grp[f"{family}_score"], q=q)
            hi_idx = grp.index[grp[f"{family}_score"] >= hi]
            lo_idx = grp.index[grp[f"{family}_score"] <= lo]
            out.loc[hi_idx, label_col] = pos_label
            out.loc[hi_idx, id_col] = 1
            out.loc[lo_idx, label_col] = neg_label
            out.loc[lo_idx, id_col] = -1
    return out


def _fit_probe(df: pd.DataFrame, feature_cols: list[str], *, seed: int, top_k: int) -> dict[str, Any]:
    train = df[df["split"] == "train"].copy()
    eval_df = df[df["split"] != "train"].copy()
    if train.empty or eval_df.empty:
        return {"error": "train/eval splits are empty"}

    train_y = (train["source"] == "llm").astype(int).to_numpy()
    eval_y = (eval_df["source"] == "llm").astype(int).to_numpy()

    if len(np.unique(train_y)) < 2 or len(np.unique(eval_y)) < 2:
        return {"error": "probe requires both classes in train and eval"}

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=int(seed),
                    class_weight="balanced",
                ),
            ),
        ]
    )

    train_x = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    eval_x = eval_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    pipe.fit(train_x, train_y)
    eval_pred = pipe.predict(eval_x)
    eval_prob = pipe.predict_proba(eval_x)[:, 1]
    clf: LogisticRegression = pipe.named_steps["clf"]
    coef = clf.coef_[0]
    order = np.argsort(coef)

    top_negative = [
        {"feature": feature_cols[i], "coef": float(coef[i])}
        for i in order[: int(top_k)]
    ]
    top_positive = [
        {"feature": feature_cols[i], "coef": float(coef[i])}
        for i in order[-int(top_k) :][::-1]
    ]

    return {
        "n_train": int(len(train)),
        "n_eval": int(len(eval_df)),
        "eval_splits": sorted({str(s) for s in eval_df["split"].unique().tolist()}),
        "accuracy": float(accuracy_score(eval_y, eval_pred)),
        "roc_auc": float(roc_auc_score(eval_y, eval_prob)),
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
    }


def _fit_probe_by_item_type(df: pd.DataFrame, feature_cols: list[str], *, seed: int, top_k: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item_type, grp in df.groupby("item_type"):
        out[str(item_type)] = _fit_probe(grp, feature_cols, seed=seed, top_k=top_k)
    return out


def _summarize_label_prevalence(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in list(UNIPOLAR_FAMILY_SPECS) + list(BIPOLAR_FAMILY_SPECS):
        label_col = f"{family}_label"
        fam_summary: dict[str, Any] = {}
        for item_type, grp in df.groupby("item_type"):
            fam_summary[str(item_type)] = {
                str(label): int(count)
                for label, count in grp[label_col].value_counts().sort_index().items()
            }
        out[family] = fam_summary
    return out


def _make_output_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_cols = list(SURFACE_FEATURE_NAMES)
    families = list(UNIPOLAR_FAMILY_SPECS) + list(BIPOLAR_FAMILY_SPECS)
    keep_cols = (
        "example_id",
        "group_id",
        "split",
        "item_type",
        "dataset",
        "subset",
        "source",
        "title",
        "text",
        "generator",
        "prompt_name",
        "question",
        "meta",
    )
    for row in df.to_dict(orient="records"):
        out = {key: row[key] for key in keep_cols if key in row}
        out["features"] = {name: float(row[name]) for name in feature_cols}
        out["cue_scores"] = {family: float(row[f"{family}_score"]) for family in families}
        out["weak_labels"] = {family: row[f"{family}_label"] for family in families}
        out["weak_label_ids"] = {family: int(row[f"{family}_label_id"]) for family in families}
        rows.append(out)
    return rows


def main() -> None:
    args = parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(Path(args.input_jsonl))
    if not rows:
        raise ValueError(f"No rows found in {args.input_jsonl}")

    feat_rows: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        merged.update(extract_surface_features(str(row.get("text") or "")))
        feat_rows.append(merged)

    df = pd.DataFrame(feat_rows)
    feature_cols = list(SURFACE_FEATURE_NAMES)
    df = _zscore_by_item_type(df, feature_cols)
    df = _apply_family_scores(df)
    df = _assign_weak_labels(df, q=float(args.label_quantile))

    probe_overall = _fit_probe(df, feature_cols, seed=int(args.seed), top_k=int(args.top_k_features))
    probe_by_item_type = _fit_probe_by_item_type(df, feature_cols, seed=int(args.seed), top_k=int(args.top_k_features))

    out_rows = _make_output_rows(df)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "seed": int(args.seed),
        "label_quantile": float(args.label_quantile),
        "n_records": int(len(df)),
        "by_dataset_source": {
            str(dataset): {
                str(source): int(count)
                for source, count in grp["source"].value_counts().sort_index().items()
            }
            for dataset, grp in df.groupby("dataset")
        },
        "by_item_type_source": {
            str(item_type): {
                str(source): int(count)
                for source, count in grp["source"].value_counts().sort_index().items()
            }
            for item_type, grp in df.groupby("item_type")
        },
        "feature_probe_overall": probe_overall,
        "feature_probe_by_item_type": probe_by_item_type,
        "label_prevalence": _summarize_label_prevalence(df),
    }
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote scored corpus to {args.out_jsonl}")
    print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
