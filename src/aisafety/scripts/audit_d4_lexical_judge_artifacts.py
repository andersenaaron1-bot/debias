"""Mine lexical and edit-level artifacts associated with D4 judge margins."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score

from aisafety.config import PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text, token_count
from aisafety.mech.d4_io import read_jsonl, resolve_path, write_json


DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_lexical_judge_artifact_audit_v1"
WORD_TOKEN_PATTERN = r"(?u)\b[\w][\w'-]*\b"
EDIT_TOKEN_PATTERN = re.compile(r"[\w][\w'-]*|[^\w\s]", flags=re.UNICODE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--hllm-pairs-jsonl",
        action="append",
        default=[],
        help="Text-bearing human-vs-LLM pairs JSONL. May be passed multiple times.",
    )
    parser.add_argument(
        "--hllm-summary",
        action="append",
        default=[],
        help="HLLM stage-summary directory as LABEL=DIR. May be passed multiple times.",
    )
    parser.add_argument(
        "--hllm-target-contrast",
        action="append",
        default=[],
        help="Derived HLLM target as NAME=LEFT,RIGHT where LEFT and RIGHT are generated target names.",
    )
    parser.add_argument("--surface-counterfactual-jsonl", type=Path, default=None)
    parser.add_argument(
        "--surface-summary",
        action="append",
        default=[],
        help="Surface template-summary directory as LABEL=DIR. May be passed multiple times.",
    )
    parser.add_argument("--word-ngram-max", type=int, default=3)
    parser.add_argument("--char-ngram-min", type=int, default=3)
    parser.add_argument("--char-ngram-max", type=int, default=5)
    parser.add_argument("--word-max-features", type=int, default=12000)
    parser.add_argument("--char-max-features", type=int, default=12000)
    parser.add_argument("--word-min-df", type=int, default=5)
    parser.add_argument("--char-min-df", type=int, default=10)
    parser.add_argument("--elastic-alpha", type=float, default=0.001)
    parser.add_argument("--elastic-l1-ratio", type=float, default=0.8)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--max-fragment-tokens", type=int, default=4)
    parser.add_argument("--surface-fragment-min-df", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError(f"Expected nonempty LABEL=PATH, got: {value}")
    return label, Path(raw_path)


def _summary_csv(path: Path, filename: str) -> Path | None:
    if path.is_file() and path.name == filename:
        return path
    candidate = path / filename
    return candidate if candidate.is_file() and candidate.stat().st_size > 0 else None


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2 or np.std(y_true) <= 1e-12:
        return None
    return float(r2_score(y_true, y_pred))


def _target_contrast(value: str) -> tuple[str, str, str]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=LEFT,RIGHT for target contrast, got: {value}")
    name, expr = value.split("=", 1)
    parts = expr.split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected NAME=LEFT,RIGHT for target contrast, got: {value}")
    left, right = parts
    if not name.strip() or not left.strip() or not right.strip():
        raise ValueError(f"Expected nonempty NAME=LEFT,RIGHT, got: {value}")
    return name.strip(), left.strip(), right.strip()


def _pair_text_rows(paths: list[Path]) -> pd.DataFrame:
    by_pair: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            pair_id = str(row.get("pair_id") or "")
            human_text = flat_text(str(row.get("human_text") or ""))
            llm_text = flat_text(str(row.get("llm_text") or ""))
            if not pair_id or not human_text or not llm_text:
                continue
            by_pair.setdefault(
                pair_id,
                {
                    "pair_id": pair_id,
                    "source_dataset": str(row.get("source_dataset") or ""),
                    "subset": str(row.get("subset") or ""),
                    "item_type": str(row.get("item_type") or ""),
                    "human_text": human_text,
                    "llm_text": llm_text,
                    "human_tokens": int(row.get("human_tokens") or token_count(human_text)),
                    "llm_tokens": int(row.get("llm_tokens") or token_count(llm_text)),
                },
            )
    if not by_pair:
        raise ValueError("No text-bearing HLLM pairs were loaded.")
    return pd.DataFrame(by_pair.values()).sort_values("pair_id").reset_index(drop=True)


def _load_hllm_targets(summary_inputs: list[tuple[str, Path]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for input_label, path in summary_inputs:
        csv_path = _summary_csv(path, "stage_pair_summary_long.csv")
        if csv_path is None:
            raise FileNotFoundError(f"Missing stage_pair_summary_long.csv in {path}")
        df = pd.read_csv(csv_path)
        required = {"pair_id", "run_label", "mean_llm_margin"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")
        keep = df[["pair_id", "run_label", "mean_llm_margin"]].copy()
        keep["pair_id"] = keep["pair_id"].astype(str)
        keep["target_name"] = input_label + "::" + keep["run_label"].astype(str)
        keep["target_value"] = pd.to_numeric(keep["mean_llm_margin"], errors="coerce")
        rows.append(keep[["pair_id", "target_name", "target_value"]])
    if not rows:
        return pd.DataFrame(columns=["pair_id", "target_name", "target_value"])
    return pd.concat(rows, ignore_index=True).dropna(subset=["target_value"])


def _add_hllm_target_contrasts(targets: pd.DataFrame, contrasts: list[str]) -> pd.DataFrame:
    if not contrasts:
        return targets
    wide = targets.pivot_table(index="pair_id", columns="target_name", values="target_value", aggfunc="mean")
    rows = [targets]
    for raw in contrasts:
        name, left, right = _target_contrast(raw)
        if left not in wide.columns or right not in wide.columns:
            raise ValueError(f"Target contrast {raw!r} references unavailable targets.")
        values = (wide[left] - wide[right]).dropna().rename("target_value").reset_index()
        values["target_name"] = name
        rows.append(values[["pair_id", "target_name", "target_value"]])
    return pd.concat(rows, ignore_index=True)


def _vectorize_hllm_pairs(
    pair_df: pd.DataFrame,
    *,
    word_ngram_max: int,
    char_ngram_min: int,
    char_ngram_max: int,
    word_max_features: int,
    char_max_features: int,
    word_min_df: int,
    char_min_df: int,
) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    all_text = pd.concat([pair_df["human_text"], pair_df["llm_text"]], ignore_index=True).astype(str)
    matrices: list[sparse.csr_matrix] = []
    names: list[str] = []
    kinds: list[str] = []
    if int(word_max_features) > 0:
        word = CountVectorizer(
            lowercase=True,
            binary=True,
            ngram_range=(1, max(int(word_ngram_max), 1)),
            token_pattern=WORD_TOKEN_PATTERN,
            min_df=max(int(word_min_df), 1),
            max_features=int(word_max_features),
        )
        word.fit(all_text)
        matrices.append((word.transform(pair_df["llm_text"]) - word.transform(pair_df["human_text"])).tocsr())
        word_names = [f"word::{item}" for item in word.get_feature_names_out()]
        names.extend(word_names)
        kinds.extend(["word"] * len(word_names))
    if int(char_max_features) > 0:
        char = CountVectorizer(
            lowercase=True,
            binary=True,
            analyzer="char",
            ngram_range=(max(int(char_ngram_min), 1), max(int(char_ngram_max), int(char_ngram_min))),
            min_df=max(int(char_min_df), 1),
            max_features=int(char_max_features),
        )
        char.fit(all_text)
        matrices.append((char.transform(pair_df["llm_text"]) - char.transform(pair_df["human_text"])).tocsr())
        char_names = [f"char::{item}" for item in char.get_feature_names_out()]
        names.extend(char_names)
        kinds.extend(["char"] * len(char_names))
    if not matrices:
        raise ValueError("At least one HLLM vectorizer must be enabled.")
    return sparse.hstack(matrices, format="csr"), names, kinds


def _controls(pair_df: pd.DataFrame) -> np.ndarray:
    frame = pd.DataFrame(index=pair_df.index)
    human_tokens = pd.to_numeric(pair_df["human_tokens"], errors="coerce").fillna(0.0)
    llm_tokens = pd.to_numeric(pair_df["llm_tokens"], errors="coerce").fillna(0.0)
    frame["length_delta"] = llm_tokens - human_tokens
    frame["log_length_ratio"] = np.log((llm_tokens + 1.0) / (human_tokens + 1.0))
    for col in ("source_dataset", "subset", "item_type"):
        if col in pair_df.columns:
            dummies = pd.get_dummies(pair_df[col].fillna("").astype(str), prefix=col, dtype=float)
            frame = pd.concat([frame, dummies], axis=1)
    numeric = frame.astype(float)
    for col in ("length_delta", "log_length_ratio"):
        std = float(numeric[col].std())
        if std > 1e-12:
            numeric[col] = (numeric[col] - float(numeric[col].mean())) / std
    return np.column_stack([np.ones(len(numeric), dtype=float), numeric.to_numpy(dtype=float)])


def _residual_target(y: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, singular_values, _ = np.linalg.svd(controls, full_matrices=False)
    if not len(singular_values):
        return y.copy(), np.empty((len(y), 0), dtype=float)
    tolerance = max(controls.shape) * np.finfo(float).eps * float(singular_values[0])
    q = u[:, singular_values > tolerance]
    residual = y - q @ (q.T @ y)
    return residual, q


def _univariate_coefficients(
    x: sparse.csr_matrix,
    y_residual: np.ndarray,
    q: np.ndarray,
    *,
    feature_names: list[str],
    feature_kinds: list[str],
    target_name: str,
) -> pd.DataFrame:
    x = x.tocsr()
    dot = np.asarray(x.T @ y_residual).reshape(-1)
    x_sq = np.asarray(x.multiply(x).sum(axis=0)).reshape(-1)
    projected = np.asarray(q.T @ x)
    residual_x_sq = np.maximum(x_sq - np.square(projected).sum(axis=0), 0.0)
    y_sq = float(np.square(y_residual).sum())
    denom = np.sqrt(residual_x_sq * max(y_sq, 0.0))
    corr = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 1e-12)
    slope = np.divide(dot, residual_x_sq, out=np.zeros_like(dot), where=residual_x_sq > 1e-12)
    support = np.asarray((x != 0).sum(axis=0)).reshape(-1)
    mean_diff = np.asarray(x.mean(axis=0)).reshape(-1)
    return pd.DataFrame(
        {
            "target_name": target_name,
            "artifact_name": feature_names,
            "artifact_kind": feature_kinds,
            "support_pairs": support.astype(int),
            "mean_llm_minus_human_presence": mean_diff,
            "partial_corr": corr,
            "abs_partial_corr": np.abs(corr),
            "partial_slope": slope,
        }
    )


def _fold_id(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % max(int(folds), 1)


def _fit_elastic(
    x: sparse.csr_matrix,
    y: np.ndarray,
    controls: np.ndarray,
    pair_ids: list[str],
    *,
    alpha: float,
    l1_ratio: float,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    fold_ids = np.asarray([_fold_id(pair_id, folds, seed) for pair_id in pair_ids], dtype=int)
    for fold in sorted(set(fold_ids)):
        train = fold_ids != fold
        test = fold_ids == fold
        if int(train.sum()) < 5 or int(test.sum()) < 2:
            continue
        control_coef, *_ = np.linalg.lstsq(controls[train], y[train], rcond=None)
        y_train_residual = y[train] - controls[train] @ control_coef
        y_test_residual = y[test] - controls[test] @ control_coef
        model = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=True,
            max_iter=5000,
            random_state=int(seed),
            selection="cyclic",
        )
        model.fit(x[train], y_train_residual)
        pred = model.predict(x[test])
        rows.append(
            {
                "fold": int(fold),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "r2": _safe_r2(y_test_residual, pred),
                "pearson": _pearson(y_test_residual, pred),
                "mae": float(mean_absolute_error(y_test_residual, pred)),
                "baseline_mae": float(mean_absolute_error(y_test_residual, np.zeros(int(test.sum())))),
            }
        )
    y_residual, _ = _residual_target(y, controls)
    final = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=True,
        max_iter=5000,
        random_state=int(seed),
        selection="cyclic",
    )
    final.fit(x, y_residual)
    return np.asarray(final.coef_, dtype=float), rows


def audit_hllm_artifacts(
    pair_df: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_df = pair_df.copy().sort_values("pair_id").reset_index(drop=True)
    x, feature_names, feature_kinds = _vectorize_hllm_pairs(
        pair_df,
        word_ngram_max=args.word_ngram_max,
        char_ngram_min=args.char_ngram_min,
        char_ngram_max=args.char_ngram_max,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        word_min_df=args.word_min_df,
        char_min_df=args.char_min_df,
    )
    controls = _controls(pair_df)
    pair_index = {pair_id: idx for idx, pair_id in enumerate(pair_df["pair_id"].astype(str))}
    coefficient_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for target_name, target_df in targets.groupby("target_name", sort=True):
        target_df = target_df.groupby("pair_id", sort=True)["target_value"].mean().reset_index()
        target_df = target_df[target_df["pair_id"].astype(str).isin(pair_index)].copy()
        if len(target_df) < max(int(args.cv_folds) * 2, 10):
            continue
        indices = np.asarray([pair_index[str(item)] for item in target_df["pair_id"]], dtype=int)
        x_target = x[indices]
        y = pd.to_numeric(target_df["target_value"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y)
        indices = indices[valid]
        x_target = x_target[valid]
        y = y[valid]
        if len(y) < max(int(args.cv_folds) * 2, 10) or np.std(y) <= 1e-12:
            continue
        y_residual, q = _residual_target(y, controls[indices])
        coef_df = _univariate_coefficients(
            x_target,
            y_residual,
            q,
            feature_names=feature_names,
            feature_kinds=feature_kinds,
            target_name=str(target_name),
        )
        elastic_coef, cv_rows = _fit_elastic(
            x_target,
            y,
            controls[indices],
            target_df.loc[valid, "pair_id"].astype(str).tolist(),
            alpha=args.elastic_alpha,
            l1_ratio=args.elastic_l1_ratio,
            folds=args.cv_folds,
            seed=args.seed,
        )
        coef_df["elastic_coef"] = elastic_coef
        coef_df["abs_elastic_coef"] = np.abs(elastic_coef)
        coef_df["artifact_rank"] = coef_df["abs_partial_corr"].rank(method="first", ascending=False).astype(int)
        coefficient_rows.append(coef_df.nsmallest(max(int(args.top_k), 1), "artifact_rank"))
        for row in cv_rows:
            metric_rows.append({"target_name": str(target_name), **row})
        target_rows.append(
            {
                "target_name": str(target_name),
                "n_pairs": int(len(y)),
                "target_mean": float(y.mean()),
                "target_std": float(y.std()),
                "residual_std": float(y_residual.std()),
                "n_features": int(len(feature_names)),
                "n_nonzero_elastic": int(np.count_nonzero(np.abs(elastic_coef) > 1e-12)),
            }
        )
    coefficients = pd.concat(coefficient_rows, ignore_index=True) if coefficient_rows else pd.DataFrame()
    return coefficients, pd.DataFrame(metric_rows), pd.DataFrame(target_rows)


def _edit_tokens(text: str) -> list[str]:
    return [item.lower() for item in EDIT_TOKEN_PATTERN.findall(flat_text(text))]


def _fragment_ngrams(tokens: list[str], *, max_tokens: int) -> set[str]:
    out: set[str] = set()
    for width in range(1, min(max(int(max_tokens), 1), len(tokens)) + 1):
        for start in range(0, len(tokens) - width + 1):
            fragment = " ".join(tokens[start : start + width]).strip()
            if fragment:
                out.add(fragment)
    return out


def _counterfactual_fragments(rows: list[dict[str, Any]], *, max_tokens: int) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for row in rows:
        base = _edit_tokens(str(row.get("base_text") or ""))
        variant = _edit_tokens(str(row.get("variant_text") or ""))
        matcher = SequenceMatcher(a=base, b=variant, autojunk=False)
        features: set[str] = set()
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                continue
            if op in {"delete", "replace"}:
                features.update(f"delete::{frag}" for frag in _fragment_ngrams(base[i1:i2], max_tokens=max_tokens))
            if op in {"insert", "replace"}:
                features.update(f"insert::{frag}" for frag in _fragment_ngrams(variant[j1:j2], max_tokens=max_tokens))
        for artifact_name in sorted(features):
            out.append(
                {
                    "counterfactual_id": str(row.get("counterfactual_id") or ""),
                    "pair_id": str(row.get("pair_id") or ""),
                    "source_dataset": str(row.get("source_dataset") or ""),
                    "subset": str(row.get("subset") or ""),
                    "item_type": str(row.get("item_type") or ""),
                    "axis": str(row.get("axis") or ""),
                    "direction": str(row.get("direction") or ""),
                    "role": str(row.get("role") or ""),
                    "transform_id": str(row.get("transform_id") or ""),
                    "artifact_name": artifact_name,
                }
            )
    return pd.DataFrame(out)


def _load_surface_targets(summary_inputs: list[tuple[str, Path]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    specs = (
        ("bt_pair_summary_long.csv", "mean_cue_plus_margin", ("template_label", "run_label"), "cue_margin"),
        ("stage_contrast_pair_deltas.csv", "delta_cue_plus_margin", ("template_label", "contrast"), "stage_delta"),
        ("template_sensitivity_pair_deltas.csv", "template_delta_cue_plus_margin", ("template_contrast", "run_label"), "template_delta"),
        (
            "template_stage_interaction_pair_deltas.csv",
            "stage_template_interaction_cue_plus_margin",
            ("stage_contrast", "template_contrast"),
            "interaction",
        ),
    )
    for input_label, path in summary_inputs:
        for filename, value_col, dynamic_cols, family in specs:
            csv_path = _summary_csv(path, filename)
            if csv_path is None:
                continue
            df = pd.read_csv(csv_path)
            if "counterfactual_id" not in df.columns or value_col not in df.columns:
                continue
            selected = df[["counterfactual_id", value_col, *[col for col in dynamic_cols if col in df.columns]]].copy()
            dynamic = selected[[col for col in dynamic_cols if col in selected.columns]].fillna("").astype(str)
            suffix = dynamic.apply(lambda row: "::".join(item for item in row if item), axis=1)
            selected["target_name"] = input_label + "::" + family + "::" + suffix
            selected["target_value"] = pd.to_numeric(selected[value_col], errors="coerce")
            frames.append(selected[["counterfactual_id", "target_name", "target_value"]])
    if not frames:
        return pd.DataFrame(columns=["counterfactual_id", "target_name", "target_value"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["target_value"])


def audit_surface_fragments(
    counterfactual_rows: list[dict[str, Any]],
    targets: pd.DataFrame,
    *,
    max_fragment_tokens: int,
    min_df: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fragments = _counterfactual_fragments(counterfactual_rows, max_tokens=max_fragment_tokens)
    if fragments.empty or targets.empty:
        return pd.DataFrame(), fragments
    fragment_support = fragments.groupby("artifact_name")["counterfactual_id"].nunique()
    keep_artifacts = set(fragment_support[fragment_support >= max(int(min_df), 1)].index.astype(str))
    fragments = fragments[fragments["artifact_name"].astype(str).isin(keep_artifacts)].copy()
    if fragments.empty:
        return pd.DataFrame(), fragments
    merged = fragments.merge(targets, on="counterfactual_id", how="inner")
    strata = ["target_name", "source_dataset", "subset", "item_type", "axis", "direction", "role"]
    merged["stratum_adjusted_target_value"] = merged["target_value"] - merged.groupby(
        strata,
        dropna=False,
    )["target_value"].transform("mean")
    rows: list[dict[str, Any]] = []
    global_target = targets.groupby("target_name", sort=True)["target_value"].mean()
    for (target_name, artifact_name), group in merged.groupby(["target_name", "artifact_name"], sort=True):
        values = pd.to_numeric(group["target_value"], errors="coerce").dropna()
        if values.empty:
            continue
        global_mean = float(global_target.get(target_name, math.nan))
        rows.append(
            {
                "target_name": str(target_name),
                "artifact_name": str(artifact_name),
                "support_counterfactuals": int(group["counterfactual_id"].nunique()),
                "support_sources": int(group["source_dataset"].nunique()),
                "support_axes": int(group["axis"].nunique()),
                "support_roles": int(group["role"].nunique()),
                "mean_target_value": float(values.mean()),
                "median_target_value": float(values.median()),
                "mean_abs_target_value": float(values.abs().mean()),
                "positive_rate": float((values > 0).mean()),
                "global_target_mean": global_mean,
                "artifact_minus_global_mean": float(values.mean() - global_mean),
                "abs_artifact_minus_global_mean": float(abs(values.mean() - global_mean)),
                "mean_stratum_adjusted_target_value": float(group["stratum_adjusted_target_value"].mean()),
                "abs_mean_stratum_adjusted_target_value": float(abs(group["stratum_adjusted_target_value"].mean())),
            }
        )
    effects = pd.DataFrame(rows)
    if not effects.empty:
        effects["artifact_rank"] = effects.groupby("target_name")["abs_artifact_minus_global_mean"].rank(
            method="first",
            ascending=False,
        ).astype(int)
    return effects, fragments


def _candidate_artifacts(
    hllm_coefficients: pd.DataFrame,
    surface_effects: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not hllm_coefficients.empty:
        hllm = hllm_coefficients.copy()
        hllm["audit_family"] = "hllm_lexical"
        hllm["signal"] = hllm["partial_corr"]
        hllm["abs_signal"] = hllm["abs_partial_corr"]
        frames.append(hllm)
    if not surface_effects.empty:
        surface = surface_effects[surface_effects["artifact_rank"] <= max(int(top_k), 1)].copy()
        surface["audit_family"] = "surface_edit_fragment"
        surface["signal"] = surface["artifact_minus_global_mean"]
        surface["abs_signal"] = surface["abs_artifact_minus_global_mean"]
        frames.append(surface)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.sort_values(["audit_family", "target_name", "abs_signal"], ascending=[True, True, False])


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    out_dir = _resolve(workspace_root, args.out_dir)
    hllm_pair_paths = [_resolve(workspace_root, Path(item)) for item in args.hllm_pairs_jsonl]
    hllm_inputs = [(label, _resolve(workspace_root, path)) for label, path in map(_parse_label_path, args.hllm_summary)]
    surface_inputs = [(label, _resolve(workspace_root, path)) for label, path in map(_parse_label_path, args.surface_summary)]

    if not hllm_inputs and not surface_inputs:
        raise ValueError("Pass at least one --hllm-summary or --surface-summary input.")
    if hllm_inputs and not hllm_pair_paths:
        raise ValueError("--hllm-summary requires at least one --hllm-pairs-jsonl text source.")
    if surface_inputs and args.surface_counterfactual_jsonl is None:
        raise ValueError("--surface-summary requires --surface-counterfactual-jsonl.")

    outputs: dict[str, str] = {}
    hllm_coefficients = pd.DataFrame()
    hllm_metrics = pd.DataFrame()
    hllm_targets = pd.DataFrame()
    if hllm_inputs:
        pairs = _pair_text_rows(hllm_pair_paths)
        hllm_target_values = _load_hllm_targets(hllm_inputs)
        hllm_target_values = _add_hllm_target_contrasts(
            hllm_target_values,
            [str(item) for item in args.hllm_target_contrast],
        )
        hllm_coefficients, hllm_metrics, hllm_targets = audit_hllm_artifacts(
            pairs,
            hllm_target_values,
            args=args,
        )
        for name, frame in (
            ("hllm_artifact_coefficients", hllm_coefficients),
            ("hllm_heldout_metrics", hllm_metrics),
            ("hllm_target_summary", hllm_targets),
        ):
            path = out_dir / f"{name}.csv"
            _write_csv(path, frame)
            outputs[f"{name}_csv"] = str(path)

    surface_effects = pd.DataFrame()
    surface_fragments = pd.DataFrame()
    if surface_inputs:
        counterfactual_path = _resolve(workspace_root, args.surface_counterfactual_jsonl)
        counterfactual_rows = read_jsonl(counterfactual_path)
        surface_target_values = _load_surface_targets(surface_inputs)
        surface_effects, surface_fragments = audit_surface_fragments(
            counterfactual_rows,
            surface_target_values,
            max_fragment_tokens=args.max_fragment_tokens,
            min_df=args.surface_fragment_min_df,
        )
        for name, frame in (
            ("surface_edit_fragment_effects", surface_effects),
            ("surface_edit_fragments", surface_fragments),
        ):
            path = out_dir / f"{name}.csv"
            _write_csv(path, frame)
            outputs[f"{name}_csv"] = str(path)

    candidates = _candidate_artifacts(hllm_coefficients, surface_effects, top_k=args.top_k)
    candidate_path = out_dir / "candidate_artifacts.csv"
    _write_csv(candidate_path, candidates)
    outputs["candidate_artifacts_csv"] = str(candidate_path)
    summary_path = out_dir / "summary.json"
    outputs["summary_json"] = str(summary_path)
    write_json(
        summary_path,
        {
            "stage": "D4-lexical-judge-artifact-audit",
            "out_dir": str(out_dir),
            "hllm_pairs_jsonl": [str(path) for path in hllm_pair_paths],
            "hllm_summaries": {label: str(path) for label, path in hllm_inputs},
            "hllm_target_contrasts": [str(item) for item in args.hllm_target_contrast],
            "surface_counterfactual_jsonl": None
            if args.surface_counterfactual_jsonl is None
            else str(_resolve(workspace_root, args.surface_counterfactual_jsonl)),
            "surface_summaries": {label: str(path) for label, path in surface_inputs},
            "settings": {
                "word_ngram_max": int(args.word_ngram_max),
                "char_ngram_min": int(args.char_ngram_min),
                "char_ngram_max": int(args.char_ngram_max),
                "word_max_features": int(args.word_max_features),
                "char_max_features": int(args.char_max_features),
                "word_min_df": int(args.word_min_df),
                "char_min_df": int(args.char_min_df),
                "elastic_alpha": float(args.elastic_alpha),
                "elastic_l1_ratio": float(args.elastic_l1_ratio),
                "cv_folds": int(args.cv_folds),
                "top_k": int(args.top_k),
                "max_fragment_tokens": int(args.max_fragment_tokens),
                "surface_fragment_min_df": int(args.surface_fragment_min_df),
                "seed": int(args.seed),
            },
            "counts": {
                "hllm_artifact_coefficients": int(len(hllm_coefficients)),
                "hllm_heldout_metrics": int(len(hllm_metrics)),
                "hllm_targets": int(len(hllm_targets)),
                "surface_edit_fragment_effects": int(len(surface_effects)),
                "surface_edit_fragments": int(len(surface_fragments)),
                "candidate_artifacts": int(len(candidates)),
            },
            "outputs": outputs,
        },
    )
    print(f"out_dir={out_dir}")
    print(f"hllm_coefficients={len(hllm_coefficients)}")
    print(f"hllm_metrics={len(hllm_metrics)}")
    print(f"surface_fragment_effects={len(surface_effects)}")
    print(f"candidate_artifacts={len(candidates)}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
