"""Analyze commitment, geometry, and branch stability in judge reasoning traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, write_json
from aisafety.mech.judge_reasoning import (
    deterministic_group_fold,
    direction_angle_degrees,
    first_persistent_threshold,
    normalize_choice,
)


DEFAULT_TRACE_DIR = Path("artifacts") / "mechanistic" / "judge_reasoning_trajectories_v1"
DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "judge_reasoning_analysis_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument(
        "--probe-targets",
        default=(
            "final_choice,target_option,target_selected,"
            "condition_label,presentation_order"
        ),
    )
    parser.add_argument(
        "--group-columns",
        default=(
            "comparison_dimension,source_dataset,validity_type,"
            "difficulty_tier,analysis_split"
        ),
        help=(
            "Trace or metadata columns used to emit stratum-specific probes in "
            "addition to the all-trace analysis."
        ),
    )
    parser.add_argument("--positive-condition-label", default="")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--probe-c", type=float, default=0.1)
    parser.add_argument("--min-probe-rows", type=int, default=40)
    parser.add_argument("--min-dimension-rows", type=int, default=80)
    parser.add_argument("--commitment-auc", type=float, default=0.75)
    parser.add_argument("--commitment-persistence", type=int, default=3)
    parser.add_argument("--choice-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--target-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


class TraceArtifact:
    """Lazy layer-wise access to sharded trajectory states."""

    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.rows = read_jsonl(self.trace_dir / "traces.jsonl")
        if not self.rows:
            raise ValueError(f"No traces found in {self.trace_dir}")
        self.frame = pd.DataFrame(self.rows)
        with np.load(self.trace_dir / str(self.rows[0]["trajectory_shard"])) as first_shard:
            self.hidden_layers = [int(value) for value in first_shard["hidden_layers"].tolist()]
            self.n_points = int(first_shard["states"].shape[1])
            self.hidden_size = int(first_shard["states"].shape[-1])
        self._locations: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self._locations[str(row["trajectory_shard"])].append(
                (int(index), int(row["trajectory_shard_row"]))
            )
        self.point_mask = np.zeros((len(self.rows), self.n_points), dtype=bool)
        self.step_indices = np.full((len(self.rows), self.n_points), -1, dtype=np.int32)
        self.positions = np.full((len(self.rows), self.n_points), np.nan, dtype=np.float32)
        self.label_margins = np.full((len(self.rows), self.n_points), np.nan, dtype=np.float32)
        for shard_name, locations in self._locations.items():
            with np.load(self.trace_dir / shard_name) as shard:
                for trace_index, shard_row in locations:
                    self.point_mask[trace_index] = shard["point_mask"][shard_row]
                    self.step_indices[trace_index] = shard["step_indices"][shard_row]
                    self.positions[trace_index] = shard["positions"][shard_row]
                    self.label_margins[trace_index] = shard["label_margins"][shard_row]

    def layer_states(self, hidden_layer: int) -> np.ndarray:
        if int(hidden_layer) not in self.hidden_layers:
            raise KeyError(hidden_layer)
        layer_index = self.hidden_layers.index(int(hidden_layer))
        states = np.zeros(
            (len(self.rows), self.n_points, self.hidden_size),
            dtype=np.float32,
        )
        for shard_name, locations in self._locations.items():
            with np.load(self.trace_dir / shard_name) as shard:
                layer = shard["states"][:, :, layer_index, :].astype(np.float32)
                for trace_index, shard_row in locations:
                    states[trace_index] = layer[shard_row]
        return states


def _binary_target(
    frame: pd.DataFrame,
    *,
    target: str,
    positive_condition_label: str,
    scope: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    scope_mask = (
        np.ones((len(frame),), dtype=bool)
        if scope is None
        else np.asarray(scope, dtype=bool)
    )
    if target == "final_choice":
        values = frame.get("final_choice", pd.Series([""] * len(frame))).map(normalize_choice)
        valid = values.isin(["A", "B"]).to_numpy() & scope_mask
        return valid, (values.to_numpy() == "A").astype(int), "A"
    if target == "target_selected":
        values = frame.get("target_selected", pd.Series([None] * len(frame)))
        valid = values.notna().to_numpy() & scope_mask
        return valid, values.eq(True).astype(int).to_numpy(), "true"
    if target == "target_option":
        values = frame.get("target_option", pd.Series([""] * len(frame))).map(normalize_choice)
        valid = values.isin(["A", "B"]).to_numpy() & scope_mask
        return valid, (values.to_numpy() == "A").astype(int), "A"
    if target not in frame.columns:
        return None
    values = frame[target].fillna("").astype(str)
    labels = sorted(value for value in set(values[scope_mask]) if value)
    if len(labels) != 2:
        return None
    positive = (
        str(positive_condition_label)
        if target == "condition_label" and str(positive_condition_label) in labels
        else labels[-1]
    )
    valid = values.isin(labels).to_numpy() & scope_mask
    return valid, (values.to_numpy() == positive).astype(int), positive


def _normalize_rows(x: np.ndarray, *, center: np.ndarray | None = None) -> np.ndarray:
    center_value = (
        np.mean(x, axis=0, keepdims=True)
        if center is None
        else np.asarray(center, dtype=np.float32).reshape(1, -1)
    )
    centered = x - center_value
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 1e-12)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 4:
        return None
    a = a[valid]
    b = b[valid]
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def descriptive_mediation_row(
    *,
    x: np.ndarray,
    direction: np.ndarray,
    mediator: np.ndarray,
    outcome: np.ndarray,
) -> dict[str, Any] | None:
    """Summarize a cue-direction/outcome association without a causal claim."""

    mediator = np.asarray(mediator, dtype=int)
    outcome = np.asarray(outcome, dtype=int)
    if len(set(mediator.tolist())) < 2 or len(set(outcome.tolist())) < 2:
        return None
    projection = _normalize_rows(np.asarray(x, dtype=np.float32)) @ np.asarray(
        direction,
        dtype=np.float32,
    )
    projection_gap = float(np.mean(projection[mediator == 1]) - np.mean(projection[mediator == 0]))
    outcome_gap = float(np.mean(outcome[mediator == 1]) - np.mean(outcome[mediator == 0]))
    projection_residual = projection.copy()
    outcome_residual = outcome.astype(float)
    for label in (0, 1):
        group = mediator == label
        projection_residual[group] -= float(np.mean(projection[group]))
        outcome_residual[group] -= float(np.mean(outcome[group]))
    return {
        "n_rows": int(len(outcome)),
        "mediator_positive_rate": float(np.mean(mediator)),
        "outcome_positive_rate": float(np.mean(outcome)),
        "condition_projection_gap": projection_gap,
        "condition_outcome_rate_gap": outcome_gap,
        "projection_outcome_correlation": _safe_correlation(projection, outcome),
        "within_condition_projection_outcome_correlation": _safe_correlation(
            projection_residual,
            outcome_residual,
        ),
    }


def grouped_probe_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    *,
    folds: int,
    c_value: float,
    seed: int,
    salt: str,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray, np.ndarray]:
    """Fit a pair-grouped probe and retain strictly out-of-fold probabilities."""

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or len(x) != len(y) or len(x) != len(groups):
        raise ValueError("Probe arrays and groups must have matching row counts.")
    if len(set(y.tolist())) < 2:
        return (
            {"status": "single_class"},
            None,
            np.full((len(y),), np.nan, dtype=float),
            np.full((len(y),), -1, dtype=int),
        )
    fold_ids = np.asarray(
        [
            deterministic_group_fold(group, n_folds=int(folds), seed=int(seed), salt=salt)
            for group in groups
        ],
        dtype=int,
    )
    probabilities = np.full((len(y),), np.nan, dtype=float)
    predictions = np.full((len(y),), -1, dtype=int)
    used_folds = 0
    for fold in sorted(set(fold_ids.tolist())):
        train = fold_ids != fold
        test = fold_ids == fold
        if int(train.sum()) < 4 or int(test.sum()) < 1:
            continue
        if len(set(y[train].tolist())) < 2:
            continue
        train_center = np.mean(x[train], axis=0)
        x_train = _normalize_rows(x[train], center=train_center)
        x_test = _normalize_rows(x[test], center=train_center)
        model = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=int(seed),
        )
        model.fit(x_train, y[train])
        probabilities[test] = model.predict_proba(x_test)[:, 1]
        predictions[test] = model.predict(x_test)
        used_folds += 1
    valid = np.isfinite(probabilities)
    if int(valid.sum()) < 4 or len(set(y[valid].tolist())) < 2:
        return (
            {"status": "insufficient_heldout", "n_heldout": int(valid.sum())},
            None,
            probabilities,
            fold_ids,
        )
    final_center = np.mean(x, axis=0)
    x_final = _normalize_rows(x, center=final_center)
    final = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=int(seed),
    )
    final.fit(x_final, y)
    return (
        {
            "status": "ok",
            "n_rows": int(len(y)),
            "n_groups": int(len(set(groups))),
            "n_positive": int(y.sum()),
            "n_heldout": int(valid.sum()),
            "cv_folds_used": int(used_folds),
            "roc_auc": float(roc_auc_score(y[valid], probabilities[valid])),
            "balanced_accuracy": float(
                balanced_accuracy_score(y[valid], predictions[valid])
            ),
            "direction_norm": float(np.linalg.norm(final.coef_[0])),
        },
        final.coef_[0].astype(np.float32),
        probabilities,
        fold_ids,
    )


def grouped_probe(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    *,
    folds: int,
    c_value: float,
    seed: int,
    salt: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Backward-compatible aggregate interface for the grouped probe."""

    metrics, direction, _, _ = grouped_probe_oof(
        x,
        y,
        groups,
        folds=folds,
        c_value=c_value,
        seed=seed,
        salt=salt,
    )
    return metrics, direction


def _metadata_augmented_frame(
    frame: pd.DataFrame,
    *,
    columns: list[str],
) -> pd.DataFrame:
    out = frame.copy()
    metadata = out.get("metadata", pd.Series([{}] * len(out)))
    for column in columns:
        if column in out.columns:
            continue
        out[column] = [
            value.get(column) if isinstance(value, dict) else None
            for value in metadata
        ]
    return out


def _analysis_groups(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    min_group_rows: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = [
        {
            "label": "all",
            "group_type": "all",
            "group_value": "all",
            "mask": np.ones((len(frame),), dtype=bool),
        }
    ]
    for column in group_columns:
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str)
        for value, indices in values[values != ""].groupby(values[values != ""]).groups.items():
            mask = np.zeros((len(frame),), dtype=bool)
            mask[np.asarray(list(indices), dtype=int)] = True
            if int(mask.sum()) >= int(min_group_rows):
                groups.append(
                    {
                        "label": f"{column}:{value}",
                        "group_type": str(column),
                        "group_value": str(value),
                        "mask": mask,
                    }
                )
    return groups


def _last_valid_rows(states: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros((len(states), states.shape[-1]), dtype=np.float32)
    for index in range(len(states)):
        valid = np.flatnonzero(mask[index])
        if len(valid):
            out[index] = states[index, valid[-1]]
    return out


def branch_convergence(
    endpoints: np.ndarray,
    frame: pd.DataFrame,
    *,
    valid_mask: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    normalized = endpoints / np.maximum(np.linalg.norm(endpoints, axis=1, keepdims=True), 1e-12)
    working = frame.copy()
    working["_index"] = np.arange(len(frame))
    working = working.loc[valid_mask]
    for keys, group in working.groupby(
        ["reasoning_mode", "comparison_dimension", "comparison_id"],
        sort=True,
    ):
        indices = group["_index"].astype(int).to_numpy()
        if len(indices) < 2:
            continue
        similarities = normalized[indices] @ normalized[indices].T
        upper = similarities[np.triu_indices(len(indices), k=1)]
        rows.append(
            {
                "reasoning_mode": keys[0],
                "comparison_dimension": keys[1],
                "comparison_id": keys[2],
                "n_branches": int(len(indices)),
                "mean_endpoint_cosine": float(np.mean(upper)),
                "std_endpoint_cosine": float(np.std(upper)),
            }
        )
    return pd.DataFrame(rows)


def margin_revision_rows(
    frame: pd.DataFrame,
    margins: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, metadata in frame.iterrows():
        valid = mask[int(index)] & np.isfinite(margins[int(index)])
        values = margins[int(index), valid]
        signs = np.sign(values)
        nonzero = signs[signs != 0]
        revisions = int(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
        final_choice = normalize_choice(metadata.get("final_choice"))
        final_sign = 1 if final_choice == "A" else (-1 if final_choice == "B" else 0)
        rows.append(
            {
                "trace_id": metadata.get("trace_id"),
                "pair_id": metadata.get("pair_id"),
                "comparison_id": metadata.get("comparison_id"),
                "reasoning_mode": metadata.get("reasoning_mode"),
                "comparison_dimension": metadata.get("comparison_dimension"),
                "n_margin_points": int(len(values)),
                "margin_sign_revisions": revisions,
                "initial_A_minus_B_margin": None if not len(values) else float(values[0]),
                "final_A_minus_B_margin": None if not len(values) else float(values[-1]),
                "final_margin_agrees_with_choice": None
                if not len(values) or final_sign == 0
                else bool(int(np.sign(values[-1])) == final_sign),
            }
        )
    return pd.DataFrame(rows)


def trajectory_geometry_rows(
    frame: pd.DataFrame,
    states: np.ndarray,
    mask: np.ndarray,
    step_indices: np.ndarray,
    *,
    hidden_layer: int,
) -> pd.DataFrame:
    """Compute per-trace path geometry using generated-token distance as time."""

    rows: list[dict[str, Any]] = []
    for index, metadata in frame.iterrows():
        valid = mask[int(index)]
        points = states[int(index), valid]
        tokens = step_indices[int(index), valid].astype(float)
        if not len(points):
            continue
        deltas = np.diff(points, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        token_deltas = np.diff(tokens)
        per_token = np.divide(
            distances,
            token_deltas,
            out=np.full_like(distances, np.nan, dtype=float),
            where=token_deltas > 0,
        )
        path_length = float(np.sum(distances))
        displacement = (
            0.0 if len(points) < 2 else float(np.linalg.norm(points[-1] - points[0]))
        )
        rows.append(
            {
                "trace_id": metadata.get("trace_id"),
                "pair_id": metadata.get("pair_id"),
                "comparison_id": metadata.get("comparison_id"),
                "run_label": metadata.get("run_label"),
                "reasoning_mode": metadata.get("reasoning_mode"),
                "source_dataset": metadata.get("source_dataset"),
                "comparison_dimension": metadata.get("comparison_dimension"),
                "hidden_layer": int(hidden_layer),
                "n_trajectory_points": int(len(points)),
                "first_generated_tokens_before_state": int(tokens[0]),
                "last_generated_tokens_before_state": int(tokens[-1]),
                "path_length": path_length,
                "endpoint_displacement": displacement,
                "path_efficiency": None
                if path_length <= 1e-12
                else float(displacement / path_length),
                "mean_step_norm": None
                if not len(distances)
                else float(np.mean(distances)),
                "mean_step_norm_per_token": None
                if not np.isfinite(per_token).any()
                else float(np.nanmean(per_token)),
                "max_step_norm_per_token": None
                if not np.isfinite(per_token).any()
                else float(np.nanmax(per_token)),
            }
        )
    return pd.DataFrame(rows)


def _series_event(
    values: np.ndarray,
    tokens: np.ndarray,
    positions: np.ndarray,
    *,
    threshold: float,
    persistence: int,
) -> tuple[int | None, float | None, float | None]:
    point = first_persistent_threshold(
        values,
        threshold=float(threshold),
        persistence=int(persistence),
    )
    if point is None:
        return None, None, None
    return int(point), float(tokens[point]), float(positions[point])


def _rate(values: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros((0,), dtype=float)
    token_delta = np.diff(tokens.astype(float))
    return np.divide(
        np.diff(values.astype(float)),
        token_delta,
        out=np.full((len(values) - 1,), np.nan, dtype=float),
        where=token_delta > 0,
    )


def _sign_revisions(values: np.ndarray) -> int:
    signs = np.sign(np.asarray(values, dtype=float))
    nonzero = signs[signs != 0]
    return int(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0


def _prediction_static_row(
    frame: pd.DataFrame,
    index: int,
    *,
    metadata_columns: list[str],
) -> dict[str, Any]:
    metadata = frame.iloc[int(index)]
    keys = [
        "trace_id",
        "pair_id",
        "comparison_id",
        "run_label",
        "model_id",
        "reasoning_mode",
        "source_dataset",
        "comparison_dimension",
        "task_type",
        "target_option",
        "target_kind",
        "target_selected",
        "final_choice",
        "valid_choice",
        "presentation_order",
        "condition_label",
        "split",
    ]
    row = {key: metadata.get(key) for key in keys}
    for column in metadata_columns:
        row[column] = metadata.get(column)
    return row


def decision_dynamics_rows(
    predictions: pd.DataFrame,
    *,
    choice_confidence_threshold: float,
    target_confidence_threshold: float,
    persistence: int,
) -> pd.DataFrame:
    """Derive per-trace decision timing only from out-of-fold probabilities."""

    if predictions.empty:
        return pd.DataFrame()
    keys = ["trace_id", "analysis_group", "hidden_layer"]
    rows: list[dict[str, Any]] = []
    for group_keys, group in predictions.groupby(keys, sort=True, dropna=False):
        choice = group[group["probe_target"] == "final_choice"].sort_values("point_index")
        if choice.empty:
            continue
        first = choice.iloc[0]
        choice_prob_a = choice["prob_positive"].astype(float).to_numpy()
        tokens = choice["generated_tokens_before_state"].astype(float).to_numpy()
        positions = choice["position"].astype(float).to_numpy()
        choice_confidence = 2.0 * np.abs(choice_prob_a - 0.5)
        choice_signed_evidence = 2.0 * (choice_prob_a - 0.5)
        final_choice = normalize_choice(first.get("final_choice"))
        eventual_choice_support = (
            choice_prob_a
            if final_choice == "A"
            else (
                1.0 - choice_prob_a
                if final_choice == "B"
                else np.full_like(choice_prob_a, np.nan)
            )
        )
        choice_point, choice_token, choice_position = _series_event(
            choice_confidence,
            tokens,
            positions,
            threshold=float(choice_confidence_threshold),
            persistence=int(persistence),
        )
        choice_velocity = _rate(choice_confidence, tokens)

        target = group[group["probe_target"] == "target_option"].sort_values("point_index")
        target_confidence = np.zeros((0,), dtype=float)
        target_velocity = np.zeros((0,), dtype=float)
        target_point: int | None = None
        target_token: float | None = None
        target_position: float | None = None
        if not target.empty:
            target_prob_a = target["prob_positive"].astype(float).to_numpy()
            target_tokens = target["generated_tokens_before_state"].astype(float).to_numpy()
            target_positions = target["position"].astype(float).to_numpy()
            target_option = normalize_choice(first.get("target_option"))
            target_confidence = (
                target_prob_a
                if target_option == "A"
                else (
                    1.0 - target_prob_a
                    if target_option == "B"
                    else np.full_like(target_prob_a, np.nan)
                )
            )
            target_point, target_token, target_position = _series_event(
                target_confidence,
                target_tokens,
                target_positions,
                threshold=float(target_confidence_threshold),
                persistence=int(persistence),
            )
            target_velocity = _rate(target_confidence, target_tokens)

        shortcut_gap = (
            None
            if choice_token is None or target_token is None
            else float(target_token - choice_token)
        )
        rows.append(
            {
                **{
                    key: first.get(key)
                    for key in (
                        "trace_id",
                        "pair_id",
                        "comparison_id",
                        "run_label",
                        "model_id",
                        "reasoning_mode",
                        "source_dataset",
                        "comparison_dimension",
                        "task_type",
                        "target_kind",
                        "target_selected",
                        "final_choice",
                        "valid_choice",
                        "presentation_order",
                        "condition_label",
                        "split",
                        "validity_type",
                        "difficulty_tier",
                        "analysis_split",
                        "analysis_group",
                        "analysis_group_type",
                        "analysis_group_value",
                    )
                },
                "hidden_layer": int(group_keys[2]),
                "n_choice_points": int(len(choice_confidence)),
                "choice_confidence_threshold": float(choice_confidence_threshold),
                "target_confidence_threshold": float(target_confidence_threshold),
                "persistence": int(persistence),
                "choice_commitment_point_index": choice_point,
                "choice_commitment_generated_tokens": choice_token,
                "choice_commitment_position": choice_position,
                "target_emergence_point_index": target_point,
                "target_emergence_generated_tokens": target_token,
                "target_emergence_position": target_position,
                "shortcut_gap_tokens": shortcut_gap,
                "premature_commitment": None
                if choice_token is None
                else bool(target_token is None or choice_token < target_token),
                "committed_without_target_emergence": bool(
                    choice_token is not None and target_token is None
                ),
                "initial_choice_confidence": float(choice_confidence[0]),
                "final_choice_confidence": float(choice_confidence[-1]),
                "final_eventual_choice_support": None
                if not np.isfinite(eventual_choice_support[-1])
                else float(eventual_choice_support[-1]),
                "choice_evidence_sign_revisions": _sign_revisions(choice_signed_evidence),
                "mean_choice_confidence_velocity": None
                if not np.isfinite(choice_velocity).any()
                else float(np.nanmean(choice_velocity)),
                "max_choice_confidence_velocity": None
                if not np.isfinite(choice_velocity).any()
                else float(np.nanmax(choice_velocity)),
                "largest_choice_confidence_drop": None
                if not np.isfinite(choice_velocity).any()
                else float(np.nanmin(choice_velocity)),
                "initial_target_confidence": None
                if not len(target_confidence)
                else float(target_confidence[0]),
                "final_target_confidence": None
                if not len(target_confidence)
                else float(target_confidence[-1]),
                "mean_target_confidence_velocity": None
                if not np.isfinite(target_velocity).any()
                else float(np.nanmean(target_velocity)),
                "max_target_confidence_velocity": None
                if not np.isfinite(target_velocity).any()
                else float(np.nanmax(target_velocity)),
                "largest_target_confidence_drop": None
                if not np.isfinite(target_velocity).any()
                else float(np.nanmin(target_velocity)),
                "n_productive_target_updates": int(np.sum(target_velocity > 0))
                if len(target_velocity)
                else 0,
                "n_harmful_target_updates": int(np.sum(target_velocity < 0))
                if len(target_velocity)
                else 0,
            }
        )
    return pd.DataFrame(rows)


def summarize_decision_dynamics(dynamics: pd.DataFrame) -> pd.DataFrame:
    if dynamics.empty:
        return pd.DataFrame()
    working = dynamics.copy()
    working["_choice_committed"] = working[
        "choice_commitment_generated_tokens"
    ].notna()
    working["_target_emerged"] = working[
        "target_emergence_generated_tokens"
    ].notna()
    group_columns = [
        "run_label",
        "reasoning_mode",
        "analysis_group",
        "analysis_group_type",
        "analysis_group_value",
        "hidden_layer",
    ]
    available = [column for column in group_columns if column in dynamics.columns]

    def median_or_nan(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        return float(numeric.median()) if len(numeric) else float("nan")

    return (
        working.groupby(available, sort=True, dropna=False)
        .agg(
            n_traces=("trace_id", "count"),
            target_selection_rate=("target_selected", "mean"),
            choice_commitment_rate=("_choice_committed", "mean"),
            target_emergence_rate=("_target_emerged", "mean"),
            median_choice_commitment_tokens=(
                "choice_commitment_generated_tokens",
                median_or_nan,
            ),
            median_target_emergence_tokens=(
                "target_emergence_generated_tokens",
                median_or_nan,
            ),
            mean_shortcut_gap_tokens=("shortcut_gap_tokens", "mean"),
            premature_commitment_rate=("premature_commitment", "mean"),
            committed_without_target_rate=(
                "committed_without_target_emergence",
                "mean",
            ),
            mean_final_choice_confidence=("final_choice_confidence", "mean"),
            mean_final_target_confidence=("final_target_confidence", "mean"),
            mean_choice_confidence_velocity=(
                "mean_choice_confidence_velocity",
                "mean",
            ),
            mean_target_confidence_velocity=(
                "mean_target_confidence_velocity",
                "mean",
            ),
            mean_choice_evidence_revisions=(
                "choice_evidence_sign_revisions",
                "mean",
            ),
            mean_productive_target_updates=(
                "n_productive_target_updates",
                "mean",
            ),
            mean_harmful_target_updates=("n_harmful_target_updates", "mean"),
        )
        .reset_index()
    )


def analyze(
    artifact: TraceArtifact,
    *,
    probe_targets: list[str],
    group_columns: list[str] | None = None,
    positive_condition_label: str,
    cv_folds: int,
    probe_c: float,
    min_probe_rows: int,
    min_dimension_rows: int,
    commitment_auc: float,
    commitment_persistence: int,
    choice_confidence_threshold: float = 0.8,
    target_confidence_threshold: float = 0.8,
    seed: int,
) -> dict[str, Any]:
    group_columns = list(group_columns or ["comparison_dimension"])
    frame = _metadata_augmented_frame(
        artifact.frame.reset_index(drop=True),
        columns=group_columns,
    )
    probe_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    direction_arrays: list[np.ndarray] = []
    mediation_rows: list[dict[str, Any]] = []
    velocity_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    geometry_frames: list[pd.DataFrame] = []
    convergence_frames: list[pd.DataFrame] = []
    groups = _analysis_groups(
        frame,
        group_columns=group_columns,
        min_group_rows=int(min_dimension_rows),
    )

    for hidden_layer in artifact.hidden_layers:
        states = artifact.layer_states(hidden_layer)
        endpoint = _last_valid_rows(states, artifact.point_mask)
        convergence = branch_convergence(
            endpoint,
            frame,
            valid_mask=artifact.point_mask.any(axis=1),
        )
        if not convergence.empty:
            convergence["hidden_layer"] = int(hidden_layer)
            convergence_frames.append(convergence)
        geometry_frames.append(
            trajectory_geometry_rows(
                frame,
                states,
                artifact.point_mask,
                artifact.step_indices,
                hidden_layer=int(hidden_layer),
            )
        )

        for reasoning_mode in sorted(set(frame["reasoning_mode"].astype(str))):
            mode_mask = frame["reasoning_mode"].astype(str).to_numpy() == reasoning_mode
            for point_index in range(artifact.n_points):
                valid_point = artifact.point_mask[:, point_index]
                current = states[:, point_index]
                if point_index > 0:
                    both = mode_mask & valid_point & artifact.point_mask[:, point_index - 1]
                    if int(both.sum()) > 0:
                        velocity = np.linalg.norm(
                            current[both] - states[both, point_index - 1],
                            axis=1,
                        )
                        velocity_rows.append(
                            {
                                "reasoning_mode": reasoning_mode,
                                "hidden_layer": int(hidden_layer),
                                "point_index": int(point_index),
                                "mean_position": float(
                                    np.nanmean(artifact.positions[both, point_index])
                                ),
                                "n_traces": int(both.sum()),
                                "mean_step_norm": float(np.mean(velocity)),
                                "median_step_norm": float(np.median(velocity)),
                            }
                        )
                for analysis_group in groups:
                    group_label = str(analysis_group["label"])
                    group_mask = np.asarray(analysis_group["mask"], dtype=bool)
                    base_mask = mode_mask & group_mask & valid_point
                    for target in probe_targets:
                        encoded = _binary_target(
                            frame,
                            target=target,
                            positive_condition_label=positive_condition_label,
                            scope=base_mask,
                        )
                        if encoded is None:
                            continue
                        target_valid, y_all, positive_label = encoded
                        mask = base_mask & target_valid
                        if int(mask.sum()) < int(min_probe_rows):
                            continue
                        y = y_all[mask]
                        if len(set(y.tolist())) < 2:
                            continue
                        indices = np.flatnonzero(mask)
                        metrics, direction, probabilities, fold_ids = grouped_probe_oof(
                            current[mask],
                            y,
                            frame.loc[mask, "pair_id"].astype(str).tolist(),
                            folds=int(cv_folds),
                            c_value=float(probe_c),
                            seed=int(seed),
                            salt=f"{reasoning_mode}:{group_label}",
                        )
                        row = {
                            "reasoning_mode": reasoning_mode,
                            "analysis_group": group_label,
                            "analysis_group_type": analysis_group["group_type"],
                            "analysis_group_value": analysis_group["group_value"],
                            "probe_target": target,
                            "positive_label": positive_label,
                            "hidden_layer": int(hidden_layer),
                            "point_index": int(point_index),
                            "mean_position": float(
                                np.nanmean(artifact.positions[indices, point_index])
                            ),
                            **metrics,
                        }
                        probe_rows.append(row)
                        heldout = np.isfinite(probabilities)
                        for local_index in np.flatnonzero(heldout):
                            trace_index = int(indices[int(local_index)])
                            prediction_rows.append(
                                {
                                    **_prediction_static_row(
                                        frame,
                                        trace_index,
                                        metadata_columns=group_columns,
                                    ),
                                    "analysis_group": group_label,
                                    "analysis_group_type": analysis_group["group_type"],
                                    "analysis_group_value": analysis_group["group_value"],
                                    "probe_target": target,
                                    "positive_label": positive_label,
                                    "hidden_layer": int(hidden_layer),
                                    "point_index": int(point_index),
                                    "position": float(
                                        artifact.positions[trace_index, point_index]
                                    ),
                                    "generated_tokens_before_state": int(
                                        artifact.step_indices[trace_index, point_index]
                                    ),
                                    "fold_id": int(fold_ids[int(local_index)]),
                                    "observed_label": int(y[int(local_index)]),
                                    "prob_positive": float(
                                        probabilities[int(local_index)]
                                    ),
                                    "predicted_label": int(
                                        probabilities[int(local_index)] >= 0.5
                                    ),
                                }
                            )
                        if direction is not None:
                            direction_index = len(direction_arrays)
                            direction_arrays.append(direction)
                            direction_rows.append(
                                {
                                    **{
                                        key: row[key]
                                        for key in (
                                            "reasoning_mode",
                                            "analysis_group",
                                            "probe_target",
                                            "positive_label",
                                            "hidden_layer",
                                            "point_index",
                                            "mean_position",
                                        )
                                    },
                                    "direction_index": int(direction_index),
                                }
                            )
                            if target in {"condition_label", "presentation_order"}:
                                for outcome_target in ("final_choice", "target_selected"):
                                    outcome_encoded = _binary_target(
                                        frame,
                                        target=outcome_target,
                                        positive_condition_label="",
                                        scope=mask,
                                    )
                                    if outcome_encoded is None:
                                        continue
                                    outcome_valid, outcome_all, outcome_positive = outcome_encoded
                                    mediation_mask = mask & outcome_valid
                                    if int(mediation_mask.sum()) < int(min_probe_rows):
                                        continue
                                    summary = descriptive_mediation_row(
                                        x=current[mediation_mask],
                                        direction=direction,
                                        mediator=y_all[mediation_mask],
                                        outcome=outcome_all[mediation_mask],
                                    )
                                    if summary is None:
                                        continue
                                    mediation_rows.append(
                                        {
                                            "reasoning_mode": reasoning_mode,
                                            "analysis_group": group_label,
                                            "mediator_target": target,
                                            "mediator_positive_label": positive_label,
                                            "outcome_target": outcome_target,
                                            "outcome_positive_label": outcome_positive,
                                            "hidden_layer": int(hidden_layer),
                                            "point_index": int(point_index),
                                            "mean_position": float(
                                                np.nanmean(
                                                    artifact.positions[
                                                        mediation_mask,
                                                        point_index,
                                                    ]
                                                )
                                            ),
                                            **summary,
                                        }
                                    )
        del states

    probe_df = pd.DataFrame(probe_rows)
    direction_df = pd.DataFrame(direction_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    dynamics_df = decision_dynamics_rows(
        prediction_df,
        choice_confidence_threshold=float(choice_confidence_threshold),
        target_confidence_threshold=float(target_confidence_threshold),
        persistence=int(commitment_persistence),
    )
    commitment_rows: list[dict[str, Any]] = []
    if not probe_df.empty:
        ok = probe_df[probe_df["status"] == "ok"].copy()
        group_cols = [
            "reasoning_mode",
            "analysis_group",
            "probe_target",
            "positive_label",
            "hidden_layer",
        ]
        for keys, group in ok.groupby(group_cols, sort=True):
            ordered = group.sort_values("point_index")
            point = first_persistent_threshold(
                ordered["roc_auc"].astype(float).tolist(),
                threshold=float(commitment_auc),
                persistence=int(commitment_persistence),
            )
            selected = None if point is None else ordered.iloc[int(point)]
            commitment_rows.append(
                {
                    **dict(zip(group_cols, keys)),
                    "commitment_point_index": None
                    if selected is None
                    else int(selected["point_index"]),
                    "commitment_position": None
                    if selected is None
                    else float(selected["mean_position"]),
                    "commitment_auc": None
                    if selected is None
                    else float(selected["roc_auc"]),
                    "threshold": float(commitment_auc),
                    "persistence": int(commitment_persistence),
                    "max_auc": float(ordered["roc_auc"].max()),
                    "metric_basis": "heldout_probe_roc_auc",
                }
            )

    angle_rows: list[dict[str, Any]] = []
    if len(direction_df) and direction_arrays:
        by_key: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for index, row in direction_df.iterrows():
            key = (
                row["reasoning_mode"],
                row["analysis_group"],
                int(row["hidden_layer"]),
                int(row["point_index"]),
            )
            by_key[key].append(int(index))
        for key, indices in by_key.items():
            for left_pos in range(len(indices)):
                for right_pos in range(left_pos + 1, len(indices)):
                    left = direction_df.iloc[indices[left_pos]]
                    right = direction_df.iloc[indices[right_pos]]
                    if left["probe_target"] == right["probe_target"]:
                        continue
                    left_vector = direction_arrays[int(left["direction_index"])]
                    right_vector = direction_arrays[int(right["direction_index"])]
                    angle_rows.append(
                        {
                            "reasoning_mode": key[0],
                            "analysis_group": key[1],
                            "hidden_layer": key[2],
                            "point_index": key[3],
                            "left_target": left["probe_target"],
                            "right_target": right["probe_target"],
                            "angle_degrees": direction_angle_degrees(
                                left_vector,
                                right_vector,
                            ),
                        }
                    )

    return {
        "probe_metrics": probe_df,
        "commitment_summary": pd.DataFrame(commitment_rows),
        "probe_oof_predictions": prediction_df,
        "decision_dynamics": dynamics_df,
        "decision_dynamics_summary": summarize_decision_dynamics(dynamics_df),
        "probe_direction_index": direction_df,
        "probe_directions": np.stack(direction_arrays, axis=0)
        if direction_arrays
        else np.zeros((0, artifact.hidden_size), dtype=np.float32),
        "subspace_angles": pd.DataFrame(angle_rows),
        "trajectory_mediation_screen": pd.DataFrame(mediation_rows),
        "trajectory_velocity": pd.DataFrame(velocity_rows),
        "trajectory_geometry": pd.concat(geometry_frames, ignore_index=True)
        if geometry_frames
        else pd.DataFrame(),
        "branch_convergence": pd.concat(convergence_frames, ignore_index=True)
        if convergence_frames
        else pd.DataFrame(),
        "margin_revisions": margin_revision_rows(
            frame,
            artifact.label_margins,
            artifact.point_mask,
        ),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    trace_dir = _resolve(workspace_root, args.trace_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    artifact = TraceArtifact(trace_dir)
    outputs = analyze(
        artifact,
        probe_targets=_csv_list(str(args.probe_targets)),
        group_columns=_csv_list(str(args.group_columns)),
        positive_condition_label=str(args.positive_condition_label),
        cv_folds=int(args.cv_folds),
        probe_c=float(args.probe_c),
        min_probe_rows=int(args.min_probe_rows),
        min_dimension_rows=int(args.min_dimension_rows),
        commitment_auc=float(args.commitment_auc),
        commitment_persistence=int(args.commitment_persistence),
        choice_confidence_threshold=float(args.choice_confidence_threshold),
        target_confidence_threshold=float(args.target_confidence_threshold),
        seed=int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for name, value in outputs.items():
        if name == "probe_directions":
            path = out_dir / "probe_directions.npz"
            np.savez(path, directions=value)
        else:
            path = out_dir / f"{name}.csv"
            value.to_csv(path, index=False)
        output_paths[name] = str(path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-reasoning-trajectory-analysis",
            "trace_dir": str(trace_dir),
            "out_dir": str(out_dir),
            "n_traces": int(len(artifact.frame)),
            "hidden_layers": artifact.hidden_layers,
            "trajectory_points": int(artifact.n_points),
            "hidden_size": int(artifact.hidden_size),
            "probe_targets": _csv_list(str(args.probe_targets)),
            "group_columns": _csv_list(str(args.group_columns)),
            "positive_condition_label": str(args.positive_condition_label),
            "cv_folds": int(args.cv_folds),
            "probe_c": float(args.probe_c),
            "min_probe_rows": int(args.min_probe_rows),
            "min_dimension_rows": int(args.min_dimension_rows),
            "commitment_auc": float(args.commitment_auc),
            "commitment_persistence": int(args.commitment_persistence),
            "choice_confidence_threshold": float(
                args.choice_confidence_threshold
            ),
            "target_confidence_threshold": float(
                args.target_confidence_threshold
            ),
            "seed": int(args.seed),
            "outputs": output_paths,
        },
    )
    print(f"out_dir={out_dir}")
    print(f"n_traces={len(artifact.frame)}")
    probes = outputs["probe_metrics"]
    print(f"n_probe_rows={len(probes)}")


if __name__ == "__main__":
    main()
