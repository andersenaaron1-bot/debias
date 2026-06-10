"""Run pair-grouped temporal decoding for criterion-switch activations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json
from aisafety.mech.judge_reasoning import deterministic_group_fold
from aisafety.scripts.analyze_judge_criterion_switch_decoders import (
    CombinedArtifact,
    TARGETS,
    _fit,
    _point_value,
)
from aisafety.scripts.analyze_judge_reasoning_trajectories import _normalize_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1.0")
    parser.add_argument("--fit-split", default="fit")
    parser.add_argument("--selection-split", default="selection")
    parser.add_argument(
        "--estimation-splits",
        default="intervention",
        help="Pair splits used for cross-fitted reported metrics.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-fit-rows", type=int, default=24)
    parser.add_argument("--min-selection-rows", type=int, default=8)
    parser.add_argument("--min-fold-train-rows", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/criterion_switch_pair_analysis_v1"
        ),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _csv(raw: str) -> list[str]:
    return [value.strip() for value in str(raw).split(",") if value.strip()]


def _labels(
    frame: pd.DataFrame,
    *,
    target: str,
    point_index: int,
) -> np.ndarray:
    return np.asarray(
        [
            _point_value(row, target, point_index)
            for _, row in frame.iterrows()
        ],
        dtype=object,
    )


def _predict(
    model: Any,
    center: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(_normalize_rows(x, center=center))
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    return predictions, probabilities


def _candidate_selection(
    *,
    frame: pd.DataFrame,
    point_mask: np.ndarray,
    layer_states: dict[int, np.ndarray],
    targets: list[str],
    c_values: list[float],
    fit_split: str,
    selection_split: str,
    min_fit_rows: int,
    min_selection_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = frame["analysis_split"].fillna("").astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for target in targets:
        for point_index in range(point_mask.shape[1]):
            labels = _labels(
                frame,
                target=target,
                point_index=point_index,
            )
            valid = (labels != "") & point_mask[:, point_index]
            fit = valid & (split == str(fit_split))
            selection = valid & (split == str(selection_split))
            if (
                int(fit.sum()) < int(min_fit_rows)
                or int(selection.sum()) < int(min_selection_rows)
                or len(set(labels[fit].tolist())) < 2
            ):
                continue
            for hidden_layer, states in layer_states.items():
                for c_value in c_values:
                    center, model = _fit(
                        states[fit, point_index],
                        labels[fit],
                        c_value=float(c_value),
                        seed=int(seed),
                    )
                    predictions, probabilities = _predict(
                        model,
                        center,
                        states[selection, point_index],
                    )
                    observed = labels[selection]
                    auc = _macro_auc(
                        observed,
                        probabilities,
                        model.classes_,
                    )
                    rows.append(
                        {
                            "probe_target": target,
                            "train_point_index": int(point_index),
                            "train_point_name": (
                                frame.iloc[0].get("point_names") or []
                            )[point_index],
                            "hidden_layer": int(hidden_layer),
                            "c_value": float(c_value),
                            "n_fit": int(fit.sum()),
                            "n_selection": int(selection.sum()),
                            "balanced_accuracy": float(
                                balanced_accuracy_score(
                                    observed,
                                    predictions,
                                )
                            ),
                            "macro_roc_auc": auc,
                        }
                    )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No pair-analysis decoder candidates were fitted.")
    selected = (
        candidates.assign(
            finite_auc=candidates["macro_roc_auc"].fillna(-1.0)
        )
        .sort_values(
            [
                "probe_target",
                "train_point_index",
                "balanced_accuracy",
                "finite_auc",
                "hidden_layer",
                "c_value",
            ],
            ascending=[True, True, False, False, True, True],
        )
        .groupby(
            ["probe_target", "train_point_index"],
            as_index=False,
            sort=True,
        )
        .head(1)
        .drop(columns=["finite_auc"])
        .reset_index(drop=True)
    )
    return candidates, selected


def _macro_auc(
    observed: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> float:
    try:
        if len(classes) == 2:
            binary = (observed == classes[1]).astype(int)
            return float(roc_auc_score(binary, probabilities[:, 1]))
        return float(
            roc_auc_score(
                observed,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return np.nan


def _cross_fitted_predictions(
    *,
    frame: pd.DataFrame,
    point_mask: np.ndarray,
    layer_states: dict[int, np.ndarray],
    selected: pd.DataFrame,
    estimation_splits: set[str],
    cv_folds: int,
    min_fold_train_rows: int,
    seed: int,
) -> pd.DataFrame:
    split = frame["analysis_split"].fillna("").astype(str).to_numpy()
    pair_ids = frame["pair_id"].fillna("").astype(str).to_numpy()
    estimation = np.isin(split, sorted(estimation_splits))
    folds = np.asarray(
        [
            deterministic_group_fold(
                pair_id,
                n_folds=int(cv_folds),
                seed=int(seed),
                salt="criterion-switch-pair-cv",
            )
            for pair_id in pair_ids
        ],
        dtype=int,
    )
    rows: list[dict[str, Any]] = []
    label_cache: dict[tuple[str, int], np.ndarray] = {}
    for spec in selected.to_dict(orient="records"):
        target = str(spec["probe_target"])
        train_point = int(spec["train_point_index"])
        hidden_layer = int(spec["hidden_layer"])
        c_value = float(spec["c_value"])
        states = layer_states[hidden_layer]
        train_labels = label_cache.setdefault(
            (target, train_point),
            _labels(frame, target=target, point_index=train_point),
        )
        for fold in range(int(cv_folds)):
            train = (
                estimation
                & (folds != fold)
                & point_mask[:, train_point]
                & (train_labels != "")
            )
            if (
                int(train.sum()) < int(min_fold_train_rows)
                or len(set(train_labels[train].tolist())) < 2
            ):
                continue
            center, model = _fit(
                states[train, train_point],
                train_labels[train],
                c_value=c_value,
                seed=int(seed) + int(fold),
            )
            for test_point in range(point_mask.shape[1]):
                test_labels = label_cache.setdefault(
                    (target, test_point),
                    _labels(
                        frame,
                        target=target,
                        point_index=test_point,
                    ),
                )
                test = (
                    estimation
                    & (folds == fold)
                    & point_mask[:, test_point]
                    & (test_labels != "")
                )
                indices = np.flatnonzero(test)
                if not len(indices):
                    continue
                predictions, probabilities = _predict(
                    model,
                    center,
                    states[indices, test_point],
                )
                for local_index, trace_index in enumerate(indices):
                    source = frame.iloc[int(trace_index)]
                    payload = {
                        "trace_id": source.get("trace_id"),
                        "pair_id": source.get("pair_id"),
                        "source_dataset": source.get("source_dataset", ""),
                        "condition_id": source.get("condition_id", ""),
                        "transition_type": source.get("transition_type", ""),
                        "presentation_order": source.get(
                            "presentation_order",
                            "",
                        ),
                        "branch_index": source.get("branch_index", ""),
                        "analysis_split": source.get("analysis_split", ""),
                        "cv_fold": int(fold),
                        "probe_target": target,
                        "hidden_layer": hidden_layer,
                        "c_value": c_value,
                        "train_point_index": train_point,
                        "train_point_name": (
                            source.get("point_names") or []
                        )[train_point],
                        "test_point_index": int(test_point),
                        "test_point_name": (
                            source.get("point_names") or []
                        )[test_point],
                        "observed_label": str(test_labels[trace_index]),
                        "predicted_label": str(predictions[local_index]),
                        "correct": bool(
                            str(test_labels[trace_index])
                            == str(predictions[local_index])
                        ),
                    }
                    for class_index, class_name in enumerate(model.classes_):
                        payload[f"prob_{class_name}"] = float(
                            probabilities[local_index, class_index]
                        )
                    rows.append(payload)
    return pd.DataFrame(rows)


def _prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        "probe_target",
        "hidden_layer",
        "train_point_index",
        "train_point_name",
        "test_point_index",
        "test_point_name",
    ]
    for keys, group in predictions.groupby(groups, sort=True):
        classes = sorted(
            {
                column.removeprefix("prob_")
                for column in group.columns
                if column.startswith("prob_")
                and group[column].notna().any()
            }
        )
        probabilities = (
            group[[f"prob_{value}" for value in classes]]
            .fillna(0.0)
            .to_numpy()
        )
        rows.append(
            {
                **dict(zip(groups, keys, strict=True)),
                "n_traces": int(len(group)),
                "n_pairs": int(group["pair_id"].nunique()),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        group["observed_label"],
                        group["predicted_label"],
                    )
                ),
                "accuracy": float(group["correct"].mean()),
                "macro_roc_auc": _macro_auc(
                    group["observed_label"].to_numpy(),
                    probabilities,
                    np.asarray(classes, dtype=object),
                ),
            }
        )
    return pd.DataFrame(rows)


def _stratified_point_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    diagonal = predictions[
        predictions["train_point_index"] == predictions["test_point_index"]
    ].copy()
    rows: list[dict[str, Any]] = []
    for column in (
        "source_dataset",
        "condition_id",
        "transition_type",
    ):
        if column not in diagonal.columns:
            continue
        values = diagonal[column].fillna("").astype(str)
        for keys, group in diagonal[values != ""].groupby(
            ["probe_target", "test_point_name", column],
            sort=True,
        ):
            rows.append(
                {
                    "group_type": column,
                    "group_value": keys[2],
                    "probe_target": keys[0],
                    "point_name": keys[1],
                    "n_traces": int(len(group)),
                    "n_pairs": int(group["pair_id"].nunique()),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(
                            group["observed_label"],
                            group["predicted_label"],
                        )
                    ),
                    "accuracy": float(group["correct"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _pair_bootstrap(
    predictions: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    diagonal = predictions[
        predictions["train_point_index"] == predictions["test_point_index"]
    ]
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for keys, group in diagonal.groupby(
        ["probe_target", "test_point_index", "test_point_name"],
        sort=True,
    ):
        by_pair = {
            str(pair_id): values
            for pair_id, values in group.groupby("pair_id", sort=False)
        }
        pair_ids = np.asarray(sorted(by_pair), dtype=object)
        if not len(pair_ids):
            continue
        values: list[float] = []
        for _ in range(max(int(samples), 0)):
            sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
            boot = pd.concat(
                [by_pair[str(pair_id)] for pair_id in sampled],
                ignore_index=True,
            )
            values.append(
                float(
                    balanced_accuracy_score(
                        boot["observed_label"],
                        boot["predicted_label"],
                    )
                )
            )
        point = float(
            balanced_accuracy_score(
                group["observed_label"],
                group["predicted_label"],
            )
        )
        rows.append(
            {
                "probe_target": keys[0],
                "point_index": int(keys[1]),
                "point_name": keys[2],
                "n_traces": int(len(group)),
                "n_pairs": int(len(pair_ids)),
                "balanced_accuracy": point,
                "ci95_low": (
                    float(np.quantile(values, 0.025))
                    if values
                    else np.nan
                ),
                "ci95_high": (
                    float(np.quantile(values, 0.975))
                    if values
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_difference_artifact(
    artifact: CombinedArtifact,
) -> tuple[pd.DataFrame, np.ndarray, dict[int, np.ndarray]] | None:
    frame = artifact.frame.reset_index(drop=True)
    key_columns = ["pair_id", "presentation_order", "branch_index"]
    reminder = frame[frame["condition_id"].astype(str) == "reminder"]
    switched = frame[frame["condition_id"].astype(str) == "switch"]
    reminder_map = {
        tuple(row[column] for column in key_columns): int(index)
        for index, row in reminder.iterrows()
    }
    pairs: list[tuple[int, int]] = []
    rows: list[dict[str, Any]] = []
    for switch_index, row in switched.iterrows():
        key = tuple(row[column] for column in key_columns)
        reminder_index = reminder_map.get(key)
        if reminder_index is None:
            continue
        pairs.append((int(switch_index), int(reminder_index)))
        rows.append(row.to_dict())
    if not rows:
        return None
    point_mask = np.stack(
        [
            artifact.point_mask[switch_index]
            & artifact.point_mask[reminder_index]
            for switch_index, reminder_index in pairs
        ],
        axis=0,
    )
    layer_states = {}
    for hidden_layer in artifact.hidden_layers:
        states = artifact.layer_states(hidden_layer)
        layer_states[int(hidden_layer)] = np.stack(
            [
                states[switch_index] - states[reminder_index]
                for switch_index, reminder_index in pairs
            ],
            axis=0,
        )
    return pd.DataFrame(rows), point_mask, layer_states


def _analyze(
    *,
    frame: pd.DataFrame,
    point_mask: np.ndarray,
    layer_states: dict[int, np.ndarray],
    targets: list[str],
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    candidates, selected = _candidate_selection(
        frame=frame,
        point_mask=point_mask,
        layer_states=layer_states,
        targets=targets,
        c_values=[float(value) for value in _csv(args.c_values)],
        fit_split=str(args.fit_split),
        selection_split=str(args.selection_split),
        min_fit_rows=int(args.min_fit_rows),
        min_selection_rows=int(args.min_selection_rows),
        seed=int(args.seed),
    )
    predictions = _cross_fitted_predictions(
        frame=frame,
        point_mask=point_mask,
        layer_states=layer_states,
        selected=selected,
        estimation_splits=set(_csv(args.estimation_splits)),
        cv_folds=int(args.cv_folds),
        min_fold_train_rows=int(args.min_fold_train_rows),
        seed=int(args.seed),
    )
    return {
        "candidate_selection": candidates,
        "selected_decoders": selected,
        "cross_time_predictions": predictions,
        "cross_time_metrics": _prediction_metrics(predictions),
        "point_metrics_by_group": _stratified_point_metrics(predictions),
        "point_pair_bootstrap": _pair_bootstrap(
            predictions,
            samples=int(args.bootstrap),
            seed=int(args.seed),
        ),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    trace_dirs = [_resolve(workspace_root, path) for path in args.trace_dir]
    out_dir = _resolve(workspace_root, args.out_dir)
    artifact = CombinedArtifact(trace_dirs)
    layer_states = {
        int(layer): artifact.layer_states(int(layer))
        for layer in artifact.hidden_layers
    }
    targets = _csv(args.targets)
    outputs = _analyze(
        frame=artifact.frame.reset_index(drop=True),
        point_mask=artifact.point_mask,
        layer_states=layer_states,
        targets=targets,
        args=args,
    )
    difference = _paired_difference_artifact(artifact)
    if difference is not None:
        diff_frame, diff_mask, diff_states = difference
        diff_targets = [
            target
            for target in (
                "active_criterion",
                "criterion_target",
                "current_choice",
            )
            if target in targets
        ]
        if diff_targets:
            for name, table in _analyze(
                frame=diff_frame,
                point_mask=diff_mask,
                layer_states=diff_states,
                targets=diff_targets,
                args=args,
            ).items():
                outputs[f"switch_minus_reminder_{name}"] = table

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for name, table in outputs.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        output_paths[name] = str(path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-criterion-switch-pair-temporal-analysis",
            "trace_dirs": [str(path) for path in trace_dirs],
            "out_dir": str(out_dir),
            "targets": targets,
            "fit_split": str(args.fit_split),
            "selection_split": str(args.selection_split),
            "estimation_splits": _csv(args.estimation_splits),
            "cv_folds": int(args.cv_folds),
            "bootstrap_samples": int(args.bootstrap),
            "seed": int(args.seed),
            "outputs": output_paths,
        },
    )
    print(f"out_dir={out_dir}")
    print(outputs["point_pair_bootstrap"].to_string(index=False))


if __name__ == "__main__":
    main()
