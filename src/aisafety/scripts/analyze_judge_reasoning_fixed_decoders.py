"""Analyze judge trajectories with endpoint-trained fixed temporal decoders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json
from aisafety.scripts.analyze_judge_reasoning_trajectories import (
    TraceArtifact,
    _binary_target,
    _metadata_augmented_frame,
    _normalize_rows,
    decision_dynamics_rows,
    summarize_decision_dynamics,
)


DEFAULT_TRACE_DIR = (
    Path("artifacts") / "mechanistic" / "judge_reasoning_trajectories_v1"
)
DEFAULT_OUT_DIR = (
    Path("artifacts") / "mechanistic" / "judge_reasoning_fixed_decoders_v1"
)
META_COLUMNS = [
    "criterion_id",
    "criterion_family",
    "criterion_determinacy",
    "determinacy_level",
    "analysis_split",
    "validity_type",
    "difficulty_tier",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--reasoning-mode", default="thinking")
    parser.add_argument("--anchor-target", default="final_choice")
    parser.add_argument(
        "--probe-targets",
        default="final_choice,target_option,presentation_order",
    )
    parser.add_argument("--positive-condition-label", default="")
    parser.add_argument("--c-values", default="0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--fit-split", default="fit")
    parser.add_argument("--selection-split", default="selection")
    parser.add_argument("--evaluation-split", default="intervention")
    parser.add_argument("--min-fit-rows", type=int, default=40)
    parser.add_argument("--min-selection-rows", type=int, default=20)
    parser.add_argument("--min-evaluation-rows", type=int, default=20)
    parser.add_argument("--choice-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--target-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--commitment-persistence", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _csv_list(raw: str) -> list[str]:
    return [value.strip() for value in str(raw).split(",") if value.strip()]


def _last_valid_rows(states: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros((len(states), states.shape[-1]), dtype=np.float32)
    for index in range(len(states)):
        valid = np.flatnonzero(mask[index])
        if len(valid):
            out[index] = states[index, valid[-1]]
    return out


def _fit_decoder(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, LogisticRegression]:
    center = np.mean(x, axis=0).astype(np.float32)
    normalized = _normalize_rows(x, center=center)
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=int(seed),
    )
    model.fit(normalized, y)
    return center, model


def _evaluate(
    model: LogisticRegression,
    center: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    probabilities = model.predict_proba(_normalize_rows(x, center=center))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
    }


def select_anchor_decoder(
    *,
    layer_endpoints: dict[int, np.ndarray],
    y: np.ndarray,
    fit_mask: np.ndarray,
    selection_mask: np.ndarray,
    c_values: list[float],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hidden_layer, endpoint in layer_endpoints.items():
        for c_value in c_values:
            center, model = _fit_decoder(
                endpoint[fit_mask],
                y[fit_mask],
                c_value=float(c_value),
                seed=int(seed),
            )
            metrics = _evaluate(
                model,
                center,
                endpoint[selection_mask],
                y[selection_mask],
            )
            rows.append(
                {
                    "hidden_layer": int(hidden_layer),
                    "c_value": float(c_value),
                    "n_fit": int(fit_mask.sum()),
                    "n_selection": int(selection_mask.sum()),
                    **metrics,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No fixed-decoder candidates were fitted.")
    selected = (
        table.sort_values(
            ["roc_auc", "balanced_accuracy", "hidden_layer", "c_value"],
            ascending=[False, False, True, True],
        )
        .iloc[0]
        .to_dict()
    )
    return table, selected


def _target_encoding(
    frame: pd.DataFrame,
    *,
    target: str,
    positive_condition_label: str,
    mode_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    encoded = _binary_target(
        frame,
        target=target,
        positive_condition_label=positive_condition_label,
        scope=mode_mask,
    )
    if encoded is None:
        return None
    valid, y, positive = encoded
    return np.asarray(valid, dtype=bool), np.asarray(y, dtype=int), str(positive)


def _check_split(
    *,
    target: str,
    name: str,
    mask: np.ndarray,
    y: np.ndarray,
    minimum: int,
) -> None:
    if int(mask.sum()) < int(minimum):
        raise ValueError(
            f"{target} has {int(mask.sum())} rows in {name}; need {int(minimum)}."
        )
    if len(set(y[mask].tolist())) < 2:
        raise ValueError(f"{target} is single-class in {name}.")


def analyze_fixed_decoders(
    artifact: TraceArtifact,
    *,
    reasoning_mode: str,
    anchor_target: str,
    probe_targets: list[str],
    positive_condition_label: str,
    c_values: list[float],
    fit_split: str,
    selection_split: str,
    evaluation_split: str,
    min_fit_rows: int,
    min_selection_rows: int,
    min_evaluation_rows: int,
    choice_confidence_threshold: float,
    target_confidence_threshold: float,
    commitment_persistence: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]:
    frame = _metadata_augmented_frame(
        artifact.frame.reset_index(drop=True),
        columns=META_COLUMNS,
    )
    mode_mask = frame["reasoning_mode"].astype(str).eq(str(reasoning_mode)).to_numpy()
    split_values = frame["analysis_split"].fillna("").astype(str).to_numpy()
    point_valid = artifact.point_mask.any(axis=1)
    anchor = _target_encoding(
        frame,
        target=anchor_target,
        positive_condition_label=positive_condition_label,
        mode_mask=mode_mask,
    )
    if anchor is None:
        raise ValueError(f"Could not encode anchor target {anchor_target!r}.")
    anchor_valid, anchor_y, anchor_positive = anchor
    fit_mask = (
        mode_mask
        & point_valid
        & anchor_valid
        & (split_values == str(fit_split))
    )
    selection_mask = (
        mode_mask
        & point_valid
        & anchor_valid
        & (split_values == str(selection_split))
    )
    evaluation_mask = (
        mode_mask
        & point_valid
        & anchor_valid
        & (split_values == str(evaluation_split))
    )
    _check_split(
        target=anchor_target,
        name=fit_split,
        mask=fit_mask,
        y=anchor_y,
        minimum=min_fit_rows,
    )
    _check_split(
        target=anchor_target,
        name=selection_split,
        mask=selection_mask,
        y=anchor_y,
        minimum=min_selection_rows,
    )
    _check_split(
        target=anchor_target,
        name=evaluation_split,
        mask=evaluation_mask,
        y=anchor_y,
        minimum=min_evaluation_rows,
    )

    layer_states = {
        int(layer): artifact.layer_states(int(layer))
        for layer in artifact.hidden_layers
    }
    endpoints = {
        layer: _last_valid_rows(states, artifact.point_mask)
        for layer, states in layer_states.items()
    }
    selection_table, selected = select_anchor_decoder(
        layer_endpoints=endpoints,
        y=anchor_y,
        fit_mask=fit_mask,
        selection_mask=selection_mask,
        c_values=c_values,
        seed=seed,
    )
    hidden_layer = int(selected["hidden_layer"])
    c_value = float(selected["c_value"])
    states = layer_states[hidden_layer]
    endpoint = endpoints[hidden_layer]

    prediction_rows: list[dict[str, Any]] = []
    target_metrics: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    trained_targets: list[str] = []
    train_split_mask = np.isin(split_values, [str(fit_split), str(selection_split)])
    for target in probe_targets:
        encoded = _target_encoding(
            frame,
            target=target,
            positive_condition_label=positive_condition_label,
            mode_mask=mode_mask,
        )
        if encoded is None:
            continue
        target_valid, y, positive_label = encoded
        train_mask = mode_mask & point_valid & target_valid & train_split_mask
        test_mask = (
            mode_mask
            & point_valid
            & target_valid
            & (split_values == str(evaluation_split))
        )
        if (
            int(train_mask.sum()) < int(min_fit_rows + min_selection_rows)
            or int(test_mask.sum()) < int(min_evaluation_rows)
            or len(set(y[train_mask].tolist())) < 2
            or len(set(y[test_mask].tolist())) < 2
        ):
            continue
        center, model = _fit_decoder(
            endpoint[train_mask],
            y[train_mask],
            c_value=c_value,
            seed=seed,
        )
        endpoint_metrics = _evaluate(
            model,
            center,
            endpoint[test_mask],
            y[test_mask],
        )
        target_metrics.append(
            {
                "probe_target": target,
                "positive_label": positive_label,
                "hidden_layer": hidden_layer,
                "c_value": c_value,
                "n_train": int(train_mask.sum()),
                "n_evaluation": int(test_mask.sum()),
                **endpoint_metrics,
            }
        )
        arrays[f"{target}_center"] = center.astype(np.float32)
        arrays[f"{target}_direction"] = model.coef_[0].astype(np.float32)
        arrays[f"{target}_intercept"] = np.asarray(
            model.intercept_,
            dtype=np.float32,
        )
        trained_targets.append(target)
        test_indices = np.flatnonzero(test_mask)
        for point_index in range(artifact.n_points):
            valid_indices = test_indices[
                artifact.point_mask[test_indices, point_index]
            ]
            if not len(valid_indices):
                continue
            probabilities = model.predict_proba(
                _normalize_rows(states[valid_indices, point_index], center=center)
            )[:, 1]
            for trace_index, probability in zip(
                valid_indices,
                probabilities,
                strict=True,
            ):
                metadata = frame.iloc[int(trace_index)]
                prediction_rows.append(
                    {
                        **{
                            key: metadata.get(key)
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
                                "target_option",
                                "target_kind",
                                "target_selected",
                                "final_choice",
                                "valid_choice",
                                "presentation_order",
                                "condition_label",
                                "split",
                                *META_COLUMNS,
                            )
                        },
                        "analysis_group": "all",
                        "analysis_group_type": "all",
                        "analysis_group_value": "all",
                        "probe_target": target,
                        "positive_label": positive_label,
                        "hidden_layer": hidden_layer,
                        "point_index": int(point_index),
                        "position": float(
                            artifact.positions[int(trace_index), point_index]
                        ),
                        "generated_tokens_before_state": int(
                            artifact.step_indices[int(trace_index), point_index]
                        ),
                        "observed_label": int(y[int(trace_index)]),
                        "prob_positive": float(probability),
                        "predicted_label": int(probability >= 0.5),
                        "decoder_training": "endpoint_fit_plus_selection",
                        "decoder_evaluation_split": str(evaluation_split),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise ValueError("No fixed-decoder held-out predictions were emitted.")
    point_metric_rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(
        ["probe_target", "hidden_layer", "point_index"],
        sort=True,
        dropna=False,
    ):
        y = group["observed_label"].astype(int).to_numpy()
        probability = group["prob_positive"].astype(float).to_numpy()
        if len(set(y.tolist())) < 2:
            continue
        point_metric_rows.append(
            {
                "probe_target": keys[0],
                "hidden_layer": int(keys[1]),
                "point_index": int(keys[2]),
                "mean_position": float(group["position"].mean()),
                "mean_generated_tokens_before_state": float(
                    group["generated_tokens_before_state"].mean()
                ),
                "n_evaluation": int(len(group)),
                "roc_auc": float(roc_auc_score(y, probability)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y, probability >= 0.5)
                ),
            }
        )
    dynamics = decision_dynamics_rows(
        predictions,
        choice_confidence_threshold=float(choice_confidence_threshold),
        target_confidence_threshold=float(target_confidence_threshold),
        persistence=int(commitment_persistence),
    )
    outputs = {
        "anchor_model_selection": selection_table,
        "selected_decoder_targets": pd.DataFrame(target_metrics),
        "fixed_decoder_predictions": predictions,
        "fixed_decoder_point_metrics": pd.DataFrame(point_metric_rows),
        "fixed_decoder_dynamics": dynamics,
        "fixed_decoder_dynamics_summary": summarize_decision_dynamics(dynamics),
    }
    metadata = {
        "reasoning_mode": str(reasoning_mode),
        "anchor_target": str(anchor_target),
        "anchor_positive_label": anchor_positive,
        "selected_hidden_layer": hidden_layer,
        "selected_c_value": c_value,
        "selection_roc_auc": float(selected["roc_auc"]),
        "selection_balanced_accuracy": float(selected["balanced_accuracy"]),
        "fit_split": str(fit_split),
        "selection_split": str(selection_split),
        "evaluation_split": str(evaluation_split),
        "trained_targets": trained_targets,
    }
    return outputs, arrays, metadata


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    trace_dir = _resolve(workspace_root, args.trace_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    artifact = TraceArtifact(trace_dir)
    outputs, arrays, metadata = analyze_fixed_decoders(
        artifact,
        reasoning_mode=str(args.reasoning_mode),
        anchor_target=str(args.anchor_target),
        probe_targets=_csv_list(str(args.probe_targets)),
        positive_condition_label=str(args.positive_condition_label),
        c_values=[float(value) for value in _csv_list(str(args.c_values))],
        fit_split=str(args.fit_split),
        selection_split=str(args.selection_split),
        evaluation_split=str(args.evaluation_split),
        min_fit_rows=int(args.min_fit_rows),
        min_selection_rows=int(args.min_selection_rows),
        min_evaluation_rows=int(args.min_evaluation_rows),
        choice_confidence_threshold=float(args.choice_confidence_threshold),
        target_confidence_threshold=float(args.target_confidence_threshold),
        commitment_persistence=int(args.commitment_persistence),
        seed=int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = str(path)
    arrays_path = out_dir / "fixed_decoder_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-reasoning-fixed-endpoint-decoder-analysis",
            "trace_dir": str(trace_dir),
            "out_dir": str(out_dir),
            "seed": int(args.seed),
            "choice_confidence_threshold": float(
                args.choice_confidence_threshold
            ),
            "target_confidence_threshold": float(
                args.target_confidence_threshold
            ),
            "commitment_persistence": int(args.commitment_persistence),
            **metadata,
            "decoder_arrays": str(arrays_path),
            "outputs": output_paths,
        },
    )
    print(f"out_dir={out_dir}")
    print(f"selected_hidden_layer={metadata['selected_hidden_layer']}")
    print(f"selected_c_value={metadata['selected_c_value']}")
    print(f"trained_targets={','.join(metadata['trained_targets'])}")


if __name__ == "__main__":
    main()
