"""Fit held-out multiclass criterion, target, choice, and order decoders."""

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
    _normalize_rows,
)


TARGETS = (
    "active_criterion",
    "criterion_target",
    "final_choice",
    "presentation_order",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--c-values", default="0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--fit-split", default="fit")
    parser.add_argument("--selection-split", default="selection")
    parser.add_argument("--evaluation-split", default="intervention")
    parser.add_argument("--min-fit-rows", type=int, default=30)
    parser.add_argument("--min-selection-rows", type=int, default=10)
    parser.add_argument("--min-evaluation-rows", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/mechanistic/criterion_switch_decoders_v1"),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _csv(raw: str) -> list[str]:
    return [value.strip() for value in str(raw).split(",") if value.strip()]


class CombinedArtifact:
    def __init__(self, paths: list[Path]):
        self.artifacts = [TraceArtifact(path) for path in paths]
        reference = self.artifacts[0]
        self.hidden_layers = list(reference.hidden_layers)
        self.n_points = int(reference.n_points)
        for artifact in self.artifacts[1:]:
            if artifact.hidden_layers != self.hidden_layers:
                raise ValueError("Activation shards have different hidden layers.")
            if artifact.n_points != self.n_points:
                raise ValueError("Activation shards have different point counts.")
        self.frame = pd.concat(
            [artifact.frame for artifact in self.artifacts],
            ignore_index=True,
        )
        self.point_mask = np.concatenate(
            [artifact.point_mask for artifact in self.artifacts], axis=0
        )
        self.positions = np.concatenate(
            [artifact.positions for artifact in self.artifacts], axis=0
        )
        self.step_indices = np.concatenate(
            [artifact.step_indices for artifact in self.artifacts], axis=0
        )

    def layer_states(self, hidden_layer: int) -> np.ndarray:
        return np.concatenate(
            [
                artifact.layer_states(hidden_layer)
                for artifact in self.artifacts
            ],
            axis=0,
        )


def _point_value(row: pd.Series, target: str, point_index: int) -> str:
    if target == "active_criterion":
        values = row.get("point_active_criteria") or []
        return str(values[point_index]) if point_index < len(values) else ""
    if target == "criterion_target":
        values = row.get("point_target_semantics") or []
        return str(values[point_index]) if point_index < len(values) else ""
    if target == "final_choice":
        return str(row.get("decoder_final_choice_semantic") or "")
    if target == "presentation_order":
        return str(row.get("presentation_order") or "")
    raise ValueError(f"Unknown decoder target: {target}")


def _endpoint_labels(frame: pd.DataFrame, target: str, point_index: int) -> np.ndarray:
    return np.asarray(
        [_point_value(row, target, point_index) for _, row in frame.iterrows()],
        dtype=object,
    )


def _fit(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, LogisticRegression]:
    center = np.mean(x, axis=0).astype(np.float32)
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=int(seed),
    )
    model.fit(_normalize_rows(x, center=center), labels)
    return center, model


def _metrics(
    model: LogisticRegression,
    center: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    normalized = _normalize_rows(x, center=center)
    predictions = model.predict(normalized)
    probabilities = model.predict_proba(normalized)
    auc = np.nan
    try:
        if len(model.classes_) == 2:
            binary = (labels == model.classes_[1]).astype(int)
            auc = float(roc_auc_score(binary, probabilities[:, 1]))
        else:
            auc = float(
                roc_auc_score(
                    labels,
                    probabilities,
                    labels=model.classes_,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        pass
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_roc_auc": auc,
    }


def _check(
    target: str,
    split_name: str,
    mask: np.ndarray,
    labels: np.ndarray,
    minimum: int,
) -> None:
    if int(mask.sum()) < int(minimum):
        raise ValueError(
            f"{target} has {int(mask.sum())} {split_name} rows; "
            f"need {minimum}."
        )
    if len(set(labels[mask].tolist())) < 2:
        raise ValueError(f"{target} is single-class in {split_name}.")


def analyze(
    artifact: CombinedArtifact,
    *,
    targets: list[str],
    c_values: list[float],
    fit_split: str,
    selection_split: str,
    evaluation_split: str,
    min_fit_rows: int,
    min_selection_rows: int,
    min_evaluation_rows: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]:
    frame = artifact.frame.reset_index(drop=True)
    splits = frame["analysis_split"].fillna("").astype(str).to_numpy()
    endpoint_index = artifact.n_points - 1
    layer_states = {
        layer: artifact.layer_states(layer) for layer in artifact.hidden_layers
    }
    selection_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    decoder_specs: dict[str, Any] = {}

    for target in targets:
        labels = _endpoint_labels(frame, target, endpoint_index)
        valid = labels != ""
        fit_mask = valid & (splits == str(fit_split))
        selection_mask = valid & (splits == str(selection_split))
        evaluation_mask = valid & (splits == str(evaluation_split))
        _check(target, fit_split, fit_mask, labels, min_fit_rows)
        _check(
            target,
            selection_split,
            selection_mask,
            labels,
            min_selection_rows,
        )
        _check(
            target,
            evaluation_split,
            evaluation_mask,
            labels,
            min_evaluation_rows,
        )
        for hidden_layer, states in layer_states.items():
            endpoint = states[:, endpoint_index]
            for c_value in c_values:
                center, model = _fit(
                    endpoint[fit_mask],
                    labels[fit_mask],
                    c_value=c_value,
                    seed=seed,
                )
                metrics = _metrics(
                    model,
                    center,
                    endpoint[selection_mask],
                    labels[selection_mask],
                )
                selection_rows.append(
                    {
                        "probe_target": target,
                        "hidden_layer": int(hidden_layer),
                        "c_value": float(c_value),
                        "n_fit": int(fit_mask.sum()),
                        "n_selection": int(selection_mask.sum()),
                        "n_classes": int(len(model.classes_)),
                        **metrics,
                    }
                )
        candidates = pd.DataFrame(
            [row for row in selection_rows if row["probe_target"] == target]
        )
        selected = (
            candidates.assign(
                finite_auc=candidates["macro_roc_auc"].fillna(-1.0)
            )
            .sort_values(
                [
                    "balanced_accuracy",
                    "finite_auc",
                    "hidden_layer",
                    "c_value",
                ],
                ascending=[False, False, True, True],
            )
            .iloc[0]
        )
        hidden_layer = int(selected["hidden_layer"])
        c_value = float(selected["c_value"])
        states = layer_states[hidden_layer]
        endpoint = states[:, endpoint_index]
        train_mask = valid & np.isin(
            splits, [str(fit_split), str(selection_split)]
        )
        center, model = _fit(
            endpoint[train_mask],
            labels[train_mask],
            c_value=c_value,
            seed=seed,
        )
        endpoint_metrics = _metrics(
            model,
            center,
            endpoint[evaluation_mask],
            labels[evaluation_mask],
        )
        selected_rows.append(
            {
                "probe_target": target,
                "hidden_layer": hidden_layer,
                "c_value": c_value,
                "n_train": int(train_mask.sum()),
                "n_evaluation": int(evaluation_mask.sum()),
                "classes": "|".join(str(value) for value in model.classes_),
                **endpoint_metrics,
            }
        )
        arrays[f"{target}_center"] = center
        arrays[f"{target}_coef"] = model.coef_.astype(np.float32)
        arrays[f"{target}_intercept"] = model.intercept_.astype(np.float32)
        arrays[f"{target}_classes"] = np.asarray(model.classes_, dtype=str)
        decoder_specs[target] = {
            "hidden_layer": hidden_layer,
            "c_value": c_value,
            "classes": [str(value) for value in model.classes_],
        }
        evaluation_indices = np.flatnonzero(evaluation_mask)
        for point_index in range(artifact.n_points):
            point_labels = np.asarray(
                [
                    _point_value(frame.iloc[index], target, point_index)
                    for index in evaluation_indices
                ],
                dtype=object,
            )
            point_valid = (
                artifact.point_mask[evaluation_indices, point_index]
                & (point_labels != "")
            )
            indices = evaluation_indices[point_valid]
            observed = point_labels[point_valid]
            if not len(indices):
                continue
            probabilities = model.predict_proba(
                _normalize_rows(states[indices, point_index], center=center)
            )
            predictions = model.classes_[np.argmax(probabilities, axis=1)]
            for local_index, trace_index in enumerate(indices):
                row = frame.iloc[int(trace_index)]
                payload = {
                    "trace_id": row.get("trace_id"),
                    "pair_id": row.get("pair_id"),
                    "condition_id": row.get("condition_id"),
                    "transition_type": row.get("transition_type"),
                    "presentation_order": row.get("presentation_order"),
                    "branch_index": row.get("branch_index"),
                    "analysis_split": row.get("analysis_split"),
                    "probe_target": target,
                    "hidden_layer": hidden_layer,
                    "point_index": int(point_index),
                    "point_name": (row.get("point_names") or [])[point_index],
                    "observed_label": str(observed[local_index]),
                    "predicted_label": str(predictions[local_index]),
                    "correct": bool(
                        str(observed[local_index])
                        == str(predictions[local_index])
                    ),
                }
                for class_index, class_name in enumerate(model.classes_):
                    payload[f"prob_{class_name}"] = float(
                        probabilities[local_index, class_index]
                    )
                prediction_rows.append(payload)

    predictions = pd.DataFrame(prediction_rows)
    metric_rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(
        ["probe_target", "hidden_layer", "point_index", "point_name"],
        sort=True,
    ):
        metric_rows.append(
            {
                "probe_target": keys[0],
                "hidden_layer": int(keys[1]),
                "point_index": int(keys[2]),
                "point_name": keys[3],
                "n_evaluation": int(len(group)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        group["observed_label"],
                        group["predicted_label"],
                    )
                ),
                "accuracy": float(group["correct"].mean()),
            }
        )
    outputs = {
        "decoder_selection": pd.DataFrame(selection_rows),
        "selected_decoders": pd.DataFrame(selected_rows),
        "decoder_predictions": predictions,
        "decoder_point_metrics": pd.DataFrame(metric_rows),
    }
    metadata = {
        "fit_split": fit_split,
        "selection_split": selection_split,
        "evaluation_split": evaluation_split,
        "decoder_specs": decoder_specs,
    }
    return outputs, arrays, metadata


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    trace_dirs = [_resolve(workspace_root, path) for path in args.trace_dir]
    out_dir = _resolve(workspace_root, args.out_dir)
    artifact = CombinedArtifact(trace_dirs)
    outputs, arrays, metadata = analyze(
        artifact,
        targets=_csv(args.targets),
        c_values=[float(value) for value in _csv(args.c_values)],
        fit_split=str(args.fit_split),
        selection_split=str(args.selection_split),
        evaluation_split=str(args.evaluation_split),
        min_fit_rows=int(args.min_fit_rows),
        min_selection_rows=int(args.min_selection_rows),
        min_evaluation_rows=int(args.min_evaluation_rows),
        seed=int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = str(path)
    arrays_path = out_dir / "decoder_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-criterion-switch-multiclass-decoders",
            "trace_dirs": [str(path) for path in trace_dirs],
            "out_dir": str(out_dir),
            "decoder_arrays": str(arrays_path),
            "outputs": output_paths,
            "seed": int(args.seed),
            **metadata,
        },
    )
    print(f"out_dir={out_dir}")
    print(outputs["selected_decoders"].to_string(index=False))


if __name__ == "__main__":
    main()
