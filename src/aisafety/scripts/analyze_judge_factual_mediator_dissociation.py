"""Analyze factual mediator/decision coupling from forced readout activations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, sha1_hex, write_json
from aisafety.mech.judge_reasoning import deterministic_group_fold
from aisafety.scripts.analyze_judge_criterion_switch_decoders import (
    CombinedArtifact,
    _fit,
)
from aisafety.scripts.analyze_judge_criterion_switch_pairs import _predict
from aisafety.scripts.analyze_judge_reasoning_trajectories import _normalize_rows


DEFAULT_DATASETS = (
    "arc_challenge",
    "bbh_logical_deduction",
    "gsm8k_verification",
    "math500_verification",
    "truthfulqa",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--point-name", default="readout_2048")
    parser.add_argument("--include-datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument(
        "--probe-targets",
        default="criterion_target,current_choice,final_choice,target_reached",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--min-train-rows", type=int, default=24)
    parser.add_argument("--min-test-rows", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/judge_factual_mediator_dissociation_v1"
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


def _point_index(artifact: CombinedArtifact, point_name: str) -> int:
    for _, row in artifact.frame.iterrows():
        names = list(row.get("point_names") or [])
        if point_name in names:
            return names.index(point_name)
    return int(artifact.n_points - 1)


def _point_value(row: pd.Series, key: str, point_index: int) -> str:
    values = row.get(key)
    if isinstance(values, list) and 0 <= int(point_index) < len(values):
        return str(values[int(point_index)] or "").strip().upper()
    return ""


def labels_for_target(
    frame: pd.DataFrame,
    target: str,
    point_index: int,
) -> np.ndarray:
    if target == "criterion_target":
        return np.asarray(
            [
                _point_value(row, "point_target_semantics", point_index)
                for _, row in frame.iterrows()
            ],
            dtype=object,
        )
    if target == "current_choice":
        return np.asarray(
            [
                _point_value(row, "point_forced_choices_semantic", point_index)
                for _, row in frame.iterrows()
            ],
            dtype=object,
        )
    if target == "final_choice":
        return np.asarray(
            [
                str(row.get("decoder_final_choice_semantic") or "").strip().upper()
                for _, row in frame.iterrows()
            ],
            dtype=object,
        )
    if target == "target_reached":
        target_values = labels_for_target(frame, "criterion_target", point_index)
        choice_values = labels_for_target(frame, "current_choice", point_index)
        return np.asarray(
            [
                int(t in {"A", "B"} and c in {"A", "B"} and t == c)
                for t, c in zip(target_values, choice_values, strict=True)
            ],
            dtype=int,
        )
    raise ValueError(f"Unknown factual probe target: {target}")


def _valid_labels(labels: np.ndarray, target: str) -> np.ndarray:
    if target == "target_reached":
        return np.isin(labels, [0, 1])
    return np.asarray([str(value) in {"A", "B"} for value in labels], dtype=bool)


def _folds(frame: pd.DataFrame, *, cv_folds: int, seed: int) -> np.ndarray:
    return np.asarray(
        [
            deterministic_group_fold(
                str(pair_id),
                n_folds=int(cv_folds),
                seed=int(seed),
                salt="judge-factual-mediator-dissociation",
            )
            for pair_id in frame["pair_id"]
        ],
        dtype=int,
    )


def _auc(labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    try:
        if len(classes) == 2:
            return float(roc_auc_score(labels == classes[1], probabilities[:, 1]))
        return float(
            roc_auc_score(
                labels,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return np.nan


def _bootstrap_mean(
    values: np.ndarray,
    *,
    clusters: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters, dtype=object)
    valid = np.isfinite(values)
    values = values[valid]
    clusters = clusters[valid]
    point = float(np.mean(values)) if len(values) else np.nan
    if int(samples) <= 0 or not len(values):
        return point, np.nan, np.nan
    unique = np.asarray(sorted(set(clusters.tolist())), dtype=object)
    grouped = {cluster: values[clusters == cluster] for cluster in unique}
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(samples)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        draw = np.concatenate([grouped[cluster] for cluster in chosen])
        draws.append(float(np.mean(draw)))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def factual_mediator_summary(
    frame: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    n_points = len(list(frame.iloc[0].get("point_names") or []))
    for point_index in range(n_points):
        point_name = str((frame.iloc[0].get("point_names") or [])[point_index])
        target = labels_for_target(frame, "criterion_target", point_index)
        choice = labels_for_target(frame, "current_choice", point_index)
        reached = np.asarray(
            [
                float(t in {"A", "B"} and c in {"A", "B"} and t == c)
                for t, c in zip(target, choice, strict=True)
            ],
            dtype=float,
        )
        for dataset, group_indices in (
            [("all", np.arange(len(frame)))]
            + [
                (str(dataset), group.index.to_numpy(dtype=int))
                for dataset, group in frame.groupby("source_dataset", sort=True)
            ]
        ):
            selected = reached[group_indices]
            mean, low, high = _bootstrap_mean(
                selected,
                clusters=frame.iloc[group_indices]["pair_id"].to_numpy(dtype=object),
                samples=int(bootstrap),
                seed=int(seed)
                + int(sha1_hex(f"factual:{point_name}:{dataset}")[:6], 16),
            )
            rows.append(
                {
                    "source_dataset": dataset,
                    "point_index": int(point_index),
                    "point_name": point_name,
                    "metric": "target_reached",
                    "n_traces": int(len(group_indices)),
                    "n_pairs": int(frame.iloc[group_indices]["pair_id"].nunique()),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def fit_oof_probes(
    artifact: CombinedArtifact,
    *,
    targets: list[str],
    point_index: int,
    c_value: float,
    cv_folds: int,
    min_train_rows: int,
    min_test_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    frame = artifact.frame.reset_index(drop=True)
    folds = _folds(frame, cv_folds=cv_folds, seed=seed)
    layer_states = {layer: artifact.layer_states(layer) for layer in artifact.hidden_layers}
    selection_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    specs: dict[str, Any] = {}
    for target in targets:
        labels = labels_for_target(frame, target, point_index)
        valid = _valid_labels(labels, target)
        if valid.sum() < int(min_train_rows) + int(min_test_rows):
            continue
        if len(set(labels[valid].tolist())) < 2:
            continue
        candidates: list[dict[str, Any]] = []
        for hidden_layer, states in layer_states.items():
            fold_rows: list[dict[str, Any]] = []
            for fold in range(int(cv_folds)):
                train = valid & (folds != fold)
                test = valid & (folds == fold)
                if (
                    int(train.sum()) < int(min_train_rows)
                    or int(test.sum()) < int(min_test_rows)
                    or len(set(labels[train].tolist())) < 2
                ):
                    continue
                center, model = _fit(
                    states[train, point_index],
                    labels[train],
                    c_value=float(c_value),
                    seed=int(seed) + fold,
                )
                predictions, probabilities = _predict(
                    model,
                    center,
                    states[test, point_index],
                )
                fold_rows.append(
                    {
                        "fold": fold,
                        "n_test": int(test.sum()),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(labels[test], predictions)
                        ),
                        "macro_roc_auc": _auc(labels[test], probabilities, model.classes_),
                    }
                )
            if fold_rows:
                layer_metrics = pd.DataFrame(fold_rows)
                candidates.append(
                    {
                        "probe_target": target,
                        "hidden_layer": int(hidden_layer),
                        "point_index": int(point_index),
                        "point_name": (frame.iloc[0].get("point_names") or [])[point_index],
                        "n_valid": int(valid.sum()),
                        "balanced_accuracy": float(
                            np.average(
                                layer_metrics["balanced_accuracy"],
                                weights=layer_metrics["n_test"],
                            )
                        ),
                        "macro_roc_auc": float(np.nanmean(layer_metrics["macro_roc_auc"])),
                    }
                )
        if not candidates:
            continue
        selection_rows.extend(candidates)
        selected = (
            pd.DataFrame(candidates)
            .assign(finite_auc=lambda df: df["macro_roc_auc"].fillna(-1.0))
            .sort_values(
                ["balanced_accuracy", "finite_auc", "hidden_layer"],
                ascending=[False, False, True],
            )
            .iloc[0]
        )
        hidden_layer = int(selected["hidden_layer"])
        states = layer_states[hidden_layer]
        for fold in range(int(cv_folds)):
            train = valid & (folds != fold)
            test = valid & (folds == fold)
            if (
                int(train.sum()) < int(min_train_rows)
                or int(test.sum()) < int(min_test_rows)
                or len(set(labels[train].tolist())) < 2
            ):
                continue
            center, model = _fit(
                states[train, point_index],
                labels[train],
                c_value=float(c_value),
                seed=int(seed) + fold,
            )
            indices = np.flatnonzero(test)
            predictions, probabilities = _predict(
                model,
                center,
                states[indices, point_index],
            )
            for local_index, trace_index in enumerate(indices):
                source = frame.iloc[int(trace_index)]
                payload = {
                    "trace_id": source["trace_id"],
                    "pair_id": source["pair_id"],
                    "source_dataset": source.get("source_dataset", ""),
                    "cv_fold": int(fold),
                    "probe_target": target,
                    "hidden_layer": hidden_layer,
                    "point_index": int(point_index),
                    "observed_label": str(labels[trace_index]),
                    "predicted_label": str(predictions[local_index]),
                    "correct": bool(
                        str(labels[trace_index]) == str(predictions[local_index])
                    ),
                }
                for class_index, class_name in enumerate(model.classes_):
                    payload[f"prob_{class_name}"] = float(
                        probabilities[local_index, class_index]
                    )
                prediction_rows.append(payload)
        center, model = _fit(
            states[valid, point_index],
            labels[valid],
            c_value=float(c_value),
            seed=int(seed),
        )
        selected_rows.append(
            {
                "probe_target": target,
                "hidden_layer": hidden_layer,
                "point_index": int(point_index),
                "point_name": (frame.iloc[0].get("point_names") or [])[point_index],
                "n_train": int(valid.sum()),
                "classes": "|".join(str(value) for value in model.classes_),
                "balanced_accuracy": float(selected["balanced_accuracy"]),
                "macro_roc_auc": float(selected["macro_roc_auc"]),
            }
        )
        arrays[f"{target}_center"] = center
        arrays[f"{target}_coef"] = model.coef_.astype(np.float32)
        arrays[f"{target}_intercept"] = model.intercept_.astype(np.float32)
        arrays[f"{target}_classes"] = np.asarray(model.classes_, dtype=str)
        specs[target] = {
            "hidden_layer": hidden_layer,
            "point_index": int(point_index),
            "classes": [str(value) for value in model.classes_],
        }
    predictions = pd.DataFrame(prediction_rows)
    return (
        pd.DataFrame(selection_rows),
        pd.DataFrame(selected_rows),
        predictions,
        {"probe_specs": specs, "point_index": int(point_index)},
        arrays,
    )


def _binary_direction(arrays: dict[str, np.ndarray], target: str) -> np.ndarray | None:
    coef = arrays.get(f"{target}_coef")
    if coef is None:
        return None
    if coef.shape[0] == 1:
        direction = coef[0]
    elif coef.shape[0] == 2:
        direction = coef[1] - coef[0]
    else:
        return None
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    return (direction / norm).astype(np.float32)


def subspace_alignment(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = metadata.get("probe_specs") or {}
    targets = sorted(specs)
    for i, left in enumerate(targets):
        left_dir = _binary_direction(arrays, left)
        if left_dir is None:
            continue
        for right in targets[i + 1 :]:
            right_dir = _binary_direction(arrays, right)
            if right_dir is None:
                continue
            same_layer = int(specs[left]["hidden_layer"]) == int(specs[right]["hidden_layer"])
            same_dim = len(left_dir) == len(right_dir)
            rows.append(
                {
                    "left_probe": left,
                    "right_probe": right,
                    "left_hidden_layer": specs[left]["hidden_layer"],
                    "right_hidden_layer": specs[right]["hidden_layer"],
                    "same_layer": bool(same_layer),
                    "cosine": float(np.dot(left_dir, right_dir)) if same_layer and same_dim else np.nan,
                    "abs_cosine": float(abs(np.dot(left_dir, right_dir))) if same_layer and same_dim else np.nan,
                }
            )
    return pd.DataFrame(rows)


def controlled_projection_diagnostics(
    artifact: CombinedArtifact,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    specs = metadata.get("probe_specs") or {}
    frame = artifact.frame.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    controls = ("criterion_target",)
    outcomes = ("current_choice", "final_choice", "target_reached")
    for control in controls:
        direction = _binary_direction(arrays, control)
        if direction is None or control not in specs:
            continue
        layer = int(specs[control]["hidden_layer"])
        point_index = int(specs[control]["point_index"])
        states = artifact.layer_states(layer)[:, point_index]
        center = arrays[f"{control}_center"]
        projection = _normalize_rows(states, center=center) @ direction
        for outcome in outcomes:
            labels = labels_for_target(frame, outcome, point_index)
            valid = _valid_labels(labels, outcome) & np.isfinite(projection)
            if valid.sum() < 8 or len(set(labels[valid].tolist())) < 2:
                continue
            x = projection[valid].reshape(-1, 1)
            model = LogisticRegression(class_weight="balanced", max_iter=1000)
            try:
                model.fit(x, labels[valid])
                predictions = model.predict(x)
                score = float(balanced_accuracy_score(labels[valid], predictions))
            except ValueError:
                score = np.nan
            rows.append(
                {
                    "control_probe": control,
                    "outcome": outcome,
                    "hidden_layer": layer,
                    "point_index": point_index,
                    "n_rows": int(valid.sum()),
                    "n_pairs": int(frame.loc[valid, "pair_id"].nunique()),
                    "projection_balanced_accuracy": score,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    trace_dirs = [_resolve(workspace_root, path) for path in args.trace_dir]
    out_dir = _resolve(workspace_root, args.out_dir)
    artifact = CombinedArtifact(trace_dirs)
    frame = artifact.frame.reset_index(drop=True)
    datasets = set(_csv(args.include_datasets))
    if datasets and "source_dataset" in frame.columns:
        keep = frame["source_dataset"].astype(str).isin(datasets).to_numpy()
        frame = frame.loc[keep].reset_index(drop=True)
        row_indices = np.flatnonzero(keep)
    else:
        row_indices = np.arange(len(frame))
    if not len(frame):
        raise ValueError("No factual traces remain after dataset filtering.")

    class MaskedArtifact:
        def __init__(self, base: CombinedArtifact, filtered: pd.DataFrame, indices: np.ndarray) -> None:
            self._base = base
            self._indices = np.asarray(indices, dtype=int)
            self.frame = filtered
            self.hidden_layers = list(base.hidden_layers)
            self.n_points = int(base.n_points)
            self.point_mask = base.point_mask[self._indices]

        def layer_states(self, hidden_layer: int) -> np.ndarray:
            return self._base.layer_states(hidden_layer)[self._indices]

    masked = MaskedArtifact(artifact, frame, row_indices)
    point_index = _point_index(masked, str(args.point_name))
    selection, selected, predictions, metadata, arrays = fit_oof_probes(
        masked,
        targets=_csv(args.probe_targets),
        point_index=point_index,
        c_value=float(args.c_value),
        cv_folds=int(args.cv_folds),
        min_train_rows=int(args.min_train_rows),
        min_test_rows=int(args.min_test_rows),
        seed=int(args.seed),
    )
    summary = factual_mediator_summary(
        frame,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    align = subspace_alignment(arrays, metadata)
    controls = controlled_projection_diagnostics(masked, arrays, metadata)

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "factual_mediator_summary": summary,
        "probe_selection": selection,
        "selected_probes": selected,
        "probe_predictions": predictions,
        "subspace_alignment": align,
        "controlled_projection_diagnostics": controls,
    }
    output_paths: dict[str, str] = {}
    for name, table in outputs.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        output_paths[name] = str(path)
    arrays_path = out_dir / "probe_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    metadata.update(
        {
            "stage": "judge-factual-mediator-dissociation-analysis",
            "trace_dirs": [str(path) for path in trace_dirs],
            "out_dir": str(out_dir),
            "include_datasets": sorted(datasets),
            "point_name": str(args.point_name),
            "outputs": output_paths,
            "probe_arrays": str(arrays_path),
            "seed": int(args.seed),
        }
    )
    write_json(out_dir / "manifest.json", metadata)
    print(f"out_dir={out_dir}")
    if not summary.empty:
        print("\n=== FACTUAL MEDIATOR SUMMARY ===")
        print(summary.round(3).to_string(index=False))
    if not selected.empty:
        print("\n=== FACTUAL SELECTED PROBES ===")
        print(selected.round(3).to_string(index=False))
    if not align.empty:
        print("\n=== FACTUAL SUBSPACE ALIGNMENT ===")
        print(align.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
