"""Cross-fit fixed-layer activation readouts for criterion confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json
from aisafety.mech.judge_reasoning import deterministic_group_fold
from aisafety.scripts.analyze_judge_criterion_switch_decoders import (
    CombinedArtifact,
    _fit,
    _point_value,
)
from aisafety.scripts.analyze_judge_criterion_switch_pairs import (
    _macro_auc,
    _predict,
)


DEFAULT_TARGET_LAYERS = {
    "active_criterion": 20,
    "criterion_target": 32,
    "current_choice": 28,
    "final_choice": 32,
    "presentation_order": 12,
}
DIFFERENCE_SPECS = (
    {
        "difference_type": "criterion_update",
        "donor_condition": "early_criterion",
        "recipient_condition": "late_criterion",
        "point_names": ("phase1_readout_128",),
        "targets": ("active_criterion", "criterion_target"),
    },
    {
        "difference_type": "evidence_operationalization",
        "donor_condition": "late_evidence",
        "recipient_condition": "late_criterion",
        "point_names": (
            "phase2_readout_0",
            "phase2_readout_32",
            "phase2_readout_128",
            "phase2_readout_384",
        ),
        "targets": ("criterion_target", "current_choice", "final_choice"),
    },
    {
        "difference_type": "explicit_target",
        "donor_condition": "late_explicit_target",
        "recipient_condition": "late_criterion",
        "point_names": ("phase2_readout_0",),
        "targets": ("criterion_target", "current_choice", "final_choice"),
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument(
        "--target-layers",
        default="active_criterion:20,criterion_target:32,current_choice:28,"
        "final_choice:32,presentation_order:12",
    )
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-fold-train-rows", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/"
            "criterion_confirmation_activation_analysis_v1"
        ),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _target_layers(raw: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for item in str(raw).split(","):
        if not item.strip():
            continue
        target, layer = item.split(":", 1)
        values[target.strip()] = int(layer)
    if not values:
        raise ValueError("No target layers were provided.")
    return values


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


def _folds(frame: pd.DataFrame, *, cv_folds: int, seed: int) -> np.ndarray:
    return np.asarray(
        [
            deterministic_group_fold(
                str(pair_id),
                n_folds=int(cv_folds),
                seed=int(seed),
                salt="criterion-confirmation-fixed-readout",
            )
            for pair_id in frame["pair_id"]
        ],
        dtype=int,
    )


def _oof_predictions(
    *,
    frame: pd.DataFrame,
    point_mask: np.ndarray,
    layer_states: dict[int, np.ndarray],
    target_layers: dict[str, int],
    c_value: float,
    cv_folds: int,
    min_fold_train_rows: int,
    seed: int,
) -> pd.DataFrame:
    folds = _folds(frame, cv_folds=cv_folds, seed=seed)
    rows: list[dict[str, Any]] = []
    for target, hidden_layer in target_layers.items():
        if int(hidden_layer) not in layer_states:
            raise KeyError(f"Layer {hidden_layer} for {target} was not captured.")
        states = layer_states[int(hidden_layer)]
        for point_index in range(point_mask.shape[1]):
            labels = _labels(
                frame,
                target=target,
                point_index=point_index,
            )
            valid = point_mask[:, point_index] & (labels != "")
            for fold in range(int(cv_folds)):
                train = valid & (folds != fold)
                test = valid & (folds == fold)
                if (
                    int(train.sum()) < int(min_fold_train_rows)
                    or not test.any()
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
                        "trace_id": source.get("trace_id"),
                        "pair_id": source.get("pair_id"),
                        "condition_id": source.get("condition_id"),
                        "transition_type": source.get("transition_type"),
                        "presentation_order": source.get(
                            "presentation_order"
                        ),
                        "branch_index": source.get("branch_index"),
                        "cv_fold": fold,
                        "probe_target": target,
                        "hidden_layer": int(hidden_layer),
                        "point_index": point_index,
                        "point_name": (
                            source.get("point_names") or []
                        )[point_index],
                        "observed_label": str(labels[trace_index]),
                        "predicted_label": str(predictions[local_index]),
                        "correct": bool(
                            str(labels[trace_index])
                            == str(predictions[local_index])
                        ),
                    }
                    for class_index, class_name in enumerate(model.classes_):
                        payload[f"prob_{class_name}"] = float(
                            probabilities[local_index, class_index]
                        )
                    rows.append(payload)
    return pd.DataFrame(rows)


def _metrics(
    predictions: pd.DataFrame,
    *,
    group_columns: list[str],
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["probe_target", "hidden_layer", "point_index", "point_name"]
    keys.extend(group_columns)
    for values, group in predictions.groupby(keys, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        classes = sorted(
            column.removeprefix("prob_")
            for column in group.columns
            if column.startswith("prob_") and group[column].notna().any()
        )
        probabilities = group[
            [f"prob_{class_name}" for class_name in classes]
        ].fillna(0.0).to_numpy()
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n_traces": int(len(group)),
                "n_pairs": int(group["pair_id"].nunique()),
                "n_classes": int(group["observed_label"].nunique()),
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


def _bootstrap_accuracy(
    predictions: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(
        ["probe_target", "hidden_layer", "point_index", "point_name"],
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
                "hidden_layer": int(keys[1]),
                "point_index": int(keys[2]),
                "point_name": keys[3],
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


def _condition_map(
    frame: pd.DataFrame,
    condition: str,
) -> dict[tuple[str, str, int], int]:
    selected = frame[frame["condition_id"].astype(str).eq(condition)]
    return {
        (
            str(row["pair_id"]),
            str(row["presentation_order"]),
            int(row["branch_index"]),
        ): int(index)
        for index, row in selected.iterrows()
    }


def difference_rows(
    artifact: CombinedArtifact,
    *,
    target_layers: dict[str, int],
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    frame = artifact.frame.reset_index(drop=True)
    point_names = list(frame.iloc[0].get("point_names") or [])
    states = {
        layer: artifact.layer_states(layer)
        for layer in sorted(set(target_layers.values()))
    }
    rows: list[dict[str, Any]] = []
    vectors: dict[int, list[np.ndarray]] = {
        layer: [] for layer in states
    }
    for spec in DIFFERENCE_SPECS:
        donor = _condition_map(frame, str(spec["donor_condition"]))
        recipient = _condition_map(frame, str(spec["recipient_condition"]))
        for key in sorted(set(donor) & set(recipient)):
            donor_index = donor[key]
            recipient_index = recipient[key]
            source = frame.iloc[donor_index]
            for point_name in spec["point_names"]:
                if point_name not in point_names:
                    continue
                point_index = point_names.index(point_name)
                row = {
                    "difference_id": (
                        f"{spec['difference_type']}|{key[0]}|{key[1]}|"
                        f"{key[2]}|{point_name}"
                    ),
                    "difference_type": str(spec["difference_type"]),
                    "pair_id": key[0],
                    "presentation_order": key[1],
                    "branch_index": key[2],
                    "transition_type": source.get("transition_type"),
                    "point_index": point_index,
                    "point_name": point_name,
                    "targets": list(spec["targets"]),
                    "active_criterion": str(
                        source.get("phase2_criterion_id")
                        if point_name.startswith("phase2")
                        else source.get("phase1_criterion_id")
                    ),
                    "criterion_target": str(
                        source.get("phase2_target_semantic")
                        if point_name.startswith("phase2")
                        else source.get("phase1_target_semantic")
                    ),
                    "current_choice": (
                        source.get("point_forced_choices_semantic") or []
                    )[point_index],
                    "final_choice": str(
                        source.get("decoder_final_choice_semantic") or ""
                    ),
                }
                rows.append(row)
                for layer, values in states.items():
                    delta = (
                        values[donor_index, point_index]
                        - values[recipient_index, point_index]
                    )
                    vectors[layer].append(delta.astype(np.float32))
                    row[f"delta_norm_l{layer}"] = float(
                        np.linalg.norm(delta)
                    )
                    denominator = float(
                        np.linalg.norm(values[recipient_index, point_index])
                    )
                    row[f"relative_delta_norm_l{layer}"] = (
                        float(np.linalg.norm(delta)) / denominator
                        if denominator > 0
                        else np.nan
                    )
    return (
        pd.DataFrame(rows),
        {
            layer: np.stack(layer_values, axis=0)
            for layer, layer_values in vectors.items()
            if layer_values
        },
    )


def _difference_oof(
    frame: pd.DataFrame,
    states: dict[int, np.ndarray],
    *,
    target_layers: dict[str, int],
    c_value: float,
    cv_folds: int,
    min_fold_train_rows: int,
    seed: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    folds = _folds(frame, cv_folds=cv_folds, seed=seed)
    rows: list[dict[str, Any]] = []
    for difference_type, group in frame.groupby("difference_type", sort=True):
        indices_all = group.index.to_numpy(dtype=int)
        allowed_targets = set(
            target
            for values in group["targets"]
            for target in values
        )
        for target in sorted(allowed_targets):
            hidden_layer = target_layers[target]
            labels = group[target].fillna("").astype(str).to_numpy()
            valid_local = labels != ""
            for fold in range(int(cv_folds)):
                local_folds = folds[indices_all]
                train_local = valid_local & (local_folds != fold)
                test_local = valid_local & (local_folds == fold)
                if (
                    int(train_local.sum()) < int(min_fold_train_rows)
                    or not test_local.any()
                    or len(set(labels[train_local].tolist())) < 2
                ):
                    continue
                train_indices = indices_all[train_local]
                test_indices = indices_all[test_local]
                center, model = _fit(
                    states[hidden_layer][train_indices],
                    labels[train_local],
                    c_value=float(c_value),
                    seed=int(seed) + fold,
                )
                predictions, probabilities = _predict(
                    model,
                    center,
                    states[hidden_layer][test_indices],
                )
                for local_index, row_index in enumerate(test_indices):
                    source = frame.iloc[int(row_index)]
                    payload = {
                        "difference_id": source["difference_id"],
                        "difference_type": difference_type,
                        "pair_id": source["pair_id"],
                        "presentation_order": source[
                            "presentation_order"
                        ],
                        "branch_index": source["branch_index"],
                        "transition_type": source["transition_type"],
                        "point_index": int(source["point_index"]),
                        "point_name": source["point_name"],
                        "probe_target": target,
                        "hidden_layer": hidden_layer,
                        "cv_fold": fold,
                        "observed_label": labels[test_local][local_index],
                        "predicted_label": str(predictions[local_index]),
                    }
                    payload["correct"] = (
                        payload["observed_label"]
                        == payload["predicted_label"]
                    )
                    for class_index, class_name in enumerate(model.classes_):
                        payload[f"prob_{class_name}"] = float(
                            probabilities[local_index, class_index]
                        )
                    rows.append(payload)
    return pd.DataFrame(rows)


def _difference_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    return _metrics(
        predictions,
        group_columns=["difference_type"],
    )


def _difference_norm_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    norm_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("delta_norm_l")
    )
    for keys, group in frame.groupby(
        ["difference_type", "point_index", "point_name"],
        sort=True,
    ):
        for column in norm_columns:
            hidden_layer = int(column.removeprefix("delta_norm_l"))
            relative = f"relative_delta_norm_l{hidden_layer}"
            rows.append(
                {
                    "difference_type": keys[0],
                    "point_index": int(keys[1]),
                    "point_name": keys[2],
                    "hidden_layer": hidden_layer,
                    "n_rows": int(len(group)),
                    "n_pairs": int(group["pair_id"].nunique()),
                    "mean_delta_norm": float(group[column].mean()),
                    "median_delta_norm": float(group[column].median()),
                    "mean_relative_delta_norm": float(
                        group[relative].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def analyze(
    artifact: CombinedArtifact,
    *,
    target_layers: dict[str, int],
    c_value: float,
    cv_folds: int,
    min_fold_train_rows: int,
    bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    frame = artifact.frame.reset_index(drop=True)
    layer_states = {
        layer: artifact.layer_states(layer)
        for layer in sorted(set(target_layers.values()))
    }
    predictions = _oof_predictions(
        frame=frame,
        point_mask=artifact.point_mask,
        layer_states=layer_states,
        target_layers=target_layers,
        c_value=float(c_value),
        cv_folds=int(cv_folds),
        min_fold_train_rows=int(min_fold_train_rows),
        seed=int(seed),
    )
    differences, difference_states = difference_rows(
        artifact,
        target_layers=target_layers,
    )
    difference_predictions = _difference_oof(
        differences,
        difference_states,
        target_layers=target_layers,
        c_value=float(c_value),
        cv_folds=int(cv_folds),
        min_fold_train_rows=max(int(min_fold_train_rows) // 2, 12),
        seed=int(seed),
    )
    return {
        "oof_predictions": predictions,
        "point_metrics": _metrics(predictions, group_columns=[]),
        "point_metrics_by_condition": _metrics(
            predictions,
            group_columns=["condition_id"],
        ),
        "point_pair_bootstrap": _bootstrap_accuracy(
            predictions,
            samples=int(bootstrap),
            seed=int(seed),
        ),
        "difference_rows": differences,
        "difference_norm_summary": _difference_norm_summary(differences),
        "difference_oof_predictions": difference_predictions,
        "difference_metrics": _difference_metrics(
            difference_predictions
        ),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    trace_dir = _resolve(workspace_root, args.trace_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    target_layers = _target_layers(args.target_layers)
    outputs = analyze(
        CombinedArtifact([trace_dir]),
        target_layers=target_layers,
        c_value=float(args.c_value),
        cv_folds=int(args.cv_folds),
        min_fold_train_rows=int(args.min_fold_train_rows),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-criterion-confirmation-activation-analysis",
            "trace_dir": str(trace_dir),
            "out_dir": str(out_dir),
            "target_layers": target_layers,
            "c_value": float(args.c_value),
            "cv_folds": int(args.cv_folds),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "outputs": paths,
        },
    )
    print(f"out_dir={out_dir}")
    print(outputs["point_pair_bootstrap"].to_string(index=False))
    print(outputs["difference_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
