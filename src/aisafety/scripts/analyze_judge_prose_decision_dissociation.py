"""Analyze prose/decision dissociation in judge CoT activation traces.

This is a CPU-side analysis over existing criterion-switch activation captures.
It scores generated rationales for criterion-fluent and target-grounded prose,
fits pair-held-out probes for prose, target, and choice variables, and reports
subspace/controlled-projection diagnostics.
"""

from __future__ import annotations

import argparse
import re
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


CRITERION_TERMS: dict[str, tuple[str, ...]] = {
    "coherence": (
        "coherence",
        "coherent",
        "organized",
        "organisation",
        "organization",
        "flow",
        "connected",
        "easy to follow",
        "structure",
    ),
    "consistency": (
        "consistency",
        "consistent",
        "source",
        "fact",
        "factual",
        "unsupported",
        "contradict",
        "faithful",
    ),
    "fluency": (
        "fluency",
        "fluent",
        "grammar",
        "grammatical",
        "readable",
        "natural",
        "sentence",
        "wording",
    ),
    "relevance": (
        "relevance",
        "relevant",
        "important",
        "central",
        "main information",
        "captures",
        "coverage",
        "omission",
    ),
    "correctness": ("correct", "correctness", "accurate", "accuracy", "right"),
    "helpfulness": ("helpful", "helpfulness", "useful", "utility"),
    "complexity": ("complex", "complexity", "detailed", "depth"),
    "verbosity": ("verbose", "verbosity", "concise", "length"),
}
OPTION_A_RE = re.compile(r"\b(option\s+)?a\b", flags=re.IGNORECASE)
OPTION_B_RE = re.compile(r"\b(option\s+)?b\b", flags=re.IGNORECASE)
TARGET_PATTERNS = {
    "A": re.compile(
        r"(?:option\s+)?a\s+(?:is|seems|appears|looks|would be|should be|"
        r"better|wins|preferred|more|stronger)|"
        r"(?:prefer|choose|select|favor)\s+(?:option\s+)?a",
        flags=re.IGNORECASE,
    ),
    "B": re.compile(
        r"(?:option\s+)?b\s+(?:is|seems|appears|looks|would be|should be|"
        r"better|wins|preferred|more|stronger)|"
        r"(?:prefer|choose|select|favor)\s+(?:option\s+)?b",
        flags=re.IGNORECASE,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--point-name",
        default="phase2_readout_384",
        help="Activation point used for probe fitting; falls back to last point.",
    )
    parser.add_argument(
        "--conditions",
        default="free_cot,criterion_scaffold,generic_scaffold,score_evidence",
        help="Condition allowlist for probe/prose summaries; empty means all.",
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
            "artifacts/mechanistic/judge_prose_decision_dissociation_v1"
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


def _response_text(row: pd.Series) -> str:
    parts = [
        str(row.get("phase1_response_text") or ""),
        str(row.get("phase2_response_text") or ""),
    ]
    return "\n\n".join(part for part in parts if part.strip())


def _criterion_id(row: pd.Series) -> str:
    for key in (
        "phase2_criterion_id",
        "updated_criterion_id",
        "criterion_id",
        "active_criterion",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    values = row.get("point_active_criteria")
    if isinstance(values, list) and values:
        return str(values[-1] or "")
    return ""


def _target_semantic(row: pd.Series) -> str:
    for key in ("phase2_target_semantic", "target_semantic", "target_option"):
        value = str(row.get(key) or "").strip().upper()
        if value in {"A", "B"}:
            return value
    values = row.get("point_target_semantics")
    if isinstance(values, list) and values:
        value = str(values[-1] or "").strip().upper()
        if value in {"A", "B"}:
            return value
    return ""


def _final_choice(row: pd.Series) -> str:
    for key in (
        "decoder_final_choice_semantic",
        "final_choice_semantic",
        "natural_choice_semantic",
    ):
        value = str(row.get(key) or "").strip().upper()
        if value in {"A", "B"}:
            return value
    return ""


def _criterion_terms(criterion_id: str, criterion_text: str) -> tuple[str, ...]:
    criterion = str(criterion_id or "").strip().lower()
    values = list(CRITERION_TERMS.get(criterion, ()))
    for token in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", str(criterion_text or "")):
        low = token.lower()
        if low not in values and len(values) < 16:
            values.append(low)
    if criterion and criterion not in values:
        values.insert(0, criterion)
    return tuple(values)


def _mentions_any(text: str, terms: tuple[str, ...]) -> bool:
    low = str(text or "").lower()
    return any(term and str(term).lower() in low for term in terms)


def score_prose_row(row: pd.Series) -> dict[str, Any]:
    text = _response_text(row)
    criterion_id = _criterion_id(row)
    criterion_text = str(
        row.get("phase2_criterion_text")
        or row.get("criterion_text")
        or ""
    )
    target = _target_semantic(row)
    final_choice = _final_choice(row)
    terms = _criterion_terms(criterion_id, criterion_text)
    criterion_mention = bool(
        criterion_id and str(criterion_id).lower() in text.lower()
    )
    criterion_semantics = _mentions_any(text, terms)
    option_a = bool(OPTION_A_RE.search(text))
    option_b = bool(OPTION_B_RE.search(text))
    option_grounding = option_a and option_b
    target_grounding = bool(
        target in TARGET_PATTERNS and TARGET_PATTERNS[target].search(text)
    )
    verdict_binding = bool(target and final_choice and target == final_choice)
    prose_score = float(
        np.mean(
            [
                criterion_mention or criterion_semantics,
                option_grounding,
                target_grounding,
            ]
        )
    )
    return {
        "criterion_id": criterion_id,
        "target_semantic": target,
        "final_choice_semantic": final_choice,
        "criterion_mention": criterion_mention,
        "criterion_semantics": criterion_semantics,
        "option_grounding": option_grounding,
        "target_grounding": target_grounding,
        "verdict_binding": verdict_binding,
        "prose_score": prose_score,
        "response_chars": int(len(text)),
    }


def prose_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "trace_id": row.get("trace_id"),
                "pair_id": row.get("pair_id"),
                "condition_id": row.get("condition_id"),
                "transition_type": row.get("transition_type"),
                "presentation_order": row.get("presentation_order"),
                "branch_index": row.get("branch_index"),
                **score_prose_row(row),
            }
        )
    return pd.DataFrame(rows)


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


def summarize_prose(frame: pd.DataFrame, *, bootstrap: int, seed: int) -> pd.DataFrame:
    metrics = (
        "criterion_mention",
        "criterion_semantics",
        "option_grounding",
        "target_grounding",
        "verdict_binding",
        "prose_score",
    )
    rows: list[dict[str, Any]] = []
    for condition, group in frame.groupby("condition_id", sort=True):
        for metric in metrics:
            mean, low, high = _bootstrap_mean(
                group[metric].to_numpy(dtype=float),
                clusters=group["pair_id"].to_numpy(dtype=object),
                samples=int(bootstrap),
                seed=int(seed) + int(sha1_hex(f"prose:{condition}:{metric}")[:6], 16),
            )
            rows.append(
                {
                    "condition_id": condition,
                    "metric": metric,
                    "n_traces": int(len(group)),
                    "n_pairs": int(group["pair_id"].nunique()),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def _point_index(artifact: CombinedArtifact, point_name: str) -> int:
    for _, row in artifact.frame.iterrows():
        names = list(row.get("point_names") or [])
        if point_name in names:
            return names.index(point_name)
    return int(artifact.n_points - 1)


def _labels(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target == "criterion_prose":
        return np.asarray(frame["criterion_semantics"].astype(int), dtype=int)
    if target == "target_grounded_prose":
        return np.asarray(frame["target_grounding"].astype(int), dtype=int)
    if target == "verdict_binding":
        return np.asarray(frame["verdict_binding"].astype(int), dtype=int)
    if target == "criterion_target":
        return np.asarray(frame["target_semantic"].astype(str), dtype=object)
    if target == "final_choice":
        return np.asarray(frame["final_choice_semantic"].astype(str), dtype=object)
    raise ValueError(f"Unknown probe target: {target}")


def _valid_labels(labels: np.ndarray, target: str) -> np.ndarray:
    if target in {"criterion_prose", "target_grounded_prose", "verdict_binding"}:
        return np.isin(labels, [0, 1])
    return np.asarray([str(value) in {"A", "B"} for value in labels], dtype=bool)


def _folds(frame: pd.DataFrame, *, cv_folds: int, seed: int) -> np.ndarray:
    return np.asarray(
        [
            deterministic_group_fold(
                str(pair_id),
                n_folds=int(cv_folds),
                seed=int(seed),
                salt="judge-prose-decision-dissociation",
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


def fit_oof_probes(
    artifact: CombinedArtifact,
    labels_frame: pd.DataFrame,
    *,
    point_index: int,
    c_value: float,
    cv_folds: int,
    min_train_rows: int,
    min_test_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    targets = (
        "criterion_prose",
        "target_grounded_prose",
        "verdict_binding",
        "criterion_target",
        "final_choice",
    )
    folds = _folds(labels_frame, cv_folds=cv_folds, seed=seed)
    layer_states = {layer: artifact.layer_states(layer) for layer in artifact.hidden_layers}
    selection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    specs: dict[str, Any] = {}
    for target in targets:
        labels = _labels(labels_frame, target)
        valid = _valid_labels(labels, target)
        if valid.sum() < int(min_train_rows) + int(min_test_rows):
            continue
        if len(set(labels[valid].tolist())) < 2:
            continue
        candidates: list[dict[str, Any]] = []
        for hidden_layer, states in layer_states.items():
            rows: list[dict[str, Any]] = []
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
                    model, center, states[test, point_index]
                )
                rows.append(
                    {
                        "fold": fold,
                        "n_test": int(test.sum()),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(labels[test], predictions)
                        ),
                        "macro_roc_auc": _auc(labels[test], probabilities, model.classes_),
                    }
                )
            if rows:
                layer_metrics = pd.DataFrame(rows)
                candidates.append(
                    {
                        "probe_target": target,
                        "hidden_layer": int(hidden_layer),
                        "n_valid": int(valid.sum()),
                        "balanced_accuracy": float(
                            np.average(
                                layer_metrics["balanced_accuracy"],
                                weights=layer_metrics["n_test"],
                            )
                        ),
                        "macro_roc_auc": float(
                            np.nanmean(layer_metrics["macro_roc_auc"])
                        ),
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
                model, center, states[indices, point_index]
            )
            for local_index, trace_index in enumerate(indices):
                source = labels_frame.iloc[int(trace_index)]
                payload = {
                    "trace_id": source["trace_id"],
                    "pair_id": source["pair_id"],
                    "condition_id": source["condition_id"],
                    "transition_type": source.get("transition_type"),
                    "presentation_order": source.get("presentation_order"),
                    "branch_index": source.get("branch_index"),
                    "cv_fold": fold,
                    "probe_target": target,
                    "hidden_layer": hidden_layer,
                    "point_index": int(point_index),
                    "observed_label": str(labels[trace_index]),
                    "predicted_label": str(predictions[local_index]),
                    "correct": bool(str(labels[trace_index]) == str(predictions[local_index])),
                }
                for class_index, class_name in enumerate(model.classes_):
                    payload[f"prob_{class_name}"] = float(probabilities[local_index, class_index])
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
    metric_rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["probe_target", "hidden_layer"], sort=True):
        classes = sorted(
            column.removeprefix("prob_")
            for column in group.columns
            if column.startswith("prob_")
        )
        probabilities = group[[f"prob_{name}" for name in classes]].to_numpy()
        metric_rows.append(
            {
                "probe_target": keys[0],
                "hidden_layer": int(keys[1]),
                "n_traces": int(len(group)),
                "n_pairs": int(group["pair_id"].nunique()),
                "balanced_accuracy": float(
                    balanced_accuracy_score(group["observed_label"], group["predicted_label"])
                ),
                "accuracy": float(group["correct"].mean()),
                "macro_roc_auc": _auc(
                    group["observed_label"].to_numpy(dtype=object),
                    probabilities,
                    np.asarray(classes, dtype=object),
                ),
            }
        )
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
            rows.append(
                {
                    "left_probe": left,
                    "right_probe": right,
                    "left_hidden_layer": specs[left]["hidden_layer"],
                    "right_hidden_layer": specs[right]["hidden_layer"],
                    "same_layer": bool(
                        int(specs[left]["hidden_layer"]) == int(specs[right]["hidden_layer"])
                    ),
                    "cosine": (
                        float(np.dot(left_dir, right_dir))
                        if int(specs[left]["hidden_layer"]) == int(specs[right]["hidden_layer"])
                        and len(left_dir) == len(right_dir)
                        else np.nan
                    ),
                    "abs_cosine": (
                        float(abs(np.dot(left_dir, right_dir)))
                        if int(specs[left]["hidden_layer"]) == int(specs[right]["hidden_layer"])
                        and len(left_dir) == len(right_dir)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def controlled_projection_diagnostics(
    artifact: CombinedArtifact,
    labels_frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    specs = metadata.get("probe_specs") or {}
    rows: list[dict[str, Any]] = []
    controls = ("criterion_prose", "target_grounded_prose")
    outcomes = ("criterion_target", "final_choice", "verdict_binding")
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
            labels = _labels(labels_frame, outcome)
            valid = _valid_labels(labels, outcome) & np.isfinite(projection)
            if valid.sum() < 8 or len(set(labels[valid].tolist())) < 2:
                continue
            for residualize in (False, True):
                x = projection[valid].reshape(-1, 1)
                if residualize:
                    x = np.column_stack([x, labels_frame.loc[valid, "condition_id"].astype("category").cat.codes.to_numpy()])
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
                        "residualized_by_condition": bool(residualize),
                        "n_rows": int(valid.sum()),
                        "n_pairs": int(labels_frame.loc[valid, "pair_id"].nunique()),
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
    conditions = set(_csv(args.conditions))
    if conditions and "condition_id" in frame.columns:
        keep = frame["condition_id"].astype(str).isin(conditions).to_numpy()
        frame = frame.loc[keep].reset_index(drop=True)
        # Rebuild a lightweight artifact view by filtering arrays in-place below.
        # CombinedArtifact itself is left intact; masks are applied to labels and states.
        row_indices = np.flatnonzero(keep)
    else:
        row_indices = np.arange(len(frame))
    if not len(frame):
        raise ValueError("No traces remain after condition filtering.")

    class MaskedArtifact:
        def __init__(
            self,
            base: CombinedArtifact,
            filtered_frame: pd.DataFrame,
            indices: np.ndarray,
        ) -> None:
            self._base = base
            self._indices = np.asarray(indices, dtype=int)
            self.frame = filtered_frame
            self.hidden_layers = list(base.hidden_layers)
            self.n_points = int(base.n_points)
            self.point_mask = base.point_mask[self._indices]

        def layer_states(self, hidden_layer: int) -> np.ndarray:
            return self._base.layer_states(hidden_layer)[self._indices]

    masked = MaskedArtifact(artifact, frame, row_indices)
    prose = prose_frame(frame)
    point_index = _point_index(masked, str(args.point_name))
    selection, selected, predictions, metadata, arrays = fit_oof_probes(
        masked,
        prose,
        point_index=point_index,
        c_value=float(args.c_value),
        cv_folds=int(args.cv_folds),
        min_train_rows=int(args.min_train_rows),
        min_test_rows=int(args.min_test_rows),
        seed=int(args.seed),
    )
    align = subspace_alignment(arrays, metadata)
    controls = controlled_projection_diagnostics(masked, prose, arrays, metadata)
    summary = summarize_prose(prose, bootstrap=int(args.bootstrap), seed=int(args.seed))

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "prose_scores": prose,
        "prose_summary": summary,
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
            "stage": "judge-prose-decision-dissociation-analysis",
            "trace_dirs": [str(path) for path in trace_dirs],
            "out_dir": str(out_dir),
            "conditions": sorted(conditions),
            "point_name": str(args.point_name),
            "outputs": output_paths,
            "probe_arrays": str(arrays_path),
            "seed": int(args.seed),
        }
    )
    write_json(out_dir / "manifest.json", metadata)
    print(f"out_dir={out_dir}")
    if not summary.empty:
        print("\n=== PROSE SUMMARY ===")
        print(summary.round(3).to_string(index=False))
    if not selected.empty:
        print("\n=== SELECTED PROBES ===")
        print(selected.round(3).to_string(index=False))
    if not align.empty:
        print("\n=== SUBSPACE ALIGNMENT ===")
        print(align.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
