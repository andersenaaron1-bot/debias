"""Steer judge forced readouts along prose/decision probe directions.

This runner consumes the output of
``analyze_judge_prose_decision_dissociation``.  It applies selected probe
directions at the final token of a forced verdict readout, then measures
whether the readout moves toward the active target.  The intended contrast is
between directions that encode criterion-fluent prose and directions that
encode decision-ready target/choice variables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import (
    read_json,
    read_jsonl,
    resolve_path,
    sha1_hex,
    write_json,
)
from aisafety.scripts.run_d4_bt_stage_contrast import _load_lm
from aisafety.scripts.run_judge_criterion_confirmation_patching import (
    _probabilities,
    _score_readout,
)
from aisafety.scripts.run_judge_criterion_switch_behavior import (
    _forced_prompt,
    _semantic_verdict,
)
from aisafety.scripts.run_judge_reasoning_budget_sweep import (
    _single_token_label_ids,
)


PATCH_METRICS = (
    "target_selected",
    "target_probability",
    "target_logit_margin",
    "choice_confidence",
)
DEFAULT_PROBES = (
    "criterion_prose,"
    "target_grounded_prose,"
    "verdict_binding,"
    "criterion_target,"
    "final_choice"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--labels", default="A,B,C")
    parser.add_argument("--probe-targets", default=DEFAULT_PROBES)
    parser.add_argument(
        "--conditions",
        default="free_cot,criterion_scaffold,generic_scaffold,score_evidence",
    )
    parser.add_argument("--transition-types", default="")
    parser.add_argument("--include-orders", default="original,swapped")
    parser.add_argument("--branch-index", type=int, default=0)
    parser.add_argument("--stage", choices=["phase1", "phase2"], default="phase2")
    parser.add_argument("--budget-tokens", type=int, default=384)
    parser.add_argument("--alphas", default="-2.0,-1.0,0.0,1.0,2.0")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-score-length", type=int, default=8192)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--raw-directions",
        action="store_true",
        help="Do not L2-normalize probe coefficients before steering.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/"
            "judge_prose_decision_direction_patching_v1"
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


def _floats(raw: str) -> list[float]:
    return [float(value) for value in _csv(raw)]


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in values:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _bootstrap_mean(
    values: pd.Series,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    point = float(np.mean(array)) if len(array) else np.nan
    if int(samples) <= 0 or not len(array):
        return point, np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    draws = [
        float(np.mean(rng.choice(array, size=len(array), replace=True)))
        for _ in range(int(samples))
    ]
    return (
        point,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _target_semantic(row: dict[str, Any]) -> str:
    for key in ("phase2_target_semantic", "target_semantic", "target_option"):
        value = str(row.get(key) or "").strip().upper()
        if value in {"A", "B"}:
            return value
    return ""


def _target_displayed(row: dict[str, Any]) -> str:
    value = str(row.get("phase2_target_option") or "").strip().upper()
    if value in {"A", "B"}:
        return value
    semantic = _target_semantic(row)
    if not semantic:
        return ""
    if str(row.get("presentation_order") or "") == "swapped":
        return {"A": "B", "B": "A"}.get(semantic, "")
    return semantic


def _forced_readout(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    stage: str,
    budget_tokens: int,
) -> str:
    token_key = f"{stage}_response_token_ids"
    text_key = f"{stage}_response_text"
    prompt_key = f"{stage}_prompt_text"
    if token_key in row and row[token_key] is not None:
        response_ids = [int(value) for value in row[token_key]]
    else:
        response_ids = tokenizer(
            str(row.get(text_key) or ""),
            add_special_tokens=False,
        )["input_ids"]
    prefix = tokenizer.decode(
        response_ids[: min(int(budget_tokens), len(response_ids))],
        skip_special_tokens=False,
    )
    return _forced_prompt(str(row[prompt_key]), prefix, thinking=True)


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    conditions: set[str],
    transition_types: set[str],
    include_orders: set[str],
    branch_index: int,
    max_pairs: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if conditions and str(row.get("condition_id") or "") not in conditions:
            continue
        if transition_types and str(row.get("transition_type") or "") not in transition_types:
            continue
        if str(row.get("presentation_order") or "") not in include_orders:
            continue
        if int(row.get("branch_index") or 0) != int(branch_index):
            continue
        if _target_displayed(row) not in {"A", "B"}:
            continue
        if _target_semantic(row) not in {"A", "B"}:
            continue
        selected.append(row)
    if int(max_pairs) <= 0:
        return selected
    pair_ids = sorted(
        {str(row["pair_id"]) for row in selected},
        key=lambda pair_id: sha1_hex(
            f"{seed}:prose-decision-patch-cap:{pair_id}"
        ),
    )[: int(max_pairs)]
    allowed = set(pair_ids)
    return [row for row in selected if str(row["pair_id"]) in allowed]


def _load_probe_payload(analysis_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = read_json(analysis_dir / "manifest.json")
    candidates: list[Path] = []
    raw_arrays = manifest.get("probe_arrays")
    if raw_arrays:
        raw_path = Path(str(raw_arrays))
        candidates.append(raw_path if raw_path.is_absolute() else analysis_dir / raw_path)
    candidates.append(analysis_dir / "probe_arrays.npz")
    arrays_path = next((path for path in candidates if path.exists()), None)
    if arrays_path is None:
        raise FileNotFoundError(
            "Could not find probe arrays; checked "
            + ", ".join(str(path) for path in candidates)
        )
    arrays_file = np.load(arrays_path, allow_pickle=False)
    arrays = {name: arrays_file[name] for name in arrays_file.files}
    specs = manifest.get("probe_specs") or {}
    if not isinstance(specs, dict) or not specs:
        raise ValueError(f"No probe_specs found in {analysis_dir / 'manifest.json'}")
    return manifest, arrays


def _class_values(arrays: dict[str, np.ndarray], probe_target: str) -> list[str]:
    values = arrays[f"{probe_target}_classes"]
    return [str(value) for value in values.tolist()]


def _direction_for_probe(
    arrays: dict[str, np.ndarray],
    probe_target: str,
    *,
    desired_label: str | None,
    normalize: bool,
) -> tuple[np.ndarray, str]:
    coef = np.asarray(arrays[f"{probe_target}_coef"], dtype=np.float32)
    classes = _class_values(arrays, probe_target)
    if coef.ndim != 2 or coef.shape[1] <= 0:
        raise ValueError(f"Invalid coefficient array for {probe_target}")
    label = str(desired_label or classes[-1])
    if coef.shape[0] == 1:
        direction = coef[0].copy()
        positive = classes[-1] if len(classes) >= 2 else label
        if str(label) != str(positive):
            if len(classes) >= 2 and str(label) == str(classes[0]):
                direction = -direction
            elif str(label) not in {"1", "True", "true", "yes"}:
                direction = -direction
    else:
        if label not in classes:
            label = classes[-1]
        index = classes.index(label)
        others = [i for i in range(coef.shape[0]) if i != index]
        baseline = np.mean(coef[others], axis=0) if others else 0.0
        direction = coef[index] - baseline
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError(f"Near-zero probe direction for {probe_target}")
    if normalize:
        direction = direction / norm
    return direction.astype(np.float32), label


def _desired_label(probe_target: str, row: dict[str, Any]) -> str | None:
    if probe_target in {"criterion_target", "final_choice"}:
        return _target_semantic(row)
    return None


def _summarize_patch_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    order_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["probe_target", "condition_id", "alpha", "pair_id"],
        sort=True,
    ):
        choices = [
            value
            for value in group["predicted_semantic"].astype(str)
            if value in {"A", "B", "C"}
        ]
        order_rows.append(
            {
                "probe_target": keys[0],
                "condition_id": keys[1],
                "alpha": float(keys[2]),
                "pair_id": keys[3],
                "order_complete": len(choices) >= 2,
                "order_consistent": len(choices) >= 2 and len(set(choices)) == 1,
            }
        )
    order_frame = pd.DataFrame(order_rows)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["probe_target", "condition_id", "alpha"],
        sort=True,
    ):
        order = order_frame[
            order_frame["probe_target"].eq(keys[0])
            & order_frame["condition_id"].eq(keys[1])
            & order_frame["alpha"].eq(float(keys[2]))
        ]
        complete = order[order["order_complete"]]
        rows.append(
            {
                "probe_target": keys[0],
                "condition_id": keys[1],
                "alpha": float(keys[2]),
                "n_rows": int(len(group)),
                "n_pairs": int(group["pair_id"].nunique()),
                "target_adoption_rate": float(group["target_selected"].mean()),
                "mean_target_probability": float(group["target_probability"].mean()),
                "mean_target_logit_margin": float(group["target_logit_margin"].mean()),
                "mean_choice_confidence": float(group["choice_confidence"].mean()),
                "order_consistent_rate": (
                    float(complete["order_consistent"].mean())
                    if not complete.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _patch_effects(
    frame: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    key_cols = [
        "probe_target",
        "condition_id",
        "pair_id",
        "presentation_order",
        "branch_index",
    ]
    for keys, group in frame.groupby(["probe_target", "condition_id"], sort=True):
        baseline = group[group["alpha"].eq(0.0)][key_cols + list(PATCH_METRICS)]
        if baseline.empty:
            continue
        for alpha in sorted(value for value in group["alpha"].unique() if float(value) != 0.0):
            selected = group[group["alpha"].eq(float(alpha))]
            merged = selected[key_cols + list(PATCH_METRICS)].merge(
                baseline,
                on=key_cols,
                suffixes=("_left", "_baseline"),
            )
            if merged.empty:
                continue
            for metric in PATCH_METRICS:
                merged["difference"] = (
                    merged[f"{metric}_left"].astype(float)
                    - merged[f"{metric}_baseline"].astype(float)
                )
                pair_values = merged.groupby("pair_id", sort=True)["difference"].mean()
                point, low, high = _bootstrap_mean(
                    pair_values,
                    samples=int(bootstrap),
                    seed=int(
                        sha1_hex(
                            f"{seed}:prose-patch:{keys[0]}:{keys[1]}:{alpha}:{metric}"
                        )[:8],
                        16,
                    ),
                )
                rows.append(
                    {
                        "probe_target": keys[0],
                        "condition_id": keys[1],
                        "alpha": float(alpha),
                        "reference_alpha": 0.0,
                        "metric": metric,
                        "n_pairs": int(len(pair_values)),
                        "mean": point,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    behavior_dir = _resolve(workspace_root, args.behavior_dir)
    analysis_dir = _resolve(workspace_root, args.analysis_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    manifest, arrays = _load_probe_payload(analysis_dir)
    specs = manifest.get("probe_specs") or {}
    probe_targets = [
        target for target in _csv(args.probe_targets) if target in specs
    ]
    if not probe_targets:
        raise ValueError("None of --probe-targets are present in analysis manifest.")
    rows = read_jsonl(behavior_dir / "switch_traces.jsonl")
    selected_rows = _filter_rows(
        rows,
        conditions=set(_csv(args.conditions)),
        transition_types=set(_csv(args.transition_types)),
        include_orders=set(_csv(args.include_orders)),
        branch_index=int(args.branch_index),
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
    )
    if not selected_rows:
        raise ValueError("No behavior rows match the patching filters.")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "patch_rows.jsonl"
    if rows_path.exists() and not bool(args.resume):
        raise FileExistsError(
            f"{rows_path} exists; use --resume or choose a new output directory."
        )
    existing = read_jsonl(rows_path) if rows_path.exists() else []
    completed = {
        (
            str(row["probe_target"]),
            str(row["trace_id"]),
            str(row["direction_label"]),
            float(row["alpha"]),
        )
        for row in existing
    }

    model, tokenizer = _load_lm(args)
    labels = _csv(args.labels)
    label_ids = _single_token_label_ids(tokenizer, labels)
    label_index = {label: index for index, label in enumerate(labels)}
    alphas = _floats(args.alphas)
    if 0.0 not in alphas:
        alphas = [0.0] + alphas
    new_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        prompt = _forced_readout(
            row,
            tokenizer=tokenizer,
            stage=str(args.stage),
            budget_tokens=int(args.budget_tokens),
        )
        target_displayed = _target_displayed(row)
        target_semantic = _target_semantic(row)
        if target_displayed not in label_index:
            continue
        target_position = label_index[target_displayed]
        for probe_target in probe_targets:
            spec = specs[probe_target]
            direction, direction_label = _direction_for_probe(
                arrays,
                probe_target,
                desired_label=_desired_label(probe_target, row),
                normalize=not bool(args.raw_directions),
            )
            for alpha in alphas:
                completion_key = (
                    probe_target,
                    str(row["trace_id"]),
                    direction_label,
                    float(alpha),
                )
                if completion_key in completed:
                    continue
                vector = None if float(alpha) == 0.0 else direction
                logits = _score_readout(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    label_ids=label_ids,
                    hidden_layer=int(spec["hidden_layer"]),
                    vector=vector,
                    alpha=float(alpha),
                    max_length=int(args.max_score_length),
                )
                probabilities = _probabilities(logits)
                predicted_displayed = labels[int(np.argmax(logits))]
                predicted_semantic = _semantic_verdict(
                    predicted_displayed,
                    str(row.get("presentation_order") or ""),
                )
                alternatives = np.delete(logits, target_position)
                sorted_probabilities = np.sort(probabilities)[::-1]
                output = {
                    "patch_id": sha1_hex(
                        f"{probe_target}:{row['trace_id']}:{direction_label}:{alpha}"
                    ),
                    "probe_target": probe_target,
                    "direction_label": str(direction_label),
                    "pair_id": str(row["pair_id"]),
                    "trace_id": str(row["trace_id"]),
                    "condition_id": str(row.get("condition_id") or ""),
                    "transition_type": str(row.get("transition_type") or ""),
                    "presentation_order": str(row.get("presentation_order") or ""),
                    "branch_index": int(row.get("branch_index") or 0),
                    "hidden_layer": int(spec["hidden_layer"]),
                    "point_index": int(spec["point_index"]),
                    "stage": str(args.stage),
                    "budget_tokens": int(args.budget_tokens),
                    "alpha": float(alpha),
                    "direction_norm": float(np.linalg.norm(direction)),
                    "target_displayed": target_displayed,
                    "target_semantic": target_semantic,
                    "predicted_displayed": predicted_displayed,
                    "predicted_semantic": predicted_semantic,
                    "target_selected": bool(predicted_semantic == target_semantic),
                    "target_probability": float(probabilities[target_position]),
                    "target_logit_margin": float(
                        logits[target_position] - float(np.max(alternatives))
                    ),
                    "choice_confidence": float(
                        sorted_probabilities[0] - sorted_probabilities[1]
                    ),
                }
                for index, label in enumerate(labels):
                    output[f"logit_{label}"] = float(logits[index])
                    output[f"prob_{label}"] = float(probabilities[index])
                _append_jsonl(rows_path, [output])
                new_rows.append(output)
                completed.add(completion_key)

    frame = pd.DataFrame(read_jsonl(rows_path))
    summary = _summarize_patch_rows(frame)
    effects = _patch_effects(
        frame,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    frame.to_csv(out_dir / "patch_rows.csv", index=False)
    summary.to_csv(out_dir / "patch_summary.csv", index=False)
    effects.to_csv(out_dir / "patch_effects.csv", index=False)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-prose-decision-direction-patching",
            "behavior_dir": str(behavior_dir),
            "analysis_dir": str(analysis_dir),
            "out_dir": str(out_dir),
            "model_id": str(args.model_id),
            "probe_targets": probe_targets,
            "conditions": _csv(args.conditions),
            "transition_types": _csv(args.transition_types),
            "include_orders": _csv(args.include_orders),
            "branch_index": int(args.branch_index),
            "stage_name": str(args.stage),
            "budget_tokens": int(args.budget_tokens),
            "alphas": alphas,
            "directions_normalized": not bool(args.raw_directions),
            "max_pairs": int(args.max_pairs),
            "n_patch_rows": int(len(frame)),
            "n_new_patch_rows": int(len(new_rows)),
            "outputs": {
                "patch_rows_jsonl": str(rows_path),
                "patch_rows_csv": str(out_dir / "patch_rows.csv"),
                "patch_summary_csv": str(out_dir / "patch_summary.csv"),
                "patch_effects_csv": str(out_dir / "patch_effects.csv"),
            },
            "seed": int(args.seed),
        },
    )
    print(f"out_dir={out_dir}")
    print(f"n_patch_rows={len(frame)}")
    if not summary.empty:
        print("\n=== DIRECTION PATCH SUMMARY ===")
        print(summary.round(3).to_string(index=False))
    if not effects.empty:
        print("\n=== DIRECTION PATCH EFFECTS ===")
        print(effects.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
