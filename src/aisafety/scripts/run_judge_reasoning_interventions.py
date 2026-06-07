"""Replay partial judge reasoning traces with fitted and matched-control steering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_json, resolve_path, sha1_hex, write_json
from aisafety.mech.interventions import find_decoder_layer_module, remove_hooks
from aisafety.mech.judge_reasoning import (
    OneShotLastTokenSteeringHook,
    normalize_choice,
    parse_final_choice,
    random_orthogonal_direction,
)
from aisafety.scripts.analyze_judge_reasoning_trajectories import TraceArtifact
from aisafety.scripts.run_d4_bt_stage_contrast import _load_lm


DEFAULT_TRACE_DIR = Path("artifacts") / "mechanistic" / "judge_reasoning_trajectories_v1"
DEFAULT_ANALYSIS_DIR = Path("artifacts") / "mechanistic" / "judge_reasoning_analysis_v1"
DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "judge_reasoning_interventions_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--allow-artifact-mismatch", action="store_true")
    parser.add_argument("--reasoning-mode", default="thinking")
    parser.add_argument("--analysis-group", default="all")
    parser.add_argument("--probe-target", default="final_choice")
    parser.add_argument("--hidden-layer", type=int, default=0)
    parser.add_argument("--point-index", type=int, default=-1)
    parser.add_argument("--alphas", default="0.5,1.0,2.0")
    parser.add_argument(
        "--include-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--random-control-seeds", default="2101,2102,2103")
    parser.add_argument("--max-traces", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--generation-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in str(value or "").split(",") if part.strip()]
    if not values:
        raise ValueError("At least one steering alpha is required.")
    return values


def _int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value or "").split(",") if part.strip()]


def select_probe_direction(
    analysis_dir: Path,
    *,
    reasoning_mode: str,
    analysis_group: str,
    probe_target: str,
    hidden_layer: int,
    point_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    metrics = pd.read_csv(Path(analysis_dir) / "probe_metrics.csv")
    index = pd.read_csv(Path(analysis_dir) / "probe_direction_index.csv")
    with np.load(Path(analysis_dir) / "probe_directions.npz") as payload:
        directions = payload["directions"].astype(np.float32)
    candidates = metrics[
        (metrics["status"] == "ok")
        & (metrics["reasoning_mode"].astype(str) == str(reasoning_mode))
        & (metrics["analysis_group"].astype(str) == str(analysis_group))
        & (metrics["probe_target"].astype(str) == str(probe_target))
    ].copy()
    if int(hidden_layer) > 0:
        candidates = candidates[candidates["hidden_layer"].astype(int) == int(hidden_layer)]
    if int(point_index) >= 0:
        candidates = candidates[candidates["point_index"].astype(int) == int(point_index)]
    if candidates.empty:
        raise ValueError("No matching successful probe direction was found.")
    selected = candidates.sort_values(
        ["roc_auc", "point_index", "hidden_layer"],
        ascending=[False, True, True],
    ).iloc[0]
    matches = index[
        (index["reasoning_mode"].astype(str) == str(selected["reasoning_mode"]))
        & (index["analysis_group"].astype(str) == str(selected["analysis_group"]))
        & (index["probe_target"].astype(str) == str(selected["probe_target"]))
        & (index["hidden_layer"].astype(int) == int(selected["hidden_layer"]))
        & (index["point_index"].astype(int) == int(selected["point_index"]))
    ]
    if matches.empty:
        raise ValueError("Probe metrics row has no corresponding stored direction.")
    direction_index = int(matches.iloc[0]["direction_index"])
    direction = directions[direction_index]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("Selected probe direction has zero norm.")
    metadata = {
        "reasoning_mode": str(selected["reasoning_mode"]),
        "analysis_group": str(selected["analysis_group"]),
        "probe_target": str(selected["probe_target"]),
        "positive_label": str(selected["positive_label"]),
        "hidden_layer": int(selected["hidden_layer"]),
        "point_index": int(selected["point_index"]),
        "mean_position": float(selected["mean_position"]),
        "roc_auc": float(selected["roc_auc"]),
        "balanced_accuracy": float(selected["balanced_accuracy"]),
        "direction_index": direction_index,
        "original_direction_norm": norm,
    }
    return direction / norm, metadata


def _cap_trace_indices(
    frame: pd.DataFrame,
    *,
    mask: np.ndarray,
    max_traces: int,
    seed: int,
) -> list[int]:
    indices = np.flatnonzero(mask).tolist()
    indices.sort(
        key=lambda index: sha1_hex(
            f"{seed}:judge-reasoning-intervention:{frame.iloc[index].get('trace_id', index)}"
        )
    )
    return indices[: int(max_traces)] if int(max_traces) > 0 else indices


def _generate_continuation(
    *,
    model: Any,
    tokenizer: Any,
    input_token_ids: list[int],
    hidden_layer: int,
    vector: np.ndarray | None,
    alpha: float,
    max_new_tokens: int,
    generation_mode: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> list[int]:
    import torch

    device = next(param for param in model.parameters() if param.device.type != "meta").device
    input_ids = torch.as_tensor([input_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles: list[Any] = []
    if vector is not None and abs(float(alpha)) > 0:
        layer = find_decoder_layer_module(model, hidden_layer=int(hidden_layer))
        hook = OneShotLastTokenSteeringHook(vector, alpha=float(alpha))
        handles.append(layer.register_forward_hook(hook))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": str(generation_mode) == "sample",
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if str(generation_mode) == "sample":
            generation_kwargs.update(
                {
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                }
            )
        with torch.inference_mode():
            output = model.generate(**generation_kwargs)
    finally:
        remove_hooks(handles)
    return output[0, len(input_token_ids):].detach().cpu().tolist()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    trace_dir = _resolve(workspace_root, args.trace_dir)
    analysis_dir = _resolve(workspace_root, args.analysis_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    trace_manifest_path = trace_dir / "manifest.json"
    analysis_manifest_path = analysis_dir / "manifest.json"
    if not trace_manifest_path.is_file() or not analysis_manifest_path.is_file():
        raise FileNotFoundError("Trace and analysis manifests are required for replay.")
    trace_manifest = read_json(trace_manifest_path)
    analysis_manifest = read_json(analysis_manifest_path)
    trace_model_id = str(trace_manifest.get("model_id") or "")
    analysis_trace_dir = Path(str(analysis_manifest.get("trace_dir") or "")).resolve()
    mismatch_reasons: list[str] = []
    if trace_model_id and trace_model_id != str(args.model_id):
        mismatch_reasons.append(
            f"trace model_id={trace_model_id!r} but replay model_id={args.model_id!r}"
        )
    if analysis_trace_dir != trace_dir.resolve():
        mismatch_reasons.append(
            f"analysis trace_dir={analysis_trace_dir} but replay trace_dir={trace_dir.resolve()}"
        )
    if mismatch_reasons and not bool(args.allow_artifact_mismatch):
        raise ValueError(
            "Artifact compatibility check failed: "
            + "; ".join(mismatch_reasons)
            + ". Use --allow-artifact-mismatch only for an intentional diagnostic."
        )
    direction, direction_metadata = select_probe_direction(
        analysis_dir,
        reasoning_mode=str(args.reasoning_mode),
        analysis_group=str(args.analysis_group),
        probe_target=str(args.probe_target),
        hidden_layer=int(args.hidden_layer),
        point_index=int(args.point_index),
    )
    artifact = TraceArtifact(trace_dir)
    frame = artifact.frame.reset_index(drop=True)
    point_index = int(direction_metadata["point_index"])
    mask = (
        (frame["reasoning_mode"].astype(str).to_numpy() == str(args.reasoning_mode))
        & artifact.point_mask[:, point_index]
        & (artifact.step_indices[:, point_index] >= 0)
    )
    if str(args.analysis_group).startswith("dimension:"):
        dimension = str(args.analysis_group).split(":", 1)[1]
        mask &= frame["comparison_dimension"].astype(str).to_numpy() == dimension
    indices = _cap_trace_indices(
        frame,
        mask=mask,
        max_traces=int(args.max_traces),
        seed=int(args.seed),
    )
    if not indices:
        raise ValueError("No replayable traces match the selected direction.")

    settings: list[tuple[str, float, int | None, np.ndarray | None]] = [
        ("baseline", 0.0, None, None)
    ]
    for alpha in _float_list(str(args.alphas)):
        settings.append(("fitted", float(alpha), None, direction))
        if bool(args.include_negative):
            settings.append(("fitted_negative", -float(alpha), None, direction))
    for control_seed in _int_list(str(args.random_control_seeds)):
        control = random_orthogonal_direction(direction, seed=int(control_seed))
        for alpha in _float_list(str(args.alphas)):
            settings.append(("random_orthogonal", float(alpha), int(control_seed), control))

    model, tokenizer = _load_lm(args)
    result_rows: list[dict[str, Any]] = []
    for trace_index in indices:
        row = frame.iloc[int(trace_index)].to_dict()
        prompt_ids = [int(value) for value in row.get("prompt_token_ids") or []]
        generated_ids = [int(value) for value in row.get("generated_token_ids") or []]
        step_index = int(artifact.step_indices[int(trace_index), point_index])
        if not prompt_ids or step_index < 0 or step_index > len(generated_ids):
            continue
        prefix_ids = generated_ids[:step_index]
        input_ids = prompt_ids + prefix_ids
        replay_seed = int(
            sha1_hex(f"{args.seed}:replay:{row.get('trace_id', trace_index)}")[:8],
            16,
        )
        baseline_choice = ""
        trace_results: list[dict[str, Any]] = []
        for setting, alpha, control_seed, vector in settings:
            continuation_ids = _generate_continuation(
                model=model,
                tokenizer=tokenizer,
                input_token_ids=input_ids,
                hidden_layer=int(direction_metadata["hidden_layer"]),
                vector=vector,
                alpha=float(alpha),
                max_new_tokens=int(args.max_new_tokens),
                generation_mode=str(args.generation_mode),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                seed=replay_seed,
            )
            response_ids = prefix_ids + continuation_ids
            response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
            final_choice = parse_final_choice(response_text)
            if setting == "baseline":
                baseline_choice = final_choice
            target_option = normalize_choice(row.get("target_option"))
            trace_results.append(
                {
                    "trace_id": row.get("trace_id"),
                    "comparison_id": row.get("comparison_id"),
                    "pair_id": row.get("pair_id"),
                    "source_dataset": row.get("source_dataset"),
                    "comparison_dimension": row.get("comparison_dimension"),
                    "condition_label": row.get("condition_label"),
                    "reasoning_mode": row.get("reasoning_mode"),
                    "original_final_choice": normalize_choice(row.get("final_choice")),
                    "setting": setting,
                    "alpha": float(alpha),
                    "control_seed": control_seed,
                    "hidden_layer": int(direction_metadata["hidden_layer"]),
                    "point_index": point_index,
                    "step_index": step_index,
                    "trajectory_position": float(artifact.positions[int(trace_index), point_index]),
                    "prefix_generated_tokens": int(len(prefix_ids)),
                    "continuation_tokens": int(len(continuation_ids)),
                    "final_choice": final_choice,
                    "valid_choice": bool(final_choice),
                    "target_selected": None
                    if not final_choice or not target_option
                    else bool(final_choice == target_option),
                    "response_text": response_text,
                    "continuation_token_ids": continuation_ids,
                }
            )
        for result in trace_results:
            result["baseline_final_choice"] = baseline_choice
            result["choice_changed_from_baseline"] = bool(
                result["final_choice"]
                and baseline_choice
                and result["final_choice"] != baseline_choice
            )
        result_rows.extend(trace_results)

    if not result_rows:
        raise ValueError("No intervention rows were produced.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "intervention_rows.jsonl"
    _write_jsonl(rows_path, result_rows)
    result_frame = pd.DataFrame(result_rows)
    summary = (
        result_frame.groupby(
            ["setting", "alpha", "control_seed", "comparison_dimension"],
            dropna=False,
            sort=True,
        )
        .agg(
            n_traces=("trace_id", "count"),
            valid_choice_rate=("valid_choice", "mean"),
            A_choice_rate=("final_choice", lambda values: float((values == "A").mean())),
            target_selected_rate=(
                "target_selected",
                lambda values: float(pd.Series(values).dropna().astype(bool).mean())
                if pd.Series(values).notna().any()
                else np.nan,
            ),
            choice_change_rate=("choice_changed_from_baseline", "mean"),
        )
        .reset_index()
    )
    summary_path = out_dir / "intervention_summary.csv"
    summary.to_csv(summary_path, index=False)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-reasoning-trajectory-intervention",
            "trace_dir": str(trace_dir),
            "analysis_dir": str(analysis_dir),
            "out_dir": str(out_dir),
            "model_id": str(args.model_id),
            "trace_model_id": trace_model_id,
            "allow_artifact_mismatch": bool(args.allow_artifact_mismatch),
            "artifact_mismatch_reasons": mismatch_reasons,
            "direction": direction_metadata,
            "alphas": _float_list(str(args.alphas)),
            "include_negative": bool(args.include_negative),
            "random_control_seeds": _int_list(str(args.random_control_seeds)),
            "max_traces": int(args.max_traces),
            "max_new_tokens": int(args.max_new_tokens),
            "generation_mode": str(args.generation_mode),
            "seed": int(args.seed),
            "n_source_traces": int(len(indices)),
            "n_intervention_rows": int(len(result_rows)),
            "intervention_rows": str(rows_path),
            "intervention_summary": str(summary_path),
            "claim_scope": (
                "Verdict changes support causal influence only for the selected residual direction, "
                "layer, point, alpha, and replay protocol; probe accuracy alone is descriptive."
            ),
        },
    )
    print(f"out_dir={out_dir}")
    print(f"n_source_traces={len(indices)}")
    print(f"n_intervention_rows={len(result_rows)}")


if __name__ == "__main__":
    main()
