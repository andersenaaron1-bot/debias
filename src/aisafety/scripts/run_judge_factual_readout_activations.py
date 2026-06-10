"""Capture exact forced-decision states from completed factual budget traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from aisafety.config import DEFAULT_CACHE_DIR, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, write_json
from aisafety.mech.judge_reasoning import (
    LastTokenTrajectoryRecorder,
    render_model_prompt,
)
from aisafety.mech.labels import parse_int_list, select_hidden_layers
from aisafety.scripts.run_d4_bt_stage_contrast import _load_lm
from aisafety.scripts.run_judge_reasoning_budget_sweep import _forced_prompt
from aisafety.scripts.run_judge_reasoning_trajectories import TraceShardWriter


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
    parser.add_argument("--budget-run-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument(
        "--include-datasets",
        default=",".join(DEFAULT_DATASETS),
    )
    parser.add_argument("--budget-tokens", default="0,128,512,2048")
    parser.add_argument("--selected-layers", default="")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--tail-layers", type=int, default=2)
    parser.add_argument("--max-score-length", type=int, default=8192)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--compress-shards", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/judge_factual_readout_activations_v1"
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


def _budgets(raw: str) -> list[int]:
    values = [int(value) for value in _csv(raw)]
    if not values or any(value < 0 for value in values):
        raise ValueError("--budget-tokens requires nonnegative integers.")
    return values


def _selected_layers(
    model: Any,
    *,
    raw: str,
    stride: int,
    tail_layers: int,
) -> list[int]:
    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if n_layers <= 0:
        raise ValueError("Model config does not expose num_hidden_layers.")
    layers = (
        parse_int_list(raw)
        if str(raw).strip()
        else select_hidden_layers(
            n_layers,
            stride=int(stride),
            tail_layers=int(tail_layers),
        )
    )
    selected = [int(layer) for layer in layers if 1 <= int(layer) <= n_layers]
    if not selected:
        raise ValueError("No hidden layers selected.")
    return selected


def _metadata_value(row: dict[str, Any], key: str, default: str = "") -> str:
    if row.get(key) is not None:
        return str(row.get(key) or default)
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get(key) is not None:
        return str(metadata.get(key) or default)
    return str(default)


def _capture(
    model: Any,
    *,
    token_ids: list[int],
    hidden_layers: list[int],
) -> np.ndarray:
    import torch

    device = next(
        parameter
        for parameter in model.parameters()
        if parameter.device.type != "meta"
    ).device
    input_ids = torch.as_tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode(), LastTokenTrajectoryRecorder(
        model,
        hidden_layers=hidden_layers,
    ) as recorder:
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    states = recorder.array()
    if states.shape[0] != 1:
        raise RuntimeError(f"Expected one readout state, got {states.shape}.")
    return states[0]


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    run_dir = _resolve(workspace_root, args.budget_run_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    requested_datasets = set(_csv(args.include_datasets))
    requested_budgets = _budgets(args.budget_tokens)
    available_budgets = {
        int(value) for value in manifest.get("budget_tokens") or []
    }
    missing_budgets = sorted(set(requested_budgets) - available_budgets)
    if missing_budgets:
        raise ValueError(
            f"Budget run is missing requested budgets: {missing_budgets}"
        )

    traces = [
        row
        for row in read_jsonl(run_dir / "reasoning_traces.jsonl")
        if str(row.get("reasoning_mode") or "") == "thinking"
        and str(row.get("source_dataset") or "") in requested_datasets
    ]
    if not traces:
        raise ValueError(
            f"No matching reasoning traces found in {run_dir} for "
            f"{sorted(requested_datasets)}"
        )
    scores = {
        (str(row["trace_id"]), int(row["budget_tokens"])): row
        for row in read_jsonl(run_dir / "budget_scores.jsonl")
        if str(row.get("reasoning_mode") or "") == "thinking"
        and str(row.get("source_dataset") or "") in requested_datasets
        and int(row.get("budget_tokens") or 0) in requested_budgets
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "traces.jsonl"
    if trace_path.exists() and not bool(args.resume):
        raise FileExistsError(
            f"{trace_path} exists; use --resume or a new output directory."
        )
    completed = (
        {str(row["trace_id"]) for row in read_jsonl(trace_path)}
        if trace_path.exists()
        else set()
    )
    existing_shards = sorted(out_dir.glob("trajectory_states_*.npz"))
    start_shard = (
        max(int(path.stem.rsplit("_", 1)[-1]) for path in existing_shards) + 1
        if existing_shards
        else 0
    )

    model, tokenizer = _load_lm(args)
    hidden_layers = _selected_layers(
        model,
        raw=str(args.selected_layers),
        stride=int(args.layer_stride),
        tail_layers=int(args.tail_layers),
    )
    writer = TraceShardWriter(
        out_dir,
        n_points=len(requested_budgets),
        hidden_layers=hidden_layers,
        shard_size=int(args.shard_size),
        compress=bool(args.compress_shards),
        start_shard_index=start_shard,
    )
    pending: list[dict[str, Any]] = []
    try:
        for trace in traces:
            trace_id = str(trace["trace_id"])
            if trace_id in completed:
                continue
            score_rows = [scores.get((trace_id, budget)) for budget in requested_budgets]
            if any(row is None for row in score_rows):
                raise ValueError(
                    f"Trace {trace_id} lacks one or more requested budget scores."
                )
            prompt = render_model_prompt(
                trace,
                tokenizer,
                prompt_style=str(manifest.get("prompt_style") or "chat_template"),
                reasoning_mode="thinking",
            )
            generated = [
                int(value) for value in trace.get("generated_token_ids") or []
            ]
            sequences: list[list[int]] = []
            for budget in requested_budgets:
                prefix = tokenizer.decode(
                    generated[: min(int(budget), len(generated))],
                    skip_special_tokens=False,
                )
                forced = _forced_prompt(prompt, prefix, thinking=True)
                encoded = tokenizer(
                    forced,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=int(args.max_score_length),
                )["input_ids"]
                sequences.append([int(value) for value in encoded])
            states = np.stack(
                [
                    _capture(
                        model,
                        token_ids=token_ids,
                        hidden_layers=hidden_layers,
                    )
                    for token_ids in sequences
                ],
                axis=0,
            )
            available = np.asarray(
                [
                    min(int(budget), len(generated))
                    for budget in requested_budgets
                ],
                dtype=np.int32,
            )
            denominator = max(max(requested_budgets), 1)
            positions = np.asarray(
                [float(value) / float(denominator) for value in requested_budgets],
                dtype=np.float32,
            )
            shard, shard_row = writer.add(
                states=states,
                step_indices=available,
                positions=positions,
                label_margins=np.asarray(
                    [
                        float(row["forced_margin_a_minus_b"])
                        for row in score_rows
                        if row is not None
                    ],
                    dtype=np.float32,
                ),
            )
            target_options = [
                str(row.get("target_option") or "")
                for row in score_rows
                if row is not None
            ]
            forced_choices = [
                str(row.get("forced_choice") or "")
                for row in score_rows
                if row is not None
            ]
            criterion_id = _metadata_value(
                trace,
                "criterion_id",
                str(trace.get("comparison_dimension") or ""),
            )
            pending.append(
                {
                    **{
                        key: value
                        for key, value in trace.items()
                        if key
                        not in {
                            "response_text",
                            "generated_token_ids",
                        }
                    },
                    "trace_id": trace_id,
                    "run_label": str(
                        args.run_label or trace.get("run_label") or ""
                    ),
                    "model_id": str(args.model_id),
                    "analysis_split": _metadata_value(
                        trace,
                        "analysis_split",
                        "fit",
                    ),
                    "criterion_id": criterion_id,
                    "point_names": [
                        f"readout_{budget}" for budget in requested_budgets
                    ],
                    "point_active_criteria": [
                        criterion_id
                    ] * len(requested_budgets),
                    "point_target_semantics": target_options,
                    "point_forced_choices_semantic": forced_choices,
                    "decoder_final_choice_semantic": forced_choices[-1],
                    "trajectory_shard": shard,
                    "trajectory_shard_row": int(shard_row),
                }
            )
            if len(pending) >= int(args.shard_size):
                _append_jsonl(trace_path, pending)
                completed.update(str(row["trace_id"]) for row in pending)
                pending.clear()
    finally:
        writer.flush()
    _append_jsonl(trace_path, pending)

    output_rows = read_jsonl(trace_path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-factual-forced-readout-activation-capture",
            "budget_run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "model_id": str(args.model_id),
            "run_label": str(args.run_label),
            "include_datasets": sorted(requested_datasets),
            "budget_tokens": requested_budgets,
            "hidden_layers": hidden_layers,
            "max_score_length": int(args.max_score_length),
            "n_traces": int(len(output_rows)),
        },
    )
    print(f"out_dir={out_dir}")
    print(f"n_traces={len(output_rows)}")
    print(f"hidden_layers={','.join(str(value) for value in hidden_layers)}")


if __name__ == "__main__":
    main()
