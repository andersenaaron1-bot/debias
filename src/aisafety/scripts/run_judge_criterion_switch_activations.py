"""Replay criterion-switch prefixes and capture point-aligned residual states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from aisafety.config import DEFAULT_CACHE_DIR, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, write_json
from aisafety.mech.judge_reasoning import LastTokenTrajectoryRecorder
from aisafety.mech.labels import parse_int_list, select_hidden_layers
from aisafety.scripts.run_d4_bt_stage_contrast import _load_lm
from aisafety.scripts.run_judge_reasoning_trajectories import TraceShardWriter


POINT_SPECS = (
    ("phase1_prompt_end", "phase1", 0),
    ("phase1_64", "phase1", 64),
    ("phase1_128", "phase1", 128),
    ("phase2_prompt_end", "phase2", 0),
    ("phase2_32", "phase2", 32),
    ("phase2_128", "phase2", 128),
    ("phase2_384", "phase2", 384),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument(
        "--include-conditions",
        default="",
        help="Optional comma-separated condition allowlist.",
    )
    parser.add_argument("--selected-layers", default="")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--tail-layers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--compress-shards", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/mechanistic/criterion_switch_activations_v1"),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _csv(raw: str) -> list[str]:
    return [value.strip() for value in str(raw).split(",") if value.strip()]


def _selected_layers(
    model: Any, *, raw: str, stride: int, tail_layers: int
) -> list[int]:
    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if n_layers <= 0:
        raise ValueError("Model config does not expose num_hidden_layers.")
    layers = (
        parse_int_list(raw)
        if str(raw).strip()
        else select_hidden_layers(
            n_layers, stride=int(stride), tail_layers=int(tail_layers)
        )
    )
    selected = [int(layer) for layer in layers if 1 <= int(layer) <= n_layers]
    if not selected:
        raise ValueError("No hidden layers selected.")
    return selected


def point_token_sequences(row: dict[str, Any]) -> list[list[int]]:
    phase1_prompt = [int(value) for value in row["phase1_prompt_token_ids"]]
    phase1_response = [
        int(value) for value in row["phase1_response_token_ids"]
    ]
    phase2_prompt = [int(value) for value in row["phase2_prompt_token_ids"]]
    phase2_response = [
        int(value) for value in row["phase2_response_token_ids"]
    ]
    return [
        phase1_prompt,
        phase1_prompt + phase1_response[:64],
        phase1_prompt + phase1_response[:128],
        phase2_prompt,
        phase2_prompt + phase2_response[:32],
        phase2_prompt + phase2_response[:128],
        phase2_prompt + phase2_response[:384],
    ]


def point_labels(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    return (
        [str(row.get("phase1_criterion_id") or "")] * 3
        + [str(row.get("phase2_criterion_id") or "")] * 4,
        [str(row.get("phase1_target_semantic") or "")] * 3
        + [str(row.get("phase2_target_semantic") or "")] * 4,
    )


def _capture(
    model: Any,
    *,
    token_ids: list[int],
    hidden_layers: list[int],
) -> np.ndarray:
    import torch

    if not token_ids:
        raise ValueError("Cannot capture an empty prefix.")
    device = next(
        parameter
        for parameter in model.parameters()
        if parameter.device.type != "meta"
    ).device
    input_ids = torch.as_tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode(), LastTokenTrajectoryRecorder(
        model, hidden_layers=hidden_layers
    ) as recorder:
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    states = recorder.array()
    if states.shape[0] != 1:
        raise RuntimeError(
            f"Expected one captured state per layer, got {states.shape}."
        )
    return states[0]


def _forced_final_choice(row: dict[str, Any]) -> str:
    natural = str(row.get("final_choice_semantic") or "")
    if natural:
        return natural
    checkpoints = row.get("phase2_checkpoints") or []
    return (
        str(checkpoints[-1].get("forced_choice_semantic") or "")
        if checkpoints
        else ""
    )


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    behavior_dir = _resolve(workspace_root, args.behavior_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    traces = read_jsonl(behavior_dir / "switch_traces.jsonl")
    include_conditions = set(_csv(args.include_conditions))
    if include_conditions:
        traces = [
            row
            for row in traces
            if str(row.get("condition_id") or "") in include_conditions
        ]
    if not traces:
        raise ValueError(f"No switch traces found in {behavior_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "traces.jsonl"
    if trace_path.exists() and not bool(args.resume):
        raise FileExistsError(
            f"{trace_path} exists; use --resume or a new output directory."
        )
    completed = {
        str(row["trace_id"]) for row in read_jsonl(trace_path)
    } if trace_path.exists() else set()
    existing_shards = sorted(out_dir.glob("trajectory_states_*.npz"))
    start_shard = (
        max(int(path.stem.rsplit("_", 1)[-1]) for path in existing_shards) + 1
        if existing_shards
        else 0
    )

    model, _tokenizer = _load_lm(args)
    hidden_layers = _selected_layers(
        model,
        raw=str(args.selected_layers),
        stride=int(args.layer_stride),
        tail_layers=int(args.tail_layers),
    )
    writer = TraceShardWriter(
        out_dir,
        n_points=len(POINT_SPECS),
        hidden_layers=hidden_layers,
        shard_size=int(args.shard_size),
        compress=bool(args.compress_shards),
        start_shard_index=start_shard,
    )
    new_rows: list[dict[str, Any]] = []
    try:
        for row in traces:
            trace_id = str(row["trace_id"])
            if trace_id in completed:
                continue
            sequences = point_token_sequences(row)
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
            phase1_n = int(row.get("phase1_generated_tokens") or 0)
            step_indices = np.asarray(
                [
                    0,
                    min(64, phase1_n),
                    min(128, phase1_n),
                    phase1_n,
                    phase1_n
                    + min(32, int(row.get("phase2_generated_tokens") or 0)),
                    phase1_n
                    + min(128, int(row.get("phase2_generated_tokens") or 0)),
                    phase1_n
                    + min(384, int(row.get("phase2_generated_tokens") or 0)),
                ],
                dtype=np.int32,
            )
            positions = np.linspace(
                0.0, 1.0, num=len(POINT_SPECS), dtype=np.float32
            )
            shard, shard_row = writer.add(
                states=states,
                step_indices=step_indices,
                positions=positions,
                label_margins=np.full(
                    (len(POINT_SPECS),), np.nan, dtype=np.float32
                ),
            )
            criteria, targets = point_labels(row)
            new_rows.append(
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "phase1_prompt_token_ids",
                            "phase1_response_token_ids",
                            "phase2_prompt_token_ids",
                            "phase2_response_token_ids",
                        }
                    },
                    "trace_id": trace_id,
                    "run_label": str(args.run_label or row.get("run_label") or ""),
                    "model_id": str(args.model_id),
                    "point_names": [name for name, _stage, _budget in POINT_SPECS],
                    "point_active_criteria": criteria,
                    "point_target_semantics": targets,
                    "decoder_final_choice_semantic": _forced_final_choice(row),
                    "trajectory_shard": shard,
                    "trajectory_shard_row": int(shard_row),
                }
            )
            if len(new_rows) >= int(args.shard_size):
                with trace_path.open("a", encoding="utf-8") as handle:
                    for output_row in new_rows:
                        handle.write(
                            json.dumps(
                                output_row,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                new_rows.clear()
    finally:
        writer.flush()
    if new_rows:
        with trace_path.open("a", encoding="utf-8") as handle:
            for output_row in new_rows:
                handle.write(
                    json.dumps(
                        output_row, ensure_ascii=False, sort_keys=True
                    )
                    + "\n"
                )
    output_rows = read_jsonl(trace_path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-criterion-switch-activation-capture",
            "behavior_dir": str(behavior_dir),
            "out_dir": str(out_dir),
            "model_id": str(args.model_id),
            "run_label": str(args.run_label),
            "include_conditions": sorted(include_conditions),
            "hidden_layers": hidden_layers,
            "point_specs": [
                {"name": name, "stage": stage, "budget_tokens": budget}
                for name, stage, budget in POINT_SPECS
            ],
            "n_traces": int(len(output_rows)),
        },
    )
    print(f"out_dir={out_dir}")
    print(f"n_traces={len(output_rows)}")
    print(f"hidden_layers={','.join(str(value) for value in hidden_layers)}")


if __name__ == "__main__":
    main()
