"""Patch within-pair criterion-switch state differences into reminder replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_json, read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.mech.judge_reasoning import parse_final_verdict
from aisafety.scripts.analyze_judge_reasoning_trajectories import TraceArtifact
from aisafety.scripts.run_d4_bt_stage_contrast import _load_lm
from aisafety.scripts.run_judge_criterion_switch_behavior import (
    _semantic_verdict,
)
from aisafety.scripts.run_judge_reasoning_interventions import (
    _generate_continuation,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--behavior-dir", type=Path, action="append", required=True)
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--decoder-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--layer-target", default="active_criterion")
    parser.add_argument("--hidden-layer", type=int, default=0)
    parser.add_argument("--point-name", default="phase2_prompt_end")
    parser.add_argument("--alphas", default="0.5,1.0")
    parser.add_argument("--include-negative", action="store_true")
    parser.add_argument("--include-shuffled", action="store_true")
    parser.add_argument("--include-placebo", action="store_true")
    parser.add_argument("--analysis-split", default="intervention")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--generation-mode", choices=["greedy", "sample"], default="greedy"
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/mechanistic/criterion_switch_patching_v1"),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _floats(raw: str) -> list[float]:
    return [float(value.strip()) for value in str(raw).split(",") if value.strip()]


class StateIndex:
    def __init__(self, trace_dirs: list[Path]):
        self.artifacts = [TraceArtifact(path) for path in trace_dirs]
        self.locations: dict[str, tuple[TraceArtifact, int]] = {}
        for artifact in self.artifacts:
            for index, row in artifact.frame.reset_index(drop=True).iterrows():
                trace_id = str(row["trace_id"])
                if trace_id in self.locations:
                    raise ValueError(f"Duplicate activation trace_id: {trace_id}")
                self.locations[trace_id] = (artifact, int(index))
        self.cache: dict[tuple[int, int], np.ndarray] = {}

    def state(self, trace_id: str, *, hidden_layer: int, point_name: str) -> np.ndarray:
        artifact, index = self.locations[str(trace_id)]
        row = artifact.frame.iloc[index]
        point_names = list(row.get("point_names") or [])
        if point_name not in point_names:
            raise KeyError(f"{point_name!r} not found for trace {trace_id}")
        point_index = point_names.index(point_name)
        key = (id(artifact), int(hidden_layer))
        if key not in self.cache:
            self.cache[key] = artifact.layer_states(int(hidden_layer))
        return self.cache[key][index, point_index].astype(np.float32)


def _behavior_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path / "switch_traces.jsonl"))
    return rows


def _paired(rows: list[dict[str, Any]], *, analysis_split: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("analysis_split") or "") != str(analysis_split):
            continue
        key = (
            str(row["pair_id"]),
            str(row["presentation_order"]),
            int(row["branch_index"]),
        )
        by_key.setdefault(key, {})[str(row["condition_id"])] = row
    pairs: list[dict[str, Any]] = []
    for key, conditions in by_key.items():
        if {"reminder", "switch"}.issubset(conditions):
            pairs.append(
                {
                    "key": key,
                    "reminder": conditions["reminder"],
                    "switch": conditions["switch"],
                }
            )
    return pairs


def _select_layer(decoder_dir: Path, *, target: str, override: int) -> int:
    if int(override) > 0:
        return int(override)
    manifest = read_json(decoder_dir / "manifest.json")
    specs = manifest.get("decoder_specs") or {}
    if str(target) not in specs:
        raise KeyError(f"Decoder target {target!r} is unavailable.")
    return int(specs[str(target)]["hidden_layer"])


def _cap_pairs(
    pairs: list[dict[str, Any]], *, maximum: int, seed: int
) -> list[dict[str, Any]]:
    ordered = sorted(
        pairs,
        key=lambda row: sha1_hex(
            f"{seed}:criterion-patch:{row['key'][0]}:{row['key'][1]}:{row['key'][2]}"
        ),
    )
    return ordered[: int(maximum)] if int(maximum) > 0 else ordered


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    behavior_dirs = [
        _resolve(workspace_root, path) for path in args.behavior_dir
    ]
    trace_dirs = [_resolve(workspace_root, path) for path in args.trace_dir]
    decoder_dir = _resolve(workspace_root, args.decoder_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    hidden_layer = _select_layer(
        decoder_dir,
        target=str(args.layer_target),
        override=int(args.hidden_layer),
    )
    rows = _behavior_rows(behavior_dirs)
    pairs = _cap_pairs(
        _paired(rows, analysis_split=str(args.analysis_split)),
        maximum=int(args.max_pairs),
        seed=int(args.seed),
    )
    if not pairs:
        raise ValueError("No matched reminder/switch activation pairs found.")
    state_index = StateIndex(trace_dirs)
    for pair in pairs:
        pair["delta"] = (
            state_index.state(
                str(pair["switch"]["trace_id"]),
                hidden_layer=hidden_layer,
                point_name=str(args.point_name),
            )
            - state_index.state(
                str(pair["reminder"]["trace_id"]),
                hidden_layer=hidden_layer,
                point_name=str(args.point_name),
            )
        )

    shuffled = pairs[1:] + pairs[:1]
    placebo_pool = [
        pair for pair in pairs
        if str(pair["switch"].get("transition_type")) == "same_target"
    ]
    if not placebo_pool:
        placebo_pool = pairs
    settings: list[tuple[str, float]] = [("baseline", 0.0)]
    settings.extend(("criterion_delta", alpha) for alpha in _floats(args.alphas))
    if bool(args.include_negative):
        settings.extend(
            ("criterion_delta_negative", -alpha)
            for alpha in _floats(args.alphas)
        )
    if bool(args.include_shuffled):
        settings.append(("shuffled_pair_delta", 1.0))
    if bool(args.include_placebo):
        settings.append(("same_target_placebo_delta", 1.0))

    model, tokenizer = _load_lm(args)
    output_rows: list[dict[str, Any]] = []
    labels = ["A", "B", "C"]
    for index, pair in enumerate(pairs):
        reminder = pair["reminder"]
        for setting, alpha in settings:
            vector: np.ndarray | None
            donor_trace_id = ""
            if setting == "baseline":
                vector = None
            elif setting == "shuffled_pair_delta":
                vector = shuffled[index]["delta"]
                donor_trace_id = str(shuffled[index]["switch"]["trace_id"])
            elif setting == "same_target_placebo_delta":
                donor = placebo_pool[index % len(placebo_pool)]
                vector = donor["delta"]
                donor_trace_id = str(donor["switch"]["trace_id"])
            else:
                vector = pair["delta"]
                donor_trace_id = str(pair["switch"]["trace_id"])
            generation_seed = int(
                sha1_hex(
                    f"{args.seed}:criterion-patch-generation:{reminder['trace_id']}"
                )[:8],
                16,
            )
            generated_ids = _generate_continuation(
                model=model,
                tokenizer=tokenizer,
                input_token_ids=[
                    int(value)
                    for value in reminder["phase2_prompt_token_ids"]
                ],
                hidden_layer=hidden_layer,
                vector=vector,
                alpha=float(alpha),
                max_new_tokens=int(args.max_new_tokens),
                generation_mode=str(args.generation_mode),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                seed=generation_seed,
            )
            text = tokenizer.decode(
                generated_ids, skip_special_tokens=False
            )
            verdict = parse_final_verdict(text, labels=labels)
            semantic = _semantic_verdict(
                verdict, str(reminder["presentation_order"])
            )
            updated_target = str(
                pair["switch"]["phase2_target_semantic"]
            )
            initial_target = str(
                reminder["phase2_target_semantic"]
            )
            output_rows.append(
                {
                    "pair_id": str(reminder["pair_id"]),
                    "trace_id": str(reminder["trace_id"]),
                    "switch_trace_id": str(pair["switch"]["trace_id"]),
                    "donor_trace_id": donor_trace_id,
                    "presentation_order": str(
                        reminder["presentation_order"]
                    ),
                    "branch_index": int(reminder["branch_index"]),
                    "transition_type": str(reminder["transition_type"]),
                    "setting": setting,
                    "alpha": float(alpha),
                    "hidden_layer": hidden_layer,
                    "point_name": str(args.point_name),
                    "generated_tokens": int(len(generated_ids)),
                    "response_text": text,
                    "valid_choice": bool(verdict),
                    "choice": verdict,
                    "choice_semantic": semantic,
                    "initial_target_semantic": initial_target,
                    "updated_target_semantic": updated_target,
                    "updated_target_selected": (
                        None if not semantic else semantic == updated_target
                    ),
                    "initial_target_retained": (
                        None if not semantic else semantic == initial_target
                    ),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "patch_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    frame = pd.DataFrame(output_rows)
    summary = (
        frame.groupby(["setting", "alpha", "transition_type"], sort=True)
        .agg(
            n=("trace_id", "size"),
            n_pairs=("pair_id", "nunique"),
            valid_choice_rate=("valid_choice", "mean"),
            updated_target_selection_rate=("updated_target_selected", "mean"),
            initial_target_retention_rate=("initial_target_retained", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "patch_summary.csv", index=False)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-criterion-switch-patching",
            "behavior_dirs": [str(path) for path in behavior_dirs],
            "trace_dirs": [str(path) for path in trace_dirs],
            "decoder_dir": str(decoder_dir),
            "out_dir": str(out_dir),
            "model_id": str(args.model_id),
            "hidden_layer": hidden_layer,
            "point_name": str(args.point_name),
            "analysis_split": str(args.analysis_split),
            "n_pairs": int(len(pairs)),
            "n_rows": int(len(frame)),
            "patch_rows": str(rows_path),
            "patch_summary": str(out_dir / "patch_summary.csv"),
        },
    )
    print(f"out_dir={out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
