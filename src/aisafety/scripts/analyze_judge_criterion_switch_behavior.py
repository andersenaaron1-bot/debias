"""Summarize staged criterion-switch behavior across one or more run shards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/mechanistic/criterion_switch_behavior_analysis_v1"),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _checkpoint_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat: list[dict[str, Any]] = []
    for row in rows:
        common = {
            key: row.get(key)
            for key in (
                "trace_id",
                "pair_id",
                "episode_id",
                "run_label",
                "model_id",
                "branch_index",
                "condition_id",
                "transition_type",
                "analysis_split",
                "presentation_order",
                "phase1_criterion_id",
                "phase2_criterion_id",
                "phase1_target_semantic",
                "phase2_target_semantic",
                "final_choice_semantic",
                "valid_choice",
            )
        }
        for stage, checkpoints in (
            ("phase1", row.get("phase1_checkpoints") or []),
            ("phase2", row.get("phase2_checkpoints") or []),
        ):
            for checkpoint in checkpoints:
                flat.append({**common, "stage": stage, **checkpoint})
    return pd.DataFrame(flat)


def _mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def analyze(run_dirs: list[Path]) -> dict[str, pd.DataFrame]:
    traces: list[dict[str, Any]] = []
    direct: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        traces.extend(read_jsonl(run_dir / "switch_traces.jsonl"))
        direct.extend(read_jsonl(run_dir / "direct_rows.jsonl"))
    if not traces:
        raise ValueError("No criterion-switch traces were found.")
    trace_frame = pd.DataFrame(traces)
    checkpoint_frame = _checkpoint_frame(traces)

    summary_rows: list[dict[str, Any]] = []
    group_columns = [
        "condition_id",
        "transition_type",
        "stage",
        "budget_tokens",
    ]
    for keys, group in checkpoint_frame.groupby(group_columns, sort=True):
        summary_rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "n_checkpoints": int(len(group)),
                "n_pairs": int(group["pair_id"].nunique()),
                "target_adoption_rate": _mean(
                    group, "forced_target_semantic_selected"
                ),
                "mean_choice_confidence": _mean(
                    group, "forced_choice_confidence"
                ),
                "choice_a_rate": float(
                    group["forced_choice_semantic"].eq("A").mean()
                ),
                "choice_b_rate": float(
                    group["forced_choice_semantic"].eq("B").mean()
                ),
                "tie_rate": float(
                    group["forced_choice_semantic"].eq("C").mean()
                ),
            }
        )

    phase1_end = checkpoint_frame[
        (checkpoint_frame["stage"] == "phase1")
        & (checkpoint_frame["budget_tokens"] == 128)
    ][["trace_id", "forced_choice_semantic"]].rename(
        columns={"forced_choice_semantic": "phase1_choice_semantic"}
    )
    switch_rows = checkpoint_frame[
        checkpoint_frame["stage"].eq("phase2")
    ].merge(phase1_end, on="trace_id", how="left")
    switch_rows["revised_from_phase1"] = (
        switch_rows["phase1_choice_semantic"].ne("")
        & switch_rows["forced_choice_semantic"].ne("")
        & switch_rows["phase1_choice_semantic"].ne(
            switch_rows["forced_choice_semantic"]
        )
    )
    switch_rows["anchored_to_phase1"] = (
        switch_rows["phase1_choice_semantic"].ne("")
        & switch_rows["phase1_choice_semantic"].eq(
            switch_rows["forced_choice_semantic"]
        )
    )
    switch_rows["adopted_updated_target"] = (
        switch_rows["forced_choice_semantic"]
        == switch_rows["phase2_target_semantic"]
    )
    adoption_summary = (
        switch_rows.groupby(
            ["condition_id", "transition_type", "budget_tokens"],
            sort=True,
        )
        .agg(
            n=("trace_id", "size"),
            n_pairs=("pair_id", "nunique"),
            updated_target_adoption_rate=("adopted_updated_target", "mean"),
            revision_rate=("revised_from_phase1", "mean"),
            phase1_anchoring_rate=("anchored_to_phase1", "mean"),
            mean_choice_confidence=("forced_choice_confidence", "mean"),
        )
        .reset_index()
    )

    final_summary = (
        trace_frame.groupby(
            ["condition_id", "transition_type"], sort=True
        )
        .agg(
            n_traces=("trace_id", "size"),
            n_pairs=("pair_id", "nunique"),
            natural_valid_rate=("valid_choice", "mean"),
            final_target_adoption_rate=(
                "final_target_semantic_selected",
                "mean",
            ),
            phase2_budget_saturation_rate=(
                "phase2_max_budget_saturated",
                "mean",
            ),
            mean_phase2_generated_tokens=(
                "phase2_generated_tokens",
                "mean",
            ),
        )
        .reset_index()
    )

    order_rows: list[dict[str, Any]] = []
    final_checkpoint = switch_rows[
        switch_rows["budget_tokens"].eq(384)
    ].copy()
    for keys, group in final_checkpoint.groupby(
        ["pair_id", "condition_id", "branch_index"], sort=True
    ):
        by_order = group.set_index("presentation_order")
        if not {"original", "swapped"}.issubset(by_order.index):
            continue
        original = str(
            by_order.loc["original", "forced_choice_semantic"]
        )
        swapped = str(by_order.loc["swapped", "forced_choice_semantic"])
        order_rows.append(
            {
                "pair_id": keys[0],
                "condition_id": keys[1],
                "branch_index": int(keys[2]),
                "order_consistent": bool(original and original == swapped),
                "original_choice_semantic": original,
                "swapped_choice_semantic": swapped,
            }
        )
    order_frame = pd.DataFrame(order_rows)
    order_summary = (
        order_frame.groupby("condition_id", sort=True)
        .agg(
            n_ordered_branches=("pair_id", "size"),
            order_consistent_rate=("order_consistent", "mean"),
        )
        .reset_index()
        if not order_frame.empty
        else pd.DataFrame()
    )
    return {
        "checkpoint_rows": checkpoint_frame,
        "checkpoint_summary": pd.DataFrame(summary_rows),
        "switch_adoption_rows": switch_rows,
        "switch_adoption_summary": adoption_summary,
        "final_outcome_summary": final_summary,
        "order_consistency_rows": order_frame,
        "order_consistency_summary": order_summary,
        "direct_rows": pd.DataFrame(direct),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    run_dirs = [_resolve(workspace_root, path) for path in args.run_dir]
    out_dir = _resolve(workspace_root, args.out_dir)
    outputs = analyze(run_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-criterion-switch-behavior-analysis",
            "run_dirs": [str(path) for path in run_dirs],
            "out_dir": str(out_dir),
            "outputs": paths,
        },
    )
    print(f"out_dir={out_dir}")
    print(outputs["final_outcome_summary"].to_string(index=False))
    print(outputs["switch_adoption_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
