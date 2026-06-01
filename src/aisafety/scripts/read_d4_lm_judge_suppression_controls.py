"""Print a compact layer-specific readout for anchored suppression sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--hidden-layer", type=int, default=27)
    return parser.parse_args()


def _rows(root: Path, *, hidden_layer: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in sorted(path for path in Path(root).iterdir() if path.is_dir() and path.name != "logs"):
        summary_path = run_dir / "subspace_suppression_summary.csv"
        manifest_path = run_dir / "manifest.json"
        if not summary_path.is_file() or not manifest_path.is_file():
            continue
        frame = pd.read_csv(summary_path)
        frame = frame[frame["hidden_layer"].astype(int) == int(hidden_layer)].copy()
        if frame.empty:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame["setting"] = run_dir.name
        frame["basis_control_seed"] = int(manifest.get("basis_control_seed", 0))
        frame["aggregate_attenuation"] = np.divide(
            frame["mean_suppressed_margin"] - frame["mean_observed_margin"],
            frame["mean_neutral_margin"] - frame["mean_observed_margin"],
        )
        frame["preference_rate_change"] = frame["mean_suppressed_preferred"] - frame["mean_observed_preferred"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def readout(root: Path, *, hidden_layer: int) -> None:
    frame = _rows(root, hidden_layer=int(hidden_layer))
    print(f"=== Anchored suppression controls: hidden layer {int(hidden_layer)} ===")
    if frame.empty:
        print("(empty)")
        return
    fitted = frame[frame["basis_control"] == "fitted"].copy()
    controls = frame[frame["basis_control"] != "fitted"].copy()
    cols = [
        "setting",
        "dataset",
        "basis_eval_split",
        "subspace_rank",
        "suppression_alpha",
        "n_counterfactuals",
        "mean_observed_margin",
        "mean_suppressed_margin",
        "aggregate_attenuation",
        "mean_observed_preferred",
        "mean_suppressed_preferred",
        "preference_rate_change",
    ]
    print("\n=== Fitted basis settings ===")
    print(fitted[cols].sort_values(["setting", "dataset", "basis_eval_split"]).to_string(index=False))
    print("\n=== Matched-rank control aggregate ===")
    if controls.empty:
        print("(empty)")
        return
    grouped = (
        controls.groupby(["basis_control", "dataset", "basis_eval_split", "subspace_rank", "suppression_alpha"], sort=True)
        .agg(
            n_runs=("setting", "count"),
            mean_attenuation=("aggregate_attenuation", "mean"),
            std_attenuation=("aggregate_attenuation", "std"),
            mean_preference_rate_change=("preference_rate_change", "mean"),
            std_preference_rate_change=("preference_rate_change", "std"),
        )
        .reset_index()
    )
    print(grouped.to_string(index=False))


def main() -> None:
    args = _parse_args()
    readout(args.input, hidden_layer=int(args.hidden_layer))


if __name__ == "__main__":
    main()
