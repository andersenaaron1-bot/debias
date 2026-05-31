"""Compare layerwise style-direction participation across base and IT checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--instruct", type=Path, required=True)
    parser.add_argument("--contrast-label", default="instruct_minus_base")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _csv(root: Path, filename: str) -> pd.DataFrame:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _direction_cosines(base_dir: Path, instruct_dir: Path) -> pd.DataFrame:
    base = np.load(base_dir / "layer_mean_directions.npz")
    instruct = np.load(instruct_dir / "layer_mean_directions.npz")
    rows: list[dict[str, Any]] = []
    for key in sorted(set(base.files) & set(instruct.files)):
        base_vec = np.asarray(base[key], dtype=float)
        instruct_vec = np.asarray(instruct[key], dtype=float)
        denom = float(np.linalg.norm(base_vec) * np.linalg.norm(instruct_vec))
        rows.append(
            {
                "hidden_layer": int(key.replace("hidden_", "")),
                "base_instruct_mean_direction_cosine": None
                if denom <= 1e-12
                else float(np.dot(base_vec, instruct_vec) / denom),
            }
        )
    return pd.DataFrame(rows)


def summarize(base_dir: Path, instruct_dir: Path, *, contrast_label: str) -> pd.DataFrame:
    base = _csv(base_dir, "layer_summary.csv")
    instruct = _csv(instruct_dir, "layer_summary.csv")
    keep = [
        "hidden_layer",
        "group_type",
        "group_value",
        "n_counterfactuals",
        "mean_delta_norm",
        "mean_direction_norm",
        "direction_concentration",
        "mean_cv_style_projection",
        "mean_cv_cosine_to_style_direction",
        "positive_cv_cosine_rate",
        "pearson_projection_with_binary_margin",
    ]
    keys = ["hidden_layer", "group_type", "group_value"]
    merged = base[keep].merge(instruct[keep], on=keys, suffixes=("_base", "_instruct"))
    merged["contrast_label"] = str(contrast_label)
    for metric in (
        "mean_delta_norm",
        "mean_direction_norm",
        "direction_concentration",
        "mean_cv_style_projection",
        "mean_cv_cosine_to_style_direction",
        "positive_cv_cosine_rate",
        "pearson_projection_with_binary_margin",
    ):
        merged[f"delta_{metric}"] = (
            pd.to_numeric(merged[f"{metric}_instruct"], errors="coerce")
            - pd.to_numeric(merged[f"{metric}_base"], errors="coerce")
        )
    direction_cosines = _direction_cosines(base_dir, instruct_dir)
    merged = merged.merge(direction_cosines, on="hidden_layer", how="left")
    merged.loc[merged["group_type"] != "all", "base_instruct_mean_direction_cosine"] = pd.NA
    return merged


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    base_dir = _resolve(workspace_root, args.base)
    instruct_dir = _resolve(workspace_root, args.instruct)
    out_dir = _resolve(workspace_root, args.out_dir)
    summary = summarize(base_dir, instruct_dir, contrast_label=str(args.contrast_label))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "layer_stage_contrast.csv", index=False)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "D4-LM-style-activation-stage-summary",
            "base_dir": str(base_dir),
            "instruct_dir": str(instruct_dir),
            "out_dir": str(out_dir),
            "contrast_label": str(args.contrast_label),
            "n_layers": int(len(summary)),
        },
    )
    cols = [
        "hidden_layer",
        "delta_direction_concentration",
        "delta_mean_cv_cosine_to_style_direction",
        "delta_pearson_projection_with_binary_margin",
        "base_instruct_mean_direction_cosine",
    ]
    print(summary[summary["group_type"] == "all"][cols].to_string(index=False))
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
