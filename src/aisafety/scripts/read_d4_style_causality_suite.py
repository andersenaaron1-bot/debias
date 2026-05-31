"""Print a compact binary and activation readout from a D4 style-causality suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def readout(root: Path, *, top_k: int) -> None:
    root = Path(root)
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "logs"):
        binary = _csv(dataset_dir / "binary_summary" / "stage_contrast_deltas.csv")
        binary_groups = _csv(dataset_dir / "binary_summary" / "stage_contrast_group_deltas.csv")
        activation = _csv(dataset_dir / "activation_summary" / "layer_stage_contrast.csv")
        print(f"\n######## {dataset_dir.name} ########")
        print("\n=== Binary judge stage delta ===")
        if binary.empty:
            print("(empty)")
        else:
            cols = [
                "contrast",
                "n_counterfactuals",
                "mean_delta_cue_plus_margin",
                "right_cue_plus_preference_rate",
                "left_cue_plus_preference_rate",
            ]
            print(binary[cols].to_string(index=False))

        print("\n=== Binary transform-level deltas ===")
        transforms = (
            binary_groups[binary_groups["group_type"] == "transform_id"].copy()
            if not binary_groups.empty
            else pd.DataFrame()
        )
        if transforms.empty:
            print("(empty)")
        else:
            transforms["abs_delta"] = transforms["mean_delta_cue_plus_margin"].abs()
            transforms = transforms.sort_values("abs_delta", ascending=False).head(max(int(top_k), 1))
            cols = ["group_value", "n_counterfactuals", "mean_delta_cue_plus_margin"]
            print(transforms[cols].to_string(index=False))

        print("\n=== Activation global layer deltas ===")
        global_activation = (
            activation[activation["group_type"] == "all"].copy()
            if not activation.empty
            else pd.DataFrame()
        )
        if global_activation.empty:
            print("(empty)")
        else:
            cols = [
                "hidden_layer",
                "delta_direction_concentration",
                "delta_mean_cv_cosine_to_style_direction",
                "delta_pearson_projection_with_binary_margin",
                "base_instruct_mean_direction_cosine",
            ]
            print(global_activation[cols].to_string(index=False))

        print("\n=== Activation transform-level deltas ===")
        transform_activation = (
            activation[activation["group_type"] == "transform_id"].copy()
            if not activation.empty
            else pd.DataFrame()
        )
        if transform_activation.empty:
            print("(empty)")
        else:
            transform_activation["abs_delta"] = transform_activation[
                "delta_mean_cv_cosine_to_style_direction"
            ].abs()
            transform_activation = transform_activation.sort_values("abs_delta", ascending=False).head(
                max(int(top_k), 1)
            )
            cols = [
                "hidden_layer",
                "group_value",
                "delta_mean_cv_cosine_to_style_direction",
                "delta_pearson_projection_with_binary_margin",
            ]
            print(transform_activation[cols].to_string(index=False))


def main() -> None:
    args = _parse_args()
    readout(args.input, top_k=int(args.top_k))


if __name__ == "__main__":
    main()
