"""Print compact readouts for D4 human-vs-LLM stage-contrast summaries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


DEFAULT_SUMMARY_DIR = (
    Path("artifacts")
    / "mechanistic"
    / "d4_human_llm_stage_contrast_summary_tulu_stage_scout_fresh_local_v1"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.environ.get("OUT", DEFAULT_SUMMARY_DIR)),
        help="Summary output directory containing stage_summary.csv and contrast CSVs.",
    )
    parser.add_argument(
        "--source-top-k",
        type=int,
        default=200,
        help="Maximum source-level contrast rows to print.",
    )
    return parser.parse_args()


def _cols(df: pd.DataFrame, names: list[str]) -> list[str]:
    return [name for name in names if name in df.columns]


def _print_table(title: str, df: pd.DataFrame, cols: list[str]) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("(empty)")
        return
    selected = _cols(df, cols)
    if not selected:
        print(df.to_string(index=False))
        return
    print(df[selected].to_string(index=False))


def print_summary(out_dir: Path, *, source_top_k: int = 200) -> None:
    out_dir = Path(out_dir).expanduser().resolve()
    print(f"out_dir = {out_dir}")

    stage_path = out_dir / "stage_summary.csv"
    contrast_path = out_dir / "stage_contrast_deltas.csv"
    group_path = out_dir / "stage_contrast_group_deltas.csv"
    for path in (stage_path, contrast_path, group_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    stage = pd.read_csv(stage_path)
    overall = stage[(stage["group_type"] == "all") & (stage["group_value"] == "all")]
    overall = overall.sort_values("run_label") if "run_label" in overall.columns else overall
    _print_table(
        "Stage Overall",
        overall,
        [
            "run_label",
            "n_pairs",
            "mean_llm_margin",
            "median_llm_margin",
            "llm_preference_rate",
            "mean_llm_prob",
        ],
    )

    contrasts = pd.read_csv(contrast_path)
    _print_table(
        "Paired Stage Deltas",
        contrasts,
        [
            "contrast",
            "n_pairs",
            "mean_right_llm_margin",
            "mean_left_llm_margin",
            "mean_delta_llm_margin",
            "right_llm_preference_rate",
            "left_llm_preference_rate",
            "pref_flip_to_llm_rate",
            "pref_flip_from_llm_rate",
        ],
    )

    groups = pd.read_csv(group_path)
    sources = groups[groups["group_type"] == "source_dataset"]
    if {"contrast", "group_value"}.issubset(sources.columns):
        sources = sources.sort_values(["contrast", "group_value"])
    sources = sources.head(max(int(source_top_k), 0))
    _print_table(
        "Source-Level Deltas",
        sources,
        [
            "contrast",
            "group_value",
            "n_pairs",
            "mean_delta_llm_margin",
            "right_llm_preference_rate",
            "left_llm_preference_rate",
        ],
    )


def main() -> None:
    args = _parse_args()
    print_summary(Path(args.input), source_top_k=int(args.source_top_k))


if __name__ == "__main__":
    main()
