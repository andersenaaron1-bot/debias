"""Print compact factual CoT-vs-direct behavior results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path


DEFAULT_INPUT = (
    Path("artifacts")
    / "mechanistic"
    / "judge_factual_cot_effect_qwen3_8b_v1"
    / "analysis"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--endpoint-budget", type=int, default=2048)
    parser.add_argument("--digits", type=int, default=3)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _read(input_dir: Path, name: str) -> pd.DataFrame:
    path = input_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _print(title: str, frame: pd.DataFrame, *, digits: int) -> None:
    print(f"\n=== {title} ===")
    if frame.empty:
        print("NO ROWS")
        return
    print(frame.round(int(digits)).to_string(index=False))


def _metric_subset(frame: pd.DataFrame) -> pd.DataFrame:
    keep = {
        "forced_target_adoption",
        "order_consistent_target_adoption",
        "order_consistent_rate",
        "natural_valid_rate",
    }
    return frame[frame["metric"].isin(keep)].copy()


def readout(input_dir: Path, *, endpoint_budget: int, digits: int) -> None:
    summary = _read(input_dir, "factual_cot_summary.csv")
    endpoint = _read(input_dir, "factual_cot_endpoint_summary.csv")
    effects = _read(input_dir, "factual_cot_effects.csv")
    endpoint_effects = _read(input_dir, "factual_cot_endpoint_effects.csv")

    endpoint_all = _metric_subset(
        endpoint[endpoint["source_dataset"].eq("all")]
    )
    _print(
        f"FACTUAL DIRECT VS {endpoint_budget}-TOKEN COT",
        endpoint_all[
            [
                "condition_id",
                "budget_tokens",
                "metric",
                "n_pairs",
                "mean",
                "ci95_low",
                "ci95_high",
            ]
        ],
        digits=digits,
    )

    _print(
        f"FACTUAL COT MINUS DIRECT AT {endpoint_budget} TOKENS",
        _metric_subset(endpoint_effects[endpoint_effects["source_dataset"].eq("all")])[
            [
                "contrast",
                "metric",
                "n_pairs",
                "mean",
                "ci95_low",
                "ci95_high",
            ]
        ],
        digits=digits,
    )

    by_budget = _metric_subset(
        effects[
            effects["source_dataset"].eq("all")
            & effects["metric"].isin(
                {
                    "forced_target_adoption",
                    "order_consistent_target_adoption",
                }
            )
        ]
    )
    _print(
        "FACTUAL COT MINUS DIRECT OVER BUDGET",
        by_budget[
            [
                "budget_tokens",
                "metric",
                "n_pairs",
                "mean",
                "ci95_low",
                "ci95_high",
            ]
        ],
        digits=digits,
    )

    by_dataset = _metric_subset(
        endpoint_effects[
            ~endpoint_effects["source_dataset"].eq("all")
            & endpoint_effects["metric"].isin(
                {
                    "forced_target_adoption",
                    "order_consistent_target_adoption",
                }
            )
        ]
    )
    _print(
        f"FACTUAL COT MINUS DIRECT BY DATASET AT {endpoint_budget} TOKENS",
        by_dataset[
            [
                "source_dataset",
                "metric",
                "n_pairs",
                "mean",
                "ci95_low",
                "ci95_high",
            ]
        ],
        digits=digits,
    )

    curves = _metric_subset(
        summary[
            summary["source_dataset"].eq("all")
            & summary["metric"].isin(
                {
                    "forced_target_adoption",
                    "order_consistent_target_adoption",
                }
            )
        ]
    )
    _print(
        "FACTUAL TARGET-ADOPTION CURVES",
        curves[
            [
                "condition_id",
                "budget_tokens",
                "metric",
                "n_pairs",
                "mean",
                "ci95_low",
                "ci95_high",
            ]
        ],
        digits=digits,
    )


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    readout(
        _resolve(workspace_root, args.input),
        endpoint_budget=int(args.endpoint_budget),
        digits=int(args.digits),
    )


if __name__ == "__main__":
    main()
