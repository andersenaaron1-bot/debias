"""Print a detailed readout for a matched HelpSteer2 criterion scout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path


DEFAULT_INPUT = (
    Path("artifacts")
    / "mechanistic"
    / "helpsteer2_matched_criterion_qwen3_8b_scout_v1"
    / "analysis"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--budget", type=int, default=512)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _boolean(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            np.nan
            if pd.isna(value)
            else float(str(value).strip().lower() in {"true", "1", "1.0"})
        )
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source: Path,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def summarize_orders(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tied = group[group["target_tie"].eq(1.0)]
        non_tied = group[group["target_tie"].eq(0.0)]
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "n": int(len(group)),
                "robust_accuracy": float(
                    group["robust_target_success"].mean()
                ),
                "consistent_coverage": float(
                    group["has_order_consistent_verdict"].mean()
                ),
                "natural_valid": float(
                    group["mean_natural_valid_rate"].mean()
                ),
                "branch_agreement": float(
                    group["mean_branch_agreement"].mean()
                ),
                "tie_recall": (
                    float(tied["robust_target_success"].mean())
                    if len(tied)
                    else float("nan")
                ),
                "unjustified_tie": (
                    float(non_tied["unjustified_tie"].mean())
                    if len(non_tied)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def criterion_transition_summary(
    orders: pd.DataFrame,
    switches: pd.DataFrame,
) -> pd.DataFrame:
    lookup = orders[
        [
            "reasoning_mode",
            "pair_stratum",
            "pair_id",
            "budget_tokens",
            "criterion_id",
            "target_tie",
        ]
    ].drop_duplicates()
    left = lookup.rename(
        columns={
            "criterion_id": "left_criterion",
            "target_tie": "left_tie",
        }
    )
    right = lookup.rename(
        columns={
            "criterion_id": "right_criterion",
            "target_tie": "right_tie",
        }
    )
    left_keys = [
        "reasoning_mode",
        "pair_stratum",
        "pair_id",
        "budget_tokens",
        "left_criterion",
    ]
    right_keys = [
        "reasoning_mode",
        "pair_stratum",
        "pair_id",
        "budget_tokens",
        "right_criterion",
    ]
    merged = switches.merge(left, on=left_keys, how="left").merge(
        right,
        on=right_keys,
        how="left",
    )
    merged["transition"] = np.where(
        merged["expected_switch"].eq(0.0),
        "same_target",
        np.where(
            merged["left_tie"].ne(merged["right_tie"]),
            "tie_vs_choice",
            "choice_vs_choice",
        ),
    )
    merged["unconditional_compliance"] = merged[
        "switch_compliance"
    ].fillna(0.0)
    return (
        merged.groupby(
            ["reasoning_mode", "budget_tokens", "transition"],
            sort=True,
            dropna=False,
        )
        .agg(
            n=("pair_id", "size"),
            consistent_coverage=("both_order_consistent", "mean"),
            conditional_compliance=("switch_compliance", "mean"),
            unconditional_compliance=("unconditional_compliance", "mean"),
            both_criteria_correct=("both_criteria_correct", "mean"),
        )
        .reset_index()
    )


def revision_summary(revisions: pd.DataFrame) -> pd.DataFrame:
    frame = revisions.copy()
    frame["pair_stratum"] = (
        frame["source_dataset"]
        .astype(str)
        .str.replace("helpsteer2_matched_", "", regex=False)
    )
    frame["any_revision"] = (
        frame["forced_choice_revisions"].astype(float) > 0
    ).astype(float)
    return (
        frame.groupby("pair_stratum", sort=True)
        .agg(
            n_traces=("comparison_id", "size"),
            any_revision_rate=("any_revision", "mean"),
            mean_revisions=("forced_choice_revisions", "mean"),
            median_first_stable_budget=("first_stable_budget", "median"),
            mean_confidence_change=("confidence_change", "mean"),
        )
        .reset_index()
    )


def readout(input_dir: Path, *, budget: int) -> None:
    orders_path = input_dir / "matched_order_rows.csv"
    switches_path = input_dir / "criterion_switch_rows.csv"
    revisions_path = input_dir / "trace_revision_rows.csv"
    for path in (orders_path, switches_path, revisions_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    orders = pd.read_csv(orders_path)
    switches = pd.read_csv(switches_path)
    revisions = pd.read_csv(revisions_path)
    _require_columns(
        orders,
        [
            "reasoning_mode",
            "budget_tokens",
            "pair_stratum",
            "criterion_id",
            "target_tie",
            "robust_target_success",
            "has_order_consistent_verdict",
            "mean_natural_valid_rate",
            "mean_branch_agreement",
            "unjustified_tie",
        ],
        source=orders_path,
    )
    for column in (
        "target_tie",
        "predicted_tie",
        "has_order_consistent_verdict",
    ):
        if column in orders:
            orders[column] = _boolean(orders[column])
    for column in (
        "both_order_consistent",
        "expected_switch",
        "switch_compliance",
        "both_criteria_correct",
    ):
        switches[column] = _boolean(switches[column])

    available_budgets = sorted(
        int(value)
        for value in orders.loc[
            orders["reasoning_mode"].eq("thinking"),
            "budget_tokens",
        ].unique()
    )
    if int(budget) not in available_budgets:
        raise ValueError(
            f"Budget {budget} is unavailable; choose from {available_budgets}."
        )

    print(f"input={input_dir}")
    print("\n=== COMPLETE BUDGET CURVES BY STRATUM ===")
    curves = summarize_orders(
        orders,
        ["reasoning_mode", "budget_tokens", "pair_stratum"],
    )
    print(curves.round(3).to_string(index=False))

    print(f"\n=== {budget}-TOKEN DETAIL BY CRITERION ===")
    selected = orders[
        (
            orders["reasoning_mode"].eq("direct")
            & orders["budget_tokens"].eq(0)
        )
        | (
            orders["reasoning_mode"].eq("thinking")
            & orders["budget_tokens"].eq(int(budget))
        )
    ]
    detail = summarize_orders(
        selected,
        [
            "reasoning_mode",
            "budget_tokens",
            "pair_stratum",
            "criterion_id",
        ],
    )
    print(detail.round(3).to_string(index=False))

    print("\n=== CRITERION SWITCHING BY TARGET TRANSITION ===")
    transitions = criterion_transition_summary(orders, switches)
    print(transitions.round(3).to_string(index=False))

    print("\n=== THINKING REVISION DYNAMICS ===")
    print(revision_summary(revisions).round(3).to_string(index=False))


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    input_dir = _resolve(workspace_root, args.input)
    readout(input_dir, budget=int(args.budget))


if __name__ == "__main__":
    main()
