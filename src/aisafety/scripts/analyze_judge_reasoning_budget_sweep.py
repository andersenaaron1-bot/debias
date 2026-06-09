"""Analyze nested token-budget effects in pairwise judge reasoning."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_json, read_jsonl, resolve_path, write_json
from aisafety.mech.judge_reasoning import normalize_verdict


DEFAULT_RUN_DIR = (
    Path("artifacts") / "mechanistic" / "judge_deliberation_budget_sweep_v1"
)
DEFAULT_OUT_DIR = (
    Path("artifacts") / "mechanistic" / "judge_deliberation_budget_analysis_v1"
)
META_COLUMNS = (
    "criterion_id",
    "criterion_family",
    "criterion_determinacy",
    "determinacy_level",
    "origin_pair_id",
    "analysis_split",
    "validity_type",
    "difficulty_tier",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def prepare_scores(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        raise ValueError("Budget run contains no score rows.")
    metadata = frame.get("metadata", pd.Series([{}] * len(frame)))
    for column in META_COLUMNS:
        if column not in frame.columns:
            frame[column] = [
                value.get(column) if isinstance(value, dict) else None
                for value in metadata
            ]
    frame["criterion_id"] = frame["criterion_id"].fillna("default").astype(str)
    frame["criterion_family"] = frame["criterion_family"].fillna("").astype(str)
    frame["origin_pair_id"] = (
        frame["origin_pair_id"].fillna(frame["pair_id"]).astype(str)
    )
    labels = ("A", "B", "C")
    frame["forced_choice"] = frame["forced_choice"].map(
        lambda value: normalize_verdict(value, labels=labels)
    )
    frame["target_option"] = frame["target_option"].map(
        lambda value: normalize_verdict(value, labels=labels)
    )
    frame["scored"] = frame["target_option"].isin(labels)
    frame["forced_target_selected_numeric"] = np.where(
        frame["scored"],
        frame["forced_choice"].eq(frame["target_option"]).astype(float),
        np.nan,
    )
    frame["natural_valid_at_budget"] = frame["natural_valid_at_budget"].eq(True)
    frame["is_thinking"] = frame["reasoning_mode"].eq("thinking")
    return frame


def _entropy(counts: Counter[str]) -> float:
    total = int(sum(counts.values()))
    if total <= 0:
        return float("nan")
    probabilities = [count / total for count in counts.values() if count > 0]
    return float(-sum(value * math.log2(value) for value in probabilities))


def branch_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "reasoning_mode",
        "source_dataset",
        "criterion_id",
        "criterion_family",
        "criterion_determinacy",
        "determinacy_level",
        "origin_pair_id",
        "pair_id",
        "comparison_id",
        "presentation_order",
        "budget_tokens",
        "target_option",
        "scored",
    ]
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        counts = Counter(group["forced_choice"].astype(str))
        majority = ""
        if counts:
            ordered = counts.most_common()
            if len(ordered) == 1 or ordered[0][1] > ordered[1][1]:
                majority = ordered[0][0]
        selected_hash = ""
        if majority:
            hashes = group.loc[
                group["forced_choice"].eq(majority),
                "forced_selected_text_hash",
            ]
            if not hashes.empty:
                selected_hash = str(hashes.mode().iloc[0])
        majority_count = int(counts.get(majority, 0)) if majority else 0
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "n_branches": int(len(group)),
                "majority_choice": majority,
                "majority_selected_text_hash": selected_hash,
                "target_selected_text_hash": str(
                    group["target_selected_text_hash"].dropna().iloc[0]
                )
                if "target_selected_text_hash" in group
                and not group["target_selected_text_hash"].dropna().empty
                else "",
                "branch_agreement": (
                    float(majority_count / len(group))
                    if len(group)
                    else float("nan")
                ),
                "branch_entropy": _entropy(counts),
                "mean_forced_confidence": float(
                    group["forced_choice_confidence"].mean()
                ),
                "natural_valid_rate": float(
                    group["natural_valid_at_budget"].mean()
                ),
                "majority_target_selected": (
                    float(majority == keys[11])
                    if bool(keys[12]) and majority
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def order_rows(branches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "reasoning_mode",
        "source_dataset",
        "criterion_id",
        "criterion_family",
        "criterion_determinacy",
        "determinacy_level",
        "origin_pair_id",
        "pair_id",
        "budget_tokens",
    ]
    for keys, group in branches.groupby(group_columns, sort=True, dropna=False):
        valid = group[
            group["majority_selected_text_hash"].fillna("").astype(str).ne("")
        ]
        hashes = valid["majority_selected_text_hash"].astype(str).tolist()
        consistent_hash = (
            hashes[0] if len(hashes) >= 2 and len(set(hashes)) == 1 else ""
        )
        target_hashes = (
            group["target_selected_text_hash"].dropna().astype(str).tolist()
            if "target_selected_text_hash" in group
            else []
        )
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "n_orders": int(group["comparison_id"].nunique()),
                "n_orders_with_majority": int(len(valid)),
                "order_consistent_majority": (
                    float(len(set(hashes)) == 1)
                    if len(hashes) >= 2
                    else float("nan")
                ),
                "order_majority_selected_text_hash": consistent_hash,
                "target_selected_text_hash": (
                    target_hashes[0]
                    if target_hashes and len(set(target_hashes)) == 1
                    else ""
                ),
                "mean_branch_agreement": float(group["branch_agreement"].mean()),
                "mean_branch_entropy": float(group["branch_entropy"].mean()),
                "mean_natural_valid_rate": float(
                    group["natural_valid_rate"].mean()
                ),
                "mean_forced_confidence": float(
                    group["mean_forced_confidence"].mean()
                ),
                "mean_majority_target_selected": (
                    float(group["majority_target_selected"].mean())
                    if group["scored"].any()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def budget_summary(frame: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    trace_groups = [
        "reasoning_mode",
        "source_dataset",
        "criterion_id",
        "criterion_determinacy",
        "determinacy_level",
        "budget_tokens",
    ]
    trace = (
        frame.groupby(trace_groups, sort=True, dropna=False)
        .agg(
            n_score_rows=("budget_eval_id", "count"),
            n_pairs=("pair_id", "nunique"),
            natural_valid_rate=("natural_valid_at_budget", "mean"),
            forced_target_success=("forced_target_selected_numeric", "mean"),
            forced_choice_a_rate=("forced_choice", lambda values: values.eq("A").mean()),
            mean_forced_confidence=("forced_choice_confidence", "mean"),
            mean_abs_forced_margin=(
                "forced_margin_a_minus_b",
                lambda values: values.abs().mean(),
            ),
            mean_full_generated_tokens=("full_generated_tokens", "mean"),
            max_budget_saturation_rate=("max_budget_saturated", "mean"),
        )
        .reset_index()
    )
    order = (
        orders.groupby(trace_groups, sort=True, dropna=False)
        .agg(
            n_order_pairs=("pair_id", "count"),
            order_consistent_majority_rate=("order_consistent_majority", "mean"),
            mean_branch_agreement=("mean_branch_agreement", "mean"),
            mean_branch_entropy=("mean_branch_entropy", "mean"),
            majority_target_success=("mean_majority_target_selected", "mean"),
        )
        .reset_index()
    )
    return trace.merge(order, on=trace_groups, how="left")


def revision_rows(frame: pd.DataFrame) -> pd.DataFrame:
    thinking = frame[frame["reasoning_mode"].eq("thinking")].copy()
    rows: list[dict[str, Any]] = []
    keys = [
        "source_dataset",
        "criterion_id",
        "criterion_determinacy",
        "determinacy_level",
        "origin_pair_id",
        "pair_id",
        "comparison_id",
        "branch_index",
    ]
    for group_keys, group in thinking.groupby(keys, sort=True, dropna=False):
        ordered = group.sort_values("budget_tokens")
        choices = ordered["forced_choice"].astype(str).tolist()
        revisions = int(
            sum(left != right for left, right in zip(choices, choices[1:]))
        )
        stable_budget = float("nan")
        for index, choice in enumerate(choices):
            if all(value == choice for value in choices[index:]):
                stable_budget = float(ordered.iloc[index]["budget_tokens"])
                break
        rows.append(
            {
                **dict(zip(keys, group_keys, strict=True)),
                "n_budget_points": int(len(ordered)),
                "forced_choice_revisions": revisions,
                "first_stable_budget": stable_budget,
                "initial_forced_choice": choices[0] if choices else "",
                "final_forced_choice": choices[-1] if choices else "",
                "initial_forced_confidence": float(
                    ordered.iloc[0]["forced_choice_confidence"]
                ),
                "final_forced_confidence": float(
                    ordered.iloc[-1]["forced_choice_confidence"]
                ),
                "confidence_change": float(
                    ordered.iloc[-1]["forced_choice_confidence"]
                    - ordered.iloc[0]["forced_choice_confidence"]
                ),
            }
        )
    return pd.DataFrame(rows)


def budget_pair_effects(orders: pd.DataFrame) -> pd.DataFrame:
    direct = orders[orders["reasoning_mode"].eq("direct")].copy()
    thinking = orders[orders["reasoning_mode"].eq("thinking")].copy()
    keys = [
        "source_dataset",
        "criterion_id",
        "criterion_family",
        "criterion_determinacy",
        "determinacy_level",
        "origin_pair_id",
        "pair_id",
    ]
    direct = direct[keys + [
        "order_consistent_majority",
        "mean_branch_agreement",
        "mean_branch_entropy",
        "mean_natural_valid_rate",
        "mean_forced_confidence",
        "mean_majority_target_selected",
    ]].rename(
        columns={
            column: f"direct_{column}"
            for column in (
                "order_consistent_majority",
                "mean_branch_agreement",
                "mean_branch_entropy",
                "mean_natural_valid_rate",
                "mean_forced_confidence",
                "mean_majority_target_selected",
            )
        }
    )
    merged = thinking.merge(direct, on=keys, how="inner")
    for metric in (
        "order_consistent_majority",
        "mean_branch_agreement",
        "mean_branch_entropy",
        "mean_natural_valid_rate",
        "mean_forced_confidence",
        "mean_majority_target_selected",
    ):
        merged[f"delta_{metric}"] = (
            merged[metric] - merged[f"direct_{metric}"]
        )
    return merged


def criterion_pair_effects(orders: pd.DataFrame) -> pd.DataFrame:
    criterion = orders[
        orders["criterion_family"].fillna("").astype(str).ne("")
        & orders["reasoning_mode"].eq("thinking")
    ].copy()
    if criterion.empty:
        return pd.DataFrame()
    values = [
        "mean_natural_valid_rate",
        "mean_forced_confidence",
        "mean_branch_agreement",
        "mean_branch_entropy",
        "order_consistent_majority",
        "mean_majority_target_selected",
    ]
    keys = [
        "criterion_family",
        "origin_pair_id",
        "pair_id",
        "budget_tokens",
    ]
    overall = criterion[criterion["criterion_id"].eq("overall")][
        keys + values
    ].rename(columns={value: f"overall_{value}" for value in values})
    explicit = criterion[~criterion["criterion_id"].eq("overall")].copy()
    merged = explicit.merge(overall, on=keys, how="inner")
    for value in values:
        merged[f"delta_vs_overall_{value}"] = (
            merged[value] - merged[f"overall_{value}"]
        )
    return merged


def pair_budget_slopes(orders: pd.DataFrame) -> pd.DataFrame:
    thinking = orders[orders["reasoning_mode"].eq("thinking")].copy()
    metrics = [
        "mean_natural_valid_rate",
        "mean_forced_confidence",
        "mean_branch_agreement",
        "mean_branch_entropy",
        "order_consistent_majority",
        "mean_majority_target_selected",
    ]
    keys = [
        "source_dataset",
        "criterion_id",
        "criterion_determinacy",
        "determinacy_level",
        "origin_pair_id",
        "pair_id",
    ]
    rows: list[dict[str, Any]] = []
    for group_keys, group in thinking.groupby(keys, sort=True, dropna=False):
        ordered = group.sort_values("budget_tokens")
        x = np.log2(1.0 + ordered["budget_tokens"].astype(float).to_numpy())
        row: dict[str, Any] = dict(zip(keys, group_keys, strict=True))
        row["n_budget_points"] = int(len(ordered))
        for metric in metrics:
            y = ordered[metric].astype(float).to_numpy()
            valid = np.isfinite(x) & np.isfinite(y)
            row[f"slope_{metric}_per_log2_token"] = (
                float(np.polyfit(x[valid], y[valid], deg=1)[0])
                if int(valid.sum()) >= 2 and float(np.std(x[valid])) > 0
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_mean(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return float("nan"), float("nan"), float("nan")
    means = np.empty((max(int(n_bootstrap), 1),), dtype=float)
    for index in range(len(means)):
        sample = rng.choice(valid, size=len(valid), replace=True)
        means[index] = float(np.mean(sample))
    return (
        float(np.mean(valid)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def bootstrap_budget_effects(
    effects: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()
    metrics = [
        "delta_order_consistent_majority",
        "delta_mean_branch_agreement",
        "delta_mean_branch_entropy",
        "delta_mean_natural_valid_rate",
        "delta_mean_forced_confidence",
        "delta_mean_majority_target_selected",
    ]
    groups = [
        "source_dataset",
        "criterion_id",
        "criterion_determinacy",
        "determinacy_level",
        "budget_tokens",
    ]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    for keys, group in effects.groupby(groups, sort=True, dropna=False):
        for metric in metrics:
            mean, low, high = _bootstrap_mean(
                group[metric].to_numpy(),
                n_bootstrap=int(n_bootstrap),
                rng=rng,
            )
            rows.append(
                {
                    **dict(zip(groups, keys, strict=True)),
                    "metric": metric,
                    "n_pairs": int(group["pair_id"].nunique()),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def grouped_bootstrap_summary(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    metrics: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for metric in metrics:
            mean, low, high = _bootstrap_mean(
                group[metric].to_numpy(),
                n_bootstrap=int(n_bootstrap),
                rng=rng,
            )
            rows.append(
                {
                    **dict(zip(group_columns, keys, strict=True)),
                    "metric": metric,
                    "n_pairs": int(group["pair_id"].nunique()),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def analyze_budget_sweep(
    rows: list[dict[str, Any]],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    frame = prepare_scores(rows)
    branches = branch_rows(frame)
    orders = order_rows(branches)
    effects = budget_pair_effects(orders)
    revisions = revision_rows(frame)
    slopes = pair_budget_slopes(orders)
    criterion_effects = criterion_pair_effects(orders)
    revision_summary = (
        revisions.groupby(
            [
                "source_dataset",
                "criterion_id",
                "criterion_determinacy",
                "determinacy_level",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            n_traces=("comparison_id", "count"),
            mean_forced_choice_revisions=("forced_choice_revisions", "mean"),
            median_first_stable_budget=("first_stable_budget", "median"),
            mean_confidence_change=("confidence_change", "mean"),
        )
        .reset_index()
    )
    return {
        "budget_summary": budget_summary(frame, orders),
        "branch_budget_rows": branches,
        "order_budget_rows": orders,
        "trace_revision_rows": revisions,
        "revision_summary": revision_summary,
        "budget_pair_effects": effects,
        "bootstrap_budget_effects": bootstrap_budget_effects(
            effects,
            n_bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "pair_budget_slopes": slopes,
        "budget_slope_summary": grouped_bootstrap_summary(
            slopes,
            group_columns=[
                "source_dataset",
                "criterion_id",
                "criterion_determinacy",
                "determinacy_level",
            ],
            metrics=[
                column
                for column in slopes.columns
                if column.startswith("slope_")
            ],
            n_bootstrap=int(bootstrap),
            seed=int(seed) + 1,
        ),
        "criterion_pair_effects": criterion_effects,
        "criterion_rescue_summary": grouped_bootstrap_summary(
            criterion_effects,
            group_columns=[
                "criterion_family",
                "criterion_id",
                "budget_tokens",
            ],
            metrics=[
                column
                for column in criterion_effects.columns
                if column.startswith("delta_vs_overall_")
            ],
            n_bootstrap=int(bootstrap),
            seed=int(seed) + 2,
        ),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    run_dir = _resolve(workspace_root, args.run_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    manifest = read_json(run_dir / "manifest.json")
    rows = read_jsonl(run_dir / "budget_scores.jsonl")
    outputs = analyze_budget_sweep(
        rows,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = str(path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-reasoning-token-budget-analysis",
            "run_dir": str(run_dir),
            "run_manifest": manifest,
            "out_dir": str(out_dir),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "n_score_rows": int(len(rows)),
            "outputs": output_paths,
        },
    )
    print(f"out_dir={out_dir}")
    print("\n=== Budget summary ===")
    print(outputs["budget_summary"].to_string(index=False))
    print("\n=== Revision summary ===")
    print(outputs["revision_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
