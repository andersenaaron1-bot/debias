"""Analyze what chain-of-thought buys on factual pairwise decisions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_json, read_jsonl, resolve_path, write_json
from aisafety.scripts.analyze_judge_reasoning_budget_sweep import (
    branch_rows,
    order_rows,
    prepare_scores,
)


DEFAULT_RUN_DIR = (
    Path("artifacts")
    / "mechanistic"
    / "judge_deliberation_qwen3_8b_budget_scout_v1"
    / "budget_sweep"
)
DEFAULT_OUT_DIR = (
    Path("artifacts")
    / "mechanistic"
    / "judge_factual_cot_effect_qwen3_8b_v1"
    / "analysis"
)
DEFAULT_FACTUAL_DATASETS = (
    "gsm8k_verification",
    "math500_verification",
    "bbh_logical_deduction",
    "arc_challenge",
    "truthfulqa",
)
PAIR_METRICS = {
    "forced_target_adoption": "mean_majority_target_selected",
    "order_consistent_target_adoption": "order_consistent_target_adoption",
    "order_consistent_rate": "order_consistent_majority",
    "natural_valid_rate": "mean_natural_valid_rate",
    "branch_agreement": "mean_branch_agreement",
    "forced_choice_confidence": "mean_forced_confidence",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--include-datasets",
        default=",".join(DEFAULT_FACTUAL_DATASETS),
        help="Comma-separated source_dataset values to treat as factual.",
    )
    parser.add_argument("--endpoint-budget", type=int, default=2048)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _csv_set(raw: str) -> set[str]:
    return {value.strip() for value in str(raw).split(",") if value.strip()}


def _bootstrap_mean(
    values: Iterable[float],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    if int(n_bootstrap) <= 0:
        mean = float(np.mean(array))
        return mean, float("nan"), float("nan")
    means = np.empty((int(n_bootstrap),), dtype=float)
    for index in range(int(n_bootstrap)):
        means[index] = float(np.mean(rng.choice(array, size=len(array), replace=True)))
    return (
        float(np.mean(array)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _with_all(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    all_rows = frame.copy()
    all_rows["source_dataset"] = "all"
    return pd.concat([frame, all_rows], ignore_index=True)


def _condition_id(reasoning_mode: str, budget_tokens: int) -> str:
    if str(reasoning_mode) == "direct":
        return "direct"
    return f"cot_{int(budget_tokens)}"


def _add_order_consistent_target(orders: pd.DataFrame) -> pd.DataFrame:
    frame = orders.copy()
    selected = frame["order_majority_selected_text_hash"].fillna("").astype(str)
    target = frame["target_selected_text_hash"].fillna("").astype(str)
    consistent = frame["order_consistent_majority"].astype(float).eq(1.0)
    frame["order_consistent_target_adoption"] = (
        consistent & selected.ne("") & selected.eq(target)
    ).astype(float)
    return frame


def _summarize(
    pairs: pd.DataFrame,
    *,
    group_columns: list[str],
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(int(seed))
    for keys, group in pairs.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for metric_name, column in PAIR_METRICS.items():
            mean, low, high = _bootstrap_mean(
                group[column].to_numpy(),
                n_bootstrap=int(bootstrap),
                rng=rng,
            )
            rows.append(
                {
                    **dict(zip(group_columns, keys, strict=True)),
                    "metric": metric_name,
                    "n_pairs": int(group["pair_id"].nunique()),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def _direct_effects(
    pairs: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    keys = [
        "source_dataset",
        "criterion_id",
        "criterion_family",
        "criterion_determinacy",
        "determinacy_level",
        "origin_pair_id",
        "pair_id",
    ]
    direct = pairs[pairs["reasoning_mode"].eq("direct")].copy()
    thinking = pairs[pairs["reasoning_mode"].eq("thinking")].copy()
    direct = direct[keys + list(PAIR_METRICS.values())].rename(
        columns={column: f"direct_{column}" for column in PAIR_METRICS.values()}
    )
    merged = thinking.merge(direct, on=keys, how="inner")
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(int(seed))
    grouped = _with_all(merged)
    group_columns = ["source_dataset", "budget_tokens"]
    for keys_tuple, group in grouped.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys_tuple, tuple):
            keys_tuple = (keys_tuple,)
        for metric_name, column in PAIR_METRICS.items():
            values = (
                group[column].astype(float)
                - group[f"direct_{column}"].astype(float)
            ).to_numpy()
            mean, low, high = _bootstrap_mean(
                values,
                n_bootstrap=int(bootstrap),
                rng=rng,
            )
            rows.append(
                {
                    **dict(zip(group_columns, keys_tuple, strict=True)),
                    "contrast": f"cot_{int(keys_tuple[1])}_minus_direct",
                    "metric": metric_name,
                    "n_pairs": int(group["pair_id"].nunique()),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def _endpoint(frame: pd.DataFrame, endpoint_budget: int) -> pd.DataFrame:
    return frame[
        frame["condition_id"].eq("direct")
        | (
            frame["reasoning_mode"].eq("thinking")
            & frame["budget_tokens"].eq(int(endpoint_budget))
        )
    ].copy()


def analyze(
    rows: list[dict[str, object]],
    *,
    include_datasets: set[str],
    endpoint_budget: int,
    bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    frame = prepare_scores(rows)
    if include_datasets:
        frame = frame[frame["source_dataset"].astype(str).isin(include_datasets)].copy()
    if frame.empty:
        raise ValueError(
            "No factual rows remain after --include-datasets filtering. "
            f"Requested: {sorted(include_datasets)}"
        )
    branches = branch_rows(frame)
    orders = _add_order_consistent_target(order_rows(branches))
    orders["condition_id"] = [
        _condition_id(mode, budget)
        for mode, budget in zip(
            orders["reasoning_mode"].astype(str),
            orders["budget_tokens"].astype(int),
            strict=True,
        )
    ]
    pairs_all = _with_all(orders)
    summary = _summarize(
        pairs_all,
        group_columns=[
            "source_dataset",
            "condition_id",
            "reasoning_mode",
            "budget_tokens",
        ],
        bootstrap=int(bootstrap),
        seed=int(seed),
    )
    endpoint_summary = _endpoint(summary, int(endpoint_budget))
    effects = _direct_effects(
        orders,
        bootstrap=int(bootstrap),
        seed=int(seed) + 1,
    )
    endpoint_effects = effects[effects["budget_tokens"].eq(int(endpoint_budget))].copy()
    return {
        "factual_pair_metrics": orders,
        "factual_cot_summary": summary,
        "factual_cot_endpoint_summary": endpoint_summary,
        "factual_cot_effects": effects,
        "factual_cot_endpoint_effects": endpoint_effects,
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    run_dir = _resolve(workspace_root, args.run_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    manifest = read_json(run_dir / "manifest.json")
    rows = read_jsonl(run_dir / "budget_scores.jsonl")
    include_datasets = _csv_set(str(args.include_datasets))
    outputs = analyze(
        rows,
        include_datasets=include_datasets,
        endpoint_budget=int(args.endpoint_budget),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-factual-cot-effect-analysis",
            "run_dir": str(run_dir),
            "run_manifest": manifest,
            "out_dir": str(out_dir),
            "include_datasets": sorted(include_datasets),
            "endpoint_budget": int(args.endpoint_budget),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "n_score_rows": int(len(rows)),
            "n_factual_pairs": int(
                outputs["factual_pair_metrics"]["pair_id"].nunique()
            ),
            "outputs": paths,
        },
    )
    print(f"out_dir={out_dir}")
    print("\n=== FACTUAL COT ENDPOINT SUMMARY ===")
    print(outputs["factual_cot_endpoint_summary"].round(3).to_string(index=False))
    print("\n=== FACTUAL COT ENDPOINT EFFECTS ===")
    print(outputs["factual_cot_endpoint_effects"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
