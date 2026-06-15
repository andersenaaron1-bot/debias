"""Analyze long and computationally enforced criterion reasoning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_json, read_jsonl, resolve_path, write_json
from aisafety.scripts.analyze_judge_criterion_confirmation import (
    _bootstrap,
    _default_audit_csv,
    _seed_offset,
    analyze_audit,
)


CONTRASTS = {
    "long_prompt_effect": ("prompted_long", "free_long"),
    "generic_staging_effect": ("enforced_generic", "free_long"),
    "criterion_staging_rescue": ("enforced_criterion", "free_long"),
    "enforcement_increment": (
        "enforced_criterion",
        "prompted_long",
    ),
    "criterion_staging_specificity": (
        "enforced_criterion",
        "enforced_generic",
    ),
}
PAIR_METRICS = (
    "forced_target_adoption",
    "order_consistent_target_adoption",
    "order_consistent_rate",
    "natural_valid_rate",
    "unconditional_natural_target_adoption",
    "valid_natural_target_adoption",
    "mean_choice_confidence",
    "analysis_budget_saturation_rate",
    "analysis_generated_tokens",
    "verdict_budget_saturation_rate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--source-suite-dir", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/"
            "judge_structured_cot_enforced_analysis_v1"
        ),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def trace_rows(traces: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        checkpoint = trace.get("decision_checkpoint") or {}
        valid = bool(trace.get("valid_choice"))
        target_selected = trace.get("final_target_semantic_selected")
        rows.append(
            {
                "trace_id": str(trace["trace_id"]),
                "pair_id": str(trace["pair_id"]),
                "condition_id": str(trace["condition_id"]),
                "transition_type": str(trace["transition_type"]),
                "presentation_order": str(
                    trace["presentation_order"]
                ),
                "branch_index": int(trace["branch_index"]),
                "target_semantic": str(trace["target_semantic"]),
                "forced_choice_semantic": str(
                    checkpoint.get("forced_choice_semantic") or ""
                ),
                "forced_choice_confidence": float(
                    checkpoint.get("forced_choice_confidence")
                    if checkpoint.get("forced_choice_confidence")
                    is not None
                    else np.nan
                ),
                "forced_target_adoption": float(
                    bool(
                        checkpoint.get(
                            "forced_target_semantic_selected"
                        )
                    )
                ),
                "natural_valid_rate": float(valid),
                "unconditional_natural_target_adoption": float(
                    valid and bool(target_selected)
                ),
                "valid_natural_target_adoption": (
                    float(bool(target_selected)) if valid else np.nan
                ),
                "analysis_budget_saturation_rate": float(
                    trace["analysis_budget_saturation_rate"]
                ),
                "analysis_generated_tokens": float(
                    trace["analysis_generated_tokens"]
                ),
                "verdict_budget_saturation_rate": float(
                    bool(trace["verdict_budget_saturated"])
                ),
            }
        )
    return pd.DataFrame(rows)


def pair_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    order_rows: list[dict[str, Any]] = []
    for keys, group in rows.groupby(
        ["pair_id", "condition_id", "branch_index"],
        sort=True,
    ):
        by_order = group.set_index("presentation_order")
        if not {"original", "swapped"}.issubset(by_order.index):
            continue
        original = str(by_order.loc["original", "forced_choice_semantic"])
        swapped = str(by_order.loc["swapped", "forced_choice_semantic"])
        target = str(by_order.loc["original", "target_semantic"])
        consistent = bool(original and original == swapped)
        order_rows.append(
            {
                "pair_id": keys[0],
                "condition_id": keys[1],
                "branch_index": int(keys[2]),
                "order_consistent_rate": float(consistent),
                "order_consistent_target_adoption": float(
                    consistent and original == target
                ),
            }
        )
    aggregation = (
        rows.groupby(
            ["pair_id", "condition_id", "transition_type"],
            sort=True,
        )
        .agg(
            forced_target_adoption=("forced_target_adoption", "mean"),
            natural_valid_rate=("natural_valid_rate", "mean"),
            unconditional_natural_target_adoption=(
                "unconditional_natural_target_adoption",
                "mean",
            ),
            valid_natural_target_adoption=(
                "valid_natural_target_adoption",
                "mean",
            ),
            mean_choice_confidence=(
                "forced_choice_confidence",
                "mean",
            ),
            analysis_budget_saturation_rate=(
                "analysis_budget_saturation_rate",
                "mean",
            ),
            analysis_generated_tokens=(
                "analysis_generated_tokens",
                "mean",
            ),
            verdict_budget_saturation_rate=(
                "verdict_budget_saturation_rate",
                "mean",
            ),
            n_traces=("trace_id", "size"),
        )
        .reset_index()
    )
    order = pd.DataFrame(order_rows)
    if order.empty:
        aggregation["order_consistent_rate"] = np.nan
        aggregation["order_consistent_target_adoption"] = np.nan
        return aggregation
    order = (
        order.groupby(["pair_id", "condition_id"], sort=True)
        .agg(
            order_consistent_rate=("order_consistent_rate", "mean"),
            order_consistent_target_adoption=(
                "order_consistent_target_adoption",
                "mean",
            ),
        )
        .reset_index()
    )
    return aggregation.merge(
        order,
        on=["pair_id", "condition_id"],
        how="left",
    )


def _strata(
    frame: pd.DataFrame,
) -> list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    transitions = sorted(
        str(value)
        for value in frame["transition_type"].dropna().unique()
    )
    return [
        ("all", lambda value: value),
        *[
            (
                transition,
                lambda value, target=transition: value[
                    value["transition_type"].eq(target)
                ],
            )
            for transition in transitions
        ],
    ]


def summary(
    frame: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for condition, base in frame.groupby("condition_id", sort=True):
        for transition, selector in _strata(frame):
            group = selector(base)
            if group.empty:
                continue
            for metric in PAIR_METRICS:
                mean, low, high = _bootstrap(
                    group[metric].to_numpy(dtype=float),
                    n_bootstrap=int(bootstrap),
                    seed=int(seed)
                    + _seed_offset(condition, transition, metric),
                )
                output.append(
                    {
                        "condition_id": condition,
                        "transition_type": transition,
                        "metric": metric,
                        "n_pairs": int(group["pair_id"].nunique()),
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(output)


def paired_effects(
    frame: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for contrast, (left, right) in CONTRASTS.items():
        selected = frame[frame["condition_id"].isin([left, right])]
        for metric in PAIR_METRICS:
            wide = selected[
                ["pair_id", "transition_type", "condition_id", metric]
            ].pivot_table(
                index=["pair_id", "transition_type"],
                columns="condition_id",
                values=metric,
                aggfunc="mean",
            )
            if left not in wide or right not in wide:
                continue
            wide = wide.dropna(subset=[left, right]).reset_index()
            wide["effect"] = wide[left] - wide[right]
            for transition, selector in _strata(frame):
                group = selector(wide)
                if group.empty:
                    continue
                mean, low, high = _bootstrap(
                    group["effect"].to_numpy(dtype=float),
                    n_bootstrap=int(bootstrap),
                    seed=int(seed)
                    + _seed_offset(contrast, transition, metric),
                )
                output.append(
                    {
                        "contrast": contrast,
                        "left_condition": left,
                        "right_condition": right,
                        "transition_type": transition,
                        "metric": metric,
                        "n_pairs": int(group["pair_id"].nunique()),
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(output)


def stage_summary(
    traces: list[dict[str, Any]],
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        for artifact in trace.get("artifacts") or []:
            rows.append(
                {
                    "pair_id": str(trace["pair_id"]),
                    "condition_id": str(trace["condition_id"]),
                    "stage_index": int(artifact["stage_index"]),
                    "stage_name": str(artifact["stage_name"]),
                    "generated_tokens": float(
                        artifact["generated_tokens"]
                    ),
                    "budget_saturated": float(
                        bool(artifact["budget_saturated"])
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["condition_id", "stage_index", "stage_name"],
        sort=True,
    ):
        pair_values = (
            group.groupby("pair_id", sort=True)
            .agg(
                generated_tokens=("generated_tokens", "mean"),
                budget_saturated=("budget_saturated", "mean"),
            )
            .reset_index()
        )
        for metric in ("generated_tokens", "budget_saturated"):
            mean, low, high = _bootstrap(
                pair_values[metric].to_numpy(dtype=float),
                n_bootstrap=int(bootstrap),
                seed=int(seed) + _seed_offset(*keys, metric),
            )
            output.append(
                {
                    "condition_id": keys[0],
                    "stage_index": int(keys[1]),
                    "stage_name": keys[2],
                    "metric": metric,
                    "n_pairs": int(pair_values["pair_id"].nunique()),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(output)


def analyze(
    *,
    run_dir: Path,
    source_suite_dir: Path,
    audit_csv: Path | None,
    bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    traces = read_jsonl(run_dir / "traces.jsonl")
    if not traces:
        raise ValueError(f"No enforced-structure traces found in {run_dir}")
    trace_frame = trace_rows(traces)
    pairs = pair_metrics(trace_frame)
    _audit_items, audit_pairs, confirmed_pairs = analyze_audit(
        suite_dir=source_suite_dir,
        audit_csv=audit_csv,
    )
    outputs = {
        "trace_rows": trace_frame,
        "pair_metrics": pairs,
        "condition_summary": summary(
            pairs,
            bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "paired_effects": paired_effects(
            pairs,
            bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "stage_summary": stage_summary(
            traces,
            bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "audit_pair_rows": audit_pairs,
    }
    if confirmed_pairs:
        confirmed = pairs[
            pairs["pair_id"].astype(str).isin(confirmed_pairs)
        ]
        outputs["audit_confirmed_effects"] = paired_effects(
            confirmed,
            bootstrap=int(bootstrap),
            seed=int(seed) + 1,
        )
    return outputs


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    run_dir = _resolve(workspace_root, args.run_dir)
    suite_dir = _resolve(workspace_root, args.suite_dir)
    source_suite_dir = _resolve(workspace_root, args.source_suite_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    audit_csv = (
        _resolve(workspace_root, args.audit_csv)
        if args.audit_csv is not None
        else _default_audit_csv(source_suite_dir)
    )
    run_manifest = read_json(run_dir / "manifest.json")
    suite_manifest = read_json(suite_dir / "manifest.json")
    expected = int(suite_manifest.get("n_planned_traces") or 0)
    observed = int(run_manifest.get("n_traces") or 0)
    if expected and expected != observed and not bool(args.allow_incomplete):
        raise ValueError(
            "Incomplete enforced-structure artifact: "
            f"traces={observed}/{expected}. Use --allow-incomplete only "
            "for diagnostics."
        )
    outputs = analyze(
        run_dir=run_dir,
        source_suite_dir=source_suite_dir,
        audit_csv=audit_csv,
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
            "stage": "judge-structured-cot-enforced-analysis",
            "run_dir": str(run_dir),
            "suite_dir": str(suite_dir),
            "source_suite_dir": str(source_suite_dir),
            "audit_csv": str(audit_csv) if audit_csv is not None else "",
            "out_dir": str(out_dir),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "expected_traces": expected,
            "observed_traces": observed,
            "allow_incomplete": bool(args.allow_incomplete),
            "contrasts": CONTRASTS,
            "outputs": paths,
        },
    )
    print(f"out_dir={out_dir}")
    print(outputs["condition_summary"].round(3).to_string(index=False))
    print(outputs["paired_effects"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
