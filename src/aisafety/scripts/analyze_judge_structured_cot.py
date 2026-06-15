"""Analyze matched reasoning-structure effects on criterion use."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import (
    read_json,
    read_jsonl,
    resolve_path,
    write_json,
)
from aisafety.scripts.analyze_judge_criterion_confirmation import (
    PAIR_METRICS,
    _bootstrap,
    _checkpoint_frame,
    _condition_summary,
    _default_audit_csv,
    _endpoint_rows,
    _pair_endpoint_metrics,
    _seed_offset,
    analyze_audit,
)


CONTRASTS = {
    "criterion_scaffold_rescue": (
        "criterion_scaffold",
        "free_cot",
    ),
    "criterion_scaffold_specificity": (
        "criterion_scaffold",
        "generic_scaffold",
    ),
    "generic_structure_effect": (
        "generic_scaffold",
        "free_cot",
    ),
    "score_evidence_rescue": (
        "score_evidence",
        "free_cot",
    ),
    "scaffold_gap_to_score_evidence": (
        "criterion_scaffold",
        "score_evidence",
    ),
    "explicit_target_vs_scaffold": (
        "explicit_target",
        "criterion_scaffold",
    ),
}
CHECKPOINT_METRICS = (
    "forced_target_adoption",
    "order_consistent_target_adoption",
    "order_consistent_rate",
    "mean_choice_confidence",
)
DIRECT_METRICS = (
    "forced_target_adoption",
    "order_consistent_target_adoption",
    "order_consistent_rate",
    "mean_choice_confidence",
    "natural_valid_rate",
    "unconditional_natural_target_adoption",
    "valid_natural_target_adoption",
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
    parser.add_argument("--endpoint-budget", type=int, default=384)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/judge_structured_cot_analysis_v1"
        ),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _strata(
    frame: pd.DataFrame,
) -> list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    values = sorted(
        str(value) for value in frame["transition_type"].dropna().unique()
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
            for transition in values
        ],
    ]


def _paired_effects(
    pair_metrics: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    bootstrap: int,
    seed: int,
    extra_index: list[str] | None = None,
) -> pd.DataFrame:
    if pair_metrics.empty:
        return pd.DataFrame()
    extra = list(extra_index or [])
    rows: list[dict[str, Any]] = []
    strata = _strata(pair_metrics)
    extra_groups: list[tuple[tuple[Any, ...], pd.DataFrame]]
    if extra:
        extra_groups = list(
            pair_metrics.groupby(extra, sort=True, dropna=False)
        )
    else:
        extra_groups = [((), pair_metrics)]
    for extra_values, base in extra_groups:
        if not isinstance(extra_values, tuple):
            extra_values = (extra_values,)
        extra_payload = dict(zip(extra, extra_values, strict=True))
        for contrast, (left, right) in CONTRASTS.items():
            selected = base[
                base["condition_id"].isin([left, right])
            ]
            for metric in metrics:
                columns = [
                    "pair_id",
                    "transition_type",
                    "condition_id",
                    metric,
                ]
                wide = selected[columns].pivot_table(
                    index=["pair_id", "transition_type"],
                    columns="condition_id",
                    values=metric,
                    aggfunc="mean",
                )
                if left not in wide or right not in wide:
                    continue
                wide = wide.dropna(subset=[left, right]).reset_index()
                wide["effect"] = wide[left] - wide[right]
                for stratum, selector in strata:
                    values = selector(wide)
                    if values.empty:
                        continue
                    mean, low, high = _bootstrap(
                        values["effect"].to_numpy(dtype=float),
                        n_bootstrap=int(bootstrap),
                        seed=int(seed)
                        + _seed_offset(
                            contrast,
                            metric,
                            stratum,
                            *extra_values,
                        ),
                    )
                    rows.append(
                        {
                            **extra_payload,
                            "contrast": contrast,
                            "left_condition": left,
                            "right_condition": right,
                            "transition_type": stratum,
                            "metric": metric,
                            "n_pairs": int(
                                values["pair_id"].nunique()
                            ),
                            "mean": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                        }
                    )
    return pd.DataFrame(rows)


def _checkpoint_pair_metrics(
    checkpoints: pd.DataFrame,
) -> pd.DataFrame:
    frame = checkpoints.copy()
    frame["target_semantic"] = np.where(
        frame["stage"].eq("phase1"),
        frame["phase1_target_semantic"],
        frame["phase2_target_semantic"],
    )
    frame["forced_target_adoption"] = frame[
        "forced_choice_semantic"
    ].eq(frame["target_semantic"])
    order_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        [
            "pair_id",
            "condition_id",
            "transition_type",
            "branch_index",
            "stage",
            "budget_tokens",
        ],
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
                "transition_type": keys[2],
                "branch_index": int(keys[3]),
                "stage": keys[4],
                "budget_tokens": int(keys[5]),
                "order_consistent_rate": float(consistent),
                "order_consistent_target_adoption": float(
                    consistent and original == target
                ),
            }
        )
    aggregate = (
        frame.groupby(
            [
                "pair_id",
                "condition_id",
                "transition_type",
                "stage",
                "budget_tokens",
            ],
            sort=True,
        )
        .agg(
            forced_target_adoption=("forced_target_adoption", "mean"),
            mean_choice_confidence=("forced_choice_confidence", "mean"),
            n_traces=("trace_id", "size"),
        )
        .reset_index()
    )
    order = pd.DataFrame(order_rows)
    if order.empty:
        aggregate["order_consistent_rate"] = np.nan
        aggregate["order_consistent_target_adoption"] = np.nan
        return aggregate
    order = (
        order.groupby(
            [
                "pair_id",
                "condition_id",
                "transition_type",
                "stage",
                "budget_tokens",
            ],
            sort=True,
        )
        .agg(
            order_consistent_rate=("order_consistent_rate", "mean"),
            order_consistent_target_adoption=(
                "order_consistent_target_adoption",
                "mean",
            ),
        )
        .reset_index()
    )
    return aggregate.merge(
        order,
        on=[
            "pair_id",
            "condition_id",
            "transition_type",
            "stage",
            "budget_tokens",
        ],
        how="left",
    )


def _summary(
    pair_metrics: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    bootstrap: int,
    seed: int,
    extra_groups: list[str] | None = None,
) -> pd.DataFrame:
    if pair_metrics.empty:
        return pd.DataFrame()
    extra = list(extra_groups or [])
    rows: list[dict[str, Any]] = []
    keys = ["condition_id", *extra]
    groups: list[tuple[tuple[Any, ...], pd.DataFrame]] = []
    for values, group in pair_metrics.groupby(
        keys,
        sort=True,
        dropna=False,
    ):
        normalized = values if isinstance(values, tuple) else (values,)
        groups.append(((*normalized, "all"), group))
    keys_with_transition = [*keys, "transition_type"]
    for values, group in pair_metrics.groupby(
        keys_with_transition,
        sort=True,
        dropna=False,
    ):
        normalized = values if isinstance(values, tuple) else (values,)
        groups.append((normalized, group))
    output_keys = [*keys, "transition_type"]
    for values, group in groups:
        if not isinstance(values, tuple):
            values = (values,)
        payload = dict(zip(output_keys, values, strict=True))
        for metric in metrics:
            mean, low, high = _bootstrap(
                pd.to_numeric(group[metric], errors="coerce").to_numpy(),
                n_bootstrap=int(bootstrap),
                seed=int(seed)
                + _seed_offset(
                    *values,
                    metric,
                ),
            )
            rows.append(
                {
                    **payload,
                    "metric": metric,
                    "n_pairs": int(group["pair_id"].nunique()),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def _direct_pair_metrics(
    direct_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    if not direct_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(direct_rows)
    frame["forced_target_adoption"] = frame[
        "forced_choice_semantic"
    ].eq(frame["target_semantic"])
    frame["natural_valid_rate"] = frame["natural_valid"].astype(float)
    frame["unconditional_natural_target_adoption"] = (
        frame["natural_valid"].astype(bool)
        & frame["natural_choice_semantic"].eq(frame["target_semantic"])
    ).astype(float)
    frame["valid_natural_target_adoption"] = np.where(
        frame["natural_valid"].astype(bool),
        frame["natural_choice_semantic"].eq(frame["target_semantic"]).astype(
            float
        ),
        np.nan,
    )
    pair_rows: list[dict[str, Any]] = []
    for pair_id, group in frame.groupby("pair_id", sort=True):
        by_order = group.set_index("presentation_order")
        if not {"original", "swapped"}.issubset(by_order.index):
            continue
        original = str(by_order.loc["original", "forced_choice_semantic"])
        swapped = str(by_order.loc["swapped", "forced_choice_semantic"])
        target = str(by_order.loc["original", "target_semantic"])
        consistent = bool(original and original == swapped)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "transition_type": str(
                    by_order.loc["original", "transition_type"]
                ),
                "forced_target_adoption": float(
                    group["forced_target_adoption"].mean()
                ),
                "order_consistent_rate": float(consistent),
                "order_consistent_target_adoption": float(
                    consistent and original == target
                ),
                "mean_choice_confidence": float(
                    group["forced_choice_confidence"].mean()
                ),
                "natural_valid_rate": float(
                    group["natural_valid_rate"].mean()
                ),
                "unconditional_natural_target_adoption": float(
                    group[
                        "unconditional_natural_target_adoption"
                    ].mean()
                ),
                "valid_natural_target_adoption": float(
                    group["valid_natural_target_adoption"].mean()
                ),
            }
        )
    return pd.DataFrame(pair_rows)


def _direct_summary(
    frame: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    with_condition = frame.assign(condition_id="direct")
    return _summary(
        with_condition,
        metrics=DIRECT_METRICS,
        bootstrap=int(bootstrap),
        seed=int(seed),
    )


def _direct_contrasts(
    endpoint: pd.DataFrame,
    direct: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if endpoint.empty or direct.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    conditions = (
        "free_cot",
        "generic_scaffold",
        "criterion_scaffold",
        "score_evidence",
    )
    metrics = tuple(
        metric
        for metric in DIRECT_METRICS
        if metric in endpoint.columns and metric in direct.columns
    )
    for condition in conditions:
        selected = endpoint[endpoint["condition_id"].eq(condition)]
        for metric in metrics:
            merged = selected[
                ["pair_id", "transition_type", metric]
            ].merge(
                direct[["pair_id", metric]],
                on="pair_id",
                suffixes=("_cot", "_direct"),
            )
            merged["effect"] = (
                merged[f"{metric}_cot"] - merged[f"{metric}_direct"]
            )
            for stratum, selector in _strata(endpoint):
                values = selector(merged)
                if values.empty:
                    continue
                mean, low, high = _bootstrap(
                    values["effect"].to_numpy(dtype=float),
                    n_bootstrap=int(bootstrap),
                    seed=int(seed)
                    + _seed_offset(
                        condition,
                        "vs_direct",
                        metric,
                        stratum,
                    ),
                )
                rows.append(
                    {
                        "condition_id": condition,
                        "reference": "direct",
                        "transition_type": stratum,
                        "metric": metric,
                        "n_pairs": int(values["pair_id"].nunique()),
                        "mean": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(rows)


def analyze(
    *,
    run_dir: Path,
    source_suite_dir: Path,
    audit_csv: Path | None,
    endpoint_budget: int,
    bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    traces = read_jsonl(run_dir / "switch_traces.jsonl")
    if not traces:
        raise ValueError(f"No structured-CoT traces found in {run_dir}")
    trace_frame = pd.DataFrame(traces)
    checkpoints = _checkpoint_frame(traces)
    endpoint_rows = _endpoint_rows(
        trace_frame,
        checkpoints,
        endpoint_budget=int(endpoint_budget),
    )
    endpoint_pairs = _pair_endpoint_metrics(endpoint_rows)
    checkpoint_pairs = _checkpoint_pair_metrics(checkpoints)
    direct_pairs = _direct_pair_metrics(
        read_jsonl(run_dir / "direct_rows.jsonl")
    )
    _audit_items, audit_pairs, confirmed_pairs = analyze_audit(
        suite_dir=source_suite_dir,
        audit_csv=audit_csv,
    )
    outputs = {
        "checkpoint_rows": checkpoints,
        "checkpoint_pair_metrics": checkpoint_pairs,
        "checkpoint_summary": _summary(
            checkpoint_pairs,
            metrics=CHECKPOINT_METRICS,
            bootstrap=int(bootstrap),
            seed=int(seed),
            extra_groups=["stage", "budget_tokens"],
        ),
        "checkpoint_effects": _paired_effects(
            checkpoint_pairs,
            metrics=CHECKPOINT_METRICS,
            bootstrap=int(bootstrap),
            seed=int(seed),
            extra_index=["stage", "budget_tokens"],
        ),
        "endpoint_rows": endpoint_rows,
        "endpoint_pair_metrics": endpoint_pairs,
        "endpoint_summary": _condition_summary(
            endpoint_pairs,
            bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "endpoint_effects": _paired_effects(
            endpoint_pairs,
            metrics=PAIR_METRICS,
            bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "direct_pair_metrics": direct_pairs,
        "direct_summary": _direct_summary(
            direct_pairs,
            bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "cot_vs_direct_effects": _direct_contrasts(
            endpoint_pairs,
            direct_pairs,
            bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "audit_pair_rows": audit_pairs,
    }
    if confirmed_pairs:
        confirmed_endpoint = endpoint_pairs[
            endpoint_pairs["pair_id"].astype(str).isin(confirmed_pairs)
        ]
        confirmed_checkpoint = checkpoint_pairs[
            checkpoint_pairs["pair_id"].astype(str).isin(confirmed_pairs)
        ]
        outputs["audit_confirmed_endpoint_effects"] = _paired_effects(
            confirmed_endpoint,
            metrics=PAIR_METRICS,
            bootstrap=int(bootstrap),
            seed=int(seed) + 1,
        )
        outputs["audit_confirmed_checkpoint_effects"] = _paired_effects(
            confirmed_checkpoint,
            metrics=CHECKPOINT_METRICS,
            bootstrap=int(bootstrap),
            seed=int(seed) + 1,
            extra_index=["stage", "budget_tokens"],
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
    observed = int(run_manifest.get("n_switch_traces") or 0)
    expected_direct = int(
        suite_manifest.get("n_planned_direct_rows") or 0
    )
    observed_direct = int(run_manifest.get("n_direct_rows") or 0)
    if (
        (expected and expected != observed)
        or (expected_direct and expected_direct != observed_direct)
    ) and not bool(args.allow_incomplete):
        raise ValueError(
            "Incomplete structured-CoT artifact: "
            f"traces={observed}/{expected}, "
            f"direct={observed_direct}/{expected_direct}. "
            "Use --allow-incomplete only for diagnostics."
        )
    outputs = analyze(
        run_dir=run_dir,
        source_suite_dir=source_suite_dir,
        audit_csv=audit_csv,
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
            "stage": "judge-structured-cot-analysis",
            "run_dir": str(run_dir),
            "suite_dir": str(suite_dir),
            "source_suite_dir": str(source_suite_dir),
            "audit_csv": str(audit_csv) if audit_csv is not None else "",
            "out_dir": str(out_dir),
            "endpoint_budget": int(args.endpoint_budget),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "expected_traces": expected,
            "observed_traces": observed,
            "expected_direct_rows": expected_direct,
            "observed_direct_rows": observed_direct,
            "allow_incomplete": bool(args.allow_incomplete),
            "contrasts": CONTRASTS,
            "outputs": paths,
        },
    )
    print(f"out_dir={out_dir}")
    print(outputs["endpoint_summary"].to_string(index=False))
    print(outputs["endpoint_effects"].to_string(index=False))


if __name__ == "__main__":
    main()
