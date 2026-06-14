"""Analyze the locked HelpSteer2 criterion-operationalization confirmation."""

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
    sha1_hex,
    write_json,
)


MAIN_CONDITIONS = (
    "early_criterion",
    "late_criterion",
    "early_evidence",
    "late_evidence",
)
CONTRASTS = {
    "early_operationalization_rescue": ("early_evidence", "early_criterion"),
    "late_operationalization_rescue": ("late_evidence", "late_criterion"),
    "criterion_commitment_penalty": ("late_criterion", "early_criterion"),
    "evidence_commitment_penalty": ("late_evidence", "early_evidence"),
    "explicit_target_vs_late_criterion": (
        "late_explicit_target",
        "late_criterion",
    ),
    "explicit_target_vs_late_evidence": (
        "late_explicit_target",
        "late_evidence",
    ),
}
PAIR_METRICS = (
    "forced_target_adoption",
    "order_consistent_target_adoption",
    "order_consistent_rate",
    "revision_rate",
    "phase1_anchoring_rate",
    "mean_choice_confidence",
    "natural_valid_rate",
    "unconditional_natural_target_adoption",
    "valid_natural_target_adoption",
    "phase2_budget_saturation_rate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument("--endpoint-budget", type=int, default=384)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/criterion_confirmation_analysis_v1"
        ),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _checkpoint_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat: list[dict[str, Any]] = []
    metadata_columns = (
        "trace_id",
        "pair_id",
        "episode_id",
        "branch_index",
        "condition_id",
        "transition_type",
        "presentation_order",
        "information_timing",
        "information_type",
        "phase1_target_semantic",
        "phase2_target_semantic",
        "final_choice_semantic",
        "valid_choice",
        "final_target_semantic_selected",
        "phase2_max_budget_saturated",
    )
    for row in rows:
        common = {column: row.get(column) for column in metadata_columns}
        for stage, checkpoints in (
            ("phase1", row.get("phase1_checkpoints") or []),
            ("phase2", row.get("phase2_checkpoints") or []),
        ):
            for checkpoint in checkpoints:
                flat.append({**common, "stage": stage, **checkpoint})
    return pd.DataFrame(flat)


def _bool_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else np.nan


def _endpoint_rows(
    trace_frame: pd.DataFrame,
    checkpoint_frame: pd.DataFrame,
    *,
    endpoint_budget: int,
) -> pd.DataFrame:
    phase1 = checkpoint_frame[
        checkpoint_frame["stage"].eq("phase1")
        & checkpoint_frame["budget_tokens"].eq(128)
    ][["trace_id", "forced_choice_semantic"]].rename(
        columns={"forced_choice_semantic": "phase1_choice_semantic"}
    )
    endpoint = checkpoint_frame[
        checkpoint_frame["stage"].eq("phase2")
        & checkpoint_frame["budget_tokens"].eq(int(endpoint_budget))
    ].merge(phase1, on="trace_id", how="left")
    endpoint["forced_target_adoption"] = endpoint[
        "forced_choice_semantic"
    ].eq(endpoint["phase2_target_semantic"])
    endpoint["revision_rate"] = (
        endpoint["phase1_choice_semantic"].fillna("").ne("")
        & endpoint["forced_choice_semantic"].fillna("").ne("")
        & endpoint["phase1_choice_semantic"].ne(
            endpoint["forced_choice_semantic"]
        )
    )
    endpoint["phase1_anchoring_rate"] = (
        endpoint["phase1_choice_semantic"].fillna("").ne("")
        & endpoint["phase1_choice_semantic"].eq(
            endpoint["forced_choice_semantic"]
        )
    )
    trace_columns = [
        "trace_id",
        "valid_choice",
        "final_target_semantic_selected",
        "phase2_max_budget_saturated",
    ]
    endpoint = endpoint.drop(
        columns=[
            column
            for column in trace_columns[1:]
            if column in endpoint.columns
        ]
    ).merge(trace_frame[trace_columns], on="trace_id", how="left")
    endpoint["natural_valid_rate"] = endpoint["valid_choice"].astype(float)
    endpoint["unconditional_natural_target_adoption"] = (
        endpoint["valid_choice"].astype(bool)
        & endpoint["final_target_semantic_selected"].fillna(False).astype(bool)
    ).astype(float)
    endpoint["valid_natural_target_adoption"] = pd.to_numeric(
        endpoint["final_target_semantic_selected"], errors="coerce"
    )
    endpoint["phase2_budget_saturation_rate"] = pd.to_numeric(
        endpoint["phase2_max_budget_saturated"], errors="coerce"
    )
    return endpoint


def _pair_endpoint_metrics(endpoint: pd.DataFrame) -> pd.DataFrame:
    order_rows: list[dict[str, Any]] = []
    for keys, group in endpoint.groupby(
        ["pair_id", "condition_id", "branch_index"], sort=True
    ):
        by_order = group.set_index("presentation_order")
        if not {"original", "swapped"}.issubset(by_order.index):
            continue
        original = str(by_order.loc["original", "forced_choice_semantic"])
        swapped = str(by_order.loc["swapped", "forced_choice_semantic"])
        target = str(by_order.loc["original", "phase2_target_semantic"])
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
    order_frame = pd.DataFrame(order_rows)
    aggregation = (
        endpoint.groupby(
            ["pair_id", "condition_id", "transition_type"], sort=True
        )
        .agg(
            forced_target_adoption=("forced_target_adoption", "mean"),
            revision_rate=("revision_rate", "mean"),
            phase1_anchoring_rate=("phase1_anchoring_rate", "mean"),
            mean_choice_confidence=("forced_choice_confidence", "mean"),
            natural_valid_rate=("natural_valid_rate", "mean"),
            unconditional_natural_target_adoption=(
                "unconditional_natural_target_adoption",
                "mean",
            ),
            valid_natural_target_adoption=(
                "valid_natural_target_adoption",
                "mean",
            ),
            phase2_budget_saturation_rate=(
                "phase2_budget_saturation_rate",
                "mean",
            ),
            n_traces=("trace_id", "size"),
        )
        .reset_index()
    )
    if order_frame.empty:
        aggregation["order_consistent_rate"] = np.nan
        aggregation["order_consistent_target_adoption"] = np.nan
        return aggregation
    order_pair = (
        order_frame.groupby(["pair_id", "condition_id"], sort=True)
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
        order_pair,
        on=["pair_id", "condition_id"],
        how="left",
    )


def _bootstrap(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(clean.mean())
    if clean.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(
        clean,
        size=(max(int(n_bootstrap), 1), clean.size),
        replace=True,
    ).mean(axis=1)
    return (
        mean,
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _seed_offset(*values: Any) -> int:
    return int(sha1_hex("|".join(str(value) for value in values))[:8], 16)


def _contrast_values(
    frame: pd.DataFrame,
    *,
    metric: str,
    left: str,
    right: str,
) -> pd.DataFrame:
    selected = frame[
        frame["condition_id"].isin([left, right])
    ][["pair_id", "transition_type", "condition_id", metric]]
    wide = selected.pivot_table(
        index=["pair_id", "transition_type"],
        columns="condition_id",
        values=metric,
        aggfunc="mean",
    )
    if left not in wide or right not in wide:
        return pd.DataFrame()
    wide = wide.dropna(subset=[left, right]).reset_index()
    wide["effect"] = wide[left] - wide[right]
    return wide


def paired_effects(
    pair_metrics: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strata: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
        ("all", lambda frame: frame),
        *[
            (
                transition,
                lambda frame, value=transition: frame[
                    frame["transition_type"].eq(value)
                ],
            )
            for transition in sorted(pair_metrics["transition_type"].unique())
        ],
    ]
    for contrast, (left, right) in CONTRASTS.items():
        for metric in PAIR_METRICS:
            values = _contrast_values(
                pair_metrics,
                metric=metric,
                left=left,
                right=right,
            )
            if values.empty:
                continue
            for stratum, selector in strata:
                selected = selector(values)
                if selected.empty:
                    continue
                effect, low, high = _bootstrap(
                    selected["effect"].to_numpy(dtype=float),
                    n_bootstrap=int(bootstrap),
                    seed=int(seed)
                    + _seed_offset(contrast, metric, stratum),
                )
                rows.append(
                    {
                        "contrast": contrast,
                        "left_condition": left,
                        "right_condition": right,
                        "transition_type": stratum,
                        "metric": metric,
                        "n_pairs": int(selected["pair_id"].nunique()),
                        "mean": effect,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    interaction_rows: list[dict[str, Any]] = []
    for metric in PAIR_METRICS:
        columns = [
            "pair_id",
            "transition_type",
            "condition_id",
            metric,
        ]
        wide = pair_metrics[
            pair_metrics["condition_id"].isin(MAIN_CONDITIONS)
        ][columns].pivot_table(
            index=["pair_id", "transition_type"],
            columns="condition_id",
            values=metric,
            aggfunc="mean",
        )
        if not set(MAIN_CONDITIONS).issubset(wide.columns):
            continue
        wide = wide.dropna(subset=list(MAIN_CONDITIONS)).reset_index()
        wide["effect"] = (
            wide["early_evidence"]
            - wide["early_criterion"]
            - wide["late_evidence"]
            + wide["late_criterion"]
        )
        for stratum, selector in strata:
            selected = selector(wide)
            if selected.empty:
                continue
            effect, low, high = _bootstrap(
                selected["effect"].to_numpy(dtype=float),
                n_bootstrap=int(bootstrap),
                seed=int(seed)
                + _seed_offset("timing_by_evidence", metric, stratum),
            )
            interaction_rows.append(
                {
                    "contrast": "timing_by_evidence_interaction",
                    "left_condition": "(early_evidence-early_criterion)",
                    "right_condition": "(late_evidence-late_criterion)",
                    "transition_type": stratum,
                    "metric": metric,
                    "n_pairs": int(selected["pair_id"].nunique()),
                    "mean": effect,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame([*rows, *interaction_rows])


def _condition_summary(
    pair_metrics: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        (
            str(condition),
            "all",
            group,
        )
        for condition, group in pair_metrics.groupby("condition_id", sort=True)
    ]
    groups.extend(
        (
            str(keys[0]),
            str(keys[1]),
            group,
        )
        for keys, group in pair_metrics.groupby(
            ["condition_id", "transition_type"], sort=True
        )
    )
    for condition, transition, group in groups:
        for metric in PAIR_METRICS:
            mean, low, high = _bootstrap(
                pd.to_numeric(group[metric], errors="coerce").to_numpy(),
                n_bootstrap=int(bootstrap),
                seed=int(seed)
                + _seed_offset(condition, transition, metric),
            )
            rows.append(
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
    return pd.DataFrame(rows)


def _audit_column(frame: pd.DataFrame) -> str | None:
    for column in ("judge", "verdict"):
        if column in frame.columns:
            values = frame[column].fillna("").astype(str).str.strip()
            if values.ne("").any():
                return column
    return None


def analyze_audit(
    *,
    suite_dir: Path,
    audit_csv: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    if audit_csv is None or not audit_csv.is_file():
        return pd.DataFrame(), pd.DataFrame(), set()
    responses = pd.read_csv(audit_csv)
    verdict_column = _audit_column(responses)
    if verdict_column is None:
        return pd.DataFrame(), pd.DataFrame(), set()
    items = pd.DataFrame(
        read_jsonl(suite_dir / "human_audit" / "audit_items.jsonl")
    )
    reference = pd.DataFrame(
        read_jsonl(
            suite_dir / "human_audit" / "private_proxy_reference.jsonl"
        )
    )
    responses["human_displayed"] = (
        responses[verdict_column].fillna("").astype(str).str.strip().str.upper()
    )
    responses = responses[responses["human_displayed"].isin(["A", "B", "C"])]
    item_rows = (
        items.merge(
            reference,
            on=[
                "audit_id",
                "pair_id",
                "criterion_role",
                "criterion_id",
                "presentation_order",
            ],
        )
        .merge(
            responses[["audit_id", "human_displayed"]],
            on="audit_id",
            how="inner",
        )
    )
    item_rows["human_semantic"] = np.where(
        item_rows["presentation_order"].eq("swapped"),
        item_rows["human_displayed"].map({"A": "B", "B": "A", "C": "C"}),
        item_rows["human_displayed"],
    )
    item_rows["proxy_agreement"] = item_rows["human_displayed"].eq(
        item_rows["proxy_target_displayed"]
    )
    pair_rows: list[dict[str, Any]] = []
    for keys, group in item_rows.groupby(
        ["pair_id", "criterion_role", "criterion_id"], sort=True
    ):
        by_order = group.set_index("presentation_order")
        complete = {"original", "swapped"}.issubset(by_order.index)
        semantic_values = (
            by_order.loc[["original", "swapped"], "human_semantic"].tolist()
            if complete
            else []
        )
        order_consistent = bool(
            complete
            and semantic_values[0]
            and semantic_values[0] == semantic_values[1]
        )
        proxy_agreement = bool(
            complete
            and by_order.loc[
                ["original", "swapped"], "proxy_agreement"
            ].all()
        )
        pair_rows.append(
            {
                "pair_id": keys[0],
                "criterion_role": keys[1],
                "criterion_id": keys[2],
                "complete": complete,
                "order_consistent": order_consistent,
                "proxy_agreement": proxy_agreement,
                "audit_confirmed": bool(order_consistent and proxy_agreement),
                "human_target_semantic": (
                    semantic_values[0] if order_consistent else ""
                ),
            }
        )
    pair_frame = pd.DataFrame(pair_rows)
    confirmed = set(
        pair_frame.groupby("pair_id")["audit_confirmed"]
        .agg(lambda values: len(values) == 2 and bool(values.all()))
        .loc[lambda values: values]
        .index.astype(str)
    )
    return item_rows, pair_frame, confirmed


def _default_audit_csv(suite_dir: Path) -> Path | None:
    root = suite_dir / "human_audit"
    for name in (
        "audit_prompts_for_judging_completed.csv",
        "audit_prompts_for_judging.csv",
        "audit_responses.csv",
    ):
        path = root / name
        if path.is_file():
            return path
    return None


def analyze(
    *,
    run_dir: Path,
    suite_dir: Path,
    audit_csv: Path | None,
    endpoint_budget: int,
    bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    traces = read_jsonl(run_dir / "switch_traces.jsonl")
    if not traces:
        raise ValueError(f"No switch traces found in {run_dir}")
    trace_frame = pd.DataFrame(traces)
    checkpoint_frame = _checkpoint_frame(traces)
    endpoint = _endpoint_rows(
        trace_frame,
        checkpoint_frame,
        endpoint_budget=int(endpoint_budget),
    )
    pair_metrics = _pair_endpoint_metrics(endpoint)
    audit_items, audit_pairs, confirmed_pairs = analyze_audit(
        suite_dir=suite_dir,
        audit_csv=audit_csv,
    )
    outputs = {
        "endpoint_rows": endpoint,
        "pair_endpoint_metrics": pair_metrics,
        "condition_summary": _condition_summary(
            pair_metrics,
            bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "paired_effects": paired_effects(
            pair_metrics,
            bootstrap=int(bootstrap),
            seed=int(seed),
        ),
        "audit_item_rows": audit_items,
        "audit_pair_rows": audit_pairs,
    }
    if confirmed_pairs:
        confirmed = pair_metrics[
            pair_metrics["pair_id"].astype(str).isin(confirmed_pairs)
        ].copy()
        outputs["audit_confirmed_condition_summary"] = _condition_summary(
            confirmed,
            bootstrap=int(bootstrap),
            seed=int(seed) + 1,
        )
        outputs["audit_confirmed_paired_effects"] = paired_effects(
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
    out_dir = _resolve(workspace_root, args.out_dir)
    audit_csv = (
        _resolve(workspace_root, args.audit_csv)
        if args.audit_csv is not None
        else _default_audit_csv(suite_dir)
    )
    run_manifest = read_json(run_dir / "manifest.json")
    suite_manifest = read_json(suite_dir / "manifest.json")
    expected_traces = int(suite_manifest.get("n_planned_traces") or 0)
    observed_traces = int(run_manifest.get("n_switch_traces") or 0)
    if (
        expected_traces
        and observed_traces != expected_traces
        and not bool(args.allow_incomplete)
    ):
        raise ValueError(
            f"Incomplete confirmation artifact: observed {observed_traces} "
            f"traces, expected {expected_traces}. Use --allow-incomplete only "
            "for a diagnostic readout."
        )
    outputs = analyze(
        run_dir=run_dir,
        suite_dir=suite_dir,
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
            "stage": "judge-criterion-confirmation-analysis",
            "run_dir": str(run_dir),
            "suite_dir": str(suite_dir),
            "audit_csv": str(audit_csv) if audit_csv is not None else "",
            "out_dir": str(out_dir),
            "endpoint_budget": int(args.endpoint_budget),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "expected_traces": expected_traces,
            "observed_traces": observed_traces,
            "allow_incomplete": bool(args.allow_incomplete),
            "outputs": paths,
        },
    )
    print(f"out_dir={out_dir}")
    print(outputs["condition_summary"].to_string(index=False))
    print(outputs["paired_effects"].to_string(index=False))


if __name__ == "__main__":
    main()
