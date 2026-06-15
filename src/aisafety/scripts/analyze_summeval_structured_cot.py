"""Analyze SummEval criterion-operationalization validation runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_json, read_jsonl, resolve_path, write_json
from aisafety.scripts.analyze_judge_criterion_confirmation import PAIR_METRICS
from aisafety.scripts.analyze_judge_structured_cot import (
    CHECKPOINT_METRICS,
    DIRECT_METRICS,
    _direct_contrasts,
    _direct_summary,
    _paired_effects,
    _summary,
    analyze as analyze_structured,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--endpoint-budget", type=int, default=384)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/summeval_structured_cot_analysis_v1"
        ),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _pair_metadata(suite_dir: Path) -> pd.DataFrame:
    rows = read_jsonl(suite_dir / "pairs.jsonl")
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    keep = [
        "pair_id",
        "origin_pair_id",
        "source_document_id",
        "initial_criterion_id",
        "updated_criterion_id",
        "transition_type",
        "analysis_split",
    ]
    return frame[[column for column in keep if column in frame.columns]].copy()


def _merge_metadata(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or metadata.empty:
        return frame
    drop = [
        column
        for column in (
            "origin_pair_id",
            "source_document_id",
            "initial_criterion_id",
            "updated_criterion_id",
            "analysis_split",
        )
        if column in frame.columns
    ]
    return frame.drop(columns=drop).merge(
        metadata,
        on=["pair_id", "transition_type"],
        how="left",
    )


def analyze(
    *,
    run_dir: Path,
    suite_dir: Path,
    endpoint_budget: int,
    bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    outputs = analyze_structured(
        run_dir=run_dir,
        source_suite_dir=suite_dir,
        audit_csv=None,
        endpoint_budget=int(endpoint_budget),
        bootstrap=int(bootstrap),
        seed=int(seed),
    )
    metadata = _pair_metadata(suite_dir)
    endpoint_pairs = _merge_metadata(
        outputs["endpoint_pair_metrics"],
        metadata,
    )
    checkpoint_pairs = _merge_metadata(
        outputs["checkpoint_pair_metrics"],
        metadata,
    )
    direct_pairs = _merge_metadata(
        outputs["direct_pair_metrics"],
        metadata,
    )

    outputs["endpoint_pair_metrics_with_metadata"] = endpoint_pairs
    outputs["checkpoint_pair_metrics_with_metadata"] = checkpoint_pairs
    outputs["direct_pair_metrics_with_metadata"] = direct_pairs
    if not endpoint_pairs.empty and "updated_criterion_id" in endpoint_pairs:
        outputs["criterion_endpoint_summary"] = _summary(
            endpoint_pairs,
            metrics=PAIR_METRICS,
            bootstrap=int(bootstrap),
            seed=int(seed),
            extra_groups=["updated_criterion_id"],
        )
        outputs["criterion_endpoint_effects"] = _paired_effects(
            endpoint_pairs,
            metrics=PAIR_METRICS,
            bootstrap=int(bootstrap),
            seed=int(seed),
            extra_index=["updated_criterion_id"],
        )
    else:
        outputs["criterion_endpoint_summary"] = pd.DataFrame()
        outputs["criterion_endpoint_effects"] = pd.DataFrame()

    if not checkpoint_pairs.empty and "updated_criterion_id" in checkpoint_pairs:
        outputs["criterion_checkpoint_summary"] = _summary(
            checkpoint_pairs,
            metrics=CHECKPOINT_METRICS,
            bootstrap=int(bootstrap),
            seed=int(seed),
            extra_groups=["updated_criterion_id", "stage", "budget_tokens"],
        )
        outputs["criterion_checkpoint_effects"] = _paired_effects(
            checkpoint_pairs,
            metrics=CHECKPOINT_METRICS,
            bootstrap=int(bootstrap),
            seed=int(seed),
            extra_index=["updated_criterion_id", "stage", "budget_tokens"],
        )
    else:
        outputs["criterion_checkpoint_summary"] = pd.DataFrame()
        outputs["criterion_checkpoint_effects"] = pd.DataFrame()

    if not direct_pairs.empty and "updated_criterion_id" in direct_pairs:
        outputs["criterion_direct_summary"] = _summary(
            direct_pairs.assign(condition_id="direct"),
            metrics=DIRECT_METRICS,
            bootstrap=int(bootstrap),
            seed=int(seed),
            extra_groups=["updated_criterion_id"],
        )
        outputs["criterion_cot_vs_direct_effects"] = _direct_contrasts(
            endpoint_pairs,
            direct_pairs,
            bootstrap=int(bootstrap),
            seed=int(seed),
        )
    else:
        outputs["criterion_direct_summary"] = pd.DataFrame()
        outputs["criterion_cot_vs_direct_effects"] = pd.DataFrame()

    return outputs


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    run_dir = _resolve(workspace_root, args.run_dir)
    suite_dir = _resolve(workspace_root, args.suite_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    run_manifest = read_json(run_dir / "manifest.json")
    suite_manifest = read_json(suite_dir / "manifest.json")
    expected = int(suite_manifest.get("n_planned_traces") or 0)
    observed = int(run_manifest.get("n_switch_traces") or 0)
    expected_direct = int(suite_manifest.get("n_planned_direct_rows") or 0)
    observed_direct = int(run_manifest.get("n_direct_rows") or 0)
    if (
        (expected and expected != observed)
        or (expected_direct and expected_direct != observed_direct)
    ) and not bool(args.allow_incomplete):
        raise ValueError(
            "Incomplete SummEval structured-CoT artifact: "
            f"traces={observed}/{expected}, "
            f"direct={observed_direct}/{expected_direct}. "
            "Use --allow-incomplete only for diagnostics."
        )

    outputs = analyze(
        run_dir=run_dir,
        suite_dir=suite_dir,
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
            "stage": "summeval-structured-cot-analysis",
            "run_dir": str(run_dir),
            "suite_dir": str(suite_dir),
            "out_dir": str(out_dir),
            "endpoint_budget": int(args.endpoint_budget),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "expected_traces": expected,
            "observed_traces": observed,
            "expected_direct_rows": expected_direct,
            "observed_direct_rows": observed_direct,
            "allow_incomplete": bool(args.allow_incomplete),
            "outputs": paths,
        },
    )
    print(f"out_dir={out_dir}")
    print(outputs["endpoint_summary"].to_string(index=False))
    print(outputs["endpoint_effects"].to_string(index=False))


if __name__ == "__main__":
    main()
