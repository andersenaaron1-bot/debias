"""Summarize comparable judge-reasoning analyses across models and conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json


DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "judge_reasoning_suite_summary_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--analysis",
        action="append",
        required=True,
        help="Repeat label=analysis_dir for each model or condition.",
    )
    parser.add_argument(
        "--intervention",
        action="append",
        default=[],
        help="Optional repeat label=intervention_dir.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, value: str | Path) -> Path:
    resolved = resolve_path(workspace_root, value)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {value}")
    return resolved


def _labeled_paths(values: list[str], workspace_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in str(value):
            raise ValueError(f"Expected label=path, got {value!r}")
        label, raw_path = str(value).split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in {value!r}")
        out.append((label, _resolve(workspace_root, raw_path.strip())))
    return out


def _read_with_label(path: Path, filename: str, label: str) -> pd.DataFrame:
    source = Path(path) / filename
    if not source.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(source)
    frame.insert(0, "run_label", label)
    return frame


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if not frame.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    analyses = _labeled_paths(list(args.analysis), workspace_root)
    interventions = _labeled_paths(list(args.intervention), workspace_root)
    out_dir = _resolve(workspace_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commitment = _combine(
        [_read_with_label(path, "commitment_summary.csv", label) for label, path in analyses]
    )
    probes = _combine(
        [_read_with_label(path, "probe_metrics.csv", label) for label, path in analyses]
    )
    convergence = _combine(
        [_read_with_label(path, "branch_convergence.csv", label) for label, path in analyses]
    )
    revisions = _combine(
        [_read_with_label(path, "margin_revisions.csv", label) for label, path in analyses]
    )
    mediation = _combine(
        [
            _read_with_label(path, "trajectory_mediation_screen.csv", label)
            for label, path in analyses
        ]
    )
    dynamics = _combine(
        [
            _read_with_label(path, "decision_dynamics.csv", label)
            for label, path in analyses
        ]
    )
    dynamics_summary = _combine(
        [
            _read_with_label(path, "decision_dynamics_summary.csv", label)
            for label, path in analyses
        ]
    )
    geometry = _combine(
        [
            _read_with_label(path, "trajectory_geometry.csv", label)
            for label, path in analyses
        ]
    )
    intervention = _combine(
        [
            _read_with_label(path, "intervention_summary.csv", label)
            for label, path in interventions
        ]
    )

    outputs: dict[str, pd.DataFrame] = {
        "commitment_summary": commitment,
        "probe_metrics": probes,
        "branch_convergence": convergence,
        "margin_revisions": revisions,
        "trajectory_mediation_screen": mediation,
        "decision_dynamics": dynamics,
        "decision_dynamics_summary": dynamics_summary,
        "trajectory_geometry": geometry,
        "intervention_summary": intervention,
    }
    if not probes.empty:
        ok = probes[probes["status"] == "ok"].copy()
        outputs["best_probe_by_target"] = (
            ok.sort_values("roc_auc", ascending=False)
            .groupby(
                ["run_label", "reasoning_mode", "analysis_group", "probe_target"],
                sort=True,
                as_index=False,
            )
            .first()
        )
    if not revisions.empty:
        outputs["margin_revision_summary"] = (
            revisions.groupby(
                ["run_label", "reasoning_mode", "comparison_dimension"],
                sort=True,
                dropna=False,
            )
            .agg(
                n_traces=("trace_id", "count"),
                mean_sign_revisions=("margin_sign_revisions", "mean"),
                final_margin_choice_agreement=(
                    "final_margin_agrees_with_choice",
                    "mean",
                ),
            )
            .reset_index()
        )
    if not geometry.empty:
        outputs["trajectory_geometry_summary"] = (
            geometry.groupby(
                [
                    "run_label",
                    "reasoning_mode",
                    "source_dataset",
                    "comparison_dimension",
                    "hidden_layer",
                ],
                sort=True,
                dropna=False,
            )
            .agg(
                n_traces=("trace_id", "count"),
                mean_path_length=("path_length", "mean"),
                mean_endpoint_displacement=("endpoint_displacement", "mean"),
                mean_path_efficiency=("path_efficiency", "mean"),
                mean_step_norm_per_token=("mean_step_norm_per_token", "mean"),
            )
            .reset_index()
        )
    output_paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = str(path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-reasoning-suite-summary",
            "out_dir": str(out_dir),
            "analyses": [{"label": label, "path": str(path)} for label, path in analyses],
            "interventions": [
                {"label": label, "path": str(path)} for label, path in interventions
            ],
            "outputs": output_paths,
        },
    )
    print(f"out_dir={out_dir}")
    print(f"n_analyses={len(analyses)}")
    print(f"n_interventions={len(interventions)}")


if __name__ == "__main__":
    main()
