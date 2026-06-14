"""Print activation and patching results for criterion confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--digits", type=int, default=3)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _print(title: str, frame: pd.DataFrame, *, digits: int) -> None:
    print(f"\n=== {title} ===")
    if frame.empty:
        print("NO ROWS")
    else:
        print(frame.round(int(digits)).to_string(index=False))


def read_results(
    analysis_dir: Path,
    patch_dir: Path,
    *,
    digits: int,
) -> None:
    points = _read(analysis_dir, "point_pair_bootstrap.csv")
    if not points.empty:
        points = points[
            points["point_name"].isin(
                {
                    "phase1_readout_128",
                    "phase2_readout_0",
                    "phase2_readout_32",
                    "phase2_readout_128",
                    "phase2_readout_384",
                }
            )
        ][
            [
                "probe_target",
                "hidden_layer",
                "point_name",
                "n_pairs",
                "balanced_accuracy",
                "ci95_low",
                "ci95_high",
            ]
        ]
    _print(
        "FIXED-LAYER PAIR-HELD-OUT DECODING",
        points,
        digits=digits,
    )

    conditions = _read(analysis_dir, "point_metrics_by_condition.csv")
    if not conditions.empty:
        conditions = conditions[
            conditions["point_name"].isin(
                {"phase2_readout_0", "phase2_readout_384"}
            )
            & conditions["probe_target"].isin(
                {
                    "active_criterion",
                    "criterion_target",
                    "current_choice",
                    "final_choice",
                }
            )
        ][
            [
                "condition_id",
                "probe_target",
                "hidden_layer",
                "point_name",
                "n_pairs",
                "balanced_accuracy",
                "macro_roc_auc",
            ]
        ]
    _print("DECODING BY CONDITION", conditions, digits=digits)

    differences = _read(analysis_dir, "difference_metrics.csv")
    if not differences.empty:
        differences = differences[
            [
                "difference_type",
                "probe_target",
                "hidden_layer",
                "point_name",
                "n_pairs",
                "balanced_accuracy",
                "macro_roc_auc",
            ]
        ]
    _print(
        "MATCHED RESIDUAL-DIFFERENCE DECODING",
        differences,
        digits=digits,
    )

    norms = _read(analysis_dir, "difference_norm_summary.csv")
    if not norms.empty:
        frozen = {
            "criterion_update": 20,
            "evidence_operationalization": 32,
            "explicit_target": 32,
        }
        norms = norms[
            norms.apply(
                lambda row: int(row["hidden_layer"])
                == frozen.get(str(row["difference_type"]), -1),
                axis=1,
            )
        ]
    _print("MATCHED UPDATE NORMS", norms, digits=digits)

    summary = _read(patch_dir, "patch_summary.csv")
    _print("PATCHED DECISION READOUTS", summary, digits=digits)

    effects = _read(patch_dir, "patch_effects.csv")
    if not effects.empty:
        effects = effects[
            effects["metric"].isin(
                {
                    "target_selected",
                    "target_probability",
                    "target_logit_margin",
                }
            )
            & (
                effects["left_setting"].eq("matched_delta")
                | effects["right_setting"].eq("baseline")
            )
        ]
    _print(
        "PAIR-BOOTSTRAP PATCH EFFECTS",
        effects,
        digits=digits,
    )


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    read_results(
        _resolve(workspace_root, args.analysis_dir),
        _resolve(workspace_root, args.patch_dir),
        digits=int(args.digits),
    )


if __name__ == "__main__":
    main()
