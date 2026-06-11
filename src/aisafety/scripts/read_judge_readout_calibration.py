"""Print compact criterion-switch and factual readout calibration results."""

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
    parser.add_argument("--criterion-analysis-dir", type=Path, required=True)
    parser.add_argument("--factual-analysis-dir", type=Path, required=True)
    parser.add_argument("--digits", type=int, default=3)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _print(title: str, frame: pd.DataFrame, *, digits: int) -> None:
    print(f"\n=== {title} ===")
    if frame.empty:
        print("NO ROWS")
        return
    print(frame.round(int(digits)).to_string(index=False))


def _diagonal_bootstrap(root: Path) -> pd.DataFrame:
    frame = _read(root, "point_pair_bootstrap.csv")
    columns = [
        "probe_target",
        "point_name",
        "n_pairs",
        "balanced_accuracy",
        "ci95_low",
        "ci95_high",
    ]
    return frame[[column for column in columns if column in frame.columns]]


def _endpoint_transfer(root: Path) -> pd.DataFrame:
    frame = _read(root, "cross_time_metrics.csv")
    if frame.empty:
        return frame
    endpoint = frame.groupby("probe_target")["train_point_index"].transform(
        "max"
    )
    frame = frame[frame["train_point_index"] == endpoint].copy()
    columns = [
        "probe_target",
        "hidden_layer",
        "train_point_name",
        "test_point_name",
        "n_pairs",
        "balanced_accuracy",
        "macro_roc_auc",
    ]
    return frame[[column for column in columns if column in frame.columns]]


def _dataset_metrics(root: Path) -> pd.DataFrame:
    frame = _read(root, "point_metrics_by_group.csv")
    if frame.empty or "group_type" not in frame:
        return pd.DataFrame()
    frame = frame[frame["group_type"] == "source_dataset"].copy()
    preferred_targets = {
        "criterion_target",
        "current_choice",
        "final_choice",
        "presentation_order",
    }
    if "probe_target" in frame:
        frame = frame[frame["probe_target"].isin(preferred_targets)]
    columns = [
        "group_value",
        "probe_target",
        "point_name",
        "n_pairs",
        "balanced_accuracy",
        "accuracy",
    ]
    return frame[[column for column in columns if column in frame.columns]]


def read_results(
    *,
    criterion_dir: Path,
    factual_dir: Path,
    digits: int,
) -> None:
    _print(
        "CRITERION SWITCH: PAIR-BOOTSTRAP POINT DECODING",
        _diagonal_bootstrap(criterion_dir),
        digits=digits,
    )
    _print(
        "CRITERION SWITCH: ENDPOINT DECODER TRANSFER",
        _endpoint_transfer(criterion_dir),
        digits=digits,
    )
    difference = criterion_dir / "switch_minus_reminder_point_pair_bootstrap.csv"
    _print(
        "SWITCH MINUS REMINDER: PAIR-BOOTSTRAP DIFFERENCE DECODING",
        pd.read_csv(difference) if difference.is_file() else pd.DataFrame(),
        digits=digits,
    )
    _print(
        "FACTUAL BASELINE: PAIR-BOOTSTRAP POINT DECODING",
        _diagonal_bootstrap(factual_dir),
        digits=digits,
    )
    _print(
        "FACTUAL BASELINE: ENDPOINT DECODER TRANSFER",
        _endpoint_transfer(factual_dir),
        digits=digits,
    )
    _print(
        "FACTUAL BASELINE: DATASET-LEVEL POINT METRICS",
        _dataset_metrics(factual_dir),
        digits=digits,
    )


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    read_results(
        criterion_dir=_resolve(
            workspace_root,
            args.criterion_analysis_dir,
        ),
        factual_dir=_resolve(
            workspace_root,
            args.factual_analysis_dir,
        ),
        digits=int(args.digits),
    )


if __name__ == "__main__":
    main()
