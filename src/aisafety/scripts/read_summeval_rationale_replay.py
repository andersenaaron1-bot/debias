"""Print the SummEval rationale-transplant replay readout."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path


PRIMARY_METRICS = {
    "recipient_target_adoption",
    "donor_target_adoption",
    "recipient_probability",
    "donor_probability",
    "recipient_minus_donor_logit_margin",
    "order_consistent_recipient_target_adoption",
    "order_consistent_donor_target_adoption",
}
PRIMARY_CONTRASTS = {
    "opposite_free_vs_baseline",
    "opposite_scaffold_vs_baseline",
    "opposite_score_vs_baseline",
    "opposite_score_vs_opposite_free",
    "opposite_score_vs_opposite_scaffold",
    "evidence_only_initial_vs_baseline",
    "same_free_vs_baseline",
    "same_scaffold_vs_baseline",
    "same_score_vs_baseline",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--replay-dir", type=Path, required=True)
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


def read_results(root: Path, *, digits: int) -> None:
    summary = _read(root, "condition_summary.csv")
    if not summary.empty:
        summary = summary[
            summary["metric"].isin(
                {
                    "recipient_target_adoption",
                    "donor_target_adoption",
                    "recipient_probability",
                    "donor_probability",
                    "order_consistent_recipient_target_adoption",
                    "order_consistent_donor_target_adoption",
                }
            )
        ]
    _print("RATIONALE REPLAY CONDITION SUMMARY", summary, digits=digits)

    effects = _read(root, "effect_summary.csv")
    if not effects.empty:
        effects = effects[
            effects["metric"].isin(PRIMARY_METRICS)
            & effects["contrast"].isin(PRIMARY_CONTRASTS)
        ]
    _print("RATIONALE REPLAY EFFECTS", effects, digits=digits)

    rows = _read(root, "replay_rows.csv")
    if not rows.empty:
        compact = (
            rows.groupby(
                [
                    "replay_mode",
                    "replay_condition",
                    "recipient_role",
                    "recipient_criterion_id",
                ],
                sort=True,
                dropna=False,
            )
            .agg(
                n_rows=("replay_id", "size"),
                n_pairs=("pair_id", "nunique"),
                recipient_target_adoption=(
                    "recipient_target_selected",
                    "mean",
                ),
                donor_target_adoption=("donor_target_selected", "mean"),
                recipient_probability=("recipient_probability", "mean"),
                donor_probability=("donor_probability", "mean"),
            )
            .reset_index()
        )
    else:
        compact = pd.DataFrame()
    _print("REPLAY BY RECIPIENT CRITERION", compact, digits=digits)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    read_results(
        _resolve(workspace_root, args.replay_dir),
        digits=int(args.digits),
    )


if __name__ == "__main__":
    main()
