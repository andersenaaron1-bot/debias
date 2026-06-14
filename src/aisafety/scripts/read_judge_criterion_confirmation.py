"""Print the claim-critical criterion confirmation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path


PRIMARY_METRICS = {
    "forced_target_adoption",
    "order_consistent_target_adoption",
    "order_consistent_rate",
    "natural_valid_rate",
    "phase2_budget_saturation_rate",
}
PRIMARY_CONTRASTS = {
    "early_operationalization_rescue",
    "late_operationalization_rescue",
    "criterion_commitment_penalty",
    "evidence_commitment_penalty",
    "timing_by_evidence_interaction",
    "explicit_target_vs_late_criterion",
    "explicit_target_vs_late_evidence",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--analysis-dir", type=Path, required=True)
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
    condition = _read(root, "condition_summary.csv")
    if not condition.empty:
        condition = condition[
            condition["metric"].isin(PRIMARY_METRICS)
            & condition["transition_type"].eq("all")
        ].copy()
    effects = _read(root, "paired_effects.csv")
    if not effects.empty:
        effects = effects[
            effects["contrast"].isin(PRIMARY_CONTRASTS)
            & effects["metric"].isin(PRIMARY_METRICS)
            & effects["transition_type"].eq("all")
        ].copy()
    _print("ENDPOINT CONDITION SUMMARY", condition, digits=digits)
    _print("PAIR-BOOTSTRAP CONFIRMATORY EFFECTS", effects, digits=digits)

    audit_pairs = _read(root, "audit_pair_rows.csv")
    if not audit_pairs.empty:
        summary = pd.DataFrame(
            [
                {
                    "n_pairs": int(audit_pairs["pair_id"].nunique()),
                    "n_criterion_checks": int(len(audit_pairs)),
                    "complete_rate": audit_pairs["complete"].mean(),
                    "order_consistent_rate": audit_pairs[
                        "order_consistent"
                    ].mean(),
                    "proxy_agreement_rate": audit_pairs[
                        "proxy_agreement"
                    ].mean(),
                    "audit_confirmed_rate": audit_pairs[
                        "audit_confirmed"
                    ].mean(),
                }
            ]
        )
        _print("HUMAN AUDIT", summary, digits=digits)
        confirmed = _read(root, "audit_confirmed_paired_effects.csv")
        if not confirmed.empty:
            confirmed = confirmed[
                confirmed["contrast"].isin(PRIMARY_CONTRASTS)
                & confirmed["metric"].isin(PRIMARY_METRICS)
                & confirmed["transition_type"].eq("all")
            ]
        _print(
            "AUDIT-CONFIRMED SENSITIVITY EFFECTS",
            confirmed,
            digits=digits,
        )


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    read_results(
        _resolve(workspace_root, args.analysis_dir),
        digits=int(args.digits),
    )


if __name__ == "__main__":
    main()
