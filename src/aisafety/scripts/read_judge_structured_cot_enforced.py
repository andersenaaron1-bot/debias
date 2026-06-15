"""Print claim-critical results from the enforced-structure follow-up."""

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
    "analysis_budget_saturation_rate",
}
PRIMARY_CONTRASTS = {
    "long_prompt_effect",
    "generic_staging_effect",
    "criterion_staging_rescue",
    "enforcement_increment",
    "criterion_staging_specificity",
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
    summary = _read(root, "condition_summary.csv")
    if not summary.empty:
        summary = summary[
            summary["transition_type"].eq("all")
            & summary["metric"].isin(PRIMARY_METRICS)
        ]
    _print("ENFORCED-STRUCTURE OUTCOMES", summary, digits=digits)

    effects = _read(root, "paired_effects.csv")
    if not effects.empty:
        effects = effects[
            effects["transition_type"].eq("all")
            & effects["metric"].isin(PRIMARY_METRICS)
            & effects["contrast"].isin(PRIMARY_CONTRASTS)
        ]
    _print("PAIR-BOOTSTRAP EFFECTS", effects, digits=digits)

    stages = _read(root, "stage_summary.csv")
    _print("STAGE TOKEN USE AND SATURATION", stages, digits=digits)

    audit = _read(root, "audit_confirmed_effects.csv")
    if not audit.empty:
        audit = audit[
            audit["transition_type"].eq("all")
            & audit["metric"].isin(PRIMARY_METRICS)
            & audit["contrast"].isin(PRIMARY_CONTRASTS)
        ]
    _print("AUDIT-CONFIRMED SENSITIVITY", audit, digits=digits)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    read_results(
        _resolve(workspace_root, args.analysis_dir),
        digits=int(args.digits),
    )


if __name__ == "__main__":
    main()
