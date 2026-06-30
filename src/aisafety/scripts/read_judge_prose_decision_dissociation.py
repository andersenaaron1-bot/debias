"""Print prose/decision dissociation and direction-patching results."""

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
    parser.add_argument("--patch-dir", type=Path)
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=80)
    return parser.parse_args()


def _resolve(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _print_table(title: str, table: pd.DataFrame, *, digits: int, max_rows: int) -> None:
    print(f"\n=== {title} ===")
    if table.empty:
        print("NO ROWS")
        return
    display = table.head(int(max_rows)).copy()
    print(display.round(int(digits)).to_string(index=False))


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    analysis_dir = _resolve(workspace_root, args.analysis_dir)
    patch_dir = _resolve(workspace_root, args.patch_dir)
    assert analysis_dir is not None

    _print_table(
        "PROSE SUMMARY",
        _read_csv(analysis_dir / "prose_summary.csv"),
        digits=int(args.round),
        max_rows=int(args.max_rows),
    )
    _print_table(
        "SELECTED PROBES",
        _read_csv(analysis_dir / "selected_probes.csv"),
        digits=int(args.round),
        max_rows=int(args.max_rows),
    )
    _print_table(
        "SUBSPACE ALIGNMENT",
        _read_csv(analysis_dir / "subspace_alignment.csv"),
        digits=int(args.round),
        max_rows=int(args.max_rows),
    )
    _print_table(
        "CONTROLLED PROJECTION DIAGNOSTICS",
        _read_csv(analysis_dir / "controlled_projection_diagnostics.csv"),
        digits=int(args.round),
        max_rows=int(args.max_rows),
    )
    if patch_dir is not None:
        _print_table(
            "DIRECTION PATCH SUMMARY",
            _read_csv(patch_dir / "patch_summary.csv"),
            digits=int(args.round),
            max_rows=int(args.max_rows),
        )
        _print_table(
            "DIRECTION PATCH EFFECTS",
            _read_csv(patch_dir / "patch_effects.csv"),
            digits=int(args.round),
            max_rows=int(args.max_rows),
        )


if __name__ == "__main__":
    main()
