"""Bootstrap confidence intervals for paired D4 mean-effect CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--value-col", required=True)
    parser.add_argument("--unit-col", required=True)
    parser.add_argument(
        "--group-cols",
        default="",
        help="Comma-separated grouping columns. Leave empty for one overall row.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--metric-name", default="")
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _group_cols(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _as_key_tuple(key: Any, n_cols: int) -> tuple[Any, ...]:
    if n_cols == 0:
        return ()
    if isinstance(key, tuple):
        return key
    return (key,)


def _bootstrap_mean(
    df: pd.DataFrame,
    *,
    value_col: str,
    unit_col: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = pd.to_numeric(df[value_col], errors="coerce")
    valid = df.loc[values.notna(), [unit_col]].copy()
    valid[value_col] = values.loc[values.notna()].astype(float)
    if valid.empty:
        return {
            "n_rows": 0,
            "n_units": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    unit_values = [
        group[value_col].to_numpy(dtype=float)
        for _, group in valid.groupby(unit_col, sort=True)
    ]
    observed = valid[value_col].to_numpy(dtype=float)
    boot_means = np.empty(int(n_bootstrap), dtype=float)
    n_units = len(unit_values)
    for idx in range(int(n_bootstrap)):
        sampled = rng.integers(0, n_units, size=n_units)
        boot_values = np.concatenate([unit_values[int(unit_idx)] for unit_idx in sampled])
        boot_means[idx] = float(np.mean(boot_values))

    return {
        "n_rows": int(len(valid)),
        "n_units": int(n_units),
        "mean": float(np.mean(observed)),
        "median": float(np.median(observed)),
        "ci95_low": float(np.quantile(boot_means, 0.025)),
        "ci95_high": float(np.quantile(boot_means, 0.975)),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    input_path = _resolve(workspace_root, args.input)
    out_csv = _resolve(workspace_root, args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input CSV has no columns: {input_path}") from exc
    group_cols = _group_cols(str(args.group_cols))
    required = [str(args.value_col), str(args.unit_col), *group_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    if group_cols:
        grouped = df.groupby(group_cols, sort=True, dropna=False)
    else:
        grouped = [((), df)]

    for key, group in grouped:
        key_tuple = _as_key_tuple(key, len(group_cols))
        row: dict[str, Any] = {
            "metric_name": str(args.metric_name or args.value_col),
            "value_col": str(args.value_col),
            "unit_col": str(args.unit_col),
            "bootstrap_samples": int(args.bootstrap),
        }
        for col, value in zip(group_cols, key_tuple):
            row[col] = value
        row.update(
            _bootstrap_mean(
                group,
                value_col=str(args.value_col),
                unit_col=str(args.unit_col),
                n_bootstrap=int(args.bootstrap),
                rng=rng,
            )
        )
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    write_json(
        out_csv.with_suffix(".json"),
        {
            "stage": "D4-mean-effect-bootstrap",
            "input": str(input_path),
            "out_csv": str(out_csv),
            "value_col": str(args.value_col),
            "unit_col": str(args.unit_col),
            "group_cols": group_cols,
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "n_rows": int(len(df)),
            "n_output_rows": int(len(out_df)),
        },
    )
    print(f"out_csv={out_csv}")


if __name__ == "__main__":
    main()
