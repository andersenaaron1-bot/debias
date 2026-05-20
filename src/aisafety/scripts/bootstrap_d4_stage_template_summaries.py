"""Bootstrap D4 stage-by-template interaction summaries in artifact folders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json
from aisafety.scripts.bootstrap_d4_mean_effects import _bootstrap_mean


INTERACTION_FILENAME = "template_stage_interaction_pair_deltas.csv"
BOOTSTRAP_FILENAME = "bootstrap_stage_template_interactions.csv"

SCHEMAS = (
    {
        "kind": "hllm",
        "value_col": "stage_template_interaction_llm_margin",
        "unit_col": "pair_id",
        "metric_name": "hllm_stage_template_interaction",
    },
    {
        "kind": "surface_bt",
        "value_col": "stage_template_interaction_cue_plus_margin",
        "unit_col": "counterfactual_id",
        "metric_name": "surface_stage_template_interaction",
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Root to recursively scan, usually $ARTROOT/artifacts/mechanistic.",
    )
    parser.add_argument(
        "--include-name",
        default="qwen,tulu,llama",
        help="Comma-separated case-insensitive substrings; only matching parent dirs are bootstrapped.",
    )
    parser.add_argument(
        "--exclude-name",
        default="",
        help="Comma-separated case-insensitive substrings; matching parent dirs are skipped.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Debug cap. Zero means no cap.",
    )
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _parts(raw: str) -> list[str]:
    return [part.strip().lower() for part in str(raw or "").split(",") if part.strip()]


def _name_matches(path: Path, *, include: list[str], exclude: list[str]) -> bool:
    name = str(path.parent.name).lower()
    if include and not any(part in name for part in include):
        return False
    if exclude and any(part in name for part in exclude):
        return False
    return True


def _detect_schema(df: pd.DataFrame) -> dict[str, str] | None:
    cols = set(df.columns)
    for schema in SCHEMAS:
        if schema["value_col"] in cols and schema["unit_col"] in cols:
            return schema
    return None


def _read_interaction_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _bootstrap_groups(
    df: pd.DataFrame,
    *,
    schema: dict[str, str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["stage_contrast", "template_contrast"]
    missing = [col for col in [*group_cols, schema["value_col"], schema["unit_col"]] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for (stage_contrast, template_contrast), group in df.groupby(group_cols, dropna=False, sort=True):
        row: dict[str, Any] = {
            "metric_name": schema["metric_name"],
            "stage_contrast": stage_contrast,
            "template_contrast": template_contrast,
            "value_col": schema["value_col"],
            "unit_col": schema["unit_col"],
            "bootstrap_samples": int(n_bootstrap),
        }
        row.update(
            _bootstrap_mean(
                group,
                value_col=schema["value_col"],
                unit_col=schema["unit_col"],
                n_bootstrap=int(n_bootstrap),
                rng=rng,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    artifact_root = _resolve(workspace_root, args.artifact_root)
    include = _parts(args.include_name)
    exclude = _parts(args.exclude_name)
    rng = np.random.default_rng(int(args.seed))

    files = [
        path
        for path in sorted(artifact_root.rglob(INTERACTION_FILENAME))
        if _name_matches(path, include=include, exclude=exclude)
    ]
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]

    rows: list[dict[str, Any]] = []
    for path in files:
        out_csv = path.parent / BOOTSTRAP_FILENAME
        status = "pending"
        detail = ""
        schema_kind = ""
        n_rows = 0
        n_bootstrap_rows = 0
        if out_csv.exists() and not bool(args.overwrite):
            status = "skipped_existing"
        else:
            df = _read_interaction_csv(path)
            if df is None:
                status = "skipped_empty_file"
            elif df.empty:
                status = "skipped_no_rows"
            else:
                schema = _detect_schema(df)
                if schema is None:
                    status = "skipped_unknown_schema"
                    detail = ",".join(map(str, df.columns))
                else:
                    schema_kind = str(schema["kind"])
                    n_rows = int(len(df))
                    if bool(args.dry_run):
                        status = "dry_run"
                    else:
                        boot_df = _bootstrap_groups(
                            df,
                            schema=schema,
                            n_bootstrap=int(args.bootstrap),
                            rng=rng,
                        )
                        if boot_df.empty:
                            status = "skipped_no_bootstrap_rows"
                        else:
                            boot_df.to_csv(out_csv, index=False)
                            write_json(
                                out_csv.with_suffix(".json"),
                                {
                                    "stage": "D4-stage-template-summary-bootstrap",
                                    "input": str(path),
                                    "out_csv": str(out_csv),
                                    "schema_kind": schema_kind,
                                    "bootstrap": int(args.bootstrap),
                                    "seed": int(args.seed),
                                    "n_rows": n_rows,
                                    "n_output_rows": int(len(boot_df)),
                                },
                            )
                            n_bootstrap_rows = int(len(boot_df))
                            status = "wrote"
        rows.append(
            {
                "status": status,
                "schema_kind": schema_kind,
                "n_rows": n_rows,
                "n_bootstrap_rows": n_bootstrap_rows,
                "input": str(path),
                "out_csv": str(out_csv),
                "detail": detail,
            }
        )
        print(f"{status}: {path}")

    summary = pd.DataFrame(rows)
    print("\n=== Bootstrap Scan Summary ===")
    if summary.empty:
        print("(no matching files)")
    else:
        counts = summary["status"].value_counts().sort_index()
        print(counts.to_string())


if __name__ == "__main__":
    main()
