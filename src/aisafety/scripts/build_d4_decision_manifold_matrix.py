"""Build a pair-level behavioral decision-manifold matrix from D4 summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json
from aisafety.scripts.summarize_d4_human_llm_stage_contrasts import _parse_contrast


DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_decision_manifold_matrix_v1"

META_COLUMNS = (
    "source_dataset",
    "subset",
    "item_type",
    "axis",
    "direction",
    "role",
    "template_label",
    "comparison_template",
    "scoring_mode",
    "model_id",
    "prompt_style",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Named summary directory as LABEL=DIR. May be passed multiple times.",
    )
    parser.add_argument(
        "--stage-contrast",
        action="append",
        default=[],
        help=(
            "Optional stage contrast NAME=LEFT-RIGHT derived from any "
            "stage_pair_summary_long.csv inputs."
        ),
    )
    parser.add_argument(
        "--unit-prefix-input",
        action="store_true",
        help="Prefix unit ids with the input label instead of merging matching pair ids across inputs.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got: {value}")
    label, raw = value.split("=", 1)
    label = label.strip()
    raw = raw.strip()
    if not label or not raw:
        raise ValueError(f"Expected nonempty LABEL=PATH, got: {value}")
    return label, Path(raw)


def _read_if_exists(root: Path, filename: str) -> pd.DataFrame | None:
    path = root / filename
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _unit_col(df: pd.DataFrame) -> str | None:
    for col in ("pair_id", "counterfactual_id"):
        if col in df.columns:
            return col
    return None


def _unit_id(raw: Any, *, input_label: str, prefix_input: bool) -> str:
    raw_id = str(raw)
    return f"{input_label}::{raw_id}" if prefix_input else raw_id


def _candidate_meta_columns(base: str) -> list[str]:
    return [
        base,
        f"{base}_left",
        f"{base}_right",
        f"{base}_left_template",
        f"{base}_right_template",
        f"{base}_left_stage",
        f"{base}_right_stage",
        f"{base}_left_template_left_stage",
        f"{base}_left_template_right_stage",
    ]


def _first_present(row: pd.Series, base: str) -> str:
    for col in _candidate_meta_columns(base):
        if col in row.index:
            value = row.get(col)
            if pd.notna(value) and str(value) != "":
                return str(value)
    return ""


def _first_present_series(df: pd.DataFrame, base: str) -> pd.Series:
    cols = [col for col in _candidate_meta_columns(base) if col in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    meta = df[cols].copy()
    meta = meta.replace("", pd.NA)
    return meta.bfill(axis=1).iloc[:, 0].fillna("").astype(str)


def _feature_name(*parts: str) -> str:
    clean = [str(part).strip() for part in parts if str(part).strip()]
    return "__".join(clean)


def _append_rows(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    *,
    input_label: str,
    prefix_input: bool,
    feature_family: str,
    value_col: str,
    feature_parts: tuple[str, ...],
    dynamic_cols: tuple[str, ...] = (),
    unit_override: str | None = None,
) -> None:
    unit = unit_override if unit_override in df.columns else _unit_col(df)
    if unit is None or value_col not in df.columns:
        return
    value = pd.to_numeric(df[value_col], errors="coerce")
    keep = value.notna()
    if not keep.any():
        return
    kept = df.loc[keep].copy()
    value = value.loc[keep].astype(float)
    raw_unit = kept[unit].astype(str)
    if prefix_input:
        unit_id = input_label + "::" + raw_unit
    else:
        unit_id = raw_unit

    prefix = _feature_name(input_label, feature_family, *feature_parts)
    if dynamic_cols:
        dynamic = kept[list(dynamic_cols)].fillna("").astype(str)
        suffix = dynamic.apply(lambda row: "__".join(part for part in row if part), axis=1)
        feature = prefix + suffix.map(lambda item: f"__{item}" if item else "")
    else:
        feature = pd.Series([prefix] * len(kept), index=kept.index)

    out = pd.DataFrame(
        {
            "unit_id": unit_id.to_numpy(),
            "raw_unit_id": raw_unit.to_numpy(),
            "unit_col": unit,
            "input_label": input_label,
            "feature_family": feature_family,
            "feature_name": feature.to_numpy(),
            "feature_value": value.to_numpy(),
            "feature_abs_value": value.abs().to_numpy(),
            "value_col": value_col,
        },
        index=kept.index,
    )
    for col in META_COLUMNS:
        out[col] = _first_present_series(kept, col).to_numpy()
    rows.extend(out.to_dict("records"))


def _add_stage_pair_rows(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    *,
    input_label: str,
    prefix_input: bool,
    stage_contrasts: list[str],
) -> None:
    if "mean_llm_margin" not in df.columns or "run_label" not in df.columns:
        return
    _append_rows(
        rows,
        df,
        input_label=input_label,
        prefix_input=prefix_input,
        feature_family="hllm_margin",
        value_col="mean_llm_margin",
        feature_parts=(),
        dynamic_cols=("run_label",),
    )
    if not stage_contrasts:
        return
    unit = _unit_col(df)
    if unit is None:
        return
    labels = set(df["run_label"].astype(str))
    for raw in stage_contrasts:
        name, left, right = _parse_contrast(raw)
        if left not in labels or right not in labels:
            continue
        left_df = df[df["run_label"].astype(str) == left].copy()
        right_df = df[df["run_label"].astype(str) == right].copy()
        keep = [unit, "mean_llm_margin", *[col for col in META_COLUMNS if col in df.columns]]
        merged = left_df[keep].merge(
            right_df[keep],
            on=unit,
            how="inner",
            suffixes=("_left", "_right"),
        )
        if merged.empty:
            continue
        merged["delta_llm_margin"] = (
            pd.to_numeric(merged["mean_llm_margin_left"], errors="coerce")
            - pd.to_numeric(merged["mean_llm_margin_right"], errors="coerce")
        )
        merged["contrast"] = name
        _append_rows(
            rows,
            merged,
            input_label=input_label,
            prefix_input=prefix_input,
            feature_family="hllm_stage_delta",
            value_col="delta_llm_margin",
            feature_parts=(),
            dynamic_cols=("contrast",),
        )


def _add_bt_pair_rows(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    *,
    input_label: str,
    prefix_input: bool,
) -> None:
    if "mean_cue_plus_margin" not in df.columns:
        return
    dynamic = tuple(col for col in ("template_label", "run_label") if col in df.columns)
    _append_rows(
        rows,
        df,
        input_label=input_label,
        prefix_input=prefix_input,
        feature_family="surface_cue_margin",
        value_col="mean_cue_plus_margin",
        feature_parts=(),
        dynamic_cols=dynamic,
        unit_override="counterfactual_id",
    )


def _add_stage_contrast_pair_rows(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    *,
    input_label: str,
    prefix_input: bool,
) -> None:
    value_col = "delta_cue_plus_margin" if "delta_cue_plus_margin" in df.columns else "delta_llm_margin"
    if value_col not in df.columns:
        return
    dynamic = tuple(col for col in ("template_label", "contrast") if col in df.columns)
    _append_rows(
        rows,
        df,
        input_label=input_label,
        prefix_input=prefix_input,
        feature_family="stage_delta",
        value_col=value_col,
        feature_parts=(),
        dynamic_cols=dynamic,
    )


def _add_template_delta_rows(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    *,
    input_label: str,
    prefix_input: bool,
) -> None:
    if "template_delta_llm_margin" in df.columns:
        value_col = "template_delta_llm_margin"
        family = "hllm_template_delta"
    elif "template_delta_cue_plus_margin" in df.columns:
        value_col = "template_delta_cue_plus_margin"
        family = "surface_template_delta"
    else:
        return
    dynamic = tuple(col for col in ("template_contrast", "run_label") if col in df.columns)
    _append_rows(
        rows,
        df,
        input_label=input_label,
        prefix_input=prefix_input,
        feature_family=family,
        value_col=value_col,
        feature_parts=(),
        dynamic_cols=dynamic,
    )


def _add_interaction_rows(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    *,
    input_label: str,
    prefix_input: bool,
) -> None:
    if "stage_template_interaction_llm_margin" in df.columns:
        value_col = "stage_template_interaction_llm_margin"
        family = "hllm_stage_template_interaction"
    elif "stage_template_interaction_cue_plus_margin" in df.columns:
        value_col = "stage_template_interaction_cue_plus_margin"
        family = "surface_stage_template_interaction"
    else:
        return
    dynamic = tuple(col for col in ("stage_contrast", "template_contrast") if col in df.columns)
    _append_rows(
        rows,
        df,
        input_label=input_label,
        prefix_input=prefix_input,
        feature_family=family,
        value_col=value_col,
        feature_parts=(),
        dynamic_cols=dynamic,
    )


def build_matrix(
    inputs: list[tuple[str, Path]],
    *,
    stage_contrasts: list[str],
    prefix_input: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_manifest: dict[str, dict[str, Any]] = {}
    for label, path in inputs:
        manifest_entry: dict[str, Any] = {"path": str(path), "files_loaded": []}
        stage_df = _read_if_exists(path, "stage_pair_summary_long.csv")
        if stage_df is not None:
            _add_stage_pair_rows(
                rows,
                stage_df,
                input_label=label,
                prefix_input=prefix_input,
                stage_contrasts=stage_contrasts,
            )
            manifest_entry["files_loaded"].append("stage_pair_summary_long.csv")

        bt_df = _read_if_exists(path, "bt_pair_summary_long.csv")
        if bt_df is not None:
            _add_bt_pair_rows(rows, bt_df, input_label=label, prefix_input=prefix_input)
            manifest_entry["files_loaded"].append("bt_pair_summary_long.csv")

        stage_delta_df = _read_if_exists(path, "stage_contrast_pair_deltas.csv")
        if stage_delta_df is not None:
            _add_stage_contrast_pair_rows(rows, stage_delta_df, input_label=label, prefix_input=prefix_input)
            manifest_entry["files_loaded"].append("stage_contrast_pair_deltas.csv")

        template_df = _read_if_exists(path, "template_sensitivity_pair_deltas.csv")
        if template_df is not None:
            _add_template_delta_rows(rows, template_df, input_label=label, prefix_input=prefix_input)
            manifest_entry["files_loaded"].append("template_sensitivity_pair_deltas.csv")

        interaction_df = _read_if_exists(path, "template_stage_interaction_pair_deltas.csv")
        if interaction_df is not None:
            _add_interaction_rows(rows, interaction_df, input_label=label, prefix_input=prefix_input)
            manifest_entry["files_loaded"].append("template_stage_interaction_pair_deltas.csv")
        input_manifest[label] = manifest_entry

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError("No pair-level decision-manifold rows were produced from the requested inputs.")

    metadata_cols = ["unit_id"]
    for col in META_COLUMNS:
        if col in long_df.columns:
            metadata_cols.append(col)
    metadata = (
        long_df[metadata_cols]
        .replace("", pd.NA)
        .groupby("unit_id", sort=True, dropna=False)
        .first()
        .reset_index()
        .fillna("")
    )
    values = long_df.pivot_table(
        index="unit_id",
        columns="feature_name",
        values="feature_value",
        aggfunc="mean",
    ).reset_index()
    wide_df = metadata.merge(values, on="unit_id", how="left")
    feature_summary = (
        long_df.groupby(["feature_family", "feature_name", "value_col"], sort=True)
        .agg(
            n_units=("unit_id", "nunique"),
            n_rows=("unit_id", "size"),
            mean=("feature_value", "mean"),
            std=("feature_value", "std"),
            median=("feature_value", "median"),
            mean_abs=("feature_abs_value", "mean"),
        )
        .reset_index()
    )
    manifest = {
        "stage": "D4-decision-manifold-matrix-build",
        "inputs": input_manifest,
        "stage_contrasts": list(stage_contrasts),
        "prefix_input": bool(prefix_input),
        "n_long_rows": int(len(long_df)),
        "n_units": int(long_df["unit_id"].nunique()),
        "n_features": int(long_df["feature_name"].nunique()),
    }
    return long_df, wide_df, feature_summary, manifest


def main() -> None:
    args = _parse_args()
    if not args.input:
        raise ValueError("Pass at least one --input LABEL=DIR.")
    workspace_root = Path(args.workspace_root).resolve()
    inputs = []
    for raw in args.input:
        label, raw_path = _parse_label_path(str(raw))
        inputs.append((label, _resolve(workspace_root, raw_path)))
    out_dir = _resolve(workspace_root, args.out_dir)

    long_df, wide_df, feature_summary, manifest = build_matrix(
        inputs,
        stage_contrasts=[str(item) for item in args.stage_contrast],
        prefix_input=bool(args.unit_prefix_input),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "manifold_long.csv"
    wide_path = out_dir / "manifold_wide.csv"
    feature_path = out_dir / "feature_summary.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    feature_summary.to_csv(feature_path, index=False)
    manifest["outputs"] = {
        "manifold_long_csv": str(long_path),
        "manifold_wide_csv": str(wide_path),
        "feature_summary_csv": str(feature_path),
        "summary_json": str(out_dir / "summary.json"),
    }
    write_json(out_dir / "summary.json", manifest)

    print(f"out_dir={out_dir}")
    print(f"n_units={manifest['n_units']}")
    print(f"n_features={manifest['n_features']}")
    print(f"long={long_path}")
    print(f"wide={wide_path}")
    print(f"feature_summary={feature_path}")


if __name__ == "__main__":
    main()
