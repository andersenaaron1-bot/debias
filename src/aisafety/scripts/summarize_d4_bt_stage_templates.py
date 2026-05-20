"""Summarize stage and template effects on D4 surface-cue BT contrasts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json
from aisafety.scripts.summarize_d4_human_llm_stage_contrasts import _parse_contrast


DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_bt_stage_template_summary_v1"
DEFAULT_STAGE_CONTRASTS = (
    "tulu3_sft_minus_base=tulu3_sft-llama31_base",
    "tulu3_dpo_minus_sft=tulu3_dpo-tulu3_sft",
    "tulu3_final_minus_dpo=tulu3_final-tulu3_dpo",
    "tulu3_final_minus_base=tulu3_final-llama31_base",
    "llama31_instruct_minus_base=llama31_instruct-llama31_base",
    "llama31_instruct_minus_tulu3_dpo=llama31_instruct-tulu3_dpo",
    "tulu3_sft_like_minus_base_like=tulu3_sft_like-llama31_base_like",
    "tulu3_dpo_like_minus_sft_like=tulu3_dpo_like-tulu3_sft_like",
)
DEFAULT_TEMPLATE_CONTRASTS = (
    "standard_minus_minimal=standard-minimal",
    "rubric_quality_minus_minimal=rubric_quality-minimal",
    "substance_only_minus_minimal=substance_only-minimal",
    "substance_only_minus_rubric_quality=substance_only-rubric_quality",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec as TEMPLATE:LABEL=DIR_OR_CSV. May be passed multiple times.",
    )
    parser.add_argument("--stage-contrast", action="append", default=[])
    parser.add_argument("--template-contrast", action="append", default=[])
    parser.add_argument("--source-top-k", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _parse_run(value: str) -> tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise ValueError(f"Expected TEMPLATE:LABEL=PATH for --run, got: {value}")
    left, raw_path = value.split("=", 1)
    template, label = left.split(":", 1)
    template = template.strip()
    label = label.strip()
    raw_path = raw_path.strip()
    if not template or not label or not raw_path:
        raise ValueError(f"Expected nonempty TEMPLATE:LABEL=PATH for --run, got: {value}")
    return template, label, Path(raw_path)


def _score_csv(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "bt_stage_scores.csv"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find bt_stage_scores.csv in {path}")


def _load_run(template_label: str, run_label: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(_score_csv(path))
    if "cue_plus_margin" not in df.columns:
        raise ValueError(f"Run {template_label}:{run_label} has no cue_plus_margin column: {path}")
    df["template_label"] = str(template_label)
    df["run_label"] = str(run_label)
    for col in (
        "counterfactual_id",
        "pair_id",
        "source_dataset",
        "subset",
        "item_type",
        "role",
        "axis",
        "direction",
        "transform_id",
        "presentation_order",
        "stage_label",
        "scoring_mode",
        "model_id",
        "prompt_style",
        "comparison_template",
    ):
        if col not in df.columns:
            df[col] = ""
    if "cue_plus_prob" not in df.columns:
        df["cue_plus_prob"] = pd.NA
    if "cue_plus_preferred" not in df.columns:
        df["cue_plus_preferred"] = pd.to_numeric(df["cue_plus_margin"], errors="coerce") > 0.0
    df["counterfactual_id"] = df["counterfactual_id"].astype(str)
    df["cue_plus_margin"] = pd.to_numeric(df["cue_plus_margin"], errors="coerce")
    df["cue_plus_preferred"] = df["cue_plus_preferred"].astype(bool)
    return df


def _pair_level(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["template_label", "run_label", "counterfactual_id"]
    meta_cols = [
        "pair_id",
        "source_dataset",
        "subset",
        "item_type",
        "role",
        "axis",
        "direction",
        "transform_id",
        "stage_label",
        "scoring_mode",
        "model_id",
        "prompt_style",
        "comparison_template",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_cols, sort=True):
        template_label, run_label, counterfactual_id = key
        margin = pd.to_numeric(group["cue_plus_margin"], errors="coerce")
        prob = pd.to_numeric(group["cue_plus_prob"], errors="coerce") if "cue_plus_prob" in group.columns else pd.Series(dtype=float)
        row: dict[str, Any] = {
            "template_label": str(template_label),
            "run_label": str(run_label),
            "counterfactual_id": str(counterfactual_id),
            "n_order_rows": int(len(group)),
            "mean_cue_plus_margin": float(margin.mean()),
            "median_cue_plus_margin": float(margin.median()),
            "cue_plus_preferred": bool(float(margin.mean()) > 0.0),
            "mean_cue_plus_prob": None if prob.empty or prob.isna().all() else float(prob.mean()),
        }
        first = group.iloc[0]
        for col in meta_cols:
            if col in group.columns:
                row[col] = str(first.get(col) or "")
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_row(df: pd.DataFrame, *, template_label: str, run_label: str, group_type: str, group_value: str) -> dict[str, Any]:
    margin = pd.to_numeric(df["mean_cue_plus_margin"], errors="coerce")
    prob = pd.to_numeric(df["mean_cue_plus_prob"], errors="coerce") if "mean_cue_plus_prob" in df.columns else pd.Series(dtype=float)
    return {
        "template_label": template_label,
        "run_label": run_label,
        "group_type": group_type,
        "group_value": group_value,
        "n_counterfactuals": int(len(df)),
        "mean_cue_plus_margin": float(margin.mean()),
        "median_cue_plus_margin": float(margin.median()),
        "mean_abs_cue_plus_margin": float(margin.abs().mean()),
        "mean_cue_plus_prob": None if prob.empty or prob.isna().all() else float(prob.mean()),
        "cue_plus_preference_rate": float(df["cue_plus_preferred"].astype(bool).mean()) if len(df) else None,
    }


def _stage_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (template_label, run_label), run_df in pair_df.groupby(["template_label", "run_label"], sort=True):
        rows.append(_summary_row(run_df, template_label=str(template_label), run_label=str(run_label), group_type="all", group_value="all"))
        for group_type, keys in (
            ("axis", ["axis"]),
            ("axis_direction", ["axis", "direction"]),
            ("axis_role", ["axis", "role"]),
            ("role", ["role"]),
            ("source_dataset", ["source_dataset"]),
            ("source_axis", ["source_dataset", "axis"]),
            ("item_type", ["item_type"]),
        ):
            for group_value, group in run_df.groupby(keys, sort=True):
                if not isinstance(group_value, tuple):
                    group_value = (group_value,)
                rows.append(
                    _summary_row(
                        group,
                        template_label=str(template_label),
                        run_label=str(run_label),
                        group_type=group_type,
                        group_value="::".join(map(str, group_value)),
                    )
                )
    return pd.DataFrame(rows)


def _available_contrasts(raw_contrasts: list[str], labels: set[str]) -> list[str]:
    out: list[str] = []
    for raw in raw_contrasts:
        name, left, right = _parse_contrast(raw)
        if left in labels and right in labels:
            out.append(f"{name}={left}-{right}")
    return out


def _delta_row(
    df: pd.DataFrame,
    *,
    contrast: str,
    left: str,
    right: str,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    delta = pd.to_numeric(df["delta_cue_plus_margin"], errors="coerce")
    return {
        "contrast": contrast,
        "left_run": left,
        "right_run": right,
        "template_label": str(df["template_label"].iloc[0]) if "template_label" in df.columns and len(df) else "",
        "group_type": group_type,
        "group_value": group_value,
        "n_counterfactuals": int(len(df)),
        "mean_right_cue_plus_margin": float(pd.to_numeric(df["mean_cue_plus_margin_right"], errors="coerce").mean()),
        "mean_left_cue_plus_margin": float(pd.to_numeric(df["mean_cue_plus_margin_left"], errors="coerce").mean()),
        "mean_delta_cue_plus_margin": float(delta.mean()),
        "median_delta_cue_plus_margin": float(delta.median()),
        "right_cue_plus_preference_rate": float(df["right_pref"].astype(bool).mean()),
        "left_cue_plus_preference_rate": float(df["left_pref"].astype(bool).mean()),
        "pref_flip_to_cue_plus_rate": float(df["pref_flip_to_cue_plus"].astype(bool).mean()),
        "pref_flip_from_cue_plus_rate": float(df["pref_flip_from_cue_plus"].astype(bool).mean()),
    }


def stage_contrast_summary(pair_df: pd.DataFrame, contrasts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for raw in contrasts:
        name, left, right = _parse_contrast(raw)
        for template_label, template_df in pair_df.groupby("template_label", sort=True):
            left_df = template_df[template_df["run_label"].astype(str) == left].copy()
            right_df = template_df[template_df["run_label"].astype(str) == right].copy()
            if left_df.empty or right_df.empty:
                continue
            keep = ["template_label", "counterfactual_id", "mean_cue_plus_margin", "cue_plus_preferred", "axis", "direction", "role", "source_dataset", "subset", "item_type"]
            merged = left_df[keep].merge(
                right_df[keep],
                on=["template_label", "counterfactual_id"],
                suffixes=("_left", "_right"),
                how="inner",
            )
            if merged.empty:
                continue
            merged["contrast"] = name
            merged["left_run"] = left
            merged["right_run"] = right
            merged["delta_cue_plus_margin"] = (
                pd.to_numeric(merged["mean_cue_plus_margin_left"], errors="coerce")
                - pd.to_numeric(merged["mean_cue_plus_margin_right"], errors="coerce")
            )
            merged["left_pref"] = merged["cue_plus_preferred_left"].astype(bool)
            merged["right_pref"] = merged["cue_plus_preferred_right"].astype(bool)
            merged["pref_flip_to_cue_plus"] = (~merged["right_pref"]) & merged["left_pref"]
            merged["pref_flip_from_cue_plus"] = merged["right_pref"] & (~merged["left_pref"])
            pair_rows.append(merged)
            rows.append(_delta_row(merged, contrast=name, left=left, right=right, group_type="all", group_value="all"))
            for group_type, keys in (
                ("axis", ["axis_left"]),
                ("axis_direction", ["axis_left", "direction_left"]),
                ("axis_role", ["axis_left", "role_left"]),
                ("role", ["role_left"]),
                ("source_dataset", ["source_dataset_left"]),
                ("source_axis", ["source_dataset_left", "axis_left"]),
            ):
                for group_value, group in merged.groupby(keys, sort=True):
                    if not isinstance(group_value, tuple):
                        group_value = (group_value,)
                    group_rows.append(
                        _delta_row(
                            group,
                            contrast=name,
                            left=left,
                            right=right,
                            group_type=group_type,
                            group_value="::".join(map(str, group_value)),
                        )
                    )
    return (
        pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame(),
        pd.DataFrame(rows),
        pd.DataFrame(group_rows),
    )


def _template_delta_pair_frame(pair_df: pd.DataFrame, raw: str) -> pd.DataFrame:
    name, left_template, right_template = _parse_contrast(raw)
    left = pair_df[pair_df["template_label"].astype(str) == left_template].copy()
    right = pair_df[pair_df["template_label"].astype(str) == right_template].copy()
    keep = ["run_label", "counterfactual_id", "mean_cue_plus_margin", "cue_plus_preferred", "axis", "direction", "role", "source_dataset", "subset", "item_type"]
    merged = left[keep].merge(
        right[keep],
        on=["run_label", "counterfactual_id"],
        suffixes=("_left_template", "_right_template"),
        how="inner",
    )
    if merged.empty:
        return merged
    merged["template_contrast"] = name
    merged["left_template"] = left_template
    merged["right_template"] = right_template
    merged["template_delta_cue_plus_margin"] = (
        pd.to_numeric(merged["mean_cue_plus_margin_left_template"], errors="coerce")
        - pd.to_numeric(merged["mean_cue_plus_margin_right_template"], errors="coerce")
    )
    merged["left_template_pref"] = merged["cue_plus_preferred_left_template"].astype(bool)
    merged["right_template_pref"] = merged["cue_plus_preferred_right_template"].astype(bool)
    return merged


def _template_delta_row(df: pd.DataFrame, *, template_contrast: str, run_label: str, group_type: str, group_value: str) -> dict[str, Any]:
    delta = pd.to_numeric(df["template_delta_cue_plus_margin"], errors="coerce")
    return {
        "template_contrast": template_contrast,
        "run_label": run_label,
        "group_type": group_type,
        "group_value": group_value,
        "n_counterfactuals": int(len(df)),
        "mean_right_template_cue_plus_margin": float(pd.to_numeric(df["mean_cue_plus_margin_right_template"], errors="coerce").mean()),
        "mean_left_template_cue_plus_margin": float(pd.to_numeric(df["mean_cue_plus_margin_left_template"], errors="coerce").mean()),
        "mean_template_delta_cue_plus_margin": float(delta.mean()),
        "median_template_delta_cue_plus_margin": float(delta.median()),
        "right_template_cue_plus_preference_rate": float(df["right_template_pref"].astype(bool).mean()),
        "left_template_cue_plus_preference_rate": float(df["left_template_pref"].astype(bool).mean()),
    }


def template_sensitivity_summary(pair_df: pd.DataFrame, contrasts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for raw in contrasts:
        delta_df = _template_delta_pair_frame(pair_df, raw)
        if delta_df.empty:
            continue
        name, _, _ = _parse_contrast(raw)
        pair_rows.append(delta_df)
        for run_label, run_df in delta_df.groupby("run_label", sort=True):
            rows.append(_template_delta_row(run_df, template_contrast=name, run_label=str(run_label), group_type="all", group_value="all"))
            for group_type, keys in (
                ("axis", ["axis_left_template"]),
                ("axis_direction", ["axis_left_template", "direction_left_template"]),
                ("axis_role", ["axis_left_template", "role_left_template"]),
                ("role", ["role_left_template"]),
                ("source_dataset", ["source_dataset_left_template"]),
                ("source_axis", ["source_dataset_left_template", "axis_left_template"]),
            ):
                for group_value, group in run_df.groupby(keys, sort=True):
                    if not isinstance(group_value, tuple):
                        group_value = (group_value,)
                    group_rows.append(
                        _template_delta_row(
                            group,
                            template_contrast=name,
                            run_label=str(run_label),
                            group_type=group_type,
                            group_value="::".join(map(str, group_value)),
                        )
                    )
    return (
        pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame(),
        pd.DataFrame(rows),
        pd.DataFrame(group_rows),
    )


def _interaction_pair_frame(template_pairs: pd.DataFrame, raw: str) -> pd.DataFrame:
    name, left_stage, right_stage = _parse_contrast(raw)
    left = template_pairs[template_pairs["run_label"].astype(str) == left_stage].copy()
    right = template_pairs[template_pairs["run_label"].astype(str) == right_stage].copy()
    keep = [
        "template_contrast",
        "counterfactual_id",
        "template_delta_cue_plus_margin",
        "axis_left_template",
        "direction_left_template",
        "role_left_template",
        "source_dataset_left_template",
    ]
    merged = left[keep].merge(
        right[keep],
        on=["template_contrast", "counterfactual_id"],
        suffixes=("_left_stage", "_right_stage"),
        how="inner",
    )
    if merged.empty:
        return merged
    merged["stage_contrast"] = name
    merged["left_stage"] = left_stage
    merged["right_stage"] = right_stage
    merged["stage_template_interaction_cue_plus_margin"] = (
        pd.to_numeric(merged["template_delta_cue_plus_margin_left_stage"], errors="coerce")
        - pd.to_numeric(merged["template_delta_cue_plus_margin_right_stage"], errors="coerce")
    )
    return merged


def _interaction_row(df: pd.DataFrame, *, stage_contrast: str, template_contrast: str, group_type: str, group_value: str) -> dict[str, Any]:
    delta = pd.to_numeric(df["stage_template_interaction_cue_plus_margin"], errors="coerce")
    return {
        "stage_template_interaction": f"{stage_contrast}__x__{template_contrast}",
        "stage_contrast": stage_contrast,
        "template_contrast": template_contrast,
        "group_type": group_type,
        "group_value": group_value,
        "n_counterfactuals": int(len(df)),
        "mean_right_stage_template_delta": float(pd.to_numeric(df["template_delta_cue_plus_margin_right_stage"], errors="coerce").mean()),
        "mean_left_stage_template_delta": float(pd.to_numeric(df["template_delta_cue_plus_margin_left_stage"], errors="coerce").mean()),
        "mean_interaction_cue_plus_margin": float(delta.mean()),
        "median_interaction_cue_plus_margin": float(delta.median()),
    }


def stage_template_interaction_summary(template_pairs: pd.DataFrame, contrasts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    if template_pairs.empty or "run_label" not in template_pairs.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for raw in contrasts:
        interaction_df = _interaction_pair_frame(template_pairs, raw)
        if interaction_df.empty:
            continue
        stage_name, _, _ = _parse_contrast(raw)
        pair_rows.append(interaction_df)
        for template_contrast, template_df in interaction_df.groupby("template_contrast", sort=True):
            rows.append(
                _interaction_row(
                    template_df,
                    stage_contrast=stage_name,
                    template_contrast=str(template_contrast),
                    group_type="all",
                    group_value="all",
                )
            )
            for group_type, keys in (
                ("axis", ["axis_left_template_left_stage"]),
                ("axis_direction", ["axis_left_template_left_stage", "direction_left_template_left_stage"]),
                ("axis_role", ["axis_left_template_left_stage", "role_left_template_left_stage"]),
                ("role", ["role_left_template_left_stage"]),
                ("source_dataset", ["source_dataset_left_template_left_stage"]),
                ("source_axis", ["source_dataset_left_template_left_stage", "axis_left_template_left_stage"]),
            ):
                for group_value, group in template_df.groupby(keys, sort=True):
                    if not isinstance(group_value, tuple):
                        group_value = (group_value,)
                    group_rows.append(
                        _interaction_row(
                            group,
                            stage_contrast=stage_name,
                            template_contrast=str(template_contrast),
                            group_type=group_type,
                            group_value="::".join(map(str, group_value)),
                        )
                    )
    return (
        pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame(),
        pd.DataFrame(rows),
        pd.DataFrame(group_rows),
    )


def _write_csv(path: str, df: pd.DataFrame) -> None:
    pd.DataFrame(df).to_csv(path, index=False)


def _print_table(title: str, df: pd.DataFrame, cols: list[str], *, max_rows: int = 80) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("(empty)")
        return
    selected = [col for col in cols if col in df.columns]
    print(df[selected].head(max(int(max_rows), 0)).to_string(index=False) if selected else df.head(max_rows).to_string(index=False))


def main() -> None:
    args = _parse_args()
    if not args.run:
        raise ValueError("Pass at least one --run TEMPLATE:LABEL=DIR_OR_CSV.")
    workspace_root = Path(args.workspace_root).resolve()
    out_dir = _resolve(workspace_root, args.out_dir)

    frames: list[pd.DataFrame] = []
    run_inputs: dict[str, str] = {}
    for raw in args.run:
        template_label, run_label, raw_path = _parse_run(str(raw))
        path = _resolve(workspace_root, raw_path)
        frames.append(_load_run(template_label, run_label, path))
        run_inputs[f"{template_label}:{run_label}"] = str(path)
    long_df = pd.concat(frames, ignore_index=True)
    pair_df = _pair_level(long_df)
    stage_summary_df = _stage_summary(pair_df)

    run_labels = set(pair_df["run_label"].astype(str))
    template_labels = set(pair_df["template_label"].astype(str))
    stage_contrasts = _available_contrasts([str(item) for item in args.stage_contrast] or list(DEFAULT_STAGE_CONTRASTS), run_labels)
    template_contrasts = _available_contrasts([str(item) for item in args.template_contrast] or list(DEFAULT_TEMPLATE_CONTRASTS), template_labels)
    stage_pair_df, stage_contrast_df, stage_group_df = stage_contrast_summary(pair_df, stage_contrasts)
    if template_contrasts:
        template_pair_df, template_summary_df, template_group_df = template_sensitivity_summary(pair_df, template_contrasts)
        interaction_pair_df, interaction_summary_df, interaction_group_df = stage_template_interaction_summary(template_pair_df, stage_contrasts)
    else:
        template_pair_df = pd.DataFrame()
        template_summary_df = pd.DataFrame()
        template_group_df = pd.DataFrame()
        interaction_pair_df = pd.DataFrame()
        interaction_summary_df = pd.DataFrame()
        interaction_group_df = pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bt_rows_long_csv": str(out_dir / "bt_rows_long.csv"),
        "bt_pair_summary_long_csv": str(out_dir / "bt_pair_summary_long.csv"),
        "stage_summary_csv": str(out_dir / "stage_summary.csv"),
        "stage_contrast_pair_deltas_csv": str(out_dir / "stage_contrast_pair_deltas.csv"),
        "stage_contrast_deltas_csv": str(out_dir / "stage_contrast_deltas.csv"),
        "stage_contrast_group_deltas_csv": str(out_dir / "stage_contrast_group_deltas.csv"),
        "template_sensitivity_pair_deltas_csv": str(out_dir / "template_sensitivity_pair_deltas.csv"),
        "template_sensitivity_deltas_csv": str(out_dir / "template_sensitivity_deltas.csv"),
        "template_sensitivity_group_deltas_csv": str(out_dir / "template_sensitivity_group_deltas.csv"),
        "template_stage_interaction_pair_deltas_csv": str(out_dir / "template_stage_interaction_pair_deltas.csv"),
        "template_stage_interactions_csv": str(out_dir / "template_stage_interactions.csv"),
        "template_stage_interaction_group_deltas_csv": str(out_dir / "template_stage_interaction_group_deltas.csv"),
        "summary_json": str(out_dir / "summary.json"),
    }
    _write_csv(outputs["bt_rows_long_csv"], long_df)
    _write_csv(outputs["bt_pair_summary_long_csv"], pair_df)
    _write_csv(outputs["stage_summary_csv"], stage_summary_df)
    _write_csv(outputs["stage_contrast_pair_deltas_csv"], stage_pair_df)
    _write_csv(outputs["stage_contrast_deltas_csv"], stage_contrast_df)
    _write_csv(outputs["stage_contrast_group_deltas_csv"], stage_group_df)
    _write_csv(outputs["template_sensitivity_pair_deltas_csv"], template_pair_df)
    _write_csv(outputs["template_sensitivity_deltas_csv"], template_summary_df)
    _write_csv(outputs["template_sensitivity_group_deltas_csv"], template_group_df)
    _write_csv(outputs["template_stage_interaction_pair_deltas_csv"], interaction_pair_df)
    _write_csv(outputs["template_stage_interactions_csv"], interaction_summary_df)
    _write_csv(outputs["template_stage_interaction_group_deltas_csv"], interaction_group_df)
    write_json(
        out_dir / "summary.json",
        {
            "stage": "D4-BT-stage-template-summary",
            "out_dir": str(out_dir),
            "runs": run_inputs,
            "templates": sorted(template_labels),
            "run_labels": sorted(run_labels),
            "stage_contrasts": stage_contrasts,
            "template_contrasts": template_contrasts,
            "n_rows": int(len(long_df)),
            "n_pair_rows": int(len(pair_df)),
            "outputs": outputs,
        },
    )

    print(f"out_dir={out_dir}")
    overall = stage_summary_df[(stage_summary_df["group_type"] == "all") & (stage_summary_df["group_value"] == "all")]
    _print_table(
        "Stage Overall",
        overall.sort_values(["template_label", "run_label"]),
        ["template_label", "run_label", "n_counterfactuals", "mean_cue_plus_margin", "cue_plus_preference_rate", "mean_cue_plus_prob"],
    )
    _print_table(
        "Stage Deltas",
        stage_contrast_df.sort_values(["template_label", "contrast"]) if not stage_contrast_df.empty else stage_contrast_df,
        ["template_label", "contrast", "n_counterfactuals", "mean_right_cue_plus_margin", "mean_left_cue_plus_margin", "mean_delta_cue_plus_margin"],
    )
    _print_table(
        "Template Sensitivity",
        template_summary_df.sort_values(["template_contrast", "run_label"]) if not template_summary_df.empty else template_summary_df,
        ["template_contrast", "run_label", "n_counterfactuals", "mean_right_template_cue_plus_margin", "mean_left_template_cue_plus_margin", "mean_template_delta_cue_plus_margin"],
    )
    _print_table(
        "Stage x Template Interactions",
        interaction_summary_df.sort_values(["template_contrast", "stage_contrast"]) if not interaction_summary_df.empty else interaction_summary_df,
        ["template_contrast", "stage_contrast", "n_counterfactuals", "mean_right_stage_template_delta", "mean_left_stage_template_delta", "mean_interaction_cue_plus_margin"],
    )
    print(f"summary={out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
