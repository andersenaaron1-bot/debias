"""Summarize template sensitivity in human-vs-LLM stage contrasts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json
from aisafety.scripts.summarize_d4_human_llm_stage_contrasts import _parse_contrast


DEFAULT_OUT_DIR = (
    Path("artifacts") / "mechanistic" / "d4_human_llm_template_sensitivity_v1"
)
DEFAULT_TEMPLATE_CONTRASTS = (
    "standard_minus_minimal=standard-minimal",
    "rubric_quality_minus_minimal=rubric_quality-minimal",
    "substance_only_minus_minimal=substance_only-minimal",
    "substance_only_minus_rubric_quality=substance_only-rubric_quality",
)
DEFAULT_STAGE_CONTRASTS = (
    "tulu3_sft_minus_base=tulu3_sft-llama31_base",
    "tulu3_dpo_minus_sft=tulu3_dpo-tulu3_sft",
    "tulu3_final_minus_base=tulu3_final-llama31_base",
    "llama31_instruct_minus_base=llama31_instruct-llama31_base",
    "llama31_instruct_minus_tulu3_dpo=llama31_instruct-tulu3_dpo",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help=(
            "Template label and stage-summary directory as TEMPLATE=DIR. DIR must contain "
            "stage_pair_summary_long.csv from summarize_d4_human_llm_stage_contrasts."
        ),
    )
    parser.add_argument(
        "--template-contrast",
        action="append",
        default=[],
        help="Template contrast as NAME=LEFT-RIGHT or LEFT-RIGHT. Defaults cover standard/minimal/rubric/substance.",
    )
    parser.add_argument(
        "--stage-contrast",
        action="append",
        default=[],
        help="Stage contrast as NAME=LEFT-RIGHT or LEFT-RIGHT. Defaults cover Tulu and Llama stage contrasts.",
    )
    parser.add_argument("--include-non-forced", action="store_true")
    parser.add_argument("--source-top-k", type=int, default=200)
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
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError(f"Expected nonempty LABEL=PATH, got: {value}")
    return label, Path(raw_path)


def _pair_summary_csv(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "stage_pair_summary_long.csv"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find stage_pair_summary_long.csv in {path}")


def _ensure_pair_columns(df: pd.DataFrame, *, template_label: str) -> pd.DataFrame:
    out = df.copy()
    out["template_label"] = str(template_label)
    if "run_label" not in out.columns:
        raise ValueError("Template summary rows need a run_label column.")
    if "pair_id" not in out.columns:
        raise ValueError("Template summary rows need a pair_id column.")
    if "mean_llm_margin" not in out.columns:
        raise ValueError("Template summary rows need a mean_llm_margin column.")
    for col in ("source_dataset", "subset", "item_type", "stage_label", "scoring_mode", "model_id", "prompt_style"):
        if col not in out.columns:
            out[col] = ""
    if "comparison_template" not in out.columns:
        out["comparison_template"] = str(template_label)
    else:
        out["comparison_template"] = out["comparison_template"].fillna("").astype(str)
        out.loc[out["comparison_template"] == "", "comparison_template"] = str(template_label)
    if "llm_preferred" not in out.columns:
        out["llm_preferred"] = pd.to_numeric(out["mean_llm_margin"], errors="coerce") > 0.0
    if "mean_llm_prob" not in out.columns:
        out["mean_llm_prob"] = pd.NA
    out["run_label"] = out["run_label"].astype(str)
    out["pair_id"] = out["pair_id"].astype(str)
    out["mean_llm_margin"] = pd.to_numeric(out["mean_llm_margin"], errors="coerce")
    out["llm_preferred"] = out["llm_preferred"].astype(bool)
    return out


def load_template_summary(
    template_label: str,
    path: Path,
    *,
    include_non_forced: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(_pair_summary_csv(path))
    df = _ensure_pair_columns(df, template_label=template_label)
    if not bool(include_non_forced) and "scoring_mode" in df.columns:
        scoring = df["scoring_mode"].fillna("").astype(str)
        df = df[(scoring == "") | (scoring == "forced_choice")].copy()
    return df


def _available_contrasts(raw_contrasts: list[str], labels: set[str]) -> list[str]:
    out: list[str] = []
    for raw in raw_contrasts:
        name, left, right = _parse_contrast(raw)
        if left in labels and right in labels:
            out.append(f"{name}={left}-{right}")
    return out


def _template_delta_frame(pair_df: pd.DataFrame, raw_contrast: str) -> pd.DataFrame:
    name, left_template, right_template = _parse_contrast(raw_contrast)
    left = pair_df[pair_df["template_label"].astype(str) == left_template].copy()
    right = pair_df[pair_df["template_label"].astype(str) == right_template].copy()
    keep = ["run_label", "pair_id", "mean_llm_margin", "llm_preferred", "source_dataset", "subset", "item_type"]
    merged = left[keep].merge(
        right[keep],
        on=["run_label", "pair_id"],
        suffixes=("_left_template", "_right_template"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"Template contrast {raw_contrast!r} has no overlapping rows.")
    merged["template_contrast"] = name
    merged["left_template"] = left_template
    merged["right_template"] = right_template
    merged["template_delta_llm_margin"] = (
        pd.to_numeric(merged["mean_llm_margin_left_template"], errors="coerce")
        - pd.to_numeric(merged["mean_llm_margin_right_template"], errors="coerce")
    )
    merged["left_template_pref"] = merged["llm_preferred_left_template"].astype(bool)
    merged["right_template_pref"] = merged["llm_preferred_right_template"].astype(bool)
    merged["pref_flip_to_llm"] = (~merged["right_template_pref"]) & merged["left_template_pref"]
    merged["pref_flip_from_llm"] = merged["right_template_pref"] & (~merged["left_template_pref"])
    return merged


def _template_delta_row(
    df: pd.DataFrame,
    *,
    template_contrast: str,
    run_label: str,
    group_type: str,
    group_value: str,
    left_template: str,
    right_template: str,
) -> dict[str, Any]:
    delta = pd.to_numeric(df["template_delta_llm_margin"], errors="coerce")
    return {
        "template_contrast": template_contrast,
        "run_label": run_label,
        "group_type": group_type,
        "group_value": group_value,
        "left_template": left_template,
        "right_template": right_template,
        "n_pairs": int(len(df)),
        "mean_left_template_llm_margin": float(
            pd.to_numeric(df["mean_llm_margin_left_template"], errors="coerce").mean()
        ),
        "mean_right_template_llm_margin": float(
            pd.to_numeric(df["mean_llm_margin_right_template"], errors="coerce").mean()
        ),
        "mean_template_delta_llm_margin": float(delta.mean()),
        "median_template_delta_llm_margin": float(delta.median()),
        "left_template_llm_preference_rate": float(df["left_template_pref"].astype(bool).mean()),
        "right_template_llm_preference_rate": float(df["right_template_pref"].astype(bool).mean()),
        "pref_flip_to_llm_rate": float(df["pref_flip_to_llm"].astype(bool).mean()),
        "pref_flip_from_llm_rate": float(df["pref_flip_from_llm"].astype(bool).mean()),
    }


def template_delta_summary(
    pair_df: pd.DataFrame,
    template_contrasts: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for raw in template_contrasts:
        delta_df = _template_delta_frame(pair_df, raw)
        name, left_template, right_template = _parse_contrast(raw)
        pair_rows.append(delta_df)
        for run_label, run_df in delta_df.groupby("run_label", sort=True):
            summary_rows.append(
                _template_delta_row(
                    run_df,
                    template_contrast=name,
                    run_label=str(run_label),
                    group_type="all",
                    group_value="all",
                    left_template=left_template,
                    right_template=right_template,
                )
            )
            for group_type, keys in (
                ("source_dataset", ["source_dataset_left_template"]),
                ("source_dataset_subset", ["source_dataset_left_template", "subset_left_template"]),
                ("item_type", ["item_type_left_template"]),
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
                            left_template=left_template,
                            right_template=right_template,
                        )
                    )
    return (
        pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame(),
        pd.DataFrame(summary_rows),
        pd.DataFrame(group_rows),
    )


def _interaction_frame(template_delta_pairs: pd.DataFrame, raw_stage_contrast: str) -> pd.DataFrame:
    name, left_stage, right_stage = _parse_contrast(raw_stage_contrast)
    left = template_delta_pairs[template_delta_pairs["run_label"].astype(str) == left_stage].copy()
    right = template_delta_pairs[template_delta_pairs["run_label"].astype(str) == right_stage].copy()
    keep = [
        "template_contrast",
        "pair_id",
        "template_delta_llm_margin",
        "source_dataset_left_template",
        "subset_left_template",
        "item_type_left_template",
    ]
    merged = left[keep].merge(
        right[keep],
        on=["template_contrast", "pair_id"],
        suffixes=("_left_stage", "_right_stage"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"Stage contrast {raw_stage_contrast!r} has no overlapping template-delta rows.")
    merged["stage_contrast"] = name
    merged["left_stage"] = left_stage
    merged["right_stage"] = right_stage
    merged["stage_template_interaction_llm_margin"] = (
        pd.to_numeric(merged["template_delta_llm_margin_left_stage"], errors="coerce")
        - pd.to_numeric(merged["template_delta_llm_margin_right_stage"], errors="coerce")
    )
    return merged


def _interaction_row(
    df: pd.DataFrame,
    *,
    stage_contrast: str,
    template_contrast: str,
    group_type: str,
    group_value: str,
    left_stage: str,
    right_stage: str,
) -> dict[str, Any]:
    delta = pd.to_numeric(df["stage_template_interaction_llm_margin"], errors="coerce")
    return {
        "stage_template_interaction": f"{stage_contrast}__x__{template_contrast}",
        "stage_contrast": stage_contrast,
        "template_contrast": template_contrast,
        "group_type": group_type,
        "group_value": group_value,
        "left_stage": left_stage,
        "right_stage": right_stage,
        "n_pairs": int(len(df)),
        "mean_left_stage_template_delta": float(
            pd.to_numeric(df["template_delta_llm_margin_left_stage"], errors="coerce").mean()
        ),
        "mean_right_stage_template_delta": float(
            pd.to_numeric(df["template_delta_llm_margin_right_stage"], errors="coerce").mean()
        ),
        "mean_interaction_llm_margin": float(delta.mean()),
        "median_interaction_llm_margin": float(delta.median()),
    }


def stage_template_interaction_summary(
    template_delta_pairs: pd.DataFrame,
    stage_contrasts: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for raw in stage_contrasts:
        interaction_df = _interaction_frame(template_delta_pairs, raw)
        stage_name, left_stage, right_stage = _parse_contrast(raw)
        pair_rows.append(interaction_df)
        for template_contrast, template_df in interaction_df.groupby("template_contrast", sort=True):
            summary_rows.append(
                _interaction_row(
                    template_df,
                    stage_contrast=stage_name,
                    template_contrast=str(template_contrast),
                    group_type="all",
                    group_value="all",
                    left_stage=left_stage,
                    right_stage=right_stage,
                )
            )
            for group_type, keys in (
                ("source_dataset", ["source_dataset_left_template_left_stage"]),
                ("source_dataset_subset", ["source_dataset_left_template_left_stage", "subset_left_template_left_stage"]),
                ("item_type", ["item_type_left_template_left_stage"]),
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
                            left_stage=left_stage,
                            right_stage=right_stage,
                        )
                    )
    return (
        pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame(),
        pd.DataFrame(summary_rows),
        pd.DataFrame(group_rows),
    )


def _print_table(title: str, df: pd.DataFrame, cols: list[str], *, max_rows: int = 200) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("(empty)")
        return
    selected = [col for col in cols if col in df.columns]
    if selected:
        print(df[selected].head(max(int(max_rows), 0)).to_string(index=False))
    else:
        print(df.head(max(int(max_rows), 0)).to_string(index=False))


def main() -> None:
    args = _parse_args()
    if not args.summary:
        raise ValueError("Pass at least two --summary TEMPLATE=DIR arguments.")
    workspace_root = Path(args.workspace_root).resolve()
    out_dir = _resolve(workspace_root, args.out_dir)

    frames: list[pd.DataFrame] = []
    summary_inputs: dict[str, str] = {}
    for raw in args.summary:
        template_label, raw_path = _parse_label_path(str(raw))
        path = _resolve(workspace_root, raw_path)
        frames.append(load_template_summary(template_label, path, include_non_forced=bool(args.include_non_forced)))
        summary_inputs[template_label] = str(path)
    pair_df = pd.concat(frames, ignore_index=True)

    templates = set(pair_df["template_label"].astype(str))
    run_labels = set(pair_df["run_label"].astype(str))
    template_contrasts = _available_contrasts(
        [str(item) for item in args.template_contrast] or list(DEFAULT_TEMPLATE_CONTRASTS),
        templates,
    )
    if not template_contrasts:
        raise ValueError(f"No template contrasts are available for templates: {sorted(templates)}")
    stage_contrasts = _available_contrasts(
        [str(item) for item in args.stage_contrast] or list(DEFAULT_STAGE_CONTRASTS),
        run_labels,
    )
    if not stage_contrasts:
        raise ValueError(f"No stage contrasts are available for run labels: {sorted(run_labels)}")

    template_pair_df, template_summary_df, template_group_df = template_delta_summary(pair_df, template_contrasts)
    interaction_pair_df, interaction_summary_df, interaction_group_df = stage_template_interaction_summary(
        template_pair_df,
        stage_contrasts,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "template_pair_rows_long_csv": str(out_dir / "template_pair_rows_long.csv"),
        "template_sensitivity_pair_deltas_csv": str(out_dir / "template_sensitivity_pair_deltas.csv"),
        "template_sensitivity_deltas_csv": str(out_dir / "template_sensitivity_deltas.csv"),
        "template_sensitivity_group_deltas_csv": str(out_dir / "template_sensitivity_group_deltas.csv"),
        "template_stage_interaction_pair_deltas_csv": str(out_dir / "template_stage_interaction_pair_deltas.csv"),
        "template_stage_interactions_csv": str(out_dir / "template_stage_interactions.csv"),
        "template_stage_interaction_group_deltas_csv": str(out_dir / "template_stage_interaction_group_deltas.csv"),
        "summary_json": str(out_dir / "summary.json"),
    }
    pair_df.to_csv(outputs["template_pair_rows_long_csv"], index=False)
    template_pair_df.to_csv(outputs["template_sensitivity_pair_deltas_csv"], index=False)
    template_summary_df.to_csv(outputs["template_sensitivity_deltas_csv"], index=False)
    template_group_df.to_csv(outputs["template_sensitivity_group_deltas_csv"], index=False)
    interaction_pair_df.to_csv(outputs["template_stage_interaction_pair_deltas_csv"], index=False)
    interaction_summary_df.to_csv(outputs["template_stage_interactions_csv"], index=False)
    interaction_group_df.to_csv(outputs["template_stage_interaction_group_deltas_csv"], index=False)
    write_json(
        out_dir / "summary.json",
        {
            "stage": "D4-human-LLM-template-sensitivity-summary",
            "out_dir": str(out_dir),
            "summaries": summary_inputs,
            "templates": sorted(templates),
            "run_labels": sorted(run_labels),
            "template_contrasts": template_contrasts,
            "stage_contrasts": stage_contrasts,
            "n_pair_rows": int(len(pair_df)),
            "n_template_delta_rows": int(len(template_pair_df)),
            "n_interaction_rows": int(len(interaction_pair_df)),
            "outputs": outputs,
        },
    )

    print(f"out_dir={out_dir}")
    _print_table(
        "Template Sensitivity Overall",
        template_summary_df.sort_values(["template_contrast", "run_label"]),
        [
            "template_contrast",
            "run_label",
            "n_pairs",
            "mean_right_template_llm_margin",
            "mean_left_template_llm_margin",
            "mean_template_delta_llm_margin",
            "right_template_llm_preference_rate",
            "left_template_llm_preference_rate",
        ],
    )
    _print_table(
        "Stage x Template Interactions",
        interaction_summary_df.sort_values(["template_contrast", "stage_contrast"]),
        [
            "template_contrast",
            "stage_contrast",
            "n_pairs",
            "mean_right_stage_template_delta",
            "mean_left_stage_template_delta",
            "mean_interaction_llm_margin",
        ],
    )
    source_groups = interaction_group_df[interaction_group_df["group_type"] == "source_dataset"].copy()
    if not source_groups.empty:
        source_groups = source_groups.sort_values(["template_contrast", "stage_contrast", "group_value"])
    _print_table(
        "Source-Level Stage x Template Interactions",
        source_groups,
        [
            "template_contrast",
            "stage_contrast",
            "group_value",
            "n_pairs",
            "mean_interaction_llm_margin",
        ],
        max_rows=int(args.source_top_k),
    )
    print(f"summary={out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
