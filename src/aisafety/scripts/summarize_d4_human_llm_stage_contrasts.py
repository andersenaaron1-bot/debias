"""Summarize human-vs-LLM stage-contrast runs and paired stage deltas."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, write_json


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
        help="Run label and path as LABEL=DIR_OR_CSV. May be passed multiple times.",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        default=[],
        help="Contrast as NAME=LEFT-RIGHT or LEFT-RIGHT, using labels passed via --run.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "mechanistic" / "d4_human_llm_stage_contrast_summary_v1")
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH for --run, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError(f"Expected nonempty LABEL=PATH for --run, got: {value}")
    return label, Path(raw_path)


def _score_csv(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("hllm_stage_scores.csv", "bt_stage_scores.csv"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find hllm_stage_scores.csv in {path}")


def _load_run(label: str, path: Path) -> pd.DataFrame:
    csv_path = _score_csv(path)
    df = pd.read_csv(csv_path)
    if "llm_margin" not in df.columns:
        if "cue_plus_margin" in df.columns:
            df["llm_margin"] = pd.to_numeric(df["cue_plus_margin"], errors="coerce")
        else:
            raise ValueError(f"Run {label} has no llm_margin column: {csv_path}")
    df["run_label"] = label
    if "stage_label" not in df.columns:
        df["stage_label"] = label
    if "scoring_mode" not in df.columns:
        df["scoring_mode"] = ""
    if "model_id" not in df.columns:
        df["model_id"] = ""
    if "prompt_style" not in df.columns:
        df["prompt_style"] = ""
    if "pair_id" not in df.columns:
        df["pair_id"] = df.get("bt_pair_id", pd.Series(range(len(df)), index=df.index)).astype(str)
    if "source_dataset" not in df.columns:
        df["source_dataset"] = ""
    if "subset" not in df.columns:
        df["subset"] = ""
    if "item_type" not in df.columns:
        df["item_type"] = ""
    if "presentation_order" not in df.columns:
        df["presentation_order"] = ""
    df["llm_margin"] = pd.to_numeric(df["llm_margin"], errors="coerce")
    if "llm_prob" not in df.columns:
        df["llm_prob"] = pd.NA
    if "llm_preferred" not in df.columns:
        df["llm_preferred"] = df["llm_margin"] > 0.0
    return df


def _pair_level(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["run_label", "pair_id"]
    meta_cols = ["stage_label", "scoring_mode", "model_id", "prompt_style", "source_dataset", "subset", "item_type"]
    rows = []
    for key, group in df.groupby(group_cols, sort=True):
        run_label, pair_id = key
        margin = pd.to_numeric(group["llm_margin"], errors="coerce")
        prob = pd.to_numeric(group["llm_prob"], errors="coerce") if "llm_prob" in group.columns else pd.Series(dtype=float)
        row: dict[str, Any] = {
            "run_label": str(run_label),
            "pair_id": str(pair_id),
            "n_order_rows": int(len(group)),
            "mean_llm_margin": float(margin.mean()),
            "median_llm_margin": float(margin.median()),
            "llm_preferred": bool(float(margin.mean()) > 0.0),
            "mean_llm_prob": None if prob.empty or prob.isna().all() else float(prob.mean()),
        }
        first = group.iloc[0]
        for col in meta_cols:
            if col in group.columns:
                row[col] = str(first.get(col) or "")
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_row(df: pd.DataFrame, *, run_label: str, group_type: str, group_value: str) -> dict[str, Any]:
    margin = pd.to_numeric(df["mean_llm_margin"], errors="coerce")
    prob = pd.to_numeric(df["mean_llm_prob"], errors="coerce") if "mean_llm_prob" in df.columns else pd.Series(dtype=float)
    return {
        "run_label": run_label,
        "group_type": group_type,
        "group_value": group_value,
        "n_pairs": int(len(df)),
        "mean_llm_margin": float(margin.mean()),
        "median_llm_margin": float(margin.median()),
        "mean_abs_llm_margin": float(margin.abs().mean()),
        "mean_llm_prob": None if prob.empty or prob.isna().all() else float(prob.mean()),
        "llm_preference_rate": float(df["llm_preferred"].astype(bool).mean()) if len(df) else None,
    }


def _stage_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_label, run_df in pair_df.groupby("run_label", sort=True):
        rows.append(_summary_row(run_df, run_label=str(run_label), group_type="all", group_value="all"))
        for group_type, keys in (
            ("source_dataset", ["source_dataset"]),
            ("source_dataset_subset", ["source_dataset", "subset"]),
            ("item_type", ["item_type"]),
        ):
            for group_value, group in run_df.groupby(keys, sort=True):
                if not isinstance(group_value, tuple):
                    group_value = (group_value,)
                rows.append(
                    _summary_row(
                        group,
                        run_label=str(run_label),
                        group_type=group_type,
                        group_value="::".join(map(str, group_value)),
                    )
                )
    return pd.DataFrame(rows)


def _parse_contrast(value: str) -> tuple[str, str, str]:
    name = ""
    expr = value
    if "=" in value:
        name, expr = value.split("=", 1)
    if "-" not in expr:
        raise ValueError(f"Expected LEFT-RIGHT contrast, got: {value}")
    left, right = expr.split("-", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(f"Expected LEFT-RIGHT contrast, got: {value}")
    if not name.strip():
        name = f"{left}_minus_{right}"
    return name.strip(), left, right


def _contrast_summary(pair_df: pd.DataFrame, contrasts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    labels = set(pair_df["run_label"].astype(str))
    for raw in contrasts:
        name, left, right = _parse_contrast(raw)
        if left not in labels:
            raise ValueError(f"Unknown left run label in contrast {raw!r}: {left}")
        if right not in labels:
            raise ValueError(f"Unknown right run label in contrast {raw!r}: {right}")
        left_df = pair_df[pair_df["run_label"].astype(str) == left].copy()
        right_df = pair_df[pair_df["run_label"].astype(str) == right].copy()
        keep = ["pair_id", "mean_llm_margin", "llm_preferred", "source_dataset", "subset", "item_type"]
        merged = left_df[keep].merge(
            right_df[keep],
            on="pair_id",
            suffixes=("_left", "_right"),
            how="inner",
        )
        if merged.empty:
            raise ValueError(f"Contrast {raw!r} has no overlapping pair_id rows.")
        merged["delta_llm_margin"] = merged["mean_llm_margin_left"] - merged["mean_llm_margin_right"]
        merged["left_pref"] = merged["llm_preferred_left"].astype(bool)
        merged["right_pref"] = merged["llm_preferred_right"].astype(bool)
        merged["pref_flip_to_llm"] = (~merged["right_pref"]) & merged["left_pref"]
        merged["pref_flip_from_llm"] = merged["right_pref"] & (~merged["left_pref"])
        all_rows.append(_contrast_row(merged, contrast=name, group_type="all", group_value="all", left=left, right=right))
        for group_type, keys in (
            ("source_dataset", ["source_dataset_left"]),
            ("source_dataset_subset", ["source_dataset_left", "subset_left"]),
            ("item_type", ["item_type_left"]),
        ):
            for group_value, group in merged.groupby(keys, sort=True):
                if not isinstance(group_value, tuple):
                    group_value = (group_value,)
                group_rows.append(
                    _contrast_row(
                        group,
                        contrast=name,
                        group_type=group_type,
                        group_value="::".join(map(str, group_value)),
                        left=left,
                        right=right,
                    )
                )
    return pd.DataFrame(all_rows), pd.DataFrame(group_rows)


def _contrast_row(df: pd.DataFrame, *, contrast: str, group_type: str, group_value: str, left: str, right: str) -> dict[str, Any]:
    delta = pd.to_numeric(df["delta_llm_margin"], errors="coerce")
    return {
        "contrast": contrast,
        "left_run": left,
        "right_run": right,
        "group_type": group_type,
        "group_value": group_value,
        "n_pairs": int(len(df)),
        "mean_left_llm_margin": float(pd.to_numeric(df["mean_llm_margin_left"], errors="coerce").mean()),
        "mean_right_llm_margin": float(pd.to_numeric(df["mean_llm_margin_right"], errors="coerce").mean()),
        "mean_delta_llm_margin": float(delta.mean()),
        "median_delta_llm_margin": float(delta.median()),
        "left_llm_preference_rate": float(df["left_pref"].astype(bool).mean()),
        "right_llm_preference_rate": float(df["right_pref"].astype(bool).mean()),
        "pref_flip_to_llm_rate": float(df["pref_flip_to_llm"].astype(bool).mean()),
        "pref_flip_from_llm_rate": float(df["pref_flip_from_llm"].astype(bool).mean()),
    }


def main() -> None:
    args = _parse_args()
    if not args.run:
        raise ValueError("Pass at least one --run LABEL=DIR_OR_CSV.")
    workspace_root = Path(args.workspace_root).resolve()
    out_dir = _resolve(workspace_root, args.out_dir)
    loaded = []
    run_inputs: dict[str, str] = {}
    for raw in args.run:
        label, raw_path = _parse_label_path(str(raw))
        path = _resolve(workspace_root, raw_path)
        loaded.append(_load_run(label, path))
        run_inputs[label] = str(path)
    long_df = pd.concat(loaded, ignore_index=True)
    pair_df = _pair_level(long_df)
    stage_summary = _stage_summary(pair_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "stage_rows_long.csv"
    pair_path = out_dir / "stage_pair_summary_long.csv"
    stage_summary_path = out_dir / "stage_summary.csv"
    long_df.to_csv(long_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    stage_summary.to_csv(stage_summary_path, index=False)

    outputs = {
        "stage_rows_long_csv": str(long_path),
        "stage_pair_summary_long_csv": str(pair_path),
        "stage_summary_csv": str(stage_summary_path),
        "summary_json": str(out_dir / "summary.json"),
    }
    manifest: dict[str, Any] = {
        "stage": "D4-human-LLM-stage-contrast-summary",
        "out_dir": str(out_dir),
        "runs": run_inputs,
        "n_stage_rows": int(len(long_df)),
        "n_pair_rows": int(len(pair_df)),
        "outputs": outputs,
    }
    if args.contrast:
        contrast_df, contrast_group_df = _contrast_summary(pair_df, [str(item) for item in args.contrast])
        contrast_path = out_dir / "stage_contrast_deltas.csv"
        contrast_group_path = out_dir / "stage_contrast_group_deltas.csv"
        contrast_df.to_csv(contrast_path, index=False)
        contrast_group_df.to_csv(contrast_group_path, index=False)
        outputs["stage_contrast_deltas_csv"] = str(contrast_path)
        outputs["stage_contrast_group_deltas_csv"] = str(contrast_group_path)
        manifest["contrasts"] = [str(item) for item in args.contrast]
    write_json(out_dir / "summary.json", manifest)
    print(f"out_dir={out_dir}")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"stage_summary={stage_summary_path}")


if __name__ == "__main__":
    main()
