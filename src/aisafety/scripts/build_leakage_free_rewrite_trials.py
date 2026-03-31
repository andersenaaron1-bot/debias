"""Build leakage-free rewritten Laurito trials from swapped rewrite CSVs.

This utility is for the common situation where you already have *two* rewritten
trials CSVs produced by `rewrite_laurito_trials_openrouter.py` with opposite
per-source target labels, e.g.:

  - file A: human -> human_plain, llm -> rlhf_ai_tone
  - file B: human -> rlhf_ai_tone, llm -> human_plain

Each of those files has severe label leakage for origin probing if you use the
rewritten A_text/B_text directly, because style is perfectly correlated with
origin. This script "stitches" them into a single trials CSV where *both*
human and llm texts are rewritten into the same target style label.

Example:

  python -m aisafety.scripts.build_leakage_free_rewrite_trials ^
    --swap-csv-a artifacts\\rewrites\\ai_tone_4omini\\trials_swap_human_human__llm_rhlf.csv ^
    --swap-csv-b artifacts\\rewrites\\ai_tone_4omini\\trials_swap_human_rlhf__llm_human.csv ^
    --target-label human_plain ^
    --out-csv artifacts\\rewrites\\ai_tone_4omini\\trials_all_human_plain.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_TRIAL_COLS = (
    "item_type",
    "title",
    "A_text",
    "B_text",
    "A_source",
    "B_source",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--swap-csv-a", type=Path, required=True, help="First swapped rewrite trials CSV.")
    p.add_argument("--swap-csv-b", type=Path, required=True, help="Second swapped rewrite trials CSV.")
    p.add_argument(
        "--target-label",
        type=str,
        required=True,
        help="Target rewrite label to use for *both* sources (e.g. human_plain or rlhf_ai_tone).",
    )
    p.add_argument("--out-csv", type=Path, required=True, help="Output trials CSV.")
    p.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep A_text_original/B_text_original in the output (if present).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Error if the two inputs are not exactly alignable by their original-text keys.",
    )
    return p.parse_args()


def _require_columns(df: pd.DataFrame, cols: tuple[str, ...], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _infer_per_source_label(df: pd.DataFrame, source: str) -> str | None:
    source = str(source).strip().lower()
    if source not in {"human", "llm"}:
        raise ValueError(f"Unexpected source {source!r}; expected 'human' or 'llm'.")

    col = "rewrite_label_human" if source == "human" else "rewrite_label_llm"
    if col in df.columns:
        vals = sorted(set(df[col].dropna().astype(str)))
        if len(vals) == 1:
            return vals[0]
        if len(vals) > 1:
            raise ValueError(f"Expected {col} to be constant, found {vals}")

    if "rewrite_label_default" in df.columns:
        vals = sorted(set(df["rewrite_label_default"].dropna().astype(str)))
        if len(vals) == 1:
            return vals[0]
        if len(vals) > 1:
            raise ValueError(f"Expected rewrite_label_default to be constant, found {vals}")

    return None


def _align_by_key(df: pd.DataFrame, *, name: str, strict: bool) -> pd.DataFrame:
    # Prefer aligning by original text if present (most robust). Otherwise, fall back
    # to (item_type, title, A_source, B_source) which assumes identical base trials.
    key_cols = ["item_type", "title", "A_source", "B_source"]
    if "A_text_original" in df.columns and "B_text_original" in df.columns:
        key_cols += ["A_text_original", "B_text_original"]
    elif strict:
        raise ValueError(
            f"{name} is missing A_text_original/B_text_original; rerun rewrites with --keep-original "
            "or omit --strict."
        )

    keys = df[key_cols].astype(str).agg("||".join, axis=1)
    if keys.duplicated().any():
        raise ValueError(f"{name} has duplicate trial keys; cannot align reliably.")
    out = df.copy()
    out["_trial_key"] = keys
    out = out.set_index("_trial_key", drop=True).sort_index()
    return out


def _require_same_rewrite_params(a: pd.DataFrame, b: pd.DataFrame) -> None:
    # If these differ and we select A rows for one class and B rows for the other,
    # we'd reintroduce a confound. Enforce equality when present.
    cols = (
        "rewrite_dimension",
        "rewrite_model",
        "rewrite_temperature",
        "rewrite_top_p",
        "rewrite_max_tokens",
        "rewrite_max_chars",
    )
    for c in cols:
        if c not in a.columns or c not in b.columns:
            continue
        av = sorted(set(a[c].dropna().astype(str)))
        bv = sorted(set(b[c].dropna().astype(str)))
        if av != bv:
            raise ValueError(f"Inputs differ on {c}: swap-csv-a={av} swap-csv-b={bv}")


def build_leakage_free_trials(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    target_label: str,
    keep_original: bool,
    strict: bool,
) -> pd.DataFrame:
    _require_columns(df_a, REQUIRED_TRIAL_COLS, name="swap-csv-a")
    _require_columns(df_b, REQUIRED_TRIAL_COLS, name="swap-csv-b")

    if len(df_a) != len(df_b):
        raise ValueError(f"Row count mismatch: swap-csv-a={len(df_a)} swap-csv-b={len(df_b)}")

    a = _align_by_key(df_a, name="swap-csv-a", strict=strict)
    b = _align_by_key(df_b, name="swap-csv-b", strict=strict)

    if a.index.tolist() != b.index.tolist():
        raise ValueError("Inputs could not be aligned by trial key (different keys/order).")

    # Ensure base trial metadata is identical after alignment.
    base_cols = ["item_type", "title", "A_source", "B_source"]
    for c in base_cols:
        if not (a[c].astype(str).values == b[c].astype(str).values).all():
            raise ValueError(f"Inputs disagree on base column {c!r}; not the same underlying trials.")

    if "A_text_original" in a.columns and "A_text_original" in b.columns:
        for c in ("A_text_original", "B_text_original"):
            if not (a[c].astype(str).values == b[c].astype(str).values).all():
                raise ValueError(f"Inputs disagree on {c!r}; not the same original texts.")

    _require_same_rewrite_params(a, b)

    labels_a = {"human": _infer_per_source_label(a, "human"), "llm": _infer_per_source_label(a, "llm")}
    labels_b = {"human": _infer_per_source_label(b, "human"), "llm": _infer_per_source_label(b, "llm")}
    if labels_a["human"] is None or labels_a["llm"] is None:
        raise ValueError(f"Could not infer per-source rewrite labels for swap-csv-a: {labels_a}")
    if labels_b["human"] is None or labels_b["llm"] is None:
        raise ValueError(f"Could not infer per-source rewrite labels for swap-csv-b: {labels_b}")

    def pick_df_for_source(source: str) -> pd.DataFrame:
        source = str(source).strip().lower()
        candidates: list[str] = []
        if labels_a[source] == target_label:
            candidates.append("a")
        if labels_b[source] == target_label:
            candidates.append("b")
        if not candidates:
            raise ValueError(
                f"No input provides target_label={target_label!r} for source={source!r} "
                f"(swap-csv-a={labels_a[source]!r}, swap-csv-b={labels_b[source]!r})."
            )
        if len(candidates) > 1:
            # Both inputs already have the desired label for this source; prefer A.
            return a
        return a if candidates[0] == "a" else b

    df_human = pick_df_for_source("human")
    df_llm = pick_df_for_source("llm")

    out = a.reset_index(drop=True).copy()

    a_src = out["A_source"].astype(str).str.lower()
    b_src = out["B_source"].astype(str).str.lower()
    if not set(a_src.unique()).issubset({"human", "llm"}):
        raise ValueError(f"Unexpected A_source values: {sorted(set(a_src.unique()))}")
    if not set(b_src.unique()).issubset({"human", "llm"}):
        raise ValueError(f"Unexpected B_source values: {sorted(set(b_src.unique()))}")

    out["A_text"] = np.where(
        a_src == "human",
        df_human["A_text"].to_numpy(copy=False),
        df_llm["A_text"].to_numpy(copy=False),
    )
    out["B_text"] = np.where(
        b_src == "human",
        df_human["B_text"].to_numpy(copy=False),
        df_llm["B_text"].to_numpy(copy=False),
    )

    out["rewrite_label_default"] = str(target_label)
    out["rewrite_label_human"] = str(target_label)
    out["rewrite_label_llm"] = str(target_label)
    out["rewrite_stitched_from"] = f"{labels_a} + {labels_b}"

    if not keep_original:
        drop_cols = [c for c in ("A_text_original", "B_text_original") if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

    return out


def main() -> None:
    args = parse_args()
    df_a = pd.read_csv(args.swap_csv_a)
    df_b = pd.read_csv(args.swap_csv_b)
    out = build_leakage_free_trials(
        df_a,
        df_b,
        target_label=str(args.target_label),
        keep_original=bool(args.keep_original),
        strict=bool(args.strict),
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} trials to {args.out_csv}")


if __name__ == "__main__":
    main()
