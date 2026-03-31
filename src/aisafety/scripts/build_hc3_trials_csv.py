"""Build HC3 A/B trials CSV for reward-model evaluation.

This script reads local HC3 JSONL files and writes a trials CSV with columns:
  item_type, title, question, hc3_source, hc3_subset, A_text, B_text, A_source, B_source

The output can be passed directly to:
  python -m aisafety.scripts.eval_laurito_bias_reward --trials-csv <out.csv> ...
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from aisafety.config import DATA_DIR, DEFAULT_SEED
from aisafety.data.hc3 import load_hc3_descriptions
from aisafety.data.trials import build_hc3_trials


def _parse_csv_list(value: str) -> set[str]:
    return {x.strip() for x in str(value or "").split(",") if x.strip()}


def _iter_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _sample_desc_rows(
    df_desc: pd.DataFrame,
    *,
    max_pairs: int | None,
    rng: random.Random,
) -> pd.DataFrame:
    if max_pairs is None or max_pairs <= 0 or df_desc.empty:
        return df_desc

    titles = sorted(set(df_desc["title"].astype(str).tolist()))
    if len(titles) <= int(max_pairs):
        return df_desc

    picked = set(rng.sample(titles, k=int(max_pairs)))
    out = df_desc[df_desc["title"].astype(str).isin(picked)].copy()
    return out.reset_index(drop=True)


def _add_swapped_rows(df_trials: pd.DataFrame) -> pd.DataFrame:
    if df_trials.empty:
        return df_trials

    swapped = df_trials.copy()
    swapped["A_text"], swapped["B_text"] = df_trials["B_text"], df_trials["A_text"]
    swapped["A_source"], swapped["B_source"] = df_trials["B_source"], df_trials["A_source"]
    out = pd.concat([df_trials, swapped], ignore_index=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hc3-dir",
        type=Path,
        default=DATA_DIR / "HC3",
        help="Directory containing HC3 JSONL files.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="",
        help="Optional comma-separated subset stems (defaults to all JSONL files).",
    )
    parser.add_argument(
        "--max-pairs-per-subset",
        type=int,
        default=None,
        help="Optional cap on number of title-pairs per subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for sampling and A/B order.",
    )
    parser.add_argument(
        "--balance-order",
        action="store_true",
        help="Emit both (human,llm) and swapped (llm,human) rows per trial.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DATA_DIR / "derived" / "hc3_trials" / "hc3_trials.csv",
        help="Output trials CSV path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DATA_DIR / "derived" / "hc3_trials" / "summary.json",
        help="Summary JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hc3_dir = Path(args.hc3_dir)
    if not hc3_dir.exists():
        raise SystemExit(f"HC3 directory not found: {hc3_dir}")

    wanted = _parse_csv_list(args.subsets)
    paths = sorted(p for p in hc3_dir.rglob("*.jsonl") if p.is_file())
    if wanted:
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise SystemExit(f"No HC3 JSONL files found under {hc3_dir} for subsets={sorted(wanted) or 'all'}")

    rng = random.Random(int(args.seed))
    all_trials: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for path in paths:
        rows = _iter_jsonl(path)
        if not rows:
            summary_rows.append({"subset": path.stem, "n_json_rows": 0, "n_desc_rows": 0, "n_trials": 0})
            continue

        df_desc = load_hc3_descriptions(rows)
        if df_desc.empty:
            summary_rows.append(
                {"subset": path.stem, "n_json_rows": len(rows), "n_desc_rows": 0, "n_trials": 0}
            )
            continue

        df_desc = _sample_desc_rows(
            df_desc,
            max_pairs=int(args.max_pairs_per_subset) if args.max_pairs_per_subset else None,
            rng=rng,
        )

        df_trials = build_hc3_trials(df_desc, seed=int(args.seed))
        if bool(args.balance_order):
            df_trials = _add_swapped_rows(df_trials)

        if not df_trials.empty:
            df_trials["hc3_subset"] = path.stem
            all_trials.append(df_trials)

        summary_rows.append(
            {
                "subset": path.stem,
                "n_json_rows": int(len(rows)),
                "n_desc_rows": int(len(df_desc)),
                "n_trials": int(len(df_trials)),
            }
        )

    if not all_trials:
        raise SystemExit("No HC3 trials were generated from the provided inputs.")

    out_df = pd.concat(all_trials, ignore_index=True)
    out_df = out_df[
        [
            "item_type",
            "title",
            "question",
            "hc3_source",
            "hc3_subset",
            "A_text",
            "B_text",
            "A_source",
            "B_source",
        ]
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("subset").reset_index(drop=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_json(args.summary_json, orient="records", indent=2)

    print(f"Wrote trials: {args.out_csv} ({len(out_df)} rows)")
    print(f"Wrote summary: {args.summary_json}")


if __name__ == "__main__":
    main()
