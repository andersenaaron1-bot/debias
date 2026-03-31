"""Build A/B trials from local human vs. LLM corpora."""

from __future__ import annotations

import argparse
from pathlib import Path

from aisafety.config import DEFAULT_SEED
from aisafety.data import DOMAINS, build_all_trials


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for shuffling pairs.",
    )
    p.add_argument(
        "--include-item-types",
        type=str,
        default="movie,paper,product",
        help="Comma-separated item types to include from the Laurito datasets.",
    )
    p.add_argument(
        "--balance-order",
        dest="balance_order",
        action="store_true",
        help="Emit paired trials with human first and second to control position bias.",
    )
    p.add_argument(
        "--no-balance-order",
        dest="balance_order",
        action="store_false",
        help="Disable paired ordering.",
    )
    p.set_defaults(balance_order=True)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/trials.csv"),
        help="Where to write the compiled trials CSV.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    include_types = [t.strip() for t in args.include_item_types.split(",") if t.strip()]
    df = build_all_trials(DOMAINS, seed=args.seed, balance_order=args.balance_order)
    if include_types:
        df = df[df["item_type"].isin(include_types)].reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} trials to {args.out}")


if __name__ == "__main__":
    main()
