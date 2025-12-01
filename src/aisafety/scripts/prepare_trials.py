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
        "--out",
        type=Path,
        default=Path("artifacts/trials.csv"),
        help="Where to write the compiled trials CSV.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = build_all_trials(DOMAINS, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} trials to {args.out}")


if __name__ == "__main__":
    main()
