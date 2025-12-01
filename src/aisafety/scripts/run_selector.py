"""Run the selector model on local trials with optional steering."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_ID, DEFAULT_SEED
from aisafety.data import DOMAINS, build_all_trials
from aisafety.eval import evaluate_by_domain
from aisafety.selectors import build_selector_backend


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, choices=["hf", "openrouter"], default="hf")
    p.add_argument("--trials-csv", type=Path, default=None, help="Existing trials CSV; build if omitted.")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Model id for the selected backend.")
    p.add_argument("--api-key", type=str, default=None, help="API key for OpenRouter (otherwise uses env).")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--persona", type=str, choices=["neutral", "human", "ai"], default="neutral")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-desc-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--use-4bit", action="store_true", help="Load model with 4-bit quantization.")
    p.add_argument("--steer-vector", type=Path, default=None, help="Path to .npy steering vector (HF backend only).")
    p.add_argument("--steer-layer", type=int, default=None, help="Layer index for steering vector (HF backend only).")
    p.add_argument("--steer-alpha", type=float, default=0.0, help="Scaling factor for steering vector (HF backend only).")
    p.add_argument("--out", type=Path, default=Path("artifacts/selector_results.csv"))
    return p.parse_args()


def build_trials_if_needed(path: Path | None, seed: int):
    if path is not None and path.exists():
        return pd.read_csv(path)
    df = build_all_trials(DOMAINS, seed=seed)
    return df


def main():
    args = parse_args()
    df_trials = build_trials_if_needed(args.trials_csv, seed=args.seed)

    backend = build_selector_backend(
        backend=args.backend,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        use_4bit=args.use_4bit,
        api_key=args.api_key,
    )

    steer_config = None
    if args.backend == "hf" and args.steer_vector and args.steer_layer is not None:
        v = np.load(args.steer_vector)
        steer_config = {"layer_idx": args.steer_layer, "vec": v, "alpha": args.steer_alpha}

    results = backend.run(
        df_trials,
        persona=args.persona,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_desc_tokens=args.max_desc_tokens,
        steer_config=steer_config,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(f"Wrote {len(results)} selector rows to {args.out}")

    stats = evaluate_by_domain(results)
    for dom, m in stats.items():
        print(dom, m)


if __name__ == "__main__":
    main()
