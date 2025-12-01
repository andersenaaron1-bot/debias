"""Run selector with SAE-based steering vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_ID, DEFAULT_SEED
from aisafety.data import DOMAINS, build_all_trials
from aisafety.eval import evaluate_by_domain
from aisafety.features import LinearSAE, load_sae_safetensors
from aisafety.selectors import HFSelectorBackend
from aisafety.steering.sae import latent_vector_to_hidden


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials-csv", type=Path, default=None, help="Existing trials CSV; build if omitted.")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--persona", type=str, choices=["neutral", "human", "ai"], default="neutral")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sae-weights", type=Path, required=True)
    p.add_argument("--sae-layer", type=int, required=True)
    p.add_argument("--latent-mask", type=Path, required=True, help="Path to .npy vector of latent weights.")
    p.add_argument("--alpha", type=float, default=1.0, help="Scaling for steering contribution.")
    p.add_argument("--pool", type=str, choices=["mean", "last", "max"], default="mean")
    p.add_argument("--out", type=Path, default=Path("artifacts/selector_results_sae.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    backend = HFSelectorBackend(model_id=args.model_id, cache_dir=args.cache_dir, use_4bit=False)
    df_trials = build_all_trials(DOMAINS, seed=args.seed) if args.trials_csv is None else None
    if args.trials_csv and args.trials_csv.exists():
        import pandas as pd

        df_trials = pd.read_csv(args.trials_csv)

    sae_weights = load_sae_safetensors(args.sae_weights, layer_idx=args.sae_layer)
    sae = LinearSAE(sae_weights, device=backend.model.device, dtype=backend.model.dtype)

    latent_mask = np.load(args.latent_mask)
    hidden_vec = latent_vector_to_hidden(decoder=sae.decoder, latents=latent_mask)

    steer_config = {"layer_idx": args.sae_layer, "vec": hidden_vec, "alpha": args.alpha}

    results = backend.run(
        df_trials=df_trials,
        persona=args.persona,
        batch_size=args.batch_size,
        max_length=args.max_length,
        steer_config=steer_config,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(f"Wrote {len(results)} rows to {args.out}")

    stats = evaluate_by_domain(results)
    for dom, m in stats.items():
        print(dom, m)


if __name__ == "__main__":
    main()
