"""Score SAE latents against stylometric features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_ID
from aisafety.data import DOMAINS, build_all_trials
from aisafety.features import (
    FEATURE_REGISTRY,
    LinearSAE,
    compute_features,
    encode_with_sae,
    layerwise_logreg_auc,
    load_sae_safetensors,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--sae-weights", type=Path, required=True, help="Path to SAE safetensors file.")
    p.add_argument("--sae-layer", type=int, required=True, help="Layer index SAE is trained on.")
    p.add_argument("--encoder-key", type=str, default="encoder.weight")
    p.add_argument("--decoder-key", type=str, default="decoder.weight")
    p.add_argument("--enc-bias-key", type=str, default="encoder.bias")
    p.add_argument("--dec-bias-key", type=str, default="decoder.bias")
    p.add_argument("--pool", type=str, choices=["mean", "last", "max"], default="mean")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sample", type=int, default=500, help="Sample size from trials to keep runtime manageable.")
    p.add_argument("--out", type=Path, default=Path("artifacts/sae_feature_scores.csv"))
    return p.parse_args()


def load_model(model_id: str, cache_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=str(cache_dir),
        attn_implementation="sdpa",
    )
    model.eval()
    return tokenizer, model


def main():
    args = parse_args()
    tokenizer, model = load_model(args.model_id, args.cache_dir)

    df_trials = build_all_trials(DOMAINS, seed=1234)
    texts = pd.concat([df_trials["A_text"], df_trials["B_text"]], ignore_index=True)
    if args.sample and len(texts) > args.sample:
        texts = texts.sample(args.sample, random_state=42).reset_index(drop=True)

    sae_weights = load_sae_safetensors(
        args.sae_weights,
        encoder_key=args.encoder_key,
        decoder_key=args.decoder_key,
        enc_bias_key=args.enc_bias_key,
        dec_bias_key=args.dec_bias_key,
        layer_idx=args.sae_layer,
    )
    sae = LinearSAE(sae_weights, device=model.device, dtype=torch.bfloat16)

    Z = encode_with_sae(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        texts=texts.tolist(),
        layer_idx=args.sae_layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pool=args.pool,
    )

    df_feats = compute_features(texts.tolist(), features=FEATURE_REGISTRY)
    scores = []
    for feat_name in df_feats.columns:
        y = df_feats[feat_name].to_numpy()
        aucs, coefs = layerwise_logreg_auc(Z[:, None, :], y > np.median(y), n_splits=5, seed=0)
        scores.append(
            {
                "feature": feat_name,
                "latent_dim": Z.shape[1],
                "layer": args.sae_layer,
                "auc_mean": float(aucs[0]),
            }
        )

    out_df = pd.DataFrame(scores)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} feature scores to {args.out}")


if __name__ == "__main__":
    main()
