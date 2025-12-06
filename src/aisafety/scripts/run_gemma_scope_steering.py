"""Compute Gemma-scope SAE directions and run steered selectors for movie vs paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR
from aisafety.data import DOMAINS, build_all_trials, build_desc_df_from_trials
from aisafety.eval import evaluate_by_domain
from aisafety.features import LinearSAE, load_sae_safetensors
from aisafety.selectors import HFSelectorBackend
from aisafety.steering.gemma_scope import (
    build_feature_splits,
    compute_mean_direction,
    download_gemma_scope_sae,
    load_feature_specs,
    load_feature_texts,
    pair_layers_by_depth,
    slugify_feature,
)


def parse_layers_list(val: str) -> list[int]:
    return [int(v.strip()) for v in val.split(",") if v.strip()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-9b", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--model-27b", type=str, default="google/gemma-2-27b-it")
    p.add_argument("--sae-repo-9b", type=str, default="google/gemma-scope-9b")
    p.add_argument("--sae-repo-27b", type=str, default="google/gemma-scope-27b")
    p.add_argument("--sae-pattern-9b", type=str, default="layer_{layer}.safetensors")
    p.add_argument("--sae-pattern-27b", type=str, default="layer_{layer}.safetensors")
    p.add_argument("--layers-9b", type=parse_layers_list, default=parse_layers_list("7,15,23"))
    p.add_argument("--layers-27b", type=parse_layers_list, default=parse_layers_list("4,12,20"))
    p.add_argument("--encoder-key", type=str, default="encoder.weight")
    p.add_argument("--decoder-key", type=str, default="decoder.weight")
    p.add_argument("--enc-bias-key", type=str, default="encoder.bias")
    p.add_argument("--dec-bias-key", type=str, default="decoder.bias")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--feature-config", type=Path, default=None, help="Optional JSON feature spec.")
    p.add_argument("--feature-sample", type=int, default=400, help="Per-side cap for feature splits.")
    p.add_argument(
        "--feature-texts",
        type=Path,
        default=None,
        help="JSON mapping of feature -> positive/negative texts or file paths to override defaults.",
    )
    p.add_argument("--pool", type=str, choices=["mean", "last", "max"], default="mean")
    p.add_argument("--max-length", type=int, default=1024, help="Token cap for SAE encoding.")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for SAE encoding.")
    p.add_argument("--selector-batch-size", type=int, default=24, help="Batch size for selector runs.")
    p.add_argument("--selector-max-length", type=int, default=2048, help="Max length for selector prompts.")
    p.add_argument("--persona", type=str, choices=["neutral", "human", "ai"], default="neutral")
    p.add_argument("--alpha", type=float, default=2.0, help="Scaling for steering vectors.")
    p.add_argument("--normalize", dest="normalize", action="store_true", help="L2 normalize latent directions.")
    p.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable latent normalization.")
    p.set_defaults(normalize=True)
    p.add_argument("--take-pre-layer", action="store_true", help="Use pre-layer hidden state for SAE.")
    p.add_argument(
        "--include-item-types",
        type=str,
        default="movie,paper",
        help="Comma-separated item types to keep when building trials.",
    )
    p.add_argument("--sample-trials", type=int, default=None, help="Optional subsample for selector trials.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/gemma_scope"))
    p.add_argument("--use-4bit", action="store_true", help="Load selector models in 4-bit quantized mode.")
    return p.parse_args()


def _load_trials(include_item_types: Sequence[str], seed: int):
    df_trials = build_all_trials(DOMAINS, seed=seed)
    df_trials = df_trials[df_trials["item_type"].isin(include_item_types)].reset_index(drop=True)
    return df_trials


def _prep_feature_splits(df_desc, args):
    specs = load_feature_specs(args.feature_config)
    external = load_feature_texts(args.feature_texts)
    return build_feature_splits(
        df_desc=df_desc,
        specs=specs,
        sample_per_side=args.feature_sample,
        seed=args.seed,
        external_texts=external,
    )


def _load_sae(repo: str, layer_idx: int, cache_dir: Path, pattern: str, args) -> LinearSAE:
    sae_path = download_gemma_scope_sae(repo, layer_idx=layer_idx, cache_dir=cache_dir, filename_pattern=pattern)
    weights = load_sae_safetensors(
        sae_path,
        encoder_key=args.encoder_key,
        decoder_key=args.decoder_key,
        enc_bias_key=args.enc_bias_key,
        dec_bias_key=args.dec_bias_key,
        layer_idx=layer_idx,
    )
    return LinearSAE(weights)


def _save_vector(path: Path, vec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vec)


def main():
    args = parse_args()
    item_types = [t.strip() for t in args.include_item_types.split(",") if t.strip()]
    df_trials = _load_trials(item_types, seed=args.seed)
    if args.sample_trials and len(df_trials) > args.sample_trials:
        df_trials = df_trials.sample(args.sample_trials, random_state=args.seed).reset_index(drop=True)
    df_desc = build_desc_df_from_trials(df_trials)

    feature_splits = _prep_feature_splits(df_desc, args)

    backend_9b = HFSelectorBackend(
        model_id=args.model_9b, cache_dir=args.cache_dir, use_4bit=args.use_4bit
    )
    backend_27b = HFSelectorBackend(
        model_id=args.model_27b, cache_dir=args.cache_dir, use_4bit=args.use_4bit
    )

    n_layers_9b = len(backend_9b.model.model.layers)
    n_layers_27b = len(backend_27b.model.model.layers)
    layer_pairs = pair_layers_by_depth(
        reference_layers=args.layers_9b,
        reference_total=n_layers_9b,
        limited_layers=args.layers_27b,
        limited_total=n_layers_27b,
    )

    pair_9_to_27 = {p[0]: p[1] for p in layer_pairs}
    pair_27_to_9 = {p[1]: p[0] for p in layer_pairs}
    print(f"Matched layers (9b -> 27b): {layer_pairs}")

    summary_rows = []

    def run_model(
        alias: str,
        backend: HFSelectorBackend,
        sae_repo: str,
        sae_pattern: str,
        layers_to_run: Sequence[int],
    ):
        for layer_idx in layers_to_run:
            sae = _load_sae(sae_repo, layer_idx, args.cache_dir, sae_pattern, args)
            # Move SAE to model device/dtype to avoid extra casting.
            sae = sae.to(device=backend.model.device, dtype=backend.model.dtype)
            for split in feature_splits:
                latent_vec, hidden_vec = compute_mean_direction(
                    model=backend.model,
                    tokenizer=backend.tokenizer,
                    sae=sae,
                    split=split,
                    layer_idx=layer_idx,
                    pool=args.pool,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    normalize=args.normalize,
                    take_post_layer=not args.take_pre_layer,
                )
                slug = slugify_feature(split.spec.name)
                dir_base = args.out_dir / alias / "directions"
                _save_vector(dir_base / f"{slug}_L{layer_idx}_latent.npy", latent_vec)
                _save_vector(dir_base / f"{slug}_L{layer_idx}_hidden.npy", hidden_vec.detach().cpu().numpy())

                steer_cfg = {"layer_idx": layer_idx, "vec": hidden_vec, "alpha": args.alpha}
                selector_out = backend.run(
                    df_trials=df_trials,
                    persona=args.persona,
                    batch_size=args.selector_batch_size,
                    max_length=args.selector_max_length,
                    steer_config=steer_cfg,
                )
                out_path = args.out_dir / alias / "selectors" / f"{slug}_L{layer_idx}.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                selector_out.to_csv(out_path, index=False)

                stats = evaluate_by_domain(selector_out)
                summary_rows.append(
                    {
                        "model_alias": alias,
                        "model_id": backend.model_id,
                        "sae_repo": sae_repo,
                        "layer_idx": layer_idx,
                        "feature": split.spec.name,
                        "alpha": args.alpha,
                        "feature_pos": len(split.positive_texts),
                        "feature_neg": len(split.negative_texts),
                        "selector_rows": len(selector_out),
                        "matched_layer_partner": pair_27_to_9.get(layer_idx)
                        if alias.endswith("27b")
                        else pair_9_to_27.get(layer_idx),
                        "stats_json": json.dumps(stats),
                        "results_path": str(out_path),
                        "latent_path": str(dir_base / f"{slug}_L{layer_idx}_latent.npy"),
                        "hidden_path": str(dir_base / f"{slug}_L{layer_idx}_hidden.npy"),
                    }
                )

    layers_9b = sorted({p[0] for p in layer_pairs})
    layers_27b = sorted({p[1] for p in layer_pairs})

    run_model("gemma_scope_9b", backend_9b, args.sae_repo_9b, args.sae_pattern_9b, layers_9b)
    run_model("gemma_scope_27b", backend_27b, args.sae_repo_27b, args.sae_pattern_27b, layers_27b)

    summary_path = args.out_dir / "steering_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote {len(summary_rows)} steering runs to {summary_path}")


if __name__ == "__main__":
    main()
