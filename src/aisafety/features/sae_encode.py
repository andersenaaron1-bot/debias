"""Encode texts into SAE latents using a HF model."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch
from tqdm import trange
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisafety.features.sae_loader import LinearSAE


PoolMode = Literal["mean", "last", "max"]


@torch.inference_mode()
def encode_with_sae(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sae: LinearSAE,
    texts: Sequence[str],
    layer_idx: int,
    batch_size: int = 8,
    max_length: int = 1024,
    pool: PoolMode = "mean",
    take_post_layer: bool = True,
) -> np.ndarray:
    """
    Return latent activations shaped (N, latent_dim) pooled per text.
    layer_idx refers to the transformer layer index; take_post_layer=True uses the output of that layer.
    """
    device = model.device
    outs = []
    for i in trange(0, len(texts), batch_size, desc="SAE encode"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        hs = out.hidden_states
        idx = layer_idx + 1 if take_post_layer else layer_idx
        h_layer = hs[idx]  # (batch, seq, hidden)
        z = sae.encode(h_layer)  # (batch, seq, latent)
        mask = enc["attention_mask"]

        if pool == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (z * mask.unsqueeze(-1)).sum(dim=1) / denom
        elif pool == "last":
            last = mask.sum(dim=1) - 1
            pooled = z[torch.arange(z.shape[0], device=device), last]
        elif pool == "max":
            masked = z.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
            pooled = masked.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool mode: {pool}")

        outs.append(pooled.cpu().numpy())
        del out, enc
        torch.cuda.empty_cache()

    return np.concatenate(outs, axis=0)
