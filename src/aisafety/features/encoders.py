"""Encode last token."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import trange


@torch.inference_mode()
def encode_last_token_hf(model, tokenizer, texts, batch_size: int = 16, max_length: int = 1024):
    """
    Return array shaped (N, L, D) where L = num layers, D = hidden size.
    Uses the final token per sequence for each layer.
    """
    feats = []
    device = model.device
    for i in trange(0, len(texts), batch_size, desc="Encoding (HF)"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        hs = out.hidden_states
        attn = enc["attention_mask"]
        last = attn.sum(dim=1) - 1
        layer_vecs = []
        for l in range(1, len(hs)):
            h = hs[l]
            v = h[torch.arange(h.shape[0], device=device), last]
            layer_vecs.append(v.float().cpu().numpy())
        feats.append(np.stack(layer_vecs, axis=0).transpose(1, 0, 2))
        del out, enc
        torch.cuda.empty_cache()
    X = np.concatenate(feats, axis=0)
    return X
