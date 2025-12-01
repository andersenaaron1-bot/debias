"""Steering utilities that map SAE latents back to hidden space."""

from __future__ import annotations

import numpy as np
import torch

from aisafety.steering import register_vector_hook


def latent_vector_to_hidden(decoder: torch.Tensor, latents: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(latents, np.ndarray):
        l = torch.from_numpy(latents)
    else:
        l = latents
    return torch.matmul(l, decoder.T)


def register_sae_latent_hook(model, decoder: torch.Tensor, latent_mask, layer_idx: int, alpha: float = 1.0):
    hidden_vec = latent_vector_to_hidden(decoder, latent_mask)
    return register_vector_hook(model=model, layer_idx=layer_idx, vec=hidden_vec, alpha=alpha)
