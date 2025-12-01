"""Helpers to load linear SAEs and encode/decoder weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open


@dataclass
class SAEWeights:
    encoder: torch.Tensor  # shape (latent, hidden)
    decoder: torch.Tensor  # shape (hidden, latent)
    enc_bias: Optional[torch.Tensor] = None  # shape (latent,)
    dec_bias: Optional[torch.Tensor] = None  # shape (hidden,)
    layer_idx: int | None = None

    @property
    def latent_size(self) -> int:
        return int(self.encoder.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.shape[1])


class LinearSAE(torch.nn.Module):
    """Minimal linear SAE wrapper with encode/decode."""

    def __init__(self, weights: SAEWeights, device: torch.device | None = None, dtype=None):
        super().__init__()
        d = device
        dt = dtype
        self.encoder = torch.nn.Parameter(weights.encoder.to(device=d, dtype=dt), requires_grad=False)
        self.decoder = torch.nn.Parameter(weights.decoder.to(device=d, dtype=dt), requires_grad=False)
        self.enc_bias = (
            torch.nn.Parameter(weights.enc_bias.to(device=d, dtype=dt), requires_grad=False)
            if weights.enc_bias is not None
            else None
        )
        self.dec_bias = (
            torch.nn.Parameter(weights.dec_bias.to(device=d, dtype=dt), requires_grad=False)
            if weights.dec_bias is not None
            else None
        )

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, seq, hidden)
        z = torch.einsum("bsh,lh->bsl", h, self.encoder)
        if self.enc_bias is not None:
            z = z + self.enc_bias
        return torch.relu(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, seq, latent)
        h = torch.einsum("bsl,hl->bsh", z, self.decoder)
        if self.dec_bias is not None:
            h = h + self.dec_bias
        return h


def load_sae_safetensors(
    path: str | Path,
    encoder_key: str = "encoder.weight",
    decoder_key: str = "decoder.weight",
    enc_bias_key: str | None = "encoder.bias",
    dec_bias_key: str | None = "decoder.bias",
    layer_idx: int | None = None,
) -> SAEWeights:
    """Load SAE weights from a safetensors file with configurable key names."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SAE file not found: {p}")
    tensors = {}
    with safe_open(p, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    try:
        enc = tensors[encoder_key]
        dec = tensors[decoder_key]
    except KeyError as e:
        raise KeyError(f"Missing key in safetensors: {e} (available: {list(tensors)})") from e

    enc_b = tensors.get(enc_bias_key) if enc_bias_key else None
    dec_b = tensors.get(dec_bias_key) if dec_bias_key else None

    return SAEWeights(encoder=enc, decoder=dec, enc_bias=enc_b, dec_bias=dec_b, layer_idx=layer_idx)
