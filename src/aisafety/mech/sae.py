"""SAE loading, mapping, and feature aggregation helpers."""

from __future__ import annotations

from typing import Any

import torch

from aisafety.features.token_positions import take_last_token


def hidden_layer_to_sae_layer(hidden_layer: int) -> int:
    """Map HF hidden_states index to Gemma Scope residual block index."""

    hidden_layer = int(hidden_layer)
    if hidden_layer <= 0:
        raise ValueError("hidden_layer must be >= 1 because hidden_states[0] is embeddings.")
    return hidden_layer - 1


def format_sae_id(template: str, *, hidden_layer: int) -> str:
    """Format an SAE id template with hidden and SAE layer ids."""

    sae_layer = hidden_layer_to_sae_layer(hidden_layer)
    return str(template).format(hidden_layer=int(hidden_layer), sae_layer=int(sae_layer))


def load_sae(*, release: str, sae_id: str, device: torch.device | str):
    """Load an SAE through sae-lens."""

    try:
        from sae_lens import SAE
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sae-lens is required for D4 SAE analysis. Install it with "
            "`pip install sae-lens` or add the repo optional dependency `.[mech]`."
        ) from exc

    loaded = SAE.from_pretrained(release=str(release), sae_id=str(sae_id), device=str(device))
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae.eval()
    return sae


def sae_d_sae(sae: Any) -> int:
    """Infer the SAE feature width."""

    cfg = getattr(sae, "cfg", None)
    if cfg is not None and hasattr(cfg, "d_sae"):
        return int(getattr(cfg, "d_sae"))
    if hasattr(sae, "W_dec"):
        return int(getattr(sae, "W_dec").shape[0])
    raise RuntimeError("Could not infer SAE width.")


def sae_encode(sae: Any, x: torch.Tensor) -> torch.Tensor:
    """Encode hidden activations into SAE feature activations."""

    acts = sae.encode(x)
    if isinstance(acts, tuple):
        acts = acts[0]
    return acts.to(dtype=torch.float32)


@torch.inference_mode()
def aggregate_sae_features(
    *,
    sae: Any,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    padding_side: str,
    aggregation: str,
    token_chunk_size: int,
) -> torch.Tensor:
    """Aggregate token-level SAE activations to one feature vector per text."""

    hidden = hidden.detach().to(dtype=torch.float32)
    aggregation = str(aggregation)
    if aggregation == "last":
        pooled = take_last_token(hidden, attention_mask, padding_side=padding_side)
        return sae_encode(sae, pooled)

    if aggregation != "max":
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    batch, seq, _dim = hidden.shape
    d_sae = sae_d_sae(sae)
    valid = attention_mask.to(dtype=torch.bool)
    flat_hidden = hidden.reshape(batch * seq, hidden.shape[-1])[valid.reshape(-1)]
    flat_ids = (
        torch.arange(batch, device=hidden.device)
        .unsqueeze(1)
        .expand(batch, seq)
        .reshape(-1)[valid.reshape(-1)]
    )
    out = torch.full((batch, d_sae), -torch.inf, device=hidden.device, dtype=torch.float32)
    chunk_size = max(1, int(token_chunk_size))
    for start in range(0, flat_hidden.shape[0], chunk_size):
        end = min(flat_hidden.shape[0], start + chunk_size)
        acts = sae_encode(sae, flat_hidden[start:end])
        ids = flat_ids[start:end]
        for item_id in torch.unique(ids).tolist():
            mask = ids == int(item_id)
            out[int(item_id)] = torch.maximum(out[int(item_id)], acts[mask].amax(dim=0))
    out[~torch.isfinite(out)] = 0.0
    return out

