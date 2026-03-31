"""Utilities for selecting token positions from padded batches."""

from __future__ import annotations

import torch


def last_non_pad_index(attention_mask: torch.Tensor, *, padding_side: str) -> torch.Tensor:
    """Return the index of the final non-padding token for each sequence.

    Args:
        attention_mask: Tensor shaped (batch, seq) with 1 for tokens and 0 for padding.
        padding_side: "left" or "right".

    Returns:
        Long tensor shaped (batch,) of indices into the seq dimension.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be 2D (batch, seq), got shape {tuple(attention_mask.shape)}")

    side = str(padding_side or "right").lower()
    if side not in {"left", "right"}:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")

    batch, seq = attention_mask.shape
    if seq <= 0:
        raise ValueError("attention_mask has empty seq dimension")

    if side == "left":
        # Left padding right-aligns tokens, so the last token is always at the final position.
        return torch.full((batch,), seq - 1, device=attention_mask.device, dtype=torch.long)

    lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
    lengths = torch.clamp(lengths, min=1)
    return lengths - 1


def take_last_token(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor, *, padding_side: str
) -> torch.Tensor:
    """Select the last non-padding token vector from hidden states.

    Args:
        hidden_states: Tensor shaped (batch, seq, dim).
        attention_mask: Tensor shaped (batch, seq).
        padding_side: "left" or "right".

    Returns:
        Tensor shaped (batch, dim).
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must be 3D (batch, seq, dim), got shape {tuple(hidden_states.shape)}"
        )
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must be 2D (batch, seq), got shape {tuple(attention_mask.shape)}"
        )
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError(
            "hidden_states and attention_mask must match in (batch, seq); "
            f"got {tuple(hidden_states.shape[:2])} vs {tuple(attention_mask.shape)}"
        )

    side = str(padding_side or "right").lower()
    if side == "left":
        return hidden_states[:, -1, :]

    idx = last_non_pad_index(attention_mask, padding_side=side)
    batch = hidden_states.shape[0]
    return hidden_states[torch.arange(batch, device=hidden_states.device), idx]

