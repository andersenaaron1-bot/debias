"""Low-rank and hook helpers for LM judge decision-state patching."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from aisafety.mech.d4_io import sha1_hex


def deterministic_fit_mask(
    counterfactual_ids: list[str],
    *,
    fit_frac: float,
    seed: int,
) -> np.ndarray:
    """Split by counterfactual id so order swaps cannot leak across splits."""

    frac = min(max(float(fit_frac), 0.05), 0.95)
    unique_ids = sorted(set(map(str, counterfactual_ids)), key=lambda item: sha1_hex(f"{seed}:judge-patch:{item}"))
    n_fit = max(1, min(len(unique_ids) - 1, int(round(len(unique_ids) * frac)))) if len(unique_ids) > 1 else 1
    fit_ids = set(unique_ids[:n_fit])
    return np.asarray([str(item) in fit_ids for item in counterfactual_ids], dtype=bool)


def normalized_recovery(
    patched: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return source-effect recovery after patching source state into target."""

    patched = np.asarray(patched, dtype=float)
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    denom = source - target
    return np.divide(
        patched - target,
        denom,
        out=np.full_like(denom, np.nan, dtype=float),
        where=np.abs(denom) > float(eps),
    )


def fit_low_rank_basis(
    deltas: np.ndarray,
    *,
    rank: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Fit an orthonormal row basis from source-minus-target state deltas."""

    matrix = np.asarray(deltas, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("deltas must have shape [n_rows, hidden_size].")
    if len(matrix) < 1:
        return np.zeros((0, matrix.shape[-1]), dtype=np.float32)
    _u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    keep = int(min(max(int(rank), 0), int(np.sum(singular > float(eps)))))
    if keep <= 0:
        return np.zeros((0, matrix.shape[-1]), dtype=np.float32)
    return vh[:keep].astype(np.float32)


def suppress_subspace(
    states: torch.Tensor,
    *,
    basis_rows: torch.Tensor,
    center: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Damp the centered projection of states into a row-wise basis."""

    if basis_rows.numel() == 0 or float(alpha) == 0.0:
        return states
    centered = states - center
    projection = (centered @ basis_rows.T) @ basis_rows
    return states - float(alpha) * projection


def replace_decision_positions(
    hidden: torch.Tensor,
    *,
    positions: torch.Tensor,
    replacements: torch.Tensor,
) -> torch.Tensor:
    """Return hidden states with one residual position replaced per batch row."""

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, seq, hidden].")
    out = hidden.clone()
    row_idx = torch.arange(hidden.shape[0], device=hidden.device)
    out[row_idx, positions] = replacements
    return out


def replace_span_positions(
    hidden: torch.Tensor,
    *,
    span_positions: list[list[int]],
    replacements: torch.Tensor,
) -> torch.Tensor:
    """Replace every selected span token with a pooled replacement state."""

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, seq, hidden].")
    out = hidden.clone()
    for row_idx, positions in enumerate(span_positions):
        if positions:
            out[row_idx, positions] = replacements[row_idx]
    return out


def _replace_output_hidden(output: Any, edited: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return edited
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (edited, *output[1:])
    raise TypeError("Expected tensor or tuple decoder-layer output.")


class DecoderOutputPatchHook:
    """Patch decoder-block residual output at a decision position or span."""

    def __init__(
        self,
        *,
        positions: torch.Tensor,
        replacements: torch.Tensor,
        span_positions: list[list[int]] | None = None,
    ):
        self.positions = positions
        self.replacements = replacements
        self.span_positions = span_positions

    def __call__(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        replacements = self.replacements.to(device=hidden.device, dtype=hidden.dtype)
        if self.span_positions is None:
            edited = replace_decision_positions(
                hidden,
                positions=self.positions.to(device=hidden.device),
                replacements=replacements,
            )
        else:
            edited = replace_span_positions(
                hidden,
                span_positions=self.span_positions,
                replacements=replacements,
            )
        return _replace_output_hidden(output, edited)


class DecoderOutputSuppressionHook:
    """Damp a fitted residual subspace at the final comparison position."""

    def __init__(
        self,
        *,
        positions: torch.Tensor,
        basis_rows: torch.Tensor,
        center: torch.Tensor,
        alpha: float,
    ):
        self.positions = positions
        self.basis_rows = basis_rows
        self.center = center
        self.alpha = float(alpha)

    def __call__(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        row_idx = torch.arange(hidden.shape[0], device=hidden.device)
        positions = self.positions.to(device=hidden.device)
        selected = hidden[row_idx, positions]
        damped = suppress_subspace(
            selected,
            basis_rows=self.basis_rows.to(device=hidden.device, dtype=hidden.dtype),
            center=self.center.to(device=hidden.device, dtype=hidden.dtype),
            alpha=float(self.alpha),
        )
        edited = hidden.clone()
        edited[row_idx, positions] = damped
        return _replace_output_hidden(output, edited)


class MlpDecisionPatchHook:
    """Replace an MLP output vector at the final comparison position."""

    def __init__(self, *, positions: torch.Tensor, replacements: torch.Tensor):
        self.positions = positions
        self.replacements = replacements

    def __call__(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        edited = replace_decision_positions(
            hidden,
            positions=self.positions.to(device=hidden.device),
            replacements=self.replacements.to(device=hidden.device, dtype=hidden.dtype),
        )
        return _replace_output_hidden(output, edited)


class AttentionHeadDecisionPatchPreHook:
    """Replace one attention-head chunk before the attention output projection."""

    def __init__(
        self,
        *,
        positions: torch.Tensor,
        replacements: torch.Tensor,
        head_index: int,
        head_dim: int,
    ):
        self.positions = positions
        self.replacements = replacements
        self.head_index = int(head_index)
        self.head_dim = int(head_dim)

    def __call__(self, _module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return inputs
        hidden = inputs[0]
        start = int(self.head_index * self.head_dim)
        stop = int(start + self.head_dim)
        edited = hidden.clone()
        row_idx = torch.arange(hidden.shape[0], device=hidden.device)
        edited[row_idx, self.positions.to(device=hidden.device), start:stop] = self.replacements.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )[:, start:stop]
        return (edited, *inputs[1:])
