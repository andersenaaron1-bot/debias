"""Readout-space surface-cue nulling helpers."""

from __future__ import annotations

from typing import Any

import torch


def orthonormal_basis(directions: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Return an orthonormal column basis for row-wise directions."""

    if directions.ndim != 2:
        raise ValueError("directions must have shape [n_directions, hidden_size].")
    if directions.numel() == 0:
        return torch.zeros((directions.shape[-1], 0), dtype=directions.dtype, device=directions.device)
    rows = []
    for row in directions:
        norm = torch.linalg.vector_norm(row)
        if float(norm.detach().cpu()) > float(eps):
            rows.append(row / norm)
    if not rows:
        return torch.zeros((directions.shape[-1], 0), dtype=directions.dtype, device=directions.device)
    mat = torch.stack(rows, dim=1)  # [hidden, n]
    q, r = torch.linalg.qr(mat, mode="reduced")
    diag = torch.abs(torch.diagonal(r))
    keep = diag > float(eps)
    if not bool(torch.any(keep)):
        return torch.zeros((directions.shape[-1], 0), dtype=directions.dtype, device=directions.device)
    return q[:, keep]


def project_out(pooled: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove the subspace spanned by basis columns from pooled states."""

    if basis.numel() == 0:
        return pooled
    if basis.ndim != 2:
        raise ValueError("basis must have shape [hidden_size, n_directions].")
    if pooled.shape[-1] != basis.shape[0]:
        raise ValueError("pooled hidden size does not match basis.")
    return pooled - (pooled @ basis) @ basis.T


@torch.no_grad()
def encode_pooled_texts(
    *,
    scorer: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode texts into RewardScorer pooled states."""

    outputs: list[torch.Tensor] = []
    scorer.eval()
    for start in range(0, len(texts), int(batch_size)):
        batch = texts[start : start + int(batch_size)]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        pooled = scorer.encode(enc["input_ids"], enc["attention_mask"])
        outputs.append(pooled.detach().cpu())
        del enc, pooled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not outputs:
        hidden_size = int(getattr(scorer, "hidden_size", 0))
        return torch.zeros((0, hidden_size), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def score_pooled(
    *,
    scorer: Any,
    pooled: torch.Tensor,
    basis: torch.Tensor | None = None,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return original and null-projected scores from pooled states."""

    if device is None:
        try:
            device = next(p for p in scorer.parameters() if p.device.type != "meta").device
        except StopIteration:
            device = torch.device("cpu")
    original_chunks: list[torch.Tensor] = []
    null_chunks: list[torch.Tensor] = []
    scorer.eval()
    basis_device = None if basis is None else basis.to(device=device, dtype=scorer.value_head.weight.dtype)
    for start in range(0, pooled.shape[0], int(batch_size)):
        chunk = pooled[start : start + int(batch_size)].to(device=device, dtype=scorer.value_head.weight.dtype)
        original = scorer.score_from_pooled(chunk).detach().cpu()
        if basis_device is None or basis_device.numel() == 0:
            nulled = chunk
        else:
            nulled = project_out(chunk, basis_device)
        null_score = scorer.score_from_pooled(nulled).detach().cpu()
        original_chunks.append(original)
        null_chunks.append(null_score)
        del chunk, original, nulled, null_score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not original_chunks:
        empty = torch.zeros((0,), dtype=torch.float32)
        return empty, empty
    return torch.cat(original_chunks, dim=0), torch.cat(null_chunks, dim=0)
