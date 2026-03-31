"""Loss functions and schedules for reward-model training."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pref_loss(scores_chosen: torch.Tensor, scores_rejected: torch.Tensor) -> torch.Tensor:
    """Pairwise preference loss: -log σ(r(chosen) - r(rejected))."""
    diff = scores_chosen - scores_rejected
    return (-F.logsigmoid(diff)).mean()


def inv_loss(scores_a: torch.Tensor, scores_b: torch.Tensor) -> torch.Tensor:
    """Style invariance loss: (r(a) - r(b))^2."""
    diff = scores_a - scores_b
    return (diff * diff).mean()


def cue_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cue-presence loss for auxiliary cue heads."""
    logits = logits.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def pointwise_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pointwise regression loss for scalar reward targets."""
    predictions = predictions.float()
    targets = targets.float()
    return F.mse_loss(predictions, targets)


def multi_head_mse_losses(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-head MSE values for a [B, H] prediction tensor."""
    predictions = predictions.float()
    targets = targets.float()
    losses = F.mse_loss(predictions, targets, reduction="none")
    if losses.ndim == 1:
        return losses.mean().reshape(1)
    return losses.mean(dim=0)


def multi_head_bce_losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-head BCE values for a [B, H] binary multi-label tensor."""
    logits = logits.float()
    targets = targets.float()
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if losses.ndim == 1:
        return losses.mean().reshape(1)
    return losses.mean(dim=0)


def group_robust_reduce(group_losses: torch.Tensor, *, strength: float = 0.0) -> torch.Tensor:
    """Mean-plus-max reduction for small group sets.

    `strength=0` returns the plain mean.
    `strength=1` returns the worst-group loss.
    """
    vals = group_losses.float().reshape(-1)
    if vals.numel() == 0:
        raise ValueError("group_losses must be non-empty")
    mean_loss = vals.mean()
    if float(strength) <= 0.0:
        return mean_loss
    worst_loss = vals.max()
    s = min(1.0, max(0.0, float(strength)))
    return mean_loss + s * (worst_loss - mean_loss)


def lambda_schedule(
    step: int,
    *,
    total_steps: int,
    lambda_max: float = 0.5,
    ramp_frac: float = 0.1,
) -> float:
    """Linear ramp for invariance weight, then hold.

    - step in [0, total_steps)
    - ramp over first (ramp_frac * total_steps) steps
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if ramp_frac <= 0:
        return float(lambda_max)
    ramp_steps = max(1, int(math.ceil(float(total_steps) * float(ramp_frac))))
    if step <= 0:
        return 0.0
    if step >= ramp_steps:
        return float(lambda_max)
    return float(lambda_max) * (float(step) / float(ramp_steps))
