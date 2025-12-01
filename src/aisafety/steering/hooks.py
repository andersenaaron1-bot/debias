"""Helper to register simple vector steering hooks on HF models."""

from __future__ import annotations

import torch
from torch.nn import Module


def register_vector_hook(model: Module, layer_idx: int, vec, alpha: float = 0.0):
    """
    Register a forward hook that adds alpha * vec to the selected layer output.
    Returns the hook handle so callers can remove it after use.
    """
    v_t = torch.tensor(vec, device=model.device)

    def steering_hook(module, input, output):
        v_cast = v_t.to(output.dtype)
        return output + alpha * v_cast

    return model.model.layers[layer_idx].register_forward_hook(steering_hook)
