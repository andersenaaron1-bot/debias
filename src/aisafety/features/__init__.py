"""Feature extraction and scoring utilities."""

from .encoders import encode_last_token_hf
from .sae_loader import LinearSAE, SAEWeights, load_sae_safetensors
from .sae_encode import encode_with_sae
from .stylometry import compute_features, FEATURE_REGISTRY
from .metrics import (
    build_choice_indices,
    layerwise_logreg_auc,
    layerwise_choice_auc,
    logreg_auc_single,
)

__all__ = [
    "encode_last_token_hf",
    "build_choice_indices",
    "layerwise_logreg_auc",
    "layerwise_choice_auc",
    "logreg_auc_single",
    "LinearSAE",
    "SAEWeights",
    "load_sae_safetensors",
    "encode_with_sae",
    "compute_features",
    "FEATURE_REGISTRY",
]
