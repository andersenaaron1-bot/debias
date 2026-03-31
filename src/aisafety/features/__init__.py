"""Feature helpers for reward-adapter analysis and surface-cue discovery."""

from .surface_cues import SURFACE_FEATURE_NAMES, extract_surface_features
from .token_positions import last_non_pad_index, take_last_token

__all__ = [
    "SURFACE_FEATURE_NAMES",
    "extract_surface_features",
    "last_non_pad_index",
    "take_last_token",
]
