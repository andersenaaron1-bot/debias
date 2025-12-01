"""Shared configuration defaults for the AI-AI bias project."""

from __future__ import annotations

from pathlib import Path

# Resolve project root relative to this file (src/aisafety/config.py -> project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

# Hugging Face model/cache defaults; override via environment or CLI args.
DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "hf"

# Default random seed for reproducibility in sampling experiments.
DEFAULT_SEED = 1234
