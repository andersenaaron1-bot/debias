"""Selector prompting and execution utilities."""

from .prompts import make_selector_prompt, system_prompt_for_persona
from .chat import build_chat_batch
from .runner import ABOnlyProcessor, run_selector_ab
from .backends import (
    SelectorBackend,
    HFSelectorBackend,
    OpenRouterSelectorBackend,
    build_selector_backend,
)

__all__ = [
    "make_selector_prompt",
    "system_prompt_for_persona",
    "build_chat_batch",
    "ABOnlyProcessor",
    "run_selector_ab",
    "SelectorBackend",
    "HFSelectorBackend",
    "OpenRouterSelectorBackend",
    "build_selector_backend",
]
