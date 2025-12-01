"""Domain configurations for human vs. LLM corpora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from aisafety.config import DATA_DIR


@dataclass(frozen=True)
class DomainConfig:
    """Locations for paired human/LLM data within the repo."""

    item_type: str
    human_dir: Path
    llm_dir: Path
    prompt_key: str | None = None

    def exists(self) -> bool:
        return self.human_dir.is_dir() and self.llm_dir.is_dir()


DOMAINS: dict[str, DomainConfig] = {
    "product": DomainConfig(
        item_type="product",
        human_dir=DATA_DIR / "product" / "human",
        llm_dir=DATA_DIR / "product" / "gpt41106preview",
        prompt_key=None,
    ),
    "movie": DomainConfig(
        item_type="movie",
        human_dir=DATA_DIR / "movie" / "human",
        llm_dir=DATA_DIR / "movie" / "gpt41106preview",
        prompt_key=None,
    ),
    "paper": DomainConfig(
        item_type="paper",
        human_dir=DATA_DIR / "paper" / "human",
        llm_dir=DATA_DIR / "paper" / "gpt41106preview",
        prompt_key=None,
    ),
}


def list_available_domains(domains: dict[str, DomainConfig] | None = None) -> Iterable[DomainConfig]:
    """Yield domains that have both human and LLM folders present."""
    doms = DOMAINS if domains is None else domains
    return (cfg for cfg in doms.values() if cfg.exists())
