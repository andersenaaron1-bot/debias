"""Selector backends for different inference surfaces (HF local, OpenRouter API, etc.)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import httpx
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from aisafety.selectors.prompts import make_selector_prompt, system_prompt_for_persona
from aisafety.selectors.runner import ABOnlyProcessor, run_selector_ab


class SelectorBackend(Protocol):
    def run(self, df_trials: pd.DataFrame, persona: str = "neutral", **kwargs) -> pd.DataFrame: ...


@dataclass
class HFSelectorBackend:
    """Local Hugging Face model backend."""

    model_id: str
    cache_dir: str | os.PathLike
    use_4bit: bool = False

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=str(self.cache_dir))
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.use_4bit:
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
                quantization_config=qconf,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=str(self.cache_dir),
                attn_implementation="sdpa",
            )
        self.model.eval()

        self.A_ID = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.B_ID = self.tokenizer.encode("B", add_special_tokens=False)[0]
        self.ab_only = ABOnlyProcessor(self.A_ID, self.B_ID)

    def run(
        self,
        df_trials: pd.DataFrame,
        persona: str = "neutral",
        batch_size: int = 24,
        max_length: int = 2048,
        max_desc_tokens: int | None = None,
        steer_config: dict | None = None,
    ) -> pd.DataFrame:
        return run_selector_ab(
            df_trials=df_trials,
            tokenizer=self.tokenizer,
            model=self.model,
            prompt_fn=make_selector_prompt,
            ab_only=self.ab_only,
            A_ID=self.A_ID,
            B_ID=self.B_ID,
            batch_size=batch_size,
            max_length=max_length,
            persona=persona,
            max_desc_tokens=max_desc_tokens,
            steer_config=steer_config,
        )


class OpenRouterSelectorBackend:
    """Remote inference via OpenRouter API."""

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self.base_url = base_url or self.OPENROUTER_URL
        self.client = httpx.Client(timeout=60)

    def _build_messages(self, row: dict, persona: str):
        sys = system_prompt_for_persona(persona)
        prompt = make_selector_prompt(row)
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ]

    def _call(self, messages):
        headers = {"Authorization": f"Bearer {self.api_key}", "HTTP-Referer": "local"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1,
        }
        resp = self.client.post(self.base_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()

    def run(self, df_trials: pd.DataFrame, persona: str = "neutral", **_) -> pd.DataFrame:
        choices = []
        raws = []
        for r in df_trials.itertuples(index=False):
            msg = self._build_messages(r._asdict(), persona=persona)
            text = self._call(msg)
            ch = "A" if text[:1].upper() == "A" else "B"
            choices.append(ch)
            raws.append(text)
        res = df_trials.copy()
        res["choice"] = choices
        res["selector_raw_output"] = raws
        return res


def build_selector_backend(
    backend: str,
    model_id: str,
    cache_dir: str | os.PathLike,
    use_4bit: bool = False,
    api_key: str | None = None,
):
    """Factory to construct selector backends."""
    if backend == "hf":
        return HFSelectorBackend(model_id=model_id, cache_dir=cache_dir, use_4bit=use_4bit)
    if backend == "openrouter":
        return OpenRouterSelectorBackend(model=model_id, api_key=api_key)
    raise ValueError(f"Unknown backend: {backend}")
