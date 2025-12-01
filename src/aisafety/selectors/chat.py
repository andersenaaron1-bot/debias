"""Chat prompt rendering helpers for selector models."""

from __future__ import annotations

from typing import Callable, Optional

from transformers import PreTrainedTokenizerBase

from aisafety.selectors.prompts import system_prompt_for_persona


def _truncate_text(tokenizer: PreTrainedTokenizerBase, txt: str, n_tokens: Optional[int]) -> str:
    if n_tokens is None:
        return txt
    ids = tokenizer.encode(txt, add_special_tokens=False)
    if len(ids) > n_tokens:
        ids = ids[:n_tokens]
    return tokenizer.decode(ids)


def build_chat_batch(
    df_rows,
    tokenizer: PreTrainedTokenizerBase,
    prompt_fn: Callable[[dict], str],
    max_desc_tokens: Optional[int] = None,
    persona: str = "neutral",
) -> list[str]:
    """Render chat prompts for a batch of trials."""
    system_text = system_prompt_for_persona(persona)

    chat_strs = []
    for r in df_rows.itertuples(index=False):
        d = r._asdict().copy()
        if max_desc_tokens is not None:
            d["A_text"] = _truncate_text(tokenizer, d["A_text"], max_desc_tokens)
            d["B_text"] = _truncate_text(tokenizer, d["B_text"], max_desc_tokens)
        prompt = prompt_fn(d)
        chat = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt},
        ]
        chat_strs.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
    return chat_strs
