"""Run A/B selector models with optional steering."""

from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import (
    LogitsProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from aisafety.selectors.chat import build_chat_batch
from aisafety.steering import register_vector_hook


class ABOnlyProcessor(LogitsProcessor):
    """Restrict generation to A/B tokens."""

    def __init__(self, a_id: int, b_id: int):
        self.a_id, self.b_id = a_id, b_id

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, [self.a_id, self.b_id]] = 0
        return scores + mask


@torch.inference_mode()
def run_selector_ab(
    df_trials,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    prompt_fn,
    ab_only: LogitsProcessor,
    A_ID: int,
    B_ID: int,
    batch_size: int = 24,
    max_length: int = 2048,
    persona: str = "neutral",
    max_desc_tokens: int | None = None,
    steer_config: dict | None = None,
):
    """Generate A/B choices for a batch of trials."""
    hook_handle = None
    if steer_config is not None:
        vec = steer_config["vec"]
        # Allow multiple vectors summed together.
        if isinstance(vec, (list, tuple)):
            vec_t = sum(torch.as_tensor(v, device=model.device) for v in vec)
        else:
            vec_t = torch.as_tensor(vec, device=model.device)
        hook_handle = register_vector_hook(
            model=model,
            layer_idx=int(steer_config["layer_idx"]),
            vec=vec_t,
            alpha=float(steer_config.get("alpha", 0.0)),
        )

    choices, raws = [], []

    try:
        for i in tqdm(range(0, len(df_trials), batch_size), desc="Selector A/B"):
            chunk = df_trials.iloc[i : i + batch_size]
            enc = tokenizer(
                build_chat_batch(
                    chunk,
                    tokenizer,
                    prompt_fn,
                    max_desc_tokens=max_desc_tokens,
                    persona=persona,
                ),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=[ab_only],
            )

            gen_tok = out[:, input_ids.shape[-1] :]
            gen_ids = gen_tok.squeeze(1).tolist()
            batch_choices = ["A" if tid == A_ID else "B" for tid in gen_ids]
            batch_raw = [tokenizer.decode(t, skip_special_tokens=True) for t in gen_tok]

            choices.extend(batch_choices)
            raws.extend(batch_raw)

    finally:
        if hook_handle is not None:
            hook_handle.remove()

    res = df_trials.copy()
    res["choice"] = choices
    res["selector_raw_output"] = raws
    return res
