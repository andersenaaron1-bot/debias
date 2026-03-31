"""Scalar reward scorer with optional auxiliary cue heads."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * float(ctx.scale), None


def grad_reverse(x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
    """Identity in the forward pass, gradient sign flip in the backward pass."""
    return _GradientReversalFn.apply(x, float(scale))


class RewardScorer(nn.Module):
    """Score text sequences with a scalar reward.

    The reward for a sequence is computed from the hidden state at the last
    non-padding token (right padding).
    """

    def __init__(
        self,
        backbone: nn.Module,
        value_head: nn.Linear,
        *,
        attribute_heads: nn.ModuleDict | None = None,
        cue_heads: nn.ModuleDict | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head
        self.attribute_heads = nn.ModuleDict() if attribute_heads is None else attribute_heads
        self.cue_heads = nn.ModuleDict() if cue_heads is None else cue_heads

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.backbone.config, "hidden_size"))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = out.last_hidden_state  # [B, T, H]
        lengths = attention_mask.to(torch.int64).sum(dim=1) - 1  # [B]
        lengths = lengths.clamp(min=0)
        bsz = last_hidden.size(0)
        idx = torch.arange(bsz, device=last_hidden.device)
        pooled = last_hidden[idx, lengths, :]  # [B, H]
        # Ensure dtype matches the value head weights (common when backbone runs in bf16/fp16
        # but the value head is loaded in fp32).
        pooled = pooled.to(dtype=self.value_head.weight.dtype)
        return pooled

    def score_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.value_head(pooled).squeeze(-1)  # [B]

    def attribute_logits_from_pooled(
        self,
        pooled: torch.Tensor,
        *,
        head_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.attribute_heads:
            return {}
        names = sorted(self.attribute_heads) if head_names is None else [n for n in head_names if n in self.attribute_heads]
        return {name: self.attribute_heads[name](pooled).squeeze(-1) for name in names}

    def cue_logits_from_pooled(
        self,
        pooled: torch.Tensor,
        *,
        head_names: list[str] | None = None,
        grl_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if not self.cue_heads:
            return {}
        names = sorted(self.cue_heads) if head_names is None else [n for n in head_names if n in self.cue_heads]
        features = grad_reverse(pooled, scale=float(grl_scale))
        return {name: self.cue_heads[name](features).squeeze(-1) for name in names}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(input_ids, attention_mask)
        return self.score_from_pooled(pooled)

    def save_value_head(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hidden_size": self.hidden_size,
            "state_dict": {k: v.detach().cpu() for k, v in self.value_head.state_dict().items()},
        }
        torch.save(payload, path)

    @staticmethod
    def load_value_head(path: Path, *, device: torch.device | str | None = None) -> nn.Linear:
        payload = torch.load(path, map_location="cpu")
        hidden_size = int(payload["hidden_size"])
        head = nn.Linear(hidden_size, 1)
        head.load_state_dict(payload["state_dict"])
        if device is not None:
            head.to(device)
        return head

    def save_attribute_heads(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hidden_size": self.hidden_size,
            "heads": {
                name: {
                    "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "out_features": int(getattr(head, "out_features", 1)),
                }
                for name, head in self.attribute_heads.items()
            },
        }
        torch.save(payload, path)

    @staticmethod
    def load_attribute_heads(
        path: Path,
        *,
        device: torch.device | str | None = None,
    ) -> nn.ModuleDict:
        payload = torch.load(path, map_location="cpu")
        hidden_size = int(payload["hidden_size"])
        heads = nn.ModuleDict()
        for name, head_payload in (payload.get("heads") or {}).items():
            head = nn.Linear(hidden_size, int(head_payload.get("out_features", 1)))
            head.load_state_dict(head_payload["state_dict"])
            if device is not None:
                head.to(device)
            heads[str(name)] = head
        return heads

    def save_cue_heads(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hidden_size": self.hidden_size,
            "heads": {
                name: {
                    "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "out_features": int(getattr(head, "out_features", 1)),
                }
                for name, head in self.cue_heads.items()
            },
        }
        torch.save(payload, path)

    @staticmethod
    def load_cue_heads(
        path: Path,
        *,
        device: torch.device | str | None = None,
    ) -> nn.ModuleDict:
        payload = torch.load(path, map_location="cpu")
        hidden_size = int(payload["hidden_size"])
        heads = nn.ModuleDict()
        for name, head_payload in (payload.get("heads") or {}).items():
            head = nn.Linear(hidden_size, int(head_payload.get("out_features", 1)))
            head.load_state_dict(head_payload["state_dict"])
            if device is not None:
                head.to(device)
            heads[str(name)] = head
        return heads


def _load_tokenizer(model_id: str, *, cache_dir: Path | None) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir) if cache_dir else None)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_reward_scorer(
    *,
    model_id: str,
    cache_dir: Path | None = None,
    lora_adapter_dir: Path | None = None,
    value_head_path: Path | None = None,
    attribute_heads_path: Path | None = None,
    cue_heads_path: Path | None = None,
    torch_dtype: torch.dtype | None = torch.bfloat16,
    use_4bit: bool = False,
    device_map: str | dict | None = None,
) -> tuple[RewardScorer, AutoTokenizer]:
    """Load backbone (+ optional LoRA) and value head, returning (scorer, tokenizer)."""
    if use_4bit:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        backbone = AutoModel.from_pretrained(
            model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            quantization_config=qconf,
            device_map=device_map,
            attn_implementation="sdpa",
        )
    else:
        backbone = AutoModel.from_pretrained(
            model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        )

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("peft is required: pip install peft>=0.12.0") from exc

    if lora_adapter_dir is not None:
        backbone = PeftModel.from_pretrained(backbone, str(lora_adapter_dir))

    if value_head_path is None:
        value_head = nn.Linear(int(backbone.config.hidden_size), 1)
    else:
        value_head = RewardScorer.load_value_head(value_head_path)
    attribute_heads = (
        RewardScorer.load_attribute_heads(attribute_heads_path) if attribute_heads_path is not None else nn.ModuleDict()
    )
    cue_heads = RewardScorer.load_cue_heads(cue_heads_path) if cue_heads_path is not None else nn.ModuleDict()
    try:
        device = next(p for p in backbone.parameters() if p.device.type != "meta").device
    except StopIteration:  # pragma: no cover
        device = torch.device("cpu")
    value_head.to(device)
    attribute_heads.to(device)
    cue_heads.to(device)

    scorer = RewardScorer(
        backbone=backbone,
        value_head=value_head,
        attribute_heads=attribute_heads,
        cue_heads=cue_heads,
    )
    tokenizer = _load_tokenizer(model_id, cache_dir=cache_dir)
    return scorer, tokenizer


def save_run_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)
