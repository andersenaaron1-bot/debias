"""Train a scalar reward scorer with preference, pointwise anchor, invariance, and cue-adversarial losses.

This trains:
  - LoRA adapters on a transformer backbone
  - a small value head (linear -> scalar)

Objective per step (multi-stream):
  - with prob p: preference batch (SHP-2) using pairwise -logsigmoid loss
  - with prob q: pointwise anchor batch using scalar + multi-attribute regression
  - with prob r: cue batch using weak-label BCE with gradient reversal
  - otherwise: invariance batch sampling two variants within meaning-group, using MSE

Example:
  python -m aisafety.scripts.train_reward_lora ^
    --model-id google/gemma-2-9b-it ^
    --pref-train-jsonl data\\derived\\pref_pairs_shp2\\pref_pairs_train.jsonl ^
    --pref-val-jsonl data\\derived\\pref_pairs_shp2\\pref_pairs_val.jsonl ^
    --style-train-jsonl data\\derived\\style_groups\\style_groups_train.jsonl ^
    --style-val-jsonl data\\derived\\style_groups\\style_groups_val.jsonl ^
    --output-dir artifacts\\reward\\invariance ^
    --max-steps 5000
"""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    set_seed,
)

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED
from aisafety.reward.jsonl_index import build_offsets, build_offsets_by_key
from aisafety.reward.losses import (
    cue_bce_loss,
    group_robust_reduce,
    inv_loss,
    lambda_schedule,
    multi_head_bce_losses,
    multi_head_mse_losses,
    pointwise_mse_loss,
    pref_loss,
)
from aisafety.reward.model import RewardScorer, save_run_config
from aisafety.reward.text_format import format_prompt_response


@dataclass(frozen=True)
class TrainConfig:
    model_id: str
    cache_dir: str
    output_dir: str
    max_length: int
    seed: int
    use_4bit: bool
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    gradient_accumulation_steps: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    eval_batch_size: int
    pref_eval_pairs: int
    anchor_eval_rows: int
    inv_eval_groups_per_axis: int
    cue_eval_rows: int
    pref_prob: float
    anchor_prob: float
    cue_prob: float
    pref_batch_pairs: int
    anchor_batch_size: int
    inv_batch_groups: int
    cue_batch_size: int
    lambda_max: float
    lambda_ramp_frac: float
    lambda_group: float
    lambda_anchor_utility: float
    lambda_anchor_attr: float
    lambda_cue: float
    cue_grl_scale: float
    exclude_axes: list[str]
    anchor_train_jsonl: str | None
    anchor_val_jsonl: str | None
    anchor_attribute_names: list[str]
    cue_train_jsonl: str | None
    cue_val_jsonl: str | None
    cue_families: list[str]
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]


class RandomAccessJsonl:
    def __init__(self, path: Path):
        self.path = path
        self._f = None

    def __enter__(self) -> "RandomAccessJsonl":
        self._f = self.path.open("rb")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def read_at(self, offset: int) -> dict[str, Any]:
        if self._f is None:
            raise RuntimeError("RandomAccessJsonl is not open")
        self._f.seek(int(offset))
        line = self._f.readline()
        if not line:
            raise EOFError(f"Offset {offset} beyond EOF for {self.path}")
        return json.loads(line.decode("utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pref-train-jsonl", type=Path, required=True)
    p.add_argument("--pref-val-jsonl", type=Path, required=True)
    p.add_argument("--style-train-jsonl", type=Path, required=True)
    p.add_argument("--style-val-jsonl", type=Path, required=True)
    p.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/reward/invariance"))
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--use-4bit", action="store_true", help="Enable QLoRA with 4-bit base model.")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=5_000)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--pref-eval-pairs", type=int, default=256)
    p.add_argument("--anchor-eval-rows", type=int, default=256)
    p.add_argument("--inv-eval-groups-per-axis", type=int, default=16)
    p.add_argument("--cue-eval-rows", type=int, default=256)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--pref-prob", type=float, default=0.7)
    p.add_argument("--anchor-prob", type=float, default=0.0)
    p.add_argument("--cue-prob", type=float, default=0.0)
    p.add_argument("--pref-batch-pairs", type=int, default=8)
    p.add_argument("--anchor-batch-size", type=int, default=16)
    p.add_argument("--inv-batch-groups", type=int, default=16)
    p.add_argument("--cue-batch-size", type=int, default=16)
    p.add_argument("--lambda-max", type=float, default=0.5)
    p.add_argument("--lambda-ramp-frac", type=float, default=0.1)
    p.add_argument("--lambda-group", type=float, default=0.0)
    p.add_argument("--lambda-anchor-utility", type=float, default=1.0)
    p.add_argument("--lambda-anchor-attr", type=float, default=1.0)
    p.add_argument("--lambda-cue", type=float, default=1.0)
    p.add_argument("--cue-grl-scale", type=float, default=1.0)
    p.add_argument("--exclude-axes", type=str, default="", help="Comma-separated style_axis values to exclude.")
    p.add_argument("--anchor-train-jsonl", type=Path, default=None)
    p.add_argument("--anchor-val-jsonl", type=Path, default=None)
    p.add_argument(
        "--anchor-attribute-names",
        type=str,
        default="helpfulness,correctness,coherence,complexity,verbosity",
        help="Comma-separated attribute names expected inside attribute_targets.",
    )
    p.add_argument("--cue-train-jsonl", type=Path, default=None)
    p.add_argument("--cue-val-jsonl", type=Path, default=None)
    p.add_argument(
        "--cue-families",
        type=str,
        default="academic_formality,safety_corporate_tone,promotional_sales_tone,narrative_packaging,template_boilerplate,verbosity_compression,hedging_certainty",
        help="Comma-separated weak-label families expected inside weak_label_ids.",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return p.parse_args()


def _parse_csv_list(val: str) -> list[str]:
    return [x.strip() for x in str(val or "").split(",") if x.strip()]


def _load_tokenizer(model_id: str, *, cache_dir: Path) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _load_backbone(
    *,
    model_id: str,
    cache_dir: Path,
    use_4bit: bool,
    bf16: bool,
    fp16: bool,
) -> nn.Module:
    torch_dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else None
    if use_4bit:
        compute_dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.bfloat16
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        backbone = AutoModel.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            quantization_config=qconf,
            device_map={"": 0} if torch.cuda.is_available() else "auto",
            attn_implementation="sdpa",
        )
    else:
        backbone = AutoModel.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            torch_dtype=torch_dtype,
            device_map={"": 0} if torch.cuda.is_available() else "auto",
            attn_implementation="sdpa",
        )
    # We never use KV caching in reward scoring; disabling saves a lot of memory.
    if hasattr(backbone, "config"):
        setattr(backbone.config, "use_cache", False)
    return backbone


def _apply_lora(backbone: nn.Module, *, args: argparse.Namespace) -> nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("peft is required: pip install peft>=0.12.0") from exc

    if bool(args.use_4bit):
        backbone = prepare_model_for_kbit_training(backbone)

    if bool(args.gradient_checkpointing):
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "config"):
            setattr(backbone.config, "use_cache", False)

    target_modules = [t.strip() for t in str(args.lora_target_modules).split(",") if t.strip()]
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )
    backbone = get_peft_model(backbone, lora_cfg)
    return backbone


def _freeze_all_but_lora_and_head(model: RewardScorer) -> None:
    for n, p in model.named_parameters():
        if (
            "lora_" in n
            or n.startswith("value_head.")
            or n.startswith("attribute_heads.")
            or n.startswith("cue_heads.")
        ):
            p.requires_grad = True
        else:
            p.requires_grad = False


def _tokenize_texts(tok: AutoTokenizer, texts: list[str], *, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def _sample_pref_batch(
    ra: RandomAccessJsonl,
    offsets: list[int],
    *,
    rng: random.Random,
    batch_pairs: int,
) -> tuple[list[str], list[str], list[str]]:
    sel = [offsets[rng.randrange(len(offsets))] for _ in range(batch_pairs)]
    prompts: list[str] = []
    chosen: list[str] = []
    rejected: list[str] = []
    for off in sel:
        row = ra.read_at(off)
        prompts.append(str(row.get("prompt") or ""))
        chosen.append(str(row.get("chosen") or ""))
        rejected.append(str(row.get("rejected") or ""))
    return prompts, chosen, rejected


def _sample_anchor_batch(
    ra: RandomAccessJsonl,
    offsets: list[int],
    *,
    rng: random.Random,
    batch_size: int,
    attribute_names: list[str],
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    sel = [offsets[rng.randrange(len(offsets))] for _ in range(batch_size)]
    texts: list[str] = []
    utility_targets: list[float] = []
    attr_targets: list[list[float]] = []
    for off in sel:
        row = ra.read_at(off)
        prompt = str(row.get("prompt") or "")
        response = str(row.get("response") or "").strip()
        utility_target = row.get("utility_target")
        attribute_targets = row.get("attribute_targets") or {}
        if not response or utility_target is None or not isinstance(attribute_targets, dict):
            continue
        target_row: list[float] = []
        ok = True
        for name in attribute_names:
            val = attribute_targets.get(name)
            if val is None:
                ok = False
                break
            target_row.append(float(val))
        if not ok:
            continue
        texts.append(format_prompt_response(prompt, response))
        utility_targets.append(float(utility_target))
        attr_targets.append(target_row)
    if not texts:
        raise RuntimeError("Could not sample anchor batch (all sampled rows invalid).")
    return (
        texts,
        torch.tensor(utility_targets, dtype=torch.float32),
        torch.tensor(attr_targets, dtype=torch.float32),
    )


def _sample_inv_batch(
    ra: RandomAccessJsonl,
    by_axis_offsets: dict[str, list[int]],
    *,
    rng: random.Random,
    batch_groups: int,
) -> tuple[str, list[str], list[str]]:
    axes = sorted(by_axis_offsets)
    if not axes:
        raise ValueError("No style axes available for invariance sampling.")
    axis = axes[rng.randrange(len(axes))]
    offsets = by_axis_offsets[axis]
    sel = [offsets[rng.randrange(len(offsets))] for _ in range(batch_groups)]
    a_texts: list[str] = []
    b_texts: list[str] = []
    for off in sel:
        row = ra.read_at(off)
        variants = row.get("variants") or []
        if not isinstance(variants, list) or len(variants) < 2:
            continue
        variants_s = [str(v) for v in variants if isinstance(v, str) and v.strip()]
        if len(variants_s) < 2:
            continue
        i, j = rng.sample(range(len(variants_s)), 2)
        a_texts.append(variants_s[i])
        b_texts.append(variants_s[j])
    if not a_texts:
        raise RuntimeError(f"Could not sample invariance batch for axis={axis!r} (all groups invalid).")
    return axis, a_texts, b_texts


def _row_has_cue_signal(row: dict[str, Any], cue_families: list[str]) -> bool:
    weak_label_ids = row.get("weak_label_ids") or {}
    if not isinstance(weak_label_ids, dict):
        return False
    for family in cue_families:
        try:
            if int(weak_label_ids.get(family, 0)) != 0:
                return True
        except Exception:
            continue
    return False


def _sample_cue_batch(
    ra: RandomAccessJsonl,
    offsets: list[int],
    *,
    rng: random.Random,
    batch_size: int,
    cue_families: list[str],
) -> tuple[list[str], torch.Tensor]:
    sel = [offsets[rng.randrange(len(offsets))] for _ in range(batch_size)]
    texts: list[str] = []
    targets: list[list[float]] = []
    for off in sel:
        row = ra.read_at(off)
        text = str(row.get("text") or "").strip()
        weak_label_ids = row.get("weak_label_ids") or {}
        if not text or not isinstance(weak_label_ids, dict):
            continue
        target_row = []
        for family in cue_families:
            try:
                target_row.append(1.0 if int(weak_label_ids.get(family, 0)) != 0 else 0.0)
            except Exception:
                target_row.append(0.0)
        texts.append(text)
        targets.append(target_row)
    if not texts:
        raise RuntimeError("Could not sample cue batch (all sampled rows invalid).")
    return texts, torch.tensor(targets, dtype=torch.float32)


@torch.no_grad()
def _eval_pref(
    model: RewardScorer,
    tok: AutoTokenizer,
    ra: RandomAccessJsonl,
    offsets: list[int],
    *,
    rng: random.Random,
    n_pairs: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, float]:
    n_pairs = min(int(n_pairs), len(offsets))
    prompts, chosen, rejected = _sample_pref_batch(ra, offsets, rng=rng, batch_pairs=n_pairs)
    chosen_texts = [format_prompt_response(p, c) for p, c in zip(prompts, chosen, strict=True)]
    rejected_texts = [format_prompt_response(p, r) for p, r in zip(prompts, rejected, strict=True)]
    sc_chunks: list[torch.Tensor] = []
    sr_chunks: list[torch.Tensor] = []
    for i in range(0, n_pairs, int(batch_size)):
        j = min(n_pairs, i + int(batch_size))
        enc = _tokenize_texts(tok, chosen_texts[i:j] + rejected_texts[i:j], max_length=max_length, device=device)
        scores = model(enc["input_ids"], enc["attention_mask"]).float()
        b = j - i
        sc_chunks.append(scores[:b].detach())
        sr_chunks.append(scores[b:].detach())
    sc = torch.cat(sc_chunks, dim=0)
    sr = torch.cat(sr_chunks, dim=0)
    loss = float(pref_loss(sc, sr).item()) if n_pairs else float("nan")
    acc = float((sc > sr).float().mean().item()) if n_pairs else float("nan")
    return {"pref_loss": loss, "pref_acc": acc}


@torch.no_grad()
def _eval_style_sensitivity(
    model: RewardScorer,
    tok: AutoTokenizer,
    ra: RandomAccessJsonl,
    by_axis_offsets: dict[str, list[int]],
    *,
    rng: random.Random,
    n_groups_per_axis: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, float]:
    axes = sorted(by_axis_offsets)
    if not axes:
        return {"inv_abs_diff": float("nan")}
    a_texts: list[str] = []
    b_texts: list[str] = []
    for axis in axes:
        offsets = by_axis_offsets[axis]
        take = min(int(n_groups_per_axis), len(offsets))
        for _ in range(take):
            off = offsets[rng.randrange(len(offsets))]
            row = ra.read_at(off)
            variants = row.get("variants") or []
            if not isinstance(variants, list) or len(variants) < 2:
                continue
            variants_s = [str(v) for v in variants if isinstance(v, str) and v.strip()]
            if len(variants_s) < 2:
                continue
            i, j = rng.sample(range(len(variants_s)), 2)
            a_texts.append(variants_s[i])
            b_texts.append(variants_s[j])
    if not a_texts:
        return {"inv_abs_diff": float("nan")}
    sa_chunks: list[torch.Tensor] = []
    sb_chunks: list[torch.Tensor] = []
    for i in range(0, len(a_texts), int(batch_size)):
        j = min(len(a_texts), i + int(batch_size))
        enc = _tokenize_texts(tok, a_texts[i:j] + b_texts[i:j], max_length=max_length, device=device)
        scores = model(enc["input_ids"], enc["attention_mask"]).float()
        b = j - i
        sa_chunks.append(scores[:b].detach())
        sb_chunks.append(scores[b:].detach())
    sa = torch.cat(sa_chunks, dim=0)
    sb = torch.cat(sb_chunks, dim=0)
    d = torch.abs(sa - sb).mean().item() if len(a_texts) else float("nan")
    return {"inv_abs_diff": float(d)}


@torch.no_grad()
def _eval_anchor(
    model: RewardScorer,
    tok: AutoTokenizer,
    ra: RandomAccessJsonl,
    offsets: list[int],
    *,
    rng: random.Random,
    n_rows: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
    attribute_names: list[str],
) -> dict[str, float]:
    if not attribute_names:
        return {"anchor_utility_mse": float("nan"), "anchor_attr_mse": float("nan")}
    n_rows = min(int(n_rows), len(offsets))
    texts, utility_targets, attr_targets = _sample_anchor_batch(
        ra,
        offsets,
        rng=rng,
        batch_size=n_rows,
        attribute_names=attribute_names,
    )
    utility_chunks: list[torch.Tensor] = []
    attr_chunks: list[torch.Tensor] = []
    utility_target_chunks: list[torch.Tensor] = []
    attr_target_chunks: list[torch.Tensor] = []
    for i in range(0, len(texts), int(batch_size)):
        j = min(len(texts), i + int(batch_size))
        enc = _tokenize_texts(tok, texts[i:j], max_length=max_length, device=device)
        pooled = model.encode(enc["input_ids"], enc["attention_mask"])
        utility_pred = model.score_from_pooled(pooled).float().detach().cpu()
        attr_map = model.attribute_logits_from_pooled(pooled, head_names=attribute_names)
        attr_pred = torch.stack([attr_map[name] for name in attribute_names], dim=1).float().detach().cpu()
        utility_chunks.append(utility_pred)
        attr_chunks.append(attr_pred)
        utility_target_chunks.append(utility_targets[i:j].detach().cpu())
        attr_target_chunks.append(attr_targets[i:j].detach().cpu())
    utility_all = torch.cat(utility_chunks, dim=0)
    attr_all = torch.cat(attr_chunks, dim=0)
    utility_target_all = torch.cat(utility_target_chunks, dim=0)
    attr_target_all = torch.cat(attr_target_chunks, dim=0)
    utility_mse = float(pointwise_mse_loss(utility_all, utility_target_all).item())
    attr_mse = float(multi_head_mse_losses(attr_all, attr_target_all).mean().item())
    return {"anchor_utility_mse": utility_mse, "anchor_attr_mse": attr_mse}


@torch.no_grad()
def _eval_cue(
    model: RewardScorer,
    tok: AutoTokenizer,
    ra: RandomAccessJsonl,
    offsets: list[int],
    *,
    rng: random.Random,
    n_rows: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
    cue_families: list[str],
    cue_grl_scale: float,
) -> dict[str, float]:
    if not cue_families:
        return {"cue_loss": float("nan"), "cue_acc": float("nan")}
    n_rows = min(int(n_rows), len(offsets))
    texts, targets = _sample_cue_batch(ra, offsets, rng=rng, batch_size=n_rows, cue_families=cue_families)
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for i in range(0, len(texts), int(batch_size)):
        j = min(len(texts), i + int(batch_size))
        enc = _tokenize_texts(tok, texts[i:j], max_length=max_length, device=device)
        pooled = model.encode(enc["input_ids"], enc["attention_mask"])
        logits_map = model.cue_logits_from_pooled(
            pooled,
            head_names=cue_families,
            grl_scale=float(cue_grl_scale),
        )
        logits = torch.stack([logits_map[f] for f in cue_families], dim=1).float()
        logits_chunks.append(logits.detach().cpu())
        target_chunks.append(targets[i:j].detach().cpu())
    logits_all = torch.cat(logits_chunks, dim=0)
    targets_all = torch.cat(target_chunks, dim=0)
    loss = float(cue_bce_loss(logits_all, targets_all).item())
    preds = (torch.sigmoid(logits_all) >= 0.5).float()
    acc = float((preds == targets_all).float().mean().item())
    return {"cue_loss": loss, "cue_acc": acc}


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    # Some Slurm/Pyxis environments expose partial distributed variables that
    # make Accelerate attempt env:// rendezvous without WORLD_SIZE being set.
    # The current training path is single-process/single-GPU, so default to a
    # clean rank-0 world when the required env vars are absent.
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if bool(args.bf16) and bool(args.fp16):
        raise ValueError("Set at most one of --bf16/--fp16.")
    if not bool(args.bf16) and not bool(args.fp16) and torch.cuda.is_available():
        args.bf16 = True

    mixed_precision = "bf16" if bool(args.bf16) else "fp16" if bool(args.fp16) else "no"
    accelerator = Accelerator(gradient_accumulation_steps=int(args.gradient_accumulation_steps), mixed_precision=mixed_precision)

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    exclude_axes = _parse_csv_list(str(args.exclude_axes))
    anchor_attribute_names = _parse_csv_list(str(args.anchor_attribute_names))
    use_anchor_stream = (
        args.anchor_train_jsonl is not None and args.anchor_val_jsonl is not None and len(anchor_attribute_names) > 0
    )
    cue_families = _parse_csv_list(str(args.cue_families))
    use_cue_stream = args.cue_train_jsonl is not None and args.cue_val_jsonl is not None and len(cue_families) > 0
    if float(args.pref_prob) < 0.0 or float(args.anchor_prob) < 0.0 or float(args.cue_prob) < 0.0:
        raise ValueError("--pref-prob, --anchor-prob, and --cue-prob must be non-negative.")
    if float(args.pref_prob) + float(args.anchor_prob) + float(args.cue_prob) > 1.0:
        raise ValueError("--pref-prob + --anchor-prob + --cue-prob must be <= 1.0.")
    if float(args.anchor_prob) > 0.0 and not use_anchor_stream:
        raise ValueError("--anchor-prob > 0 requires --anchor-train-jsonl, --anchor-val-jsonl, and --anchor-attribute-names.")
    if float(args.cue_prob) > 0.0 and not use_cue_stream:
        raise ValueError("--cue-prob > 0 requires --cue-train-jsonl, --cue-val-jsonl, and --cue-families.")

    pref_train = build_offsets(Path(args.pref_train_jsonl))
    pref_val = build_offsets(Path(args.pref_val_jsonl))
    anchor_train = build_offsets(Path(args.anchor_train_jsonl)) if use_anchor_stream else None
    anchor_val = build_offsets(Path(args.anchor_val_jsonl)) if use_anchor_stream else None
    style_train_by_axis = build_offsets_by_key(Path(args.style_train_jsonl), key="style_axis")
    style_val_by_axis = build_offsets_by_key(Path(args.style_val_jsonl), key="style_axis")
    cue_train = build_offsets(Path(args.cue_train_jsonl)) if use_cue_stream else None
    cue_val = build_offsets(Path(args.cue_val_jsonl)) if use_cue_stream else None

    style_train_offsets = {
        axis: off.offsets for axis, off in style_train_by_axis.items() if axis not in set(exclude_axes)
    }
    style_val_offsets = {
        axis: off.offsets for axis, off in style_val_by_axis.items() if axis not in set(exclude_axes)
    }
    if not style_train_offsets:
        raise ValueError("No style axes left after --exclude-axes filtering.")

    tok = _load_tokenizer(str(args.model_id), cache_dir=Path(args.cache_dir))
    backbone = _load_backbone(
        model_id=str(args.model_id),
        cache_dir=Path(args.cache_dir),
        use_4bit=bool(args.use_4bit),
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
    )
    backbone = _apply_lora(backbone, args=args)

    hidden_size = int(backbone.config.hidden_size)
    value_head = nn.Linear(hidden_size, 1)
    attribute_heads = (
        nn.ModuleDict({name: nn.Linear(hidden_size, 1) for name in anchor_attribute_names}) if use_anchor_stream else None
    )
    cue_heads = nn.ModuleDict({family: nn.Linear(hidden_size, 1) for family in cue_families}) if use_cue_stream else None
    model = RewardScorer(
        backbone=backbone,
        value_head=value_head,
        attribute_heads=attribute_heads,
        cue_heads=cue_heads,
    )
    _freeze_all_but_lora_and_head(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps),
        num_training_steps=int(args.max_steps),
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    device = accelerator.device

    cfg = TrainConfig(
        model_id=str(args.model_id),
        cache_dir=str(args.cache_dir),
        output_dir=str(args.output_dir),
        max_length=int(args.max_length),
        seed=int(args.seed),
        use_4bit=bool(args.use_4bit),
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_steps=int(args.warmup_steps),
        max_steps=int(args.max_steps),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        logging_steps=int(args.logging_steps),
        eval_steps=int(args.eval_steps),
        save_steps=int(args.save_steps),
        eval_batch_size=int(args.eval_batch_size),
        pref_eval_pairs=int(args.pref_eval_pairs),
        anchor_eval_rows=int(args.anchor_eval_rows),
        inv_eval_groups_per_axis=int(args.inv_eval_groups_per_axis),
        cue_eval_rows=int(args.cue_eval_rows),
        pref_prob=float(args.pref_prob),
        anchor_prob=float(args.anchor_prob),
        cue_prob=float(args.cue_prob),
        pref_batch_pairs=int(args.pref_batch_pairs),
        anchor_batch_size=int(args.anchor_batch_size),
        inv_batch_groups=int(args.inv_batch_groups),
        cue_batch_size=int(args.cue_batch_size),
        lambda_max=float(args.lambda_max),
        lambda_ramp_frac=float(args.lambda_ramp_frac),
        lambda_group=float(args.lambda_group),
        lambda_anchor_utility=float(args.lambda_anchor_utility),
        lambda_anchor_attr=float(args.lambda_anchor_attr),
        lambda_cue=float(args.lambda_cue),
        cue_grl_scale=float(args.cue_grl_scale),
        exclude_axes=exclude_axes,
        anchor_train_jsonl=None if args.anchor_train_jsonl is None else str(args.anchor_train_jsonl),
        anchor_val_jsonl=None if args.anchor_val_jsonl is None else str(args.anchor_val_jsonl),
        anchor_attribute_names=anchor_attribute_names,
        cue_train_jsonl=None if args.cue_train_jsonl is None else str(args.cue_train_jsonl),
        cue_val_jsonl=None if args.cue_val_jsonl is None else str(args.cue_val_jsonl),
        cue_families=cue_families,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_target_modules=[t.strip() for t in str(args.lora_target_modules).split(",") if t.strip()],
    )

    if accelerator.is_main_process:
        save_run_config(Path(args.output_dir) / "run_config.json", asdict(cfg))

    metrics_path = Path(args.output_dir) / "metrics_train.csv"
    metrics_fields = [
        "step",
        "stream",
        "loss",
        "loss_pref",
        "loss_anchor",
        "loss_anchor_utility",
        "loss_anchor_attr",
        "loss_inv",
        "loss_cue",
        "lambda",
        "lambda_group",
        "lambda_anchor_utility",
        "lambda_anchor_attr",
        "lambda_cue",
        "lr",
        "seconds_per_step",
        "pref_val_acc",
        "anchor_val_utility_mse",
        "anchor_val_attr_mse",
        "inv_val_abs_diff",
        "cue_val_acc",
        "cue_val_loss",
    ]
    if accelerator.is_main_process:
        with metrics_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=metrics_fields)
            w.writeheader()

    rng = random.Random(int(args.seed))
    last_log_t = time.time()

    with ExitStack() as stack:
        ra_pref_train = stack.enter_context(RandomAccessJsonl(Path(args.pref_train_jsonl)))
        ra_pref_val = stack.enter_context(RandomAccessJsonl(Path(args.pref_val_jsonl)))
        ra_anchor_train = stack.enter_context(RandomAccessJsonl(Path(args.anchor_train_jsonl))) if use_anchor_stream else None
        ra_anchor_val = stack.enter_context(RandomAccessJsonl(Path(args.anchor_val_jsonl))) if use_anchor_stream else None
        ra_style_train = stack.enter_context(RandomAccessJsonl(Path(args.style_train_jsonl)))
        ra_style_val = stack.enter_context(RandomAccessJsonl(Path(args.style_val_jsonl)))
        ra_cue_train = stack.enter_context(RandomAccessJsonl(Path(args.cue_train_jsonl))) if use_cue_stream else None
        ra_cue_val = stack.enter_context(RandomAccessJsonl(Path(args.cue_val_jsonl))) if use_cue_stream else None

        for step in range(int(args.max_steps)):
            draw = rng.random()
            if draw < float(args.pref_prob):
                stream = "pref"
            elif use_anchor_stream and draw < float(args.pref_prob) + float(args.anchor_prob):
                stream = "anchor"
            elif use_cue_stream and draw < float(args.pref_prob) + float(args.anchor_prob) + float(args.cue_prob):
                stream = "cue"
            else:
                stream = "inv"

            lam = lambda_schedule(
                step,
                total_steps=int(args.max_steps),
                lambda_max=float(args.lambda_max),
                ramp_frac=float(args.lambda_ramp_frac),
            )

            with accelerator.accumulate(model):
                loss_p = torch.tensor(0.0, device=device)
                loss_a = torch.tensor(0.0, device=device)
                loss_a_utility = torch.tensor(0.0, device=device)
                loss_a_attr = torch.tensor(0.0, device=device)
                loss_i = torch.tensor(0.0, device=device)
                loss_c = torch.tensor(0.0, device=device)

                if stream == "pref":
                    prompts, chosen, rejected = _sample_pref_batch(
                        ra_pref_train, pref_train.offsets, rng=rng, batch_pairs=int(args.pref_batch_pairs)
                    )
                    chosen_texts = [format_prompt_response(p, c) for p, c in zip(prompts, chosen, strict=True)]
                    rejected_texts = [format_prompt_response(p, r) for p, r in zip(prompts, rejected, strict=True)]
                    enc = _tokenize_texts(
                        tok, chosen_texts + rejected_texts, max_length=int(args.max_length), device=device
                    )
                    scores = model(enc["input_ids"], enc["attention_mask"])
                    n = len(chosen_texts)
                    loss_p = pref_loss(scores[:n], scores[n:])
                    loss = loss_p
                elif stream == "anchor":
                    if ra_anchor_train is None or anchor_train is None:
                        raise RuntimeError("Anchor stream selected without anchor training data.")
                    texts, utility_targets, attr_targets = _sample_anchor_batch(
                        ra_anchor_train,
                        anchor_train.offsets,
                        rng=rng,
                        batch_size=int(args.anchor_batch_size),
                        attribute_names=anchor_attribute_names,
                    )
                    utility_targets = utility_targets.to(device)
                    attr_targets = attr_targets.to(device)
                    enc = _tokenize_texts(tok, texts, max_length=int(args.max_length), device=device)
                    pooled = model.encode(enc["input_ids"], enc["attention_mask"])
                    utility_pred = model.score_from_pooled(pooled)
                    attr_map = model.attribute_logits_from_pooled(pooled, head_names=anchor_attribute_names)
                    attr_pred = torch.stack([attr_map[name] for name in anchor_attribute_names], dim=1)
                    loss_a_utility = pointwise_mse_loss(utility_pred, utility_targets)
                    per_attr_losses = multi_head_mse_losses(attr_pred, attr_targets)
                    loss_a_attr = group_robust_reduce(per_attr_losses, strength=float(args.lambda_group))
                    loss_a = float(args.lambda_anchor_utility) * loss_a_utility + float(args.lambda_anchor_attr) * loss_a_attr
                    loss = loss_a
                elif stream == "cue":
                    if ra_cue_train is None:
                        raise RuntimeError("Cue stream selected without cue training data.")
                    texts, cue_targets = _sample_cue_batch(
                        ra_cue_train,
                        cue_train.offsets,
                        rng=rng,
                        batch_size=int(args.cue_batch_size),
                        cue_families=cue_families,
                    )
                    cue_targets = cue_targets.to(device)
                    enc = _tokenize_texts(tok, texts, max_length=int(args.max_length), device=device)
                    pooled = model.encode(enc["input_ids"], enc["attention_mask"])
                    logits_map = model.cue_logits_from_pooled(
                        pooled,
                        head_names=cue_families,
                        grl_scale=float(args.cue_grl_scale),
                    )
                    cue_logits = torch.stack([logits_map[family] for family in cue_families], dim=1)
                    cue_loss_per_family = multi_head_bce_losses(cue_logits, cue_targets)
                    loss_c = group_robust_reduce(cue_loss_per_family, strength=float(args.lambda_group))
                    loss = float(args.lambda_cue) * loss_c
                else:
                    axis, a_texts, b_texts = _sample_inv_batch(
                        ra_style_train,
                        style_train_offsets,
                        rng=rng,
                        batch_groups=int(args.inv_batch_groups),
                    )
                    enc = _tokenize_texts(tok, a_texts + b_texts, max_length=int(args.max_length), device=device)
                    scores = model(enc["input_ids"], enc["attention_mask"])
                    n = len(a_texts)
                    loss_i = inv_loss(scores[:n], scores[n:])
                    loss = float(lam) * loss_i

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % int(args.logging_steps) == 0 or step == 0:
                now = time.time()
                sec_per_step = (now - last_log_t) / float(max(1, int(args.logging_steps) if step else 1))
                last_log_t = now
                lr = float(scheduler.get_last_lr()[0])

                pref_metrics = {"pref_acc": float("nan")}
                anchor_metrics = {"anchor_utility_mse": float("nan"), "anchor_attr_mse": float("nan")}
                inv_metrics = {"inv_abs_diff": float("nan")}
                cue_metrics = {"cue_loss": float("nan"), "cue_acc": float("nan")}
                if (step + 1) % int(args.eval_steps) == 0 or step == 0:
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.eval()
                    pref_metrics = _eval_pref(
                        unwrapped,
                        tok,
                        ra_pref_val,
                        pref_val.offsets,
                        rng=rng,
                        n_pairs=int(args.pref_eval_pairs),
                        batch_size=int(args.eval_batch_size),
                        max_length=int(args.max_length),
                        device=device,
                    )
                    if use_anchor_stream and ra_anchor_val is not None and anchor_val is not None:
                        anchor_metrics = _eval_anchor(
                            unwrapped,
                            tok,
                            ra_anchor_val,
                            anchor_val.offsets,
                            rng=rng,
                            n_rows=int(args.anchor_eval_rows),
                            batch_size=int(args.eval_batch_size),
                            max_length=int(args.max_length),
                            device=device,
                            attribute_names=anchor_attribute_names,
                        )
                    inv_metrics = _eval_style_sensitivity(
                        unwrapped,
                        tok,
                        ra_style_val,
                        style_val_offsets,
                        rng=rng,
                        n_groups_per_axis=int(args.inv_eval_groups_per_axis),
                        batch_size=int(args.eval_batch_size),
                        max_length=int(args.max_length),
                        device=device,
                    )
                    if use_cue_stream and ra_cue_val is not None and cue_val is not None:
                        cue_metrics = _eval_cue(
                            unwrapped,
                            tok,
                            ra_cue_val,
                            cue_val.offsets,
                            rng=rng,
                            n_rows=int(args.cue_eval_rows),
                            batch_size=int(args.eval_batch_size),
                            max_length=int(args.max_length),
                            device=device,
                            cue_families=cue_families,
                            cue_grl_scale=float(args.cue_grl_scale),
                        )
                    unwrapped.train()

                if accelerator.is_main_process:
                    row = {
                        "step": int(step + 1),
                        "stream": str(stream),
                        "loss": float(loss.detach().float().item()),
                        "loss_pref": float(loss_p.detach().float().item()),
                        "loss_anchor": float(loss_a.detach().float().item()),
                        "loss_anchor_utility": float(loss_a_utility.detach().float().item()),
                        "loss_anchor_attr": float(loss_a_attr.detach().float().item()),
                        "loss_inv": float(loss_i.detach().float().item()),
                        "loss_cue": float(loss_c.detach().float().item()),
                        "lambda": float(lam),
                        "lambda_group": float(args.lambda_group),
                        "lambda_anchor_utility": float(args.lambda_anchor_utility),
                        "lambda_anchor_attr": float(args.lambda_anchor_attr),
                        "lambda_cue": float(args.lambda_cue),
                        "lr": lr,
                        "seconds_per_step": float(sec_per_step),
                        "pref_val_acc": float(pref_metrics.get("pref_acc", float("nan"))),
                        "anchor_val_utility_mse": float(anchor_metrics.get("anchor_utility_mse", float("nan"))),
                        "anchor_val_attr_mse": float(anchor_metrics.get("anchor_attr_mse", float("nan"))),
                        "inv_val_abs_diff": float(inv_metrics.get("inv_abs_diff", float("nan"))),
                        "cue_val_acc": float(cue_metrics.get("cue_acc", float("nan"))),
                        "cue_val_loss": float(cue_metrics.get("cue_loss", float("nan"))),
                    }
                    with metrics_path.open("a", encoding="utf-8", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=metrics_fields)
                        w.writerow(row)
                    print(json.dumps(row, ensure_ascii=False))

            if (step + 1) % int(args.save_steps) == 0 or (step + 1) == int(args.max_steps):
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    out_dir = Path(args.output_dir)
                    adapter_dir = out_dir / "lora_adapter"
                    head_path = out_dir / "value_head.pt"
                    attribute_head_path = out_dir / "attribute_heads.pt"
                    cue_head_path = out_dir / "cue_heads.pt"
                    adapter_dir.mkdir(parents=True, exist_ok=True)
                    unwrapped.backbone.save_pretrained(adapter_dir)
                    unwrapped.save_value_head(head_path)
                    if unwrapped.attribute_heads:
                        unwrapped.save_attribute_heads(attribute_head_path)
                    if unwrapped.cue_heads:
                        unwrapped.save_cue_heads(cue_head_path)
                    tok.save_pretrained(adapter_dir)

    if accelerator.is_main_process:
        print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
