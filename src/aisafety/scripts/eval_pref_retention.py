"""Evaluate preference retention for a reward scorer on held-out preference pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from aisafety.config import DEFAULT_CACHE_DIR
from aisafety.reward.io_jsonl import iter_jsonl
from aisafety.reward.model import load_reward_scorer
from aisafety.reward.text_format import format_prompt_response


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pref-jsonl", type=Path, required=True)
    p.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--lora-adapter-dir", type=Path, default=None)
    p.add_argument("--value-head", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--out-json", type=Path, default=Path("artifacts/reward_eval/pref_retention.json"))
    return p.parse_args()


@torch.no_grad()
def _score_texts(model, tok, texts: list[str], *, max_length: int, batch_size: int, device) -> np.ndarray:
    scores: list[float] = []
    model.eval()
    for i in range(0, len(texts), int(batch_size)):
        batch = texts[i : i + int(batch_size)]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        s = model(enc["input_ids"], enc["attention_mask"]).detach().float().cpu().numpy()
        scores.extend(s.tolist())
    return np.asarray(scores, dtype=np.float32)


def main() -> None:
    args = parse_args()
    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    scorer, tok = load_reward_scorer(
        model_id=str(args.model_id),
        cache_dir=Path(args.cache_dir),
        lora_adapter_dir=args.lora_adapter_dir,
        value_head_path=args.value_head,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    device = next(p for p in scorer.parameters() if p.device.type != "meta").device

    prompts: list[str] = []
    chosen: list[str] = []
    rejected: list[str] = []
    for i, row in enumerate(iter_jsonl(args.pref_jsonl)):
        if args.max_pairs is not None and int(args.max_pairs) > 0 and i >= int(args.max_pairs):
            break
        prompts.append(str(row.get("prompt") or ""))
        chosen.append(str(row.get("chosen") or ""))
        rejected.append(str(row.get("rejected") or ""))

    chosen_texts = [format_prompt_response(p, c) for p, c in zip(prompts, chosen, strict=True)]
    rejected_texts = [format_prompt_response(p, r) for p, r in zip(prompts, rejected, strict=True)]
    scores = _score_texts(
        scorer,
        tok,
        chosen_texts + rejected_texts,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        device=device,
    )
    n = len(chosen_texts)
    sc = scores[:n]
    sr = scores[n:]
    margins = sc - sr
    acc = float(np.mean(margins > 0)) if n else float("nan")
    auc = float(roc_auc_score(([1] * n) + ([0] * n), np.concatenate([sc, sr]))) if n else float("nan")

    out = {
        "pref_jsonl": str(args.pref_jsonl),
        "model_id": str(args.model_id),
        "lora_adapter_dir": None if args.lora_adapter_dir is None else str(args.lora_adapter_dir),
        "value_head": None if args.value_head is None else str(args.value_head),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "max_pairs": None if args.max_pairs is None else int(args.max_pairs),
        "n_pairs": int(n),
        "pairwise_acc": acc,
        "separation_auc": auc,
        "mean_margin": float(np.mean(margins)) if n else float("nan"),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
