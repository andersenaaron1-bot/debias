"""Evaluate reward-model style sensitivity on held-out style groups.

For each style_axis, sample within-group variant pairs and report
  d = |r(vi) - r(vj)|
as mean/median/quantiles.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED
from aisafety.reward.io_jsonl import iter_jsonl
from aisafety.reward.model import load_reward_scorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--style-jsonl", type=Path, required=True)
    p.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--lora-adapter-dir", type=Path, default=None)
    p.add_argument("--value-head", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pairs-per-group", type=int, default=1)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-json", type=Path, default=Path("artifacts/reward_eval/style_sensitivity.json"))
    p.add_argument("--out-csv", type=Path, default=Path("artifacts/reward_eval/style_sensitivity.csv"))
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

    rng = random.Random(int(args.seed))
    pairs: list[dict] = []
    for row in iter_jsonl(args.style_jsonl):
        axis = str(row.get("style_axis") or "").strip()
        gid = str(row.get("group_id") or "").strip()
        variants = row.get("variants") or []
        if not axis or not gid or not isinstance(variants, list) or len(variants) < 2:
            continue
        variants_s = [str(v).strip() for v in variants if isinstance(v, str) and str(v).strip()]
        if len(variants_s) < 2:
            continue
        k = min(int(args.pairs_per_group), (len(variants_s) * (len(variants_s) - 1)) // 2)
        for _ in range(max(1, k)):
            i, j = rng.sample(range(len(variants_s)), 2)
            pairs.append({"style_axis": axis, "group_id": gid, "a": variants_s[i], "b": variants_s[j]})

    a_texts = [p["a"] for p in pairs]
    b_texts = [p["b"] for p in pairs]
    scores = _score_texts(
        scorer,
        tok,
        a_texts + b_texts,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        device=device,
    )
    sa = scores[: len(a_texts)]
    sb = scores[len(a_texts) :]
    d = np.abs(sa - sb)
    for p, di in zip(pairs, d.tolist(), strict=True):
        p["d_abs"] = float(di)

    df = pd.DataFrame(pairs)
    rows = []
    for axis, grp in df.groupby("style_axis"):
        vals = grp["d_abs"].astype(float).to_numpy()
        rows.append(
            {
                "style_axis": axis,
                "n_pairs": int(len(vals)),
                "mean_d": float(np.mean(vals)) if len(vals) else float("nan"),
                "median_d": float(np.median(vals)) if len(vals) else float("nan"),
                "p90_d": float(np.quantile(vals, 0.9)) if len(vals) else float("nan"),
            }
        )
    df_sum = pd.DataFrame(rows).sort_values("style_axis").reset_index(drop=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "style_jsonl": str(args.style_jsonl),
                "model_id": str(args.model_id),
                "lora_adapter_dir": None if args.lora_adapter_dir is None else str(args.lora_adapter_dir),
                "value_head": None if args.value_head is None else str(args.value_head),
                "max_length": int(args.max_length),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
                "pairs_per_group": int(args.pairs_per_group),
                "summary": df_sum.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    df_sum.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
