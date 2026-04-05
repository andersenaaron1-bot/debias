"""Score Laurito-style A/B trials with a reward model and write a scored CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aisafety.config import DATA_DIR, DEFAULT_CACHE_DIR, DEFAULT_SEED
from aisafety.data import DOMAINS, build_all_trials
from aisafety.ontology.ecology import ensure_choice_columns
from aisafety.reward.model import load_reward_scorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials-csv", type=Path, default=None)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=DATA_DIR / "derived" / "style_groups" / "d3_ecological_validation_v1" / "inputs" / "laurito_scored_trials.csv",
    )
    p.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--lora-adapter-dir", type=Path, default=None)
    p.add_argument("--value-head", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
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


def _prepare_trials_csv(path: Path, *, seed: int) -> Path:
    df = build_all_trials(DOMAINS, seed=int(seed), balance_order=True)
    if df.empty:
        raise RuntimeError("Could not build Laurito trials from local domain data.")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    trials_csv = (
        Path(args.trials_csv)
        if args.trials_csv is not None
        else _prepare_trials_csv(args.out_csv.parent / "laurito_trials.csv", seed=int(args.seed))
    )
    df = pd.read_csv(trials_csv)
    required = {"A_text", "B_text", "A_source", "B_source", "item_type", "title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required trial columns: {sorted(missing)}")

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
    scores = _score_texts(
        scorer,
        tok,
        df["A_text"].astype(str).tolist() + df["B_text"].astype(str).tolist(),
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        device=device,
    )
    n = len(df)
    out = df.copy()
    out["score_A"] = scores[:n]
    out["score_B"] = scores[n:]
    out = ensure_choice_columns(out, seed=int(args.seed))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote scored trials to {args.out_csv}")


if __name__ == "__main__":
    main()
