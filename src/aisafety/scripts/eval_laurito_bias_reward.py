"""Evaluate LLM-presented bias on Laurito A/B trials using a reward scorer.

Computes:
  - raw win rate for LLM-presented option
  - swap-debiased win rate when trials contain balanced A/B swaps
  - bootstrap 95% CIs over titles/items
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED
from aisafety.eval.bias import evaluate_by_domain
from aisafety.reward.model import load_reward_scorer
from aisafety.eval.debias import add_pairwise_debias_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials-csv", type=Path, required=True)
    p.add_argument(
        "--group-col",
        type=str,
        default="item_type",
        help=(
            "Column used for stratified reporting (default: item_type). "
            "Use hc3_source or hc3_subset for HC3 stratification."
        ),
    )
    p.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--lora-adapter-dir", type=Path, default=None)
    p.add_argument("--value-head", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--out-json", type=Path, default=Path("artifacts/reward_eval/laurito_bias.json"))
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


def _tie_break_choice(keys: list[str], *, seed: int) -> np.ndarray:
    out = []
    for k in keys:
        h = hashlib.sha1(f"{int(seed)}:{k}".encode("utf-8")).hexdigest()
        out.append("A" if (int(h[:8], 16) % 2) == 0 else "B")
    return np.asarray(out, dtype=object)


def _bootstrap_prop_llm(df: pd.DataFrame, *, seed: int, n_boot: int) -> dict:
    if "title" not in df.columns:
        return {"ci_low": float("nan"), "ci_high": float("nan")}
    valid = df[df["choice"].isin(["A", "B"])].copy()
    if valid.empty:
        return {"ci_low": float("nan"), "ci_high": float("nan")}

    chosen_source = np.where(valid["choice"] == "A", valid["A_source"], valid["B_source"]).astype(str)
    is_llm = (chosen_source == "llm").astype(int)
    stats = (
        pd.DataFrame({"title": valid["title"].astype(str), "is_llm": is_llm})
        .groupby("title")["is_llm"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "k", "count": "n"})
    )
    if stats.empty:
        return {"ci_low": float("nan"), "ci_high": float("nan")}

    k_by = stats["k"].to_numpy(dtype=np.int64)
    n_by = stats["n"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    props = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, len(stats), size=len(stats))
        k = int(k_by[idx].sum())
        n = int(n_by[idx].sum())
        props[i] = (k / n) if n else np.nan
    return {"ci_low": float(np.quantile(props, 0.025)), "ci_high": float(np.quantile(props, 0.975))}


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.trials_csv)
    required = {"item_type", "title", "A_text", "B_text", "A_source", "B_source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in trials CSV: {sorted(missing)}")
    group_col = str(args.group_col).strip()
    if not group_col:
        raise ValueError("--group-col must be non-empty.")
    if group_col not in df.columns:
        raise ValueError(f"--group-col {group_col!r} not found in trials CSV columns.")

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
    df = df.copy()
    df["score_A"] = scores[:n]
    df["score_B"] = scores[n:]
    df["score_diff"] = df["score_A"] - df["score_B"]
    tie = df["score_diff"].astype(float) == 0.0
    df["choice"] = np.where(df["score_diff"] > 0, "A", "B")
    if tie.any():
        keys = (df["item_type"].astype(str) + "||" + df["title"].astype(str)).tolist()
        choices = _tie_break_choice(keys, seed=int(args.seed))
        df.loc[tie, "choice"] = choices[tie.to_numpy()]

    # Normalize stratification key to item_type so we can reuse evaluate_by_domain.
    # This keeps output schema consistent across datasets and grouping choices.
    df_group = df.copy()
    if group_col != "item_type":
        group_vals = df_group[group_col].astype(str).str.strip()
        group_vals = group_vals.replace("", "__missing__")
        df_group["item_type"] = group_vals

    raw = evaluate_by_domain(df_group)

    swap_out = None
    try:
        df_swap = add_pairwise_debias_columns(df, logit_diff_col="score_diff", seed=int(args.seed))
        swap_summary = {}
        if group_col != "item_type":
            group_vals = df_swap[group_col].astype(str).str.strip()
            group_vals = group_vals.replace("", "__missing__")
            df_swap["item_type"] = group_vals
        for t, grp in df_swap.groupby("item_type"):
            valid = grp[grp["chosen_source_debiased"].isin(["human", "llm"])].copy()
            if valid.empty:
                swap_summary[t] = {"n_trials": 0, "prop_llm_chosen": float("nan")}
            else:
                prop = float((valid["chosen_source_debiased"] == "llm").mean())
                swap_summary[t] = {"n_trials": int(len(valid)), "prop_llm_chosen": prop, "bias": prop - 0.5}
        valid_all = df_swap[df_swap["chosen_source_debiased"].isin(["human", "llm"])].copy()
        if not valid_all.empty:
            prop_all = float((valid_all["chosen_source_debiased"] == "llm").mean())
            swap_summary["overall"] = {"n_trials": int(len(valid_all)), "prop_llm_chosen": prop_all, "bias": prop_all - 0.5}
        swap_out = swap_summary
    except Exception as exc:
        swap_out = {"error": str(exc)}

    boot = {}
    for t, grp in df_group.groupby("item_type"):
        boot[t] = _bootstrap_prop_llm(grp, seed=int(args.seed), n_boot=int(args.bootstrap))
    boot["overall"] = _bootstrap_prop_llm(df_group, seed=int(args.seed), n_boot=int(args.bootstrap))

    out = {
        "trials_csv": str(args.trials_csv),
        "group_col": group_col,
        "model_id": str(args.model_id),
        "lora_adapter_dir": None if args.lora_adapter_dir is None else str(args.lora_adapter_dir),
        "value_head": None if args.value_head is None else str(args.value_head),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "bootstrap": int(args.bootstrap),
        "raw": raw,
        "raw_bootstrap_ci": boot,
        "swap_debiased": swap_out,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
