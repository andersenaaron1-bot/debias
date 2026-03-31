"""Evaluate H/G/R triads with a reward scorer.

Triad per (item_type,title):
  H = original human text (Laurito)
  G = LLM-generated from-scratch text (Laurito)
  R = LLM rewrite of H (from OpenRouter style-pair JSONL)

Reports pairwise win rates:
  H vs G, H vs R, G vs R
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch

from aisafety.config import DATA_DIR, DEFAULT_CACHE_DIR, DEFAULT_SEED
from aisafety.data import DOMAINS
from aisafety.data.loaders import load_human_map, load_llm_all_by_title
from aisafety.reward.model import load_reward_scorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--lora-adapter-dir", type=Path, default=None)
    p.add_argument("--value-head", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--trials-csv",
        type=Path,
        default=None,
        help=(
            "Optional A/B trials CSV to source H/G pairs from. "
            "If set, H/G are aligned to these exact trial texts instead of loading "
            "Laurito domains directly."
        ),
    )
    p.add_argument(
        "--rewrite-jsonl",
        type=Path,
        default=DATA_DIR / "derived" / "openrouter_style_pairs_test" / "ai_tone.jsonl",
        help="OpenRouter style-pairs JSONL containing human rewrites.",
    )
    p.add_argument("--rewrite-dimension", type=str, default="ai_tone")
    p.add_argument("--rewrite-label", type=str, default="rlhf_ai_tone")
    p.add_argument("--out-json", type=Path, default=Path("artifacts/reward_eval/triads.json"))
    p.add_argument("--out-csv", type=Path, default=Path("artifacts/reward_eval/triads.csv"))
    return p.parse_args()


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _trial_key(item_type: str, title: str) -> str:
    return f"{item_type}||{title}"


def load_hg_map_from_trials_csv(path: Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(path)
    required = {"item_type", "title", "A_text", "B_text", "A_source", "B_source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in trials CSV: {sorted(missing)}")

    out: dict[str, dict[str, str]] = {}
    # Prefer rows where human is in A (priority=0), then fallback to swapped rows (priority=1).
    priority_by_key: dict[str, int] = {}
    for row in df.itertuples(index=False):
        item_type = str(getattr(row, "item_type", "")).strip()
        title = str(getattr(row, "title", "")).strip()
        if not item_type or not title:
            continue

        a_source = str(getattr(row, "A_source", "")).strip().lower()
        b_source = str(getattr(row, "B_source", "")).strip().lower()
        a_text = str(getattr(row, "A_text", "")).strip()
        b_text = str(getattr(row, "B_text", "")).strip()
        if not a_text or not b_text:
            continue

        if a_source == "human" and b_source == "llm":
            h_text, g_text, prio = a_text, b_text, 0
        elif a_source == "llm" and b_source == "human":
            h_text, g_text, prio = b_text, a_text, 1
        else:
            continue

        key = _trial_key(item_type, title)
        old_prio = priority_by_key.get(key)
        if old_prio is None or prio < old_prio:
            out[key] = {"item_type": item_type, "title": title, "H": h_text, "G": g_text}
            priority_by_key[key] = prio

    return out


def load_openrouter_rewrite_map(path: Path, *, dimension: str, label: str) -> dict[str, str]:
    rewrites: dict[str, str] = {}
    for row in _iter_jsonl(path):
        if str(row.get("dimension") or "") != str(dimension):
            continue
        if str(row.get("label") or "") != str(label):
            continue
        meta = row.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        item_type = str(meta.get("item_type") or "").strip()
        title = str(meta.get("title") or "").strip()
        source = str(meta.get("source") or "").strip().lower()
        if source and source != "human":
            continue
        text = str(row.get("generated_text") or "").strip()
        if not item_type or not title or not text:
            continue
        key = _trial_key(item_type, title)
        if key not in rewrites:
            rewrites[key] = text
    return rewrites


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

    rewrite_map = load_openrouter_rewrite_map(
        args.rewrite_jsonl, dimension=str(args.rewrite_dimension), label=str(args.rewrite_label)
    )
    if not rewrite_map:
        raise ValueError(f"No rewrites found for dimension={args.rewrite_dimension} label={args.rewrite_label}")

    rows = []
    if args.trials_csv is not None:
        hg_map = load_hg_map_from_trials_csv(args.trials_csv)
        for key, pair in hg_map.items():
            r = rewrite_map.get(key)
            if not r:
                continue
            rows.append(
                {
                    "item_type": str(pair["item_type"]),
                    "title": str(pair["title"]),
                    "H": str(pair["H"]),
                    "G": str(pair["G"]),
                    "R": r,
                }
            )
    else:
        for item_type, cfg in DOMAINS.items():
            if not cfg.exists():
                continue
            human = load_human_map(cfg.human_dir)
            llm_by_title = load_llm_all_by_title(cfg.llm_dir, prompt_key=cfg.prompt_key)
            shared = sorted(set(human) & set(llm_by_title))
            for title in shared:
                key = _trial_key(item_type, title)
                r = rewrite_map.get(key)
                if not r:
                    continue
                h = human[title]
                g = llm_by_title[title][0] if llm_by_title[title] else ""
                if not h or not g:
                    continue
                rows.append({"item_type": item_type, "title": title, "H": h, "G": g, "R": r})

    if not rows:
        raise ValueError("No triads constructed (check rewrite_jsonl meta item_type/title coverage).")

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

    texts = []
    for r in rows:
        texts.extend([r["H"], r["G"], r["R"]])
    scores = _score_texts(
        scorer,
        tok,
        texts,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        device=device,
    )
    for i, r in enumerate(rows):
        sH, sG, sR = scores[3 * i : 3 * i + 3].tolist()
        r["score_H"] = float(sH)
        r["score_G"] = float(sG)
        r["score_R"] = float(sR)
        r["win_H_vs_G"] = "H" if sH >= sG else "G"
        r["win_H_vs_R"] = "H" if sH >= sR else "R"
        r["win_G_vs_R"] = "G" if sG >= sR else "R"

    df = pd.DataFrame(rows)

    def win_rate(col: str, winner: str) -> float:
        return float((df[col] == winner).mean()) if len(df) else float("nan")

    summary = {
        "n_items": int(len(df)),
        "H_beats_G": win_rate("win_H_vs_G", "H"),
        "G_beats_H": win_rate("win_H_vs_G", "G"),
        "H_beats_R": win_rate("win_H_vs_R", "H"),
        "R_beats_H": win_rate("win_H_vs_R", "R"),
        "G_beats_R": win_rate("win_G_vs_R", "G"),
        "R_beats_G": win_rate("win_G_vs_R", "R"),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": str(args.model_id),
                "lora_adapter_dir": None if args.lora_adapter_dir is None else str(args.lora_adapter_dir),
                "value_head": None if args.value_head is None else str(args.value_head),
                "max_length": int(args.max_length),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
                "trials_csv": None if args.trials_csv is None else str(args.trials_csv),
                "rewrite_jsonl": str(args.rewrite_jsonl),
                "rewrite_dimension": str(args.rewrite_dimension),
                "rewrite_label": str(args.rewrite_label),
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
