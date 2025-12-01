"""Trial construction for human vs. LLM A/B comparisons."""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import pandas as pd

from aisafety.data.domains import DomainConfig
from aisafety.data.loaders import load_human_map, load_llm_all_by_title


def build_trials(
    item_type: str, human_map: dict[str, str], llm_by_title: dict[str, list[str]], rng: random.Random
) -> pd.DataFrame:
    shared = sorted(set(human_map) & set(llm_by_title))
    rows = []
    for t in shared:
        pair = [
            {"source": "human", "text": human_map[t]},
            {"source": "llm", "text": rng.choice(llm_by_title[t])},
        ]
        rng.shuffle(pair)
        rows.append(
            {
                "item_type": item_type,
                "title": t,
                "A_text": pair[0]["text"],
                "B_text": pair[1]["text"],
                "A_source": pair[0]["source"],
                "B_source": pair[1]["source"],
            }
        )
    return pd.DataFrame(rows)


def build_all_trials(domains_cfg: dict[str, DomainConfig], seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    all_trials: list[pd.DataFrame] = []
    for item_type, cfg in domains_cfg.items():
        if not cfg.exists():
            continue
        human_map = load_human_map(cfg.human_dir)
        llm_by_title = load_llm_all_by_title(cfg.llm_dir, prompt_key=cfg.prompt_key)
        df_t = build_trials(item_type, human_map, llm_by_title, rng)
        if len(df_t):
            all_trials.append(df_t)
    return pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()


def build_hc3_trials(df_desc: pd.DataFrame, seed: int = 1234) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for title, grp in df_desc.groupby("title"):
        hum = grp[grp["source"] == "human"]
        llm = grp[grp["source"] == "llm"]
        if hum.empty or llm.empty:
            continue
        h_text = hum.iloc[0]["text"]
        l_text = llm.iloc[0]["text"]
        hc3_src = hum.iloc[0].get("hc3_source")

        if rng.random() < 0.5:
            A_text, A_source = h_text, "human"
            B_text, B_source = l_text, "llm"
        else:
            A_text, A_source = l_text, "llm"
            B_text, B_source = h_text, "human"

        rows.append(
            {
                "item_type": "hc3",
                "title": title,
                "hc3_source": hc3_src,
                "A_text": A_text,
                "B_text": B_text,
                "A_source": A_source,
                "B_source": B_source,
            }
        )
    return pd.DataFrame(rows)


def build_desc_df_from_trials(df_trials: pd.DataFrame) -> pd.DataFrame:
    """Flatten trials to description-level frame with labels."""
    rows = []
    for r in df_trials.itertuples(index=False):
        hc3_src = getattr(r, "hc3_source", None)
        rows += [
            {
                "item_type": r.item_type,
                "title": r.title,
                "text": r.A_text,
                "source": r.A_source,
                "hc3_source": hc3_src,
            },
            {
                "item_type": r.item_type,
                "title": r.title,
                "text": r.B_text,
                "source": r.B_source,
                "hc3_source": hc3_src,
            },
        ]
    df = pd.DataFrame(rows).drop_duplicates(subset=["title", "source", "text"])
    df = df.sort_values(["title", "source"]).drop_duplicates(subset=["title", "source"], keep="first")
    both = df.groupby("title")["source"].nunique()
    df = df[df["title"].isin(both[both == 2].index)].reset_index(drop=True)
    df["y"] = (df["source"] == "llm").astype(int)
    return df
