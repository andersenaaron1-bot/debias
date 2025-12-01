"""Example stylometric feature extraction."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List

import pandas as pd


WORD_RE = re.compile(r"[A-Za-z]+")
SENTENCE_RE = re.compile(r"[.!?]+")


def _words(text: str) -> List[str]:
    return WORD_RE.findall(text or "")


def feat_length(text: str) -> float:
    return float(len(text or ""))


def feat_word_count(text: str) -> float:
    return float(len(_words(text)))


def feat_avg_word_len(text: str) -> float:
    w = _words(text)
    return float(sum(map(len, w)) / len(w)) if w else 0.0


def feat_type_token_ratio(text: str) -> float:
    w = _words(text)
    return float(len(set(w)) / len(w)) if w else 0.0


def feat_punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct = sum(ch in ".,;:!?" for ch in text)
    return float(punct) / len(text)


def feat_upper_ratio(text: str) -> float:
    if not text:
        return 0.0
    upp = sum(ch.isupper() for ch in text)
    return float(upp) / len(text)


def feat_digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digs = sum(ch.isdigit() for ch in text)
    return float(digs) / len(text)


def feat_sentence_count(text: str) -> float:
    return float(len(SENTENCE_RE.findall(text or "")))


FEATURE_REGISTRY: Dict[str, Callable[[str], float]] = {
    "length_chars": feat_length,
    "word_count": feat_word_count,
    "avg_word_len": feat_avg_word_len,
    "type_token_ratio": feat_type_token_ratio,
    "punct_ratio": feat_punct_ratio,
    "upper_ratio": feat_upper_ratio,
    "digit_ratio": feat_digit_ratio,
    "sentence_count": feat_sentence_count,
}


def compute_features(texts: Iterable[str], features: Dict[str, Callable[[str], float]] | None = None):
    feats = features or FEATURE_REGISTRY
    rows = []
    for t in texts:
        rows.append({name: fn(t) for name, fn in feats.items()})
    return pd.DataFrame(rows)
