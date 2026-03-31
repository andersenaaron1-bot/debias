"""Load and normalize human vs. LLM descriptions for local domains."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _first_from_list(x):
    if isinstance(x, list) and x:
        v = x[0]
        if isinstance(v, dict):
            return (v.get("text") or v.get("description") or "").strip()
        return str(v).strip()
    return None


def _strip_xml(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s)


def _extract_single_text(d: dict) -> str | None:
    if "descriptions" in d:
        t = _first_from_list(d["descriptions"])
        if t:
            return t

    for k in ("abstract",):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for k in ("synopsis", "summary", "text"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    v = d.get("abstract_xml")
    if isinstance(v, str) and ("<" in v and ">" in v):
        s = _strip_xml(v).strip()
        if s:
            return s

    return None


def _extract_all_texts(d: dict) -> list[str]:
    texts: list[str] = []

    descs = d.get("descriptions")
    if isinstance(descs, list):
        for x in descs:
            if isinstance(x, dict):
                t = (x.get("text") or x.get("description") or "").strip()
            else:
                t = str(x).strip()
            if t:
                texts.append(t)

    abs_list = d.get("abstracts")
    if isinstance(abs_list, list):
        texts += [str(a).strip() for a in abs_list if str(a).strip()]

    if isinstance(d.get("abstract"), str) and d["abstract"].strip():
        texts.append(d["abstract"].strip())

    for k in ("synopsis", "summary", "text"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            texts.append(v.strip())

    seen = set()
    uniq: list[str] = []
    for s in texts:
        if s and s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _iter_json_files(root: Path) -> Iterable[Path]:
    # Sort for reproducibility across filesystems / OS directory iteration order.
    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            yield path


def load_human_map(human_dir: str | Path) -> dict[str, str]:
    """Return mapping: title -> single human text."""
    paths = list(_iter_json_files(Path(human_dir)))
    title_to_text: dict[str, str] = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        title = d.get("title")
        txt = _extract_single_text(d)
        if title and txt:
            title_to_text[title] = txt
    return title_to_text


def load_llm_all_by_title(llm_dir: str | Path, prompt_key: str | None = None) -> dict[str, list[str]]:
    """Return mapping: title -> list of all LLM texts available."""
    root = Path(llm_dir)
    pattern = f"*{prompt_key}*.json" if prompt_key else "*.json"
    # Sort for reproducibility across filesystems / OS directory iteration order.
    paths = sorted(root.rglob(pattern))
    by_title: dict[str, list[str]] = defaultdict(list)

    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        title = d.get("title")
        if not title:
            continue
        texts = _extract_all_texts(d)
        if texts:
            by_title[title].extend(texts)

    for t, lst in by_title.items():
        seen = set()
        uniq = []
        for s in lst:
            if s and s not in seen:
                uniq.append(s)
                seen.add(s)
        by_title[t] = uniq
    return dict(by_title)
