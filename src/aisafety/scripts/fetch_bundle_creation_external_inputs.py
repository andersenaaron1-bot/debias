"""Fetch and normalize external corpora for bundle-creation ontology work.

This stages normalized excerpt-level JSONL files under ``data/external/bundle_creation_v1``.

Supported sources:
- HAP-E full corpus
- PubMed abstracts via direct NCBI baseline XML
- CMU Movie Summary Corpus
- Amazon Reviews 2023 item metadata
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
from pathlib import Path
import re
import tarfile
from typing import Iterable
import urllib.request
import xml.etree.ElementTree as ET

import httpx
from datasets import load_dataset

from aisafety.config import DATA_DIR


HAPE_HF_DATASET = "browndw/human-ai-parallel-corpus"
CMU_MOVIE_URL = "https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
PUBMED_BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
PUBMED_FILE_RE = re.compile(r'href="(pubmed\d+n\d+\.xml\.gz)"')
DEFAULT_AMAZON_CONFIGS = [
    "raw_meta_Electronics",
    "raw_meta_Home_and_Kitchen",
    "raw_meta_Office_Products",
    "raw_meta_Sports_and_Outdoors",
    "raw_meta_Beauty_and_Personal_Care",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "external" / "bundle_creation_v1",
    )
    p.add_argument("--cache-dir", type=Path, default=DATA_DIR / "external" / ".cache" / "bundle_creation_v1")
    p.add_argument("--skip-hape", action="store_true")
    p.add_argument("--skip-pubmed", action="store_true")
    p.add_argument("--skip-movie", action="store_true")
    p.add_argument("--skip-amazon", action="store_true")
    p.add_argument("--pubmed-target", type=int, default=5000)
    p.add_argument("--movie-target", type=int, default=5000)
    p.add_argument("--amazon-target", type=int, default=5000)
    p.add_argument("--amazon-configs", type=str, default=",".join(DEFAULT_AMAZON_CONFIGS))
    p.add_argument("--min-tokens", type=int, default=80)
    p.add_argument("--max-tokens", type=int, default=900)
    p.add_argument("--timeout-sec", type=float, default=120.0)
    return p.parse_args()


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r\n", "\n").replace("\r", "\n").split()).strip()


def _token_count(text: str) -> int:
    return len(_normalize_text(text).split())


def _within_len(text: str, *, min_tokens: int, max_tokens: int) -> bool:
    n = _token_count(text)
    return int(min_tokens) <= n <= int(max_tokens)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _download_to_cache(url: str, dst: Path, *, timeout_sec: float) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout_sec) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_bytes():
                if chunk:
                    f.write(chunk)
    return dst


def _jsonl_row(
    *,
    text: str,
    title: str,
    source: str,
    item_type: str,
    subset: str,
    group_id: str,
    generator: str | None = None,
    question: str | None = None,
    prompt_name: str | None = None,
    meta: dict | None = None,
) -> dict:
    return {
        "text": _normalize_text(text),
        "title": _normalize_text(title),
        "question": None if question is None else _normalize_text(question),
        "source": source,
        "generator": generator,
        "item_type": item_type,
        "subset": subset,
        "group_id": group_id,
        "prompt_name": prompt_name,
        "meta": meta or {},
    }


def build_hape_rows(*, min_tokens: int, max_tokens: int) -> list[dict]:
    ds = load_dataset(HAPE_HF_DATASET, split="train")
    grouped: dict[str, dict] = {}
    for row in ds:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("doc_id") or "").strip()
        text = _normalize_text(str(row.get("text") or ""))
        if not doc_id or not text:
            continue
        if "@" not in doc_id:
            continue
        base_id, variant = doc_id.split("@", 1)
        genre = base_id.split("_", 1)[0]
        item_type = "general"
        if genre == "acad":
            item_type = "paper"
        elif genre == "tvm":
            item_type = "movie"
        payload = grouped.setdefault(
            base_id,
            {
                "genre": genre,
                "item_type": item_type,
                "human": None,
                "llm": {},
            },
        )
        if variant == "chunk_1":
            continue
        if variant == "chunk_2":
            payload["human"] = text
        else:
            payload["llm"][variant] = text

    rows: list[dict] = []
    for base_id in sorted(grouped):
        payload = grouped[base_id]
        human_text = str(payload.get("human") or "")
        llm_variants = dict(payload.get("llm") or {})
        if not human_text or not llm_variants:
            continue
        if not _within_len(human_text, min_tokens=min_tokens, max_tokens=max_tokens):
            continue
        candidate_generators = sorted(llm_variants)
        chosen_generator = min(
            candidate_generators,
            key=lambda name: _sha1_hex(f"{base_id}|{name}"),
        )
        llm_text = _normalize_text(str(llm_variants.get(chosen_generator) or ""))
        if not _within_len(llm_text, min_tokens=min_tokens, max_tokens=max_tokens):
            continue
        genre = str(payload["genre"])
        item_type = str(payload["item_type"])
        group_id = f"hape::{base_id}"
        common_meta = {
            "origin": HAPE_HF_DATASET,
            "base_id": base_id,
            "available_generators": sorted(candidate_generators),
            "selection_rule": "one_llm_per_seed_via_stable_hash",
        }
        rows.append(
            _jsonl_row(
                text=human_text,
                title=base_id,
                source="human",
                item_type=item_type,
                subset=genre,
                group_id=group_id,
                meta={**common_meta, "variant": "chunk_2"},
            )
        )
        rows.append(
            _jsonl_row(
                text=llm_text,
                title=base_id,
                source="llm",
                generator=chosen_generator,
                item_type=item_type,
                subset=genre,
                group_id=group_id,
                meta={**common_meta, "variant": chosen_generator},
            )
        )
    return rows


def _pubmed_listing(*, timeout_sec: float) -> list[str]:
    r = httpx.get(PUBMED_BASELINE_URL, follow_redirects=True, timeout=timeout_sec)
    r.raise_for_status()
    names = sorted(set(PUBMED_FILE_RE.findall(r.text)))
    if not names:
        raise RuntimeError("Could not discover PubMed baseline files.")
    return names


def _pubmed_article_rows_from_file(
    path: Path,
    *,
    min_tokens: int,
    max_tokens: int,
) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rb") as gz:
        for _, elem in ET.iterparse(gz, events=("end",)):
            if elem.tag != "PubmedArticle":
                continue
            langs = [(_normalize_text(lang.text)).lower() for lang in elem.findall(".//Article/Language")]
            if langs and "eng" not in langs:
                elem.clear()
                continue
            pmid = _normalize_text(" ".join(node.text or "" for node in elem.findall(".//PMID")[:1]))
            title = _normalize_text("".join(elem.findtext(".//Article/ArticleTitle") or ""))
            abstract_nodes = elem.findall(".//Article/Abstract/AbstractText")
            parts: list[str] = []
            for node in abstract_nodes:
                section = _normalize_text(" ".join(node.itertext()))
                label = _normalize_text(node.attrib.get("Label", ""))
                if label and section and not section.lower().startswith(label.lower()):
                    section = f"{label}: {section}"
                if section:
                    parts.append(section)
            abstract = _normalize_text(" ".join(parts))
            if not pmid or not title or not abstract:
                elem.clear()
                continue
            if not _within_len(abstract, min_tokens=min_tokens, max_tokens=max_tokens):
                elem.clear()
                continue
            rows.append(
                _jsonl_row(
                    text=abstract,
                    title=html.unescape(title),
                    source="human",
                    item_type="paper",
                    subset="pubmed",
                    group_id=f"pubmed::{pmid}",
                    meta={"origin": "ncbi_pubmed_baseline", "pmid": pmid, "cache_file": str(path)},
                )
            )
            elem.clear()
    return rows


def build_pubmed_rows(
    *,
    cache_dir: Path,
    target_rows: int,
    min_tokens: int,
    max_tokens: int,
    timeout_sec: float,
) -> list[dict]:
    file_names = _pubmed_listing(timeout_sec=timeout_sec)
    ordered = sorted(file_names, key=lambda name: _sha1_hex(name))
    rows: list[dict] = []
    for name in ordered:
        cache_path = cache_dir / "pubmed" / name
        _download_to_cache(PUBMED_BASELINE_URL + name, cache_path, timeout_sec=timeout_sec)
        rows.extend(_pubmed_article_rows_from_file(cache_path, min_tokens=min_tokens, max_tokens=max_tokens))
        if len(rows) >= int(target_rows):
            break
    return rows[: int(target_rows)]


def build_movie_summary_rows(
    *,
    cache_dir: Path,
    target_rows: int,
    min_tokens: int,
    max_tokens: int,
    timeout_sec: float,
) -> list[dict]:
    tar_path = _download_to_cache(CMU_MOVIE_URL, cache_dir / "movie" / "MovieSummaries.tar.gz", timeout_sec=timeout_sec)
    titles_by_id: dict[str, str] = {}
    summaries_by_id: dict[str, str] = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.endswith("movie.metadata.tsv"):
                with tf.extractfile(member) as fh:
                    assert fh is not None
                    reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8"), delimiter="\t")
                    for row in reader:
                        if len(row) >= 3:
                            titles_by_id[str(row[0]).strip()] = _normalize_text(row[2])
            elif member.name.endswith("plot_summaries.txt"):
                with tf.extractfile(member) as fh:
                    assert fh is not None
                    reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8"), delimiter="\t")
                    for row in reader:
                        if len(row) >= 2:
                            summaries_by_id[str(row[0]).strip()] = _normalize_text(row[1])
    rows: list[dict] = []
    for wiki_id in sorted(summaries_by_id, key=lambda x: _sha1_hex(x)):
        summary = summaries_by_id[wiki_id]
        title = titles_by_id.get(wiki_id, wiki_id)
        if not title or not summary:
            continue
        if not _within_len(summary, min_tokens=min_tokens, max_tokens=max_tokens):
            continue
        rows.append(
            _jsonl_row(
                text=summary,
                title=title,
                source="human",
                item_type="movie",
                subset="cmu_movie_summary",
                group_id=f"cmu_movie::{wiki_id}",
                meta={"origin": "cmu_movie_summary", "wiki_movie_id": wiki_id},
            )
        )
        if len(rows) >= int(target_rows):
            break
    return rows


def _amazon_text_from_row(row: dict) -> str:
    parts: list[str] = []
    title = _normalize_text(str(row.get("title") or ""))
    subtitle = _normalize_text(str(row.get("subtitle") or ""))
    if title:
        parts.append(title)
    if subtitle and subtitle.lower() not in {"unentitled", "unentitledunentitled"}:
        parts.append(subtitle)
    description = row.get("description") or []
    if isinstance(description, list):
        parts.extend(_normalize_text(str(x)) for x in description if _normalize_text(str(x)))
    elif description:
        parts.append(_normalize_text(str(description)))
    features = row.get("features") or []
    if isinstance(features, list):
        feature_text = " ".join(_normalize_text(str(x)) for x in features if _normalize_text(str(x)))
        if feature_text:
            parts.append(feature_text)
    elif features:
        parts.append(_normalize_text(str(features)))
    return _normalize_text(" ".join(parts))


def build_amazon_rows(
    *,
    target_rows: int,
    configs: list[str],
    min_tokens: int,
    max_tokens: int,
) -> list[dict]:
    per_config = max(1, int(target_rows) // max(1, len(configs)))
    rows: list[dict] = []
    for config_name in configs:
        ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", config_name, split="full", streaming=True)
        kept = 0
        for row in ds:
            if kept >= per_config:
                break
            if not isinstance(row, dict):
                continue
            text = _amazon_text_from_row(row)
            title = _normalize_text(str(row.get("title") or ""))
            parent_asin = _normalize_text(str(row.get("parent_asin") or ""))
            if not text or not title or not parent_asin:
                continue
            if not _within_len(text, min_tokens=min_tokens, max_tokens=max_tokens):
                continue
            subset = config_name.removeprefix("raw_meta_")
            rows.append(
                _jsonl_row(
                    text=text,
                    title=title,
                    source="human",
                    item_type="product",
                    subset=subset,
                    group_id=f"amazon::{parent_asin}",
                    meta={
                        "origin": "amazon_reviews_2023_meta",
                        "config_name": config_name,
                        "parent_asin": parent_asin,
                        "main_category": row.get("main_category"),
                        "categories": row.get("categories"),
                    },
                )
            )
            kept += 1
    return rows[: int(target_rows)]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, int] = {}

    if not args.skip_hape:
        hape_rows = build_hape_rows(min_tokens=int(args.min_tokens), max_tokens=int(args.max_tokens))
        outputs["hape_excerpt.jsonl"] = _write_jsonl(out_dir / "hape_excerpt.jsonl", hape_rows)
        print(f"Wrote {outputs['hape_excerpt.jsonl']} rows to {out_dir / 'hape_excerpt.jsonl'}")

    if not args.skip_pubmed:
        pubmed_rows = build_pubmed_rows(
            cache_dir=cache_dir,
            target_rows=int(args.pubmed_target),
            min_tokens=int(args.min_tokens),
            max_tokens=int(args.max_tokens),
            timeout_sec=float(args.timeout_sec),
        )
        outputs["pubmed_abstracts_excerpt.jsonl"] = _write_jsonl(out_dir / "pubmed_abstracts_excerpt.jsonl", pubmed_rows)
        print(f"Wrote {outputs['pubmed_abstracts_excerpt.jsonl']} rows to {out_dir / 'pubmed_abstracts_excerpt.jsonl'}")

    if not args.skip_movie:
        movie_rows = build_movie_summary_rows(
            cache_dir=cache_dir,
            target_rows=int(args.movie_target),
            min_tokens=int(args.min_tokens),
            max_tokens=int(args.max_tokens),
            timeout_sec=float(args.timeout_sec),
        )
        outputs["movie_summaries_excerpt.jsonl"] = _write_jsonl(out_dir / "movie_summaries_excerpt.jsonl", movie_rows)
        print(f"Wrote {outputs['movie_summaries_excerpt.jsonl']} rows to {out_dir / 'movie_summaries_excerpt.jsonl'}")

    if not args.skip_amazon:
        amazon_rows = build_amazon_rows(
            target_rows=int(args.amazon_target),
            configs=[part.strip() for part in str(args.amazon_configs).split(",") if part.strip()],
            min_tokens=int(args.min_tokens),
            max_tokens=int(args.max_tokens),
        )
        outputs["amazon_product_descriptions_excerpt.jsonl"] = _write_jsonl(
            out_dir / "amazon_product_descriptions_excerpt.jsonl",
            amazon_rows,
        )
        print(
            "Wrote "
            f"{outputs['amazon_product_descriptions_excerpt.jsonl']} rows to "
            f"{out_dir / 'amazon_product_descriptions_excerpt.jsonl'}"
        )

    with (out_dir / "fetch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote summary to {out_dir / 'fetch_summary.json'}")


if __name__ == "__main__":
    main()
