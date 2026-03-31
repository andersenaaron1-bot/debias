"""Cue-discovery corpus builders for human-vs-LLM surface-cue analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from aisafety.data.domains import DOMAINS, DomainConfig
from aisafety.data.hc3 import first_nonempty_answer
from aisafety.data.loaders import _extract_all_texts


_WS_RE = re.compile(r"\s+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
HLLMC2_HF_DATASET = "noepsl/H-LLMC2"


@dataclass(frozen=True)
class CueCorpusRecord:
    """Normalized text record for cue discovery and weak-label induction."""

    example_id: str
    group_id: str
    split: str
    item_type: str
    dataset: str
    subset: str
    source: str
    title: str
    text: str
    generator: str | None = None
    prompt_name: str | None = None
    question: str | None = None
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "").replace("\r\n", "\n").replace("\r", "\n")).strip()


def _slug(text: str, *, max_len: int = 48) -> str:
    slug = _SLUG_RE.sub("_", str(text or "").strip().lower()).strip("_")
    return slug[:max_len] if slug else "item"


def assign_group_split(group_id: str, *, seed: int, train_frac: float = 0.8, val_frac: float = 0.1) -> str:
    """Deterministically assign a split while keeping all variants in a group together."""
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be in (0,1)")
    if not (0.0 <= float(val_frac) < 1.0):
        raise ValueError("val_frac must be in [0,1)")
    if float(train_frac) + float(val_frac) >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    h = _sha1_hex(f"{seed}:{group_id}")
    r = int(h[:8], 16) / float(2**32)
    if r < float(train_frac):
        return "train"
    if r < float(train_frac) + float(val_frac):
        return "val"
    return "test"


def _iter_json_files(root: Path):
    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            yield path


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _dedup_texts(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        norm = _normalize_text(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def iter_local_domain_records(
    cfg: DomainConfig,
    *,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    max_variants_per_group_source: int | None = 4,
) -> list[CueCorpusRecord]:
    """Return all normalized local records for a Laurito-style domain."""

    if not cfg.exists():
        return []

    records: list[CueCorpusRecord] = []
    seen_group_source_text: set[tuple[str, str, str]] = set()
    group_source_counts: Counter[tuple[str, str]] = Counter()
    source_roots = (("human", cfg.human_dir), ("llm", cfg.llm_dir))

    for source, root in source_roots:
        for path in _iter_json_files(Path(root)):
            payload = _safe_load_json(path)
            if payload is None:
                continue

            title = _normalize_text(str(payload.get("title") or path.stem))
            texts = _dedup_texts(_extract_all_texts(payload))
            if not title or not texts:
                continue

            group_id = f"local::{cfg.item_type}::{_slug(title)}::{_sha1_hex(title.lower())[:10]}"
            split = assign_group_split(group_id, seed=seed, train_frac=train_frac, val_frac=val_frac)

            for idx, text in enumerate(texts):
                if (
                    max_variants_per_group_source is not None
                    and int(max_variants_per_group_source) > 0
                    and group_source_counts[(group_id, source)] >= int(max_variants_per_group_source)
                ):
                    break
                dedup_key = (group_id, source, text)
                if dedup_key in seen_group_source_text:
                    continue
                seen_group_source_text.add(dedup_key)
                group_source_counts[(group_id, source)] += 1

                prompt_name = None if source == "human" else payload.get("generation_prompt_nickname")
                generator = "human" if source == "human" else str(payload.get("llm_engine") or "llm")
                meta = {
                    "path": str(path),
                    "origin": payload.get("origin"),
                    "source_xml": payload.get("source_xml"),
                    "variant_index": idx,
                }
                if source == "llm":
                    meta["generation_prompt_uid"] = payload.get("generation_prompt_uid")

                example_id = _sha1_hex(f"{group_id}|{source}|{path.name}|{idx}|{text}")
                records.append(
                    CueCorpusRecord(
                        example_id=example_id,
                        group_id=group_id,
                        split=split,
                        item_type=cfg.item_type,
                        dataset=f"local_{cfg.item_type}",
                        subset=cfg.item_type,
                        source=source,
                        title=title,
                        text=text,
                        generator=generator,
                        prompt_name=None if prompt_name is None else str(prompt_name),
                        question=None,
                        meta=meta,
                    )
                )
    return records


def iter_hc3_records(
    hc3_dir: str | Path,
    *,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> list[CueCorpusRecord]:
    """Return local HC3 records from JSONL files."""

    root = Path(hc3_dir)
    if not root.is_dir():
        return []

    records: list[CueCorpusRecord] = []
    seen_group_source_text: set[tuple[str, str, str]] = set()

    for path in sorted(root.glob("*.jsonl")):
        subset = path.stem
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue

                    question = _normalize_text(str(row.get("question") or ""))
                    human_text = _normalize_text(str(first_nonempty_answer(row.get("human_answers", [])) or ""))
                    llm_text = _normalize_text(str(first_nonempty_answer(row.get("chatgpt_answers", [])) or ""))
                    if not question or not human_text or not llm_text:
                        continue

                    group_id = f"hc3::{subset}::{_sha1_hex(question.lower())[:10]}"
                    split = assign_group_split(group_id, seed=seed, train_frac=train_frac, val_frac=val_frac)
                    pair = (("human", human_text, "human"), ("llm", llm_text, "chatgpt"))
                    for source, text, generator in pair:
                        dedup_key = (group_id, source, text)
                        if dedup_key in seen_group_source_text:
                            continue
                        seen_group_source_text.add(dedup_key)

                        example_id = _sha1_hex(f"{group_id}|{source}|{line_idx}|{text}")
                        records.append(
                            CueCorpusRecord(
                                example_id=example_id,
                                group_id=group_id,
                                split=split,
                                item_type="hc3",
                                dataset="hc3_local",
                                subset=subset,
                                source=source,
                                title=question,
                                question=question,
                                text=text,
                                generator=generator,
                                prompt_name=None,
                                meta={"path": str(path), "line_idx": line_idx},
                            )
                        )
        except OSError:
            continue
    return records


def iter_remote_hc3_records(
    *,
    config_names: list[str],
    max_groups_per_config: int,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    cache_dir: str | Path | None = None,
) -> list[CueCorpusRecord]:
    """Return excerpt-level HC3 records fetched from Hugging Face."""

    if not config_names or int(max_groups_per_config) <= 0:
        return []

    from datasets import load_dataset

    records: list[CueCorpusRecord] = []
    seen_group_source_text: set[tuple[str, str, str]] = set()
    for config_name in config_names:
        ds = load_dataset(
            "Hello-SimpleAI/HC3",
            str(config_name),
            split=f"train[:{int(max_groups_per_config)}]",
            trust_remote_code=True,
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
        for row_idx, row in enumerate(ds):
            if not isinstance(row, dict):
                continue
            question = _normalize_text(str(row.get("question") or ""))
            human_text = _normalize_text(str(first_nonempty_answer(row.get("human_answers", [])) or ""))
            llm_text = _normalize_text(str(first_nonempty_answer(row.get("chatgpt_answers", [])) or ""))
            if not question or not human_text or not llm_text:
                continue

            group_id = f"hc3::{config_name}::{_sha1_hex(question.lower())[:10]}"
            split = assign_group_split(group_id, seed=seed, train_frac=train_frac, val_frac=val_frac)
            pair = (("human", human_text, "human"), ("llm", llm_text, "chatgpt"))
            for source, text, generator in pair:
                dedup_key = (group_id, source, text)
                if dedup_key in seen_group_source_text:
                    continue
                seen_group_source_text.add(dedup_key)
                example_id = _sha1_hex(f"{group_id}|{source}|{row_idx}|{text}")
                records.append(
                    CueCorpusRecord(
                        example_id=example_id,
                        group_id=group_id,
                        split=split,
                        item_type="hc3",
                        dataset="hc3_remote",
                        subset=str(config_name),
                        source=source,
                        title=question,
                        question=question,
                        text=text,
                        generator=generator,
                        prompt_name=None,
                        meta={"hf_dataset": "Hello-SimpleAI/HC3", "config_name": str(config_name), "row_idx": row_idx},
                    )
                )
    return records


def _hllmc2_answer_fields(column_names: list[str]) -> list[str]:
    return [
        str(name)
        for name in column_names
        if str(name).endswith("_answers")
        and str(name) != "human_answers"
        and not str(name).endswith("_thoughts")
    ]


def iter_remote_hllmc2_records(
    *,
    source_names: list[str] | None,
    max_groups_per_source: int,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    cache_dir: str | Path | None = None,
) -> list[CueCorpusRecord]:
    """Return excerpt-level paired human/LLM records from H-LLMC2."""

    if int(max_groups_per_source) <= 0:
        return []

    from datasets import load_dataset

    ds = load_dataset(
        HLLMC2_HF_DATASET,
        split="train",
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    column_names = [str(name) for name in ds.column_names]
    answer_fields = _hllmc2_answer_fields(column_names)
    if not answer_fields:
        return []

    requested_sources = {str(name).strip() for name in (source_names or []) if str(name).strip()}
    counts_by_source: Counter[str] = Counter()
    records: list[CueCorpusRecord] = []
    seen_group_source_text: set[tuple[str, str, str]] = set()

    for row_idx, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        subset = _normalize_text(str(row.get("source") or ""))
        if not subset:
            continue
        if requested_sources and subset not in requested_sources:
            continue
        if counts_by_source[subset] >= int(max_groups_per_source):
            continue

        question = _normalize_text(str(row.get("question") or ""))
        human_text = _normalize_text(str(row.get("human_answers") or ""))
        if not question or not human_text:
            continue

        llm_answers: list[tuple[str, str]] = []
        for field_name in answer_fields:
            text = _normalize_text(str(row.get(field_name) or ""))
            if not text:
                continue
            llm_answers.append((field_name, text))
        if not llm_answers:
            continue

        counts_by_source[subset] += 1
        group_id = f"hc3::{subset}::{_sha1_hex(question.lower())[:10]}"
        split = assign_group_split(group_id, seed=seed, train_frac=train_frac, val_frac=val_frac)
        dedup_key = (group_id, "human", human_text)
        if dedup_key not in seen_group_source_text:
            seen_group_source_text.add(dedup_key)
            example_id = _sha1_hex(f"{group_id}|human|{row_idx}|{human_text}")
            records.append(
                CueCorpusRecord(
                    example_id=example_id,
                    group_id=group_id,
                    split=split,
                    item_type="hc3",
                    dataset="h_llmc2",
                    subset=subset,
                    source="human",
                    title=question,
                    question=question,
                    text=human_text,
                    generator="human",
                    prompt_name=None,
                    meta={
                        "hf_dataset": HLLMC2_HF_DATASET,
                        "row_idx": row_idx,
                        "index": row.get("index"),
                    },
                )
            )

        for field_name, text in llm_answers:
            dedup_key = (group_id, "llm", text)
            if dedup_key in seen_group_source_text:
                continue
            seen_group_source_text.add(dedup_key)
            generator = field_name.removesuffix("_answers")
            example_id = _sha1_hex(f"{group_id}|{generator}|{row_idx}|{text}")
            records.append(
                CueCorpusRecord(
                    example_id=example_id,
                    group_id=group_id,
                    split=split,
                    item_type="hc3",
                    dataset="h_llmc2",
                    subset=subset,
                    source="llm",
                    title=question,
                    question=question,
                    text=text,
                    generator=generator,
                    prompt_name=None,
                    meta={
                        "hf_dataset": HLLMC2_HF_DATASET,
                        "row_idx": row_idx,
                        "index": row.get("index"),
                        "answer_field": field_name,
                    },
                )
            )
    return records


def collect_cue_corpus_records(
    *,
    domains: dict[str, DomainConfig] | None = None,
    include_hc3: bool = True,
    hc3_dir: str | Path | None = None,
    remote_hc3_configs: list[str] | None = None,
    remote_hc3_max_groups_per_config: int = 0,
    remote_cache_dir: str | Path | None = None,
    remote_hllmc2_sources: list[str] | None = None,
    remote_hllmc2_max_groups_per_source: int = 0,
    remote_hllmc2_cache_dir: str | Path | None = None,
    seed: int = 1234,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    max_variants_per_group_source: int | None = 4,
) -> list[CueCorpusRecord]:
    """Collect all discovery-corpus records available locally."""

    doms = DOMAINS if domains is None else domains
    records: list[CueCorpusRecord] = []
    for cfg in doms.values():
        records.extend(
            iter_local_domain_records(
                cfg,
                seed=seed,
                train_frac=train_frac,
                val_frac=val_frac,
                max_variants_per_group_source=max_variants_per_group_source,
            )
        )
    if include_hc3 and hc3_dir is not None:
        records.extend(iter_hc3_records(hc3_dir, seed=seed, train_frac=train_frac, val_frac=val_frac))
    if remote_hc3_configs:
        records.extend(
            iter_remote_hc3_records(
                config_names=[str(x) for x in remote_hc3_configs if str(x).strip()],
                max_groups_per_config=int(remote_hc3_max_groups_per_config),
                seed=seed,
                train_frac=train_frac,
                val_frac=val_frac,
                cache_dir=remote_cache_dir,
            )
        )
    if int(remote_hllmc2_max_groups_per_source) > 0:
        records.extend(
            iter_remote_hllmc2_records(
                source_names=None if remote_hllmc2_sources is None else [str(x) for x in remote_hllmc2_sources],
                max_groups_per_source=int(remote_hllmc2_max_groups_per_source),
                seed=seed,
                train_frac=train_frac,
                val_frac=val_frac,
                cache_dir=remote_hllmc2_cache_dir,
            )
        )
    deduped: list[CueCorpusRecord] = []
    seen_group_source_text: set[tuple[str, str, str]] = set()
    for rec in sorted(records, key=lambda r: (r.item_type, r.group_id, r.source, r.example_id)):
        dedup_key = (rec.group_id, rec.source, _normalize_text(rec.text))
        if dedup_key in seen_group_source_text:
            continue
        seen_group_source_text.add(dedup_key)
        deduped.append(rec)
    return deduped


def limit_records_by_item_type(
    records: list[CueCorpusRecord],
    *,
    max_groups_by_item_type: dict[str, int] | None,
    seed: int,
) -> list[CueCorpusRecord]:
    """Deterministically subsample by group for each item type."""

    if not max_groups_by_item_type:
        return records

    by_item_group: dict[str, dict[str, list[CueCorpusRecord]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        by_item_group[rec.item_type][rec.group_id].append(rec)

    keep_group_ids: set[str] = set()
    for item_type, groups in by_item_group.items():
        limit = int(max_groups_by_item_type.get(item_type, 0))
        if limit <= 0 or len(groups) <= limit:
            keep_group_ids.update(groups)
            continue
        ordered = sorted(groups, key=lambda gid: _sha1_hex(f"{seed}:{item_type}:{gid}"))
        keep_group_ids.update(ordered[:limit])

    kept = [rec for rec in records if rec.group_id in keep_group_ids]
    return sorted(kept, key=lambda r: (r.item_type, r.group_id, r.source, r.example_id))


def summarize_cue_corpus(records: list[CueCorpusRecord]) -> dict[str, Any]:
    """Return compact corpus statistics for logging and provenance."""

    by_item_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_dataset_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_dataset_subset_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_subset_source: dict[str, Counter[str]] = defaultdict(Counter)
    group_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    seen_groups_by_item: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        by_item_source[rec.item_type][rec.source] += 1
        by_dataset_source[rec.dataset][rec.source] += 1
        by_dataset_subset_source[f"{rec.dataset}:{rec.subset}"][rec.source] += 1
        by_subset_source[rec.subset][rec.source] += 1
        split_counts[rec.split] += 1
        seen_groups_by_item[rec.item_type].add(rec.group_id)

    for item_type, groups in seen_groups_by_item.items():
        group_counts[item_type] = len(groups)

    return {
        "n_records": len(records),
        "by_item_type_source": {k: dict(v) for k, v in sorted(by_item_source.items())},
        "by_dataset_source": {k: dict(v) for k, v in sorted(by_dataset_source.items())},
        "by_dataset_subset_source": {k: dict(v) for k, v in sorted(by_dataset_subset_source.items())},
        "by_subset_source": {k: dict(v) for k, v in sorted(by_subset_source.items())},
        "group_counts": dict(sorted(group_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }
