"""Corpus materialization utilities for bundle creation and validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from aisafety.config import DEFAULT_SEED
from aisafety.data.cue_corpus import (
    CueCorpusRecord,
    assign_group_split,
    iter_hc3_records,
    iter_local_domain_records,
    iter_remote_hllmc2_records,
)
from aisafety.data.domains import DOMAINS


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r\n", "\n").replace("\r", "\n").split()).strip()


def load_bundle_creation_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "strata" not in payload:
        raise ValueError(f"Invalid corpus specification: {path}")
    return payload


def _dataset_targets(spec: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for stratum in spec.get("strata", []):
        for dataset in stratum.get("datasets", []):
            dataset_id = str(dataset.get("dataset_id") or "").strip()
            if not dataset_id:
                continue
            target = int(dataset.get("target_texts_preferred", dataset.get("target_texts_min", 0)) or 0)
            if target > 0:
                out[dataset_id] = target
    return out


def _annotate_record(
    rec: CueCorpusRecord,
    *,
    dataset_id: str,
    role: str,
    stratum_id: str,
    holdout_from_discovery: bool = False,
) -> CueCorpusRecord:
    meta = dict(rec.meta or {})
    meta.update(
        {
            "bundle_creation_dataset_id": dataset_id,
            "bundle_creation_role": role,
            "bundle_creation_stratum_id": stratum_id,
            "holdout_from_discovery": bool(holdout_from_discovery),
        }
    )
    return replace(rec, dataset=dataset_id, meta=meta)


def iter_excerpt_jsonl_records(
    path: str | Path,
    *,
    dataset_id: str,
    item_type_default: str,
    subset_default: str,
    source_default: str | None,
    role: str,
    stratum_id: str,
    holdout_from_discovery: bool,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    max_records: int | None = None,
) -> list[CueCorpusRecord]:
    """Load normalized excerpt-level JSONL corpora for bundle creation.

    Expected minimal schema:
    - ``text``: required

    Optional fields:
    - ``title``
    - ``question``
    - ``source``: ``human`` or ``llm``
    - ``generator``
    - ``item_type``
    - ``subset``
    - ``group_id``
    - ``prompt_name``
    - ``meta``
    """

    path = Path(path)
    if not path.is_file():
        return []

    records: list[CueCorpusRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_records is not None and len(records) >= int(max_records):
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            text = _normalize_text(str(row.get("text") or ""))
            if not text:
                continue
            title = _normalize_text(str(row.get("title") or row.get("question") or path.stem))
            question = _normalize_text(str(row.get("question") or ""))
            source = _normalize_text(str(row.get("source") or source_default or "human")).lower()
            generator = row.get("generator")
            subset = _normalize_text(str(row.get("subset") or subset_default or item_type_default))
            item_type = _normalize_text(str(row.get("item_type") or item_type_default))
            group_id = _normalize_text(str(row.get("group_id") or ""))
            if not group_id:
                base = question or title or text[:120]
                group_id = f"{dataset_id}::{_sha1_hex(base.lower())[:12]}"
            split = assign_group_split(group_id, seed=seed, train_frac=train_frac, val_frac=val_frac)
            example_id = _sha1_hex(f"{dataset_id}|{group_id}|{line_idx}|{text}")
            meta = dict(row.get("meta") or {})
            meta["path"] = str(path)
            meta["line_idx"] = int(line_idx)
            rec = CueCorpusRecord(
                example_id=example_id,
                group_id=group_id,
                split=split,
                item_type=item_type,
                dataset=dataset_id,
                subset=subset,
                source=source,
                title=title,
                text=text,
                generator=None if generator is None else str(generator),
                prompt_name=None if row.get("prompt_name") is None else str(row.get("prompt_name")),
                question=question or None,
                meta=meta,
            )
            records.append(
                _annotate_record(
                    rec,
                    dataset_id=dataset_id,
                    role=role,
                    stratum_id=stratum_id,
                    holdout_from_discovery=holdout_from_discovery,
                )
            )
    return records


def limit_records_by_dataset(
    records: list[CueCorpusRecord],
    *,
    max_records_by_dataset: dict[str, int] | None,
    seed: int,
) -> list[CueCorpusRecord]:
    """Deterministically subsample while keeping dataset/group units together."""

    if not max_records_by_dataset:
        return records

    by_dataset_group: dict[str, dict[str, list[CueCorpusRecord]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        dataset_id = str((rec.meta or {}).get("bundle_creation_dataset_id") or rec.dataset)
        by_dataset_group[dataset_id][rec.group_id].append(rec)

    kept: list[CueCorpusRecord] = []
    for dataset_id, groups in by_dataset_group.items():
        limit = int(max_records_by_dataset.get(dataset_id, 0))
        ordered_group_ids = sorted(
            groups,
            key=lambda gid: _sha1_hex(f"{seed}:{dataset_id}:{gid}"),
        )
        if limit <= 0:
            chosen = ordered_group_ids
        else:
            chosen = []
            count = 0
            for gid in ordered_group_ids:
                group_recs = groups[gid]
                if chosen and count + len(group_recs) > limit:
                    break
                chosen.append(gid)
                count += len(group_recs)
        for gid in chosen:
            kept.extend(groups[gid])

    return sorted(
        kept,
        key=lambda rec: (
            str((rec.meta or {}).get("bundle_creation_role") or ""),
            str((rec.meta or {}).get("bundle_creation_dataset_id") or rec.dataset),
            rec.group_id,
            rec.source,
            rec.example_id,
        ),
    )


def summarize_bundle_creation_records(records: list[CueCorpusRecord]) -> dict[str, Any]:
    by_role_dataset: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    by_role_item_type: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    counts_by_role = Counter()
    counts_by_dataset = Counter()
    counts_by_source = Counter()
    counts_by_split = Counter()

    for rec in records:
        meta = rec.meta or {}
        role = str(meta.get("bundle_creation_role") or "unknown")
        dataset_id = str(meta.get("bundle_creation_dataset_id") or rec.dataset)
        counts_by_role[role] += 1
        counts_by_dataset[dataset_id] += 1
        counts_by_source[rec.source] += 1
        counts_by_split[rec.split] += 1

    for role in sorted({str((rec.meta or {}).get("bundle_creation_role") or "unknown") for rec in records}):
        role_records = [rec for rec in records if str((rec.meta or {}).get("bundle_creation_role") or "unknown") == role]
        for dataset_id in sorted({str((rec.meta or {}).get("bundle_creation_dataset_id") or rec.dataset) for rec in role_records}):
            subset = [rec for rec in role_records if str((rec.meta or {}).get("bundle_creation_dataset_id") or rec.dataset) == dataset_id]
            by_role_dataset[role][dataset_id] = dict(Counter(rec.source for rec in subset))
        for item_type in sorted({rec.item_type for rec in role_records}):
            subset = [rec for rec in role_records if rec.item_type == item_type]
            by_role_item_type[role][item_type] = dict(Counter(rec.source for rec in subset))

    return {
        "n_records": int(len(records)),
        "by_role": dict(counts_by_role),
        "by_dataset": dict(counts_by_dataset),
        "by_source": dict(counts_by_source),
        "by_split": dict(counts_by_split),
        "by_role_dataset_source": {role: dict(ds) for role, ds in by_role_dataset.items()},
        "by_role_item_type_source": {role: dict(it) for role, it in by_role_item_type.items()},
    }


def materialize_bundle_creation_records(
    spec: dict[str, Any],
    *,
    hc3_dir: str | Path | None = None,
    remote_hllmc2_sources: list[str] | None = None,
    remote_hllmc2_max_groups_per_source: int = 0,
    remote_hllmc2_cache_dir: str | Path | None = None,
    hape_jsonl: str | Path | None = None,
    pubmed_jsonl: str | Path | None = None,
    movie_summary_jsonl: str | Path | None = None,
    product_jsonl: str | Path | None = None,
    paper_llm_jsonl: str | Path | None = None,
    movie_llm_jsonl: str | Path | None = None,
    product_llm_jsonl: str | Path | None = None,
    hc3_plus_jsonl: str | Path | None = None,
    rewrite_jsonl: str | Path | None = None,
    include_laurito_ecology: bool = True,
    seed: int = DEFAULT_SEED,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    max_local_variants_per_group_source: int | None = 4,
) -> tuple[list[CueCorpusRecord], dict[str, Any]]:
    """Materialize a stratified corpus from the bundle-creation spec."""

    dataset_targets = _dataset_targets(spec)
    records: list[CueCorpusRecord] = []
    missing_inputs: list[dict[str, Any]] = []

    if hc3_dir is not None and Path(hc3_dir).is_dir():
        hc3_records = [
            _annotate_record(rec, dataset_id="hc3", role="discovery_core", stratum_id="A1")
            for rec in iter_hc3_records(hc3_dir, seed=seed, train_frac=train_frac, val_frac=val_frac)
        ]
        records.extend(hc3_records)
    else:
        missing_inputs.append({"dataset_id": "hc3", "reason": "hc3_dir_missing"})

    if int(remote_hllmc2_max_groups_per_source) > 0:
        hllmc2_records = [
            _annotate_record(rec, dataset_id="h_llmc2", role="discovery_core", stratum_id="A1")
            for rec in iter_remote_hllmc2_records(
                source_names=remote_hllmc2_sources,
                max_groups_per_source=int(remote_hllmc2_max_groups_per_source),
                seed=seed,
                train_frac=train_frac,
                val_frac=val_frac,
                cache_dir=remote_hllmc2_cache_dir,
            )
        ]
        records.extend(hllmc2_records)
    else:
        missing_inputs.append({"dataset_id": "h_llmc2", "reason": "remote_sampling_not_requested"})

    generic_inputs = [
        ("hape", hape_jsonl, "general", "hape", None, "discovery_core", "A1", False),
        ("pubmed_abstracts", pubmed_jsonl, "paper", "paper", "human", "domain_bolster", "B1", False),
        ("cmu_movie_summary", movie_summary_jsonl, "movie", "movie", "human", "domain_bolster", "B1", False),
        ("amazon_product_descriptions", product_jsonl, "product", "product", "human", "domain_bolster", "B1", False),
        ("paper_llm_supplement", paper_llm_jsonl, "paper", "paper", "llm", "domain_bolster", "B1", False),
        ("movie_llm_supplement", movie_llm_jsonl, "movie", "movie", "llm", "domain_bolster", "B1", False),
        ("product_llm_supplement", product_llm_jsonl, "product", "product", "llm", "domain_bolster", "B1", False),
        ("hc3_plus", hc3_plus_jsonl, "general", "hc3_plus", None, "controlled_confirmation", "C1", False),
        ("local_rewrite_control", rewrite_jsonl, "general", "rewrites", None, "controlled_confirmation", "C1", False),
    ]
    for dataset_id, path, item_type, subset, source_default, role, stratum_id, holdout in generic_inputs:
        if path is None:
            missing_inputs.append({"dataset_id": dataset_id, "reason": "path_not_provided"})
            continue
        loaded = iter_excerpt_jsonl_records(
            path,
            dataset_id=dataset_id,
            item_type_default=item_type,
            subset_default=subset,
            source_default=source_default,
            role=role,
            stratum_id=stratum_id,
            holdout_from_discovery=holdout,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            max_records=dataset_targets.get(dataset_id),
        )
        if loaded:
            records.extend(loaded)
        else:
            missing_inputs.append({"dataset_id": dataset_id, "reason": "path_missing_or_empty"})

    if include_laurito_ecology:
        for cfg in DOMAINS.values():
            loc = iter_local_domain_records(
                cfg,
                seed=seed,
                train_frac=train_frac,
                val_frac=val_frac,
                max_variants_per_group_source=max_local_variants_per_group_source,
            )
            records.extend(
                [
                    _annotate_record(
                        rec,
                        dataset_id=f"laurito_{cfg.item_type}",
                        role="ecological_validation",
                        stratum_id="D1",
                        holdout_from_discovery=True,
                    )
                    for rec in loc
                ]
            )

    limited = limit_records_by_dataset(records, max_records_by_dataset=dataset_targets, seed=seed)
    summary = summarize_bundle_creation_records(limited)
    summary["missing_inputs"] = missing_inputs
    summary["dataset_targets"] = dataset_targets
    return limited, summary
