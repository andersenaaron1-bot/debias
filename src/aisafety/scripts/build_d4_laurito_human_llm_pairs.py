"""Build D4-compatible human-vs-LLM pairs from Laurito-style local domains.

The repo already contains Laurito-style movie/product/paper human and LLM
records. This script emits the same pair schema used by the D4 human-vs-LLM
stage-contrast scorer, so the existing base-vs-instruct and Tulu/Llama matrix
runners can be reused by pointing PAIR_JSONL at this output.

It can also ingest an existing Laurito A/B trials CSV. In that mode, order
swaps are deduplicated to one source pair per title/text combination.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re
from typing import Any

import pandas as pd

from aisafety.config import DATA_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.data import build_all_trials
from aisafety.data.domains import DomainConfig
from aisafety.data.loaders import load_human_map, load_llm_all_by_title
from aisafety.mech.counterfactuals import flat_text, token_count
from aisafety.mech.d4_io import resolve_path, sha1_hex, write_json


DEFAULT_OUT_DIR = DATA_DIR / "derived" / "d4_laurito_human_llm_pairs_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--trials-csv",
        type=Path,
        default=None,
        help=(
            "Optional Laurito A/B trials CSV with item_type,title,A_text,B_text,A_source,B_source. "
            "When omitted, pairs are rebuilt from data/{movie,product,paper}."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--include-item-types",
        type=str,
        default="movie,paper,product",
        help="Comma/colon/space-separated Laurito domains to include.",
    )
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument(
        "--max-pairs-per-item-type",
        type=int,
        default=0,
        help="Deterministic per-domain cap after pair construction. 0 disables.",
    )
    parser.add_argument(
        "--max-total-pairs",
        type=int,
        default=0,
        help="Optional deterministic global round-robin cap over item_type. 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _csv_set(value: str) -> set[str]:
    return {part.strip() for part in re.split(r"[,;:\s]+", str(value or "")) if part.strip()}


def _resolve(workspace_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _local_domain_configs(workspace_root: Path) -> dict[str, DomainConfig]:
    data_dir = Path(workspace_root) / "data"
    return {
        "product": DomainConfig(
            item_type="product",
            human_dir=data_dir / "product" / "human",
            llm_dir=data_dir / "product" / "gpt41106preview",
            prompt_key=None,
        ),
        "movie": DomainConfig(
            item_type="movie",
            human_dir=data_dir / "movie" / "human",
            llm_dir=data_dir / "movie" / "gpt41106preview",
            prompt_key=None,
        ),
        "paper": DomainConfig(
            item_type="paper",
            human_dir=data_dir / "paper" / "human",
            llm_dir=data_dir / "paper" / "gpt41106preview",
            prompt_key=None,
        ),
    }


def _json_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for candidate in path.rglob("*.json") if candidate.is_file())


def _local_domain_inventory(domains_cfg: dict[str, DomainConfig]) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for item_type, cfg in sorted(domains_cfg.items()):
        human_map = load_human_map(cfg.human_dir) if cfg.human_dir.is_dir() else {}
        llm_by_title = (
            load_llm_all_by_title(cfg.llm_dir, prompt_key=cfg.prompt_key)
            if cfg.llm_dir.is_dir()
            else {}
        )
        inventory[item_type] = {
            "human_dir": str(cfg.human_dir),
            "llm_dir": str(cfg.llm_dir),
            "human_dir_exists": bool(cfg.human_dir.is_dir()),
            "llm_dir_exists": bool(cfg.llm_dir.is_dir()),
            "human_json_files": int(_json_count(cfg.human_dir)),
            "llm_json_files": int(_json_count(cfg.llm_dir)),
            "human_titles": int(len(human_map)),
            "llm_titles": int(len(llm_by_title)),
            "shared_titles": int(len(set(human_map) & set(llm_by_title))),
        }
    return inventory


def _norm(value: Any) -> str:
    return flat_text(str(value or ""))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_preview(path: Path, rows: list[dict[str, Any]], *, limit: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = rows[: int(limit)]
    fields = [
        "pair_id",
        "source_dataset",
        "subset",
        "item_type",
        "title",
        "llm_generator",
        "human_token_count",
        "llm_token_count",
        "human_text",
        "llm_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(preview)


def _pair_id(*, item_type: str, title: str, human_text: str, llm_text: str) -> str:
    return sha1_hex(
        "|".join(
            [
                "laurito",
                str(item_type),
                str(title),
                sha1_hex(human_text),
                sha1_hex(llm_text),
            ]
        )
    )


def _pair_row(
    *,
    item_type: str,
    title: str,
    human_text: str,
    llm_text: str,
    llm_generator: str = "",
    split: str = "laurito",
) -> dict[str, Any] | None:
    human_text = _norm(human_text)
    llm_text = _norm(llm_text)
    if not human_text or not llm_text:
        return None
    human_tokens = token_count(human_text)
    llm_tokens = token_count(llm_text)
    pair_id = _pair_id(item_type=item_type, title=title, human_text=human_text, llm_text=llm_text)
    return {
        "pair_id": pair_id,
        "source_dataset": f"laurito_{item_type}",
        "bundle_creation_role": "laurito_quality_validation",
        "group_id": f"laurito::{item_type}::{sha1_hex(title)[:12]}",
        "split": split,
        "item_type": item_type,
        "subset": item_type,
        "title": str(title),
        "question": str(title),
        "prompt": str(title),
        "human_text": human_text,
        "llm_text": llm_text,
        "human_example_id": sha1_hex(f"human:{item_type}:{title}:{human_text}"),
        "llm_example_id": sha1_hex(f"llm:{item_type}:{title}:{llm_text}"),
        "llm_generator": str(llm_generator or ""),
        "human_token_count": int(human_tokens),
        "llm_token_count": int(llm_tokens),
    }


def _pairs_from_trials_csv(path: Path, *, include_item_types: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = pd.read_csv(path)
    required = {"item_type", "title", "A_text", "B_text", "A_source", "B_source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required Laurito trial columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    skipped = Counter()
    seen: set[str] = set()
    for record in df.to_dict(orient="records"):
        item_type = str(record.get("item_type") or "").strip()
        if include_item_types and item_type not in include_item_types:
            skipped["item_type_filtered"] += 1
            continue
        a_source = str(record.get("A_source") or "").strip().lower()
        b_source = str(record.get("B_source") or "").strip().lower()
        if {a_source, b_source} != {"human", "llm"}:
            skipped["source_not_human_llm"] += 1
            continue
        if a_source == "human":
            human_text = str(record.get("A_text") or "")
            llm_text = str(record.get("B_text") or "")
        else:
            human_text = str(record.get("B_text") or "")
            llm_text = str(record.get("A_text") or "")
        row = _pair_row(
            item_type=item_type,
            title=str(record.get("title") or ""),
            human_text=human_text,
            llm_text=llm_text,
            llm_generator=str(record.get("llm_generator") or record.get("generator") or ""),
            split=str(record.get("split") or "laurito_trials"),
        )
        if row is None:
            skipped["empty_text"] += 1
            continue
        if row["pair_id"] in seen:
            skipped["duplicate_order_swap"] += 1
            continue
        seen.add(str(row["pair_id"]))
        rows.append(row)
    return rows, {"source": "trials_csv", "trials_csv": str(path), "skipped": dict(skipped)}


def _pairs_from_local_domains(
    *,
    workspace_root: Path,
    include_item_types: set[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    domains = _local_domain_configs(workspace_root)
    selected_domains = {
        key: cfg for key, cfg in domains.items() if (not include_item_types or key in include_item_types)
    }
    inventory = _local_domain_inventory(selected_domains)
    df = build_all_trials(selected_domains, seed=int(seed), balance_order=False)
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for record in df.to_dict(orient="records"):
        item_type = str(record.get("item_type") or "").strip()
        a_source = str(record.get("A_source") or "").strip().lower()
        b_source = str(record.get("B_source") or "").strip().lower()
        if {a_source, b_source} != {"human", "llm"}:
            skipped["source_not_human_llm"] += 1
            continue
        if a_source == "human":
            human_text = str(record.get("A_text") or "")
            llm_text = str(record.get("B_text") or "")
        else:
            human_text = str(record.get("B_text") or "")
            llm_text = str(record.get("A_text") or "")
        row = _pair_row(
            item_type=item_type,
            title=str(record.get("title") or ""),
            human_text=human_text,
            llm_text=llm_text,
            split="laurito_local",
        )
        if row is None:
            skipped["empty_text"] += 1
            continue
        rows.append(row)
    return rows, {
        "source": "local_domains",
        "local_domain_inventory": inventory,
        "n_local_trials": int(len(df)),
        "skipped": dict(skipped),
    }


def _filter_and_cap(
    rows: list[dict[str, Any]],
    *,
    min_tokens: int,
    max_tokens: int,
    max_pairs_per_item_type: int,
    max_total_pairs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    skipped = Counter()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        human_tokens = int(row.get("human_token_count") or 0)
        llm_tokens = int(row.get("llm_token_count") or 0)
        if min(human_tokens, llm_tokens) < int(min_tokens):
            skipped["below_min_tokens"] += 1
            continue
        if max(human_tokens, llm_tokens) > int(max_tokens):
            skipped["above_max_tokens"] += 1
            continue
        filtered.append(row)

    def sort_key(row: dict[str, Any], salt: str) -> str:
        return sha1_hex(f"{seed}:{salt}:{row.get('source_dataset')}:{row.get('pair_id')}")

    capped = list(filtered)
    if int(max_pairs_per_item_type) > 0:
        by_type: dict[str, list[dict[str, Any]]] = {}
        for row in capped:
            by_type.setdefault(str(row.get("item_type") or ""), []).append(row)
        capped = []
        for item_type, group in sorted(by_type.items()):
            group = sorted(group, key=lambda row: sort_key(row, f"per-type:{item_type}"))
            capped.extend(group[: int(max_pairs_per_item_type)])

    if int(max_total_pairs) > 0 and len(capped) > int(max_total_pairs):
        by_type = {}
        for row in capped:
            by_type.setdefault(str(row.get("item_type") or ""), []).append(row)
        queues = {
            item_type: sorted(group, key=lambda row: sort_key(row, f"total:{item_type}"))
            for item_type, group in sorted(by_type.items())
        }
        selected: list[dict[str, Any]] = []
        keys = sorted(queues)
        rng = random.Random(int(seed))
        rng.shuffle(keys)
        cursor = 0
        while len(selected) < int(max_total_pairs) and any(queues.values()):
            key = keys[cursor % len(keys)]
            cursor += 1
            if queues[key]:
                selected.append(queues[key].pop(0))
        capped = selected

    capped = sorted(
        capped,
        key=lambda row: (
            str(row.get("item_type") or ""),
            str(row.get("title") or ""),
            str(row.get("pair_id") or ""),
        ),
    )
    summary = {
        "n_before_length_filter": int(len(rows)),
        "n_after_length_filter": int(len(filtered)),
        "n_pairs": int(len(capped)),
        "skipped_filter": dict(skipped),
    }
    return capped, summary


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_dataset": dict(sorted(Counter(str(row["source_dataset"]) for row in rows).items())),
        "by_item_type": dict(sorted(Counter(str(row["item_type"]) for row in rows).items())),
        "by_subset": dict(sorted(Counter(str(row["subset"]) for row in rows).items())),
        "mean_human_tokens": (
            sum(int(row["human_token_count"]) for row in rows) / max(len(rows), 1)
        ),
        "mean_llm_tokens": (
            sum(int(row["llm_token_count"]) for row in rows) / max(len(rows), 1)
        ),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    out_dir = _resolve(workspace_root, args.out_dir)
    assert out_dir is not None
    trials_csv = _resolve(workspace_root, args.trials_csv)
    include_item_types = _csv_set(str(args.include_item_types))

    if trials_csv is not None:
        rows, source_summary = _pairs_from_trials_csv(trials_csv, include_item_types=include_item_types)
    else:
        rows, source_summary = _pairs_from_local_domains(
            workspace_root=workspace_root,
            include_item_types=include_item_types,
            seed=int(args.seed),
        )
    rows, filter_summary = _filter_and_cap(
        rows,
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        max_pairs_per_item_type=int(args.max_pairs_per_item_type),
        max_total_pairs=int(args.max_total_pairs),
        seed=int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_path = out_dir / "pairs.jsonl"
    preview_path = out_dir / "pairs_preview.csv"
    summary_path = out_dir / "summary.json"
    payload = {
        "stage": "D4-Laurito-human-LLM-pair-build",
        "out_dir": str(out_dir),
        "pair_jsonl": str(pair_path),
        "preview_csv": str(preview_path),
        "seed": int(args.seed),
        "include_item_types": sorted(include_item_types),
        "min_tokens": int(args.min_tokens),
        "max_tokens": int(args.max_tokens),
        "max_pairs_per_item_type": int(args.max_pairs_per_item_type),
        "max_total_pairs": int(args.max_total_pairs),
        **source_summary,
        **filter_summary,
        **_summary(rows),
    }
    write_json(summary_path, payload)
    if not rows:
        print(f"summary={summary_path}")
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise ValueError(
            "No Laurito human-vs-LLM pairs were emitted. "
            "Check local_domain_inventory in the summary for missing directories, "
            "unmatched titles, or token filtering."
        )

    _write_jsonl(pair_path, rows)
    _write_preview(preview_path, rows)
    print(f"pairs={pair_path}")
    print(f"summary={summary_path}")
    print(f"n_pairs={len(rows)}")
    print(f"by_item_type={payload['by_item_type']}")


if __name__ == "__main__":
    main()
