"""Build D4 human-vs-LLM pair files for broad judge-alignment tests.

The input is the normalized bundle-creation corpus JSONL. Rows are grouped by
dataset and group id, then human rows are paired with one or more LLM rows from
the same group. This deliberately does not score the pairs; scoring happens in
the SAE candidate-alignment pass so the same pair file can be reused across
judges.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from aisafety.config import DATA_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import sha1_hex, write_json


DEFAULT_INCLUDED_ROLES = (
    "discovery_core",
    "controlled_confirmation",
    "heldout_transfer",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--records-jsonl",
        type=Path,
        default=DATA_DIR / "derived" / "bundle_creation_corpus_v1" / "all_records.jsonl",
        help="Normalized bundle-creation records with source=human/llm and group_id.",
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "derived" / "d4_human_llm_alignment_pairs_v1",
    )
    parser.add_argument(
        "--include-roles",
        type=str,
        default=",".join(DEFAULT_INCLUDED_ROLES),
        help="Comma-separated bundle_creation_role values to include. Empty means all roles.",
    )
    parser.add_argument(
        "--include-datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset ids to include after role filtering.",
    )
    parser.add_argument(
        "--exclude-datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset ids to exclude.",
    )
    parser.add_argument(
        "--require-datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset ids that must produce at least one pair.",
    )
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--max-human-per-group", type=int, default=1)
    parser.add_argument("--max-llm-per-group", type=int, default=2)
    parser.add_argument(
        "--max-pairs-per-dataset",
        type=int,
        default=5000,
        help="Deterministic cap after all group-level pairs are built. 0 disables the cap.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _csv_set(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def _resolve_path(base: Path, value: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(base) / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _norm(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split()).strip()


def _token_count(text: str) -> int:
    return len(_norm(text).split())


def _dataset_id(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(
        meta.get("bundle_creation_dataset_id")
        or row.get("dataset")
        or row.get("source_dataset")
        or "unknown"
    )


def _role(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(meta.get("bundle_creation_role") or row.get("role") or "unknown")


def _row_sort_key(row: dict[str, Any], *, seed: int, salt: str) -> str:
    return sha1_hex(
        "|".join(
            [
                str(seed),
                salt,
                str(row.get("example_id") or ""),
                str(row.get("generator") or ""),
                _norm(row.get("text"))[:200],
            ]
        )
    )


def _selected_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    subset = [row for row in rows if str(row.get("source") or "").lower() == source]
    subset.sort(key=lambda row: _row_sort_key(row, seed=seed, salt=source))
    if int(limit) <= 0:
        return subset
    return subset[: int(limit)]


def build_pairs_from_records(
    rows: list[dict[str, Any]],
    *,
    include_roles: set[str],
    include_datasets: set[str],
    exclude_datasets: set[str],
    min_tokens: int,
    max_tokens: int,
    max_human_per_group: int,
    max_llm_per_group: int,
    max_pairs_per_dataset: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic human/LLM pairs and a summary payload."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()

    for row in rows:
        dataset_id = _dataset_id(row)
        role = _role(row)
        if include_roles and role not in include_roles:
            skipped["role_filtered"] += 1
            continue
        if include_datasets and dataset_id not in include_datasets:
            skipped["dataset_filtered"] += 1
            continue
        if dataset_id in exclude_datasets:
            skipped["dataset_excluded"] += 1
            continue
        source = str(row.get("source") or "").lower()
        if source not in {"human", "llm"}:
            skipped["source_not_human_or_llm"] += 1
            continue
        text = _norm(row.get("text"))
        n_tokens = _token_count(text)
        if not text or n_tokens < int(min_tokens) or n_tokens > int(max_tokens):
            skipped["length_filtered"] += 1
            continue
        group_id = str(row.get("group_id") or row.get("title") or row.get("question") or "").strip()
        if not group_id:
            group_id = sha1_hex(text[:200])
        row = dict(row)
        row["text"] = text
        grouped[(dataset_id, group_id)].append(row)

    pairs: list[dict[str, Any]] = []
    for (dataset_id, group_id), group_rows in sorted(grouped.items()):
        humans = _selected_rows(
            group_rows,
            source="human",
            limit=max_human_per_group,
            seed=seed,
        )
        llms = _selected_rows(
            group_rows,
            source="llm",
            limit=max_llm_per_group,
            seed=seed,
        )
        if not humans or not llms:
            skipped["unpaired_group"] += 1
            continue

        for human in humans:
            for llm in llms:
                pair_key = "|".join(
                    [
                        dataset_id,
                        str(group_id),
                        str(human.get("example_id") or ""),
                        str(llm.get("example_id") or ""),
                        human["text"][:120],
                        llm["text"][:120],
                    ]
                )
                meta = human.get("meta") if isinstance(human.get("meta"), dict) else {}
                pair = {
                    "pair_id": sha1_hex(pair_key),
                    "source_dataset": dataset_id,
                    "bundle_creation_role": _role(human),
                    "group_id": str(group_id),
                    "split": str(human.get("split") or llm.get("split") or ""),
                    "item_type": str(human.get("item_type") or llm.get("item_type") or ""),
                    "subset": str(human.get("subset") or llm.get("subset") or ""),
                    "title": str(human.get("title") or llm.get("title") or ""),
                    "question": str(human.get("question") or llm.get("question") or ""),
                    "human_text": human["text"],
                    "llm_text": llm["text"],
                    "human_example_id": str(human.get("example_id") or ""),
                    "llm_example_id": str(llm.get("example_id") or ""),
                    "llm_generator": str(llm.get("generator") or ""),
                    "human_token_count": _token_count(human["text"]),
                    "llm_token_count": _token_count(llm["text"]),
                    "meta_path": str(meta.get("path") or ""),
                }
                pairs.append(pair)

    if int(max_pairs_per_dataset) > 0:
        limited: list[dict[str, Any]] = []
        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in pairs:
            by_dataset[str(pair["source_dataset"])].append(pair)
        for dataset_id, dataset_pairs in by_dataset.items():
            dataset_pairs.sort(
                key=lambda pair: sha1_hex(f"{seed}:pair-cap:{dataset_id}:{pair['pair_id']}")
            )
            limited.extend(dataset_pairs[: int(max_pairs_per_dataset)])
        pairs = sorted(limited, key=lambda pair: (str(pair["source_dataset"]), str(pair["pair_id"])))
    else:
        pairs = sorted(pairs, key=lambda pair: (str(pair["source_dataset"]), str(pair["pair_id"])))

    summary = {
        "n_input_rows": int(len(rows)),
        "n_pairs": int(len(pairs)),
        "n_groups_considered": int(len(grouped)),
        "skipped": dict(skipped),
        "by_dataset": dict(Counter(str(pair["source_dataset"]) for pair in pairs)),
        "by_role": dict(Counter(str(pair["bundle_creation_role"]) for pair in pairs)),
        "by_item_type": dict(Counter(str(pair["item_type"]) for pair in pairs)),
        "by_split": dict(Counter(str(pair["split"]) for pair in pairs)),
    }
    return pairs, summary


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    records_jsonl = _resolve_path(workspace_root, args.records_jsonl)
    out_dir = _resolve_path(workspace_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(records_jsonl)
    pairs, summary = build_pairs_from_records(
        rows,
        include_roles=_csv_set(str(args.include_roles)),
        include_datasets=_csv_set(str(args.include_datasets)),
        exclude_datasets=_csv_set(str(args.exclude_datasets)),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        max_human_per_group=int(args.max_human_per_group),
        max_llm_per_group=int(args.max_llm_per_group),
        max_pairs_per_dataset=int(args.max_pairs_per_dataset),
        seed=int(args.seed),
    )
    if not pairs:
        raise ValueError("No human/LLM pairs were built from the provided records.")

    pair_path = out_dir / "pairs.jsonl"
    summary_path = out_dir / "summary.json"
    _write_jsonl(pair_path, pairs)
    summary_payload = {
        "records_jsonl": str(records_jsonl),
        "out_dir": str(out_dir),
        "pair_jsonl": str(pair_path),
        "seed": int(args.seed),
        "include_roles": sorted(_csv_set(str(args.include_roles))),
        "include_datasets": sorted(_csv_set(str(args.include_datasets))),
        "exclude_datasets": sorted(_csv_set(str(args.exclude_datasets))),
        "min_tokens": int(args.min_tokens),
        "max_tokens": int(args.max_tokens),
        "max_human_per_group": int(args.max_human_per_group),
        "max_llm_per_group": int(args.max_llm_per_group),
        "max_pairs_per_dataset": int(args.max_pairs_per_dataset),
        **summary,
    }
    required_datasets = _csv_set(str(args.require_datasets))
    missing_required = sorted(ds for ds in required_datasets if int(summary["by_dataset"].get(ds, 0)) <= 0)
    if missing_required:
        summary_payload["missing_required_datasets"] = missing_required
        write_json(summary_path, summary_payload)
        raise ValueError(f"Required datasets produced no pairs: {missing_required}")
    write_json(summary_path, summary_payload)
    print(f"pairs={pair_path}")
    print(f"summary={summary_path}")
    print(f"n_pairs={len(pairs)}")


if __name__ == "__main__":
    main()
