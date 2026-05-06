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
import re
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
        help="Primary normalized bundle-creation records with source=human/llm and group_id.",
    )
    parser.add_argument(
        "--extra-records-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Additional normalized record JSONL files to append before filtering. May be passed multiple times.",
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
        "--cap-strategy",
        choices=["dataset", "dataset_subset"],
        default="dataset",
        help="Stratum used for deterministic capping.",
    )
    parser.add_argument(
        "--max-pairs-per-dataset",
        type=int,
        default=5000,
        help=(
            "Legacy deterministic cap after all group-level pairs are built. "
            "For cap-strategy=dataset_subset this caps each dataset/subset stratum. "
            "0 disables this cap."
        ),
    )
    parser.add_argument(
        "--max-total-pairs",
        type=int,
        default=0,
        help=(
            "Optional global cap. When set, pairs are selected by deterministic "
            "round-robin over the selected cap-strategy strata."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _csv_set(value: str) -> set[str]:
    return {part.strip() for part in re.split(r"[,;:\s]+", str(value or "")) if part.strip()}


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


def _read_record_inputs(workspace_root: Path, paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for raw_path in paths:
        path = _resolve_path(workspace_root, raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Record JSONL not found: {path}")
        file_rows = _read_jsonl(path)
        rows.extend(file_rows)
        inputs.append({"path": str(path), "n_rows": int(len(file_rows))})
    return rows, inputs


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


def _input_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset_source = Counter()
    by_dataset_role = Counter()
    by_dataset_subset_source = Counter()
    for row in rows:
        dataset_id = _dataset_id(row)
        source = str(row.get("source") or "").lower()
        role = _role(row)
        subset = str(row.get("subset") or row.get("item_type") or "")
        by_dataset_source[(dataset_id, source)] += 1
        by_dataset_role[(dataset_id, role)] += 1
        by_dataset_subset_source[(dataset_id, subset, source)] += 1
    return {
        "by_dataset_source": {
            f"{dataset}::{source}": int(count)
            for (dataset, source), count in sorted(by_dataset_source.items())
        },
        "by_dataset_role": {
            f"{dataset}::{role}": int(count)
            for (dataset, role), count in sorted(by_dataset_role.items())
        },
        "by_dataset_subset_source": {
            f"{dataset}::{subset}::{source}": int(count)
            for (dataset, subset, source), count in sorted(by_dataset_subset_source.items())
        },
    }


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


def _cap_key(pair: dict[str, Any], *, strategy: str) -> tuple[str, ...]:
    if str(strategy) == "dataset_subset":
        return (
            str(pair.get("source_dataset") or ""),
            str(pair.get("subset") or pair.get("item_type") or ""),
        )
    return (str(pair.get("source_dataset") or ""),)


def _stable_pair_sort(
    pairs: list[dict[str, Any]],
    *,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    return sorted(
        pairs,
        key=lambda pair: sha1_hex(
            f"{seed}:{salt}:{pair.get('source_dataset')}:{pair.get('subset')}:{pair.get('pair_id')}"
        ),
    )


def _cap_pairs(
    pairs: list[dict[str, Any]],
    *,
    cap_strategy: str,
    max_pairs_per_dataset: int,
    max_total_pairs: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically cap pairs, optionally balancing across strata."""

    if not pairs:
        return []

    capped = list(pairs)
    if int(max_pairs_per_dataset) > 0:
        by_key: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for pair in capped:
            by_key[_cap_key(pair, strategy=cap_strategy)].append(pair)
        limited: list[dict[str, Any]] = []
        for key, key_pairs in by_key.items():
            limited.extend(
                _stable_pair_sort(
                    key_pairs,
                    seed=seed,
                    salt="pair-cap:" + "::".join(key),
                )[: int(max_pairs_per_dataset)]
            )
        capped = limited

    if int(max_total_pairs) > 0 and len(capped) > int(max_total_pairs):
        by_key = defaultdict(list)
        for pair in capped:
            by_key[_cap_key(pair, strategy=cap_strategy)].append(pair)
        queues = {
            key: _stable_pair_sort(key_pairs, seed=seed, salt="total-cap:" + "::".join(key))
            for key, key_pairs in by_key.items()
        }
        ordered_keys = sorted(queues, key=lambda key: sha1_hex(f"{seed}:stratum-order:{'::'.join(key)}"))
        selected: list[dict[str, Any]] = []
        cursor = 0
        while len(selected) < int(max_total_pairs) and any(queues.values()):
            key = ordered_keys[cursor % len(ordered_keys)]
            cursor += 1
            if queues[key]:
                selected.append(queues[key].pop(0))
        capped = selected

    return sorted(
        capped,
        key=lambda pair: (
            str(pair.get("source_dataset") or ""),
            str(pair.get("subset") or ""),
            str(pair.get("pair_id") or ""),
        ),
    )


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
    cap_strategy: str,
    max_pairs_per_dataset: int,
    max_total_pairs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic human/LLM pairs and a summary payload."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    filtered_inventory_rows: list[dict[str, Any]] = []

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
        filtered_inventory_rows.append(row)

    pairs: list[dict[str, Any]] = []
    unpaired_by_dataset = Counter()
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
            unpaired_by_dataset[dataset_id] += 1
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

    uncapped_count = len(pairs)
    uncapped_by_dataset = dict(Counter(str(pair["source_dataset"]) for pair in pairs))
    uncapped_by_dataset_subset = {
        f"{dataset}::{subset}": int(count)
        for (dataset, subset), count in Counter(
            (str(pair["source_dataset"]), str(pair.get("subset") or pair.get("item_type") or ""))
            for pair in pairs
        ).items()
    }
    pairs = _cap_pairs(
        pairs,
        cap_strategy=str(cap_strategy),
        max_pairs_per_dataset=int(max_pairs_per_dataset),
        max_total_pairs=int(max_total_pairs),
        seed=int(seed),
    )

    summary = {
        "n_input_rows": int(len(rows)),
        "n_pairs": int(len(pairs)),
        "n_uncapped_pairs": int(uncapped_count),
        "n_groups_considered": int(len(grouped)),
        "cap_strategy": str(cap_strategy),
        "max_pairs_per_dataset": int(max_pairs_per_dataset),
        "max_total_pairs": int(max_total_pairs),
        "skipped": dict(skipped),
        "input_inventory": _input_inventory(rows),
        "post_filter_inventory": _input_inventory(filtered_inventory_rows),
        "unpaired_groups_by_dataset": dict(unpaired_by_dataset),
        "uncapped_by_dataset": uncapped_by_dataset,
        "uncapped_by_dataset_subset": uncapped_by_dataset_subset,
        "by_dataset": dict(Counter(str(pair["source_dataset"]) for pair in pairs)),
        "by_dataset_subset": {
            f"{dataset}::{subset}": int(count)
            for (dataset, subset), count in Counter(
                (str(pair["source_dataset"]), str(pair.get("subset") or pair.get("item_type") or ""))
                for pair in pairs
            ).items()
        },
        "by_role": dict(Counter(str(pair["bundle_creation_role"]) for pair in pairs)),
        "by_item_type": dict(Counter(str(pair["item_type"]) for pair in pairs)),
        "by_split": dict(Counter(str(pair["split"]) for pair in pairs)),
    }
    return pairs, summary


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    record_paths = [Path(args.records_jsonl), *[Path(path) for path in (args.extra_records_jsonl or [])]]
    out_dir = _resolve_path(workspace_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, record_inputs = _read_record_inputs(workspace_root, record_paths)
    pairs, summary = build_pairs_from_records(
        rows,
        include_roles=_csv_set(str(args.include_roles)),
        include_datasets=_csv_set(str(args.include_datasets)),
        exclude_datasets=_csv_set(str(args.exclude_datasets)),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        max_human_per_group=int(args.max_human_per_group),
        max_llm_per_group=int(args.max_llm_per_group),
        cap_strategy=str(args.cap_strategy),
        max_pairs_per_dataset=int(args.max_pairs_per_dataset),
        max_total_pairs=int(args.max_total_pairs),
        seed=int(args.seed),
    )
    if not pairs:
        raise ValueError("No human/LLM pairs were built from the provided records.")

    pair_path = out_dir / "pairs.jsonl"
    summary_path = out_dir / "summary.json"
    _write_jsonl(pair_path, pairs)
    summary_payload = {
        "records_jsonl": str(_resolve_path(workspace_root, args.records_jsonl)),
        "record_inputs": record_inputs,
        "extra_records_jsonl": [item["path"] for item in record_inputs[1:]],
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
        "cap_strategy": str(args.cap_strategy),
        "max_pairs_per_dataset": int(args.max_pairs_per_dataset),
        "max_total_pairs": int(args.max_total_pairs),
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
    print(f"by_dataset={summary['by_dataset']}")
    print(f"by_dataset_subset={summary['by_dataset_subset']}")


if __name__ == "__main__":
    main()
