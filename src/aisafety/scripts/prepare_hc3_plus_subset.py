"""Normalize an HC3 Plus checkout/archive into paired bundle-corpus JSONL.

The official HC3 Plus repository stores detector-style JSONL files. This script
extracts paired human/ChatGPT rows where possible and writes normalized records
that can be appended to the bundle-creation corpus before building D4
human-vs-LLM alignment pairs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import zipfile
from typing import Any

from aisafety.config import DATA_DIR, DEFAULT_SEED
from aisafety.mech.d4_io import sha1_hex, write_json


PAIR_FIELD_SETS = (
    ("human", "chatgpt"),
    ("human_text", "chatgpt_text"),
    ("human_answer", "chatgpt_answer"),
    ("human_answers", "chatgpt_answers"),
    ("human_written", "chatgpt_generated"),
    ("human-written", "ChatGPT-generated"),
)
TEXT_FIELDS = ("text", "content", "answer", "output", "target", "sentence")
GROUP_FIELDS = (
    "source_text",
    "source",
    "src",
    "input",
    "prompt",
    "question",
    "article",
    "document",
    "context",
    "origin",
    "raw",
    "id",
)
SUBSET_FIELDS = ("task", "dataset", "data_source", "source_dataset", "category", "subset")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="HC3 Plus repo checkout, unpacked archive directory, or GitHub .zip archive.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=DATA_DIR / "external" / "bundle_creation_v1" / "hc3_plus_subset.jsonl",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--languages", type=str, default="en")
    parser.add_argument(
        "--file-glob",
        type=str,
        default="*.jsonl",
        help="Glob applied below data/<language>; keep this broad to retain HC3+ QA/SI/train strata.",
    )
    parser.add_argument("--max-pairs", type=int, default=2000)
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--human-labels", type=str, default="0,human,Human,human-written,human_written")
    parser.add_argument(
        "--llm-labels",
        type=str,
        default="1,chatgpt,ChatGPT,chatgpt-generated,chatgpt_generated,llm,AI",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _csv_set(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def _norm(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(x) for x in value if str(x).strip())
    if value is None:
        value = ""
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split()).strip()


def _token_count(text: str) -> int:
    return len(_norm(text).split())


def _find_root(path: Path) -> Path:
    path = Path(path).resolve()
    if path.is_dir():
        if (path / "data").is_dir():
            return path
        matches = [p for p in path.rglob("data") if p.is_dir() and (p / "en").is_dir()]
        if matches:
            return matches[0].parent
        return path
    if path.name.endswith(".zip"):
        extract_dir = path.with_suffix("").with_suffix("")
        if not extract_dir.exists():
            with zipfile.ZipFile(path) as zf:
                for member in zf.infolist():
                    target = (extract_dir / member.filename).resolve()
                    if not str(target).startswith(str(extract_dir.resolve())):
                        raise ValueError(f"Unsafe zip member: {member.filename}")
                zf.extractall(extract_dir)
        return _find_root(extract_dir)
    raise FileNotFoundError(path)


def _iter_files(root: Path, *, languages: list[str], file_glob: str) -> list[Path]:
    files: list[Path] = []
    for lang in languages:
        lang_dir = root / "data" / lang
        if lang_dir.is_dir():
            files.extend(sorted(lang_dir.glob(file_glob)))
    return [path for path in files if path.is_file()]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _row_subset(row: dict[str, Any], *, path: Path) -> str:
    for field in SUBSET_FIELDS:
        if field in row and _norm(row[field]):
            return _norm(row[field])
    stem = path.stem.lower()
    if "summ" in stem:
        return "summarization"
    if "trans" in stem:
        return "translation"
    if "para" in stem:
        return "paraphrasing"
    return stem


def _row_group_value(row: dict[str, Any]) -> str:
    for field in GROUP_FIELDS:
        if field in row and _norm(row[field]):
            return _norm(row[field])
    return ""


def _row_text(row: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        if field in row and _norm(row[field]):
            return _norm(row[field])
    return ""


def _row_source(row: dict[str, Any], *, human_labels: set[str], llm_labels: set[str]) -> str | None:
    for field in ("source", "label", "labels", "class", "target", "y"):
        if field not in row:
            continue
        value = _norm(row[field])
        if value in human_labels:
            return "human"
        if value in llm_labels:
            return "llm"
    return None


def _emit_pair_records(
    *,
    human_text: str,
    llm_text: str,
    group_value: str,
    subset: str,
    path: Path,
    row_idx: int,
    seed: int,
) -> list[dict[str, Any]]:
    human_text = _norm(human_text)
    llm_text = _norm(llm_text)
    group_value = _norm(group_value) or f"{path.stem}:{row_idx}"
    group_id = f"hc3_plus::{subset}::{sha1_hex(group_value.lower())[:14]}"
    common = {
        "item_type": "general",
        "dataset": "hc3_plus",
        "subset": subset,
        "group_id": group_id,
        "title": group_value[:200],
        "question": group_value[:500],
        "prompt_name": None,
        "meta": {
            "path": str(path),
            "row_idx": int(row_idx),
            "origin": "suu990901/chatgpt-comparison-detection-HC3-Plus",
            "seed": int(seed),
            "bundle_creation_dataset_id": "hc3_plus",
            "bundle_creation_role": "controlled_confirmation",
            "bundle_creation_stratum_id": "C1",
            "holdout_from_discovery": False,
        },
    }
    return [
        {
            **common,
            "text": human_text,
            "source": "human",
            "generator": "human",
        },
        {
            **common,
            "text": llm_text,
            "source": "llm",
            "generator": "chatgpt",
        },
    ]


def _paired_field_records(path: Path, rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        subset = _row_subset(row, path=path)
        group_value = _row_group_value(row)
        for human_field, llm_field in PAIR_FIELD_SETS:
            if human_field not in row or llm_field not in row:
                continue
            human_text = _norm(row[human_field])
            llm_text = _norm(row[llm_field])
            if human_text and llm_text:
                out.extend(
                    _emit_pair_records(
                        human_text=human_text,
                        llm_text=llm_text,
                        group_value=group_value,
                        subset=subset,
                        path=path,
                        row_idx=row_idx,
                        seed=seed,
                    )
                )
                break
    return out


def _classifier_row_records(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    human_labels: set[str],
    llm_labels: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for row_idx, row in enumerate(rows):
        source = _row_source(row, human_labels=human_labels, llm_labels=llm_labels)
        text = _row_text(row)
        group_value = _row_group_value(row)
        if source is None or not text or not group_value:
            continue
        grouped[(_row_subset(row, path=path), group_value)][source].append((row_idx, text))

    out: list[dict[str, Any]] = []
    for (subset, group_value), by_source in grouped.items():
        humans = by_source.get("human") or []
        llms = by_source.get("llm") or []
        if not humans or not llms:
            continue
        human_idx, human_text = sorted(humans, key=lambda item: sha1_hex(f"{seed}:h:{item[0]}:{item[1][:80]}"))[0]
        llm_idx, llm_text = sorted(llms, key=lambda item: sha1_hex(f"{seed}:l:{item[0]}:{item[1][:80]}"))[0]
        out.extend(
            _emit_pair_records(
                human_text=human_text,
                llm_text=llm_text,
                group_value=group_value,
                subset=subset,
                path=path,
                row_idx=min(human_idx, llm_idx),
                seed=seed,
            )
        )
    return out


def _pick_run_item(items: list[tuple[int, str]], *, seed: int, source: str) -> tuple[int, str]:
    return sorted(items, key=lambda item: sha1_hex(f"{seed}:{source}:{item[0]}:{item[1][:120]}"))[0]


def _sequential_label_records(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    human_labels: set[str],
    llm_labels: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Pair adjacent detector-label runs when HC3+ omits prompt/group fields."""

    runs: list[tuple[str, list[tuple[int, str]], str]] = []
    current_source: str | None = None
    current_subset = ""
    current_items: list[tuple[int, str]] = []
    for row_idx, row in enumerate(rows):
        source = _row_source(row, human_labels=human_labels, llm_labels=llm_labels)
        text = _row_text(row)
        if source is None or not text:
            continue
        subset = _row_subset(row, path=path)
        if source != current_source and current_items:
            runs.append((str(current_source), current_items, current_subset))
            current_items = []
        current_source = source
        current_subset = subset
        current_items.append((row_idx, text))
    if current_items and current_source is not None:
        runs.append((current_source, current_items, current_subset))

    out: list[dict[str, Any]] = []
    idx = 0
    while idx < len(runs) - 1:
        left_source, left_items, left_subset = runs[idx]
        right_source, right_items, right_subset = runs[idx + 1]
        if left_source == right_source:
            idx += 1
            continue
        left_idx, left_text = _pick_run_item(left_items, seed=seed, source=left_source)
        right_idx, right_text = _pick_run_item(right_items, seed=seed, source=right_source)
        if left_source == "human":
            human_idx, human_text = left_idx, left_text
            llm_text = right_text
        else:
            human_idx, human_text = right_idx, right_text
            llm_text = left_text
        subset = left_subset if left_subset == right_subset else f"{left_subset}+{right_subset}"
        group_value = f"{path.stem}:{min(left_idx, right_idx)}:{max(left_idx, right_idx)}"
        out.extend(
            _emit_pair_records(
                human_text=human_text,
                llm_text=llm_text,
                group_value=group_value,
                subset=subset,
                path=path,
                row_idx=human_idx,
                seed=seed,
            )
        )
        idx += 2
    return out


def _dedup_and_cap(records: list[dict[str, Any]], *, max_pairs: int, min_tokens: int, max_tokens: int, seed: int):
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        text = _norm(row.get("text"))
        n_tokens = _token_count(text)
        if not text or n_tokens < int(min_tokens) or n_tokens > int(max_tokens):
            continue
        row = dict(row)
        row["text"] = text
        by_group[str(row["group_id"])].append(row)

    pairs: list[list[dict[str, Any]]] = []
    for group_id, rows in by_group.items():
        humans = [row for row in rows if row["source"] == "human"]
        llms = [row for row in rows if row["source"] == "llm"]
        if not humans or not llms:
            continue
        pairs.append([humans[0], llms[0]])

    pairs.sort(key=lambda pair: sha1_hex(f"{seed}:hc3-plus:{pair[0]['subset']}:{pair[0]['group_id']}"))
    if int(max_pairs) > 0:
        by_subset: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
        for pair in pairs:
            by_subset[str(pair[0]["subset"])].append(pair)
        selected: list[list[dict[str, Any]]] = []
        keys = sorted(by_subset, key=lambda key: sha1_hex(f"{seed}:subset:{key}"))
        while len(selected) < int(max_pairs) and any(by_subset.values()):
            for key in keys:
                if by_subset[key] and len(selected) < int(max_pairs):
                    selected.append(by_subset[key].pop(0))
        pairs = selected

    out: list[dict[str, Any]] = []
    for pair in pairs:
        out.extend(pair)
    return out


def main() -> None:
    args = _parse_args()
    root = _find_root(args.input)
    files = _iter_files(root, languages=[x.strip() for x in str(args.languages).split(",") if x.strip()], file_glob=str(args.file_glob))
    if not files:
        raise FileNotFoundError(f"No HC3 Plus files matched under {root}/data/<languages> with {args.file_glob!r}")

    human_labels = _csv_set(str(args.human_labels))
    llm_labels = _csv_set(str(args.llm_labels))
    records: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []
    for path in files:
        rows = _read_jsonl(path)
        paired = _paired_field_records(path, rows, seed=int(args.seed))
        if not paired:
            paired = _classifier_row_records(
                path,
                rows,
                human_labels=human_labels,
                llm_labels=llm_labels,
                seed=int(args.seed),
            )
        if not paired:
            paired = _sequential_label_records(
                path,
                rows,
                human_labels=human_labels,
                llm_labels=llm_labels,
                seed=int(args.seed),
            )
        records.extend(paired)
        file_summaries.append({"path": str(path), "raw_rows": len(rows), "normalized_records": len(paired)})

    records = _dedup_and_cap(
        records,
        max_pairs=int(args.max_pairs),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        seed=int(args.seed),
    )
    if not records:
        first_rows = _read_jsonl(files[0])[:2]
        raise ValueError(
            "No paired HC3 Plus records were produced. Inspect raw schema; "
            f"first_file={files[0]} first_rows={first_rows}"
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "input": str(args.input),
        "root": str(root),
        "files": file_summaries,
        "out_jsonl": str(args.out_jsonl),
        "n_records": len(records),
        "n_pairs": len(records) // 2,
        "by_subset": dict(Counter(str(row["subset"]) for row in records if row["source"] == "human")),
        "by_source": dict(Counter(str(row["source"]) for row in records)),
        "max_pairs": int(args.max_pairs),
        "min_tokens": int(args.min_tokens),
        "max_tokens": int(args.max_tokens),
    }
    summary_path = args.summary_json or args.out_jsonl.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(f"hc3_plus_jsonl={args.out_jsonl}")
    print(f"summary={summary_path}")
    print(f"n_pairs={summary['n_pairs']}")
    print(f"by_subset={summary['by_subset']}")


if __name__ == "__main__":
    main()
