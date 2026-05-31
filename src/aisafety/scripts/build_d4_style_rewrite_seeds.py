"""Export answer texts and metadata as seeds for generated style rewrites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text, token_count
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json


DEFAULT_PAIR_JSONL = (
    Path("data") / "derived" / "d4_human_llm_alignment_pairs_matched_lenlex_relaxed_v1" / "pairs.jsonl"
)
DEFAULT_OUT_DIR = Path("data") / "derived" / "d4_assistant_style_generated_seeds_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-seeds", type=int, default=300)
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _prompt(row: dict[str, Any]) -> str:
    for key in ("question", "title", "prompt"):
        value = flat_text(str(row.get(key) or ""))
        if value:
            return value
    return ""


def build_seed_rows(
    pair_rows: list[dict[str, Any]],
    *,
    max_seeds: int,
    min_tokens: int,
    max_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in pair_rows:
        pair_id = str(row.get("pair_id") or "")
        if not pair_id:
            continue
        for role, text_key in (("human", "human_text"), ("llm", "llm_text")):
            text = flat_text(str(row.get(text_key) or ""))
            n_tokens = token_count(text)
            key = (pair_id, role)
            if key in seen or not int(min_tokens) <= n_tokens <= int(max_tokens):
                continue
            seen.add(key)
            rows.append(
                {
                    "text": text,
                    "pair_id": pair_id,
                    "source_dataset": str(row.get("source_dataset") or ""),
                    "subset": str(row.get("subset") or ""),
                    "split": str(row.get("split") or ""),
                    "item_type": str(row.get("item_type") or ""),
                    "role": role,
                    "prompt": _prompt(row),
                    "source_tokens": int(n_tokens),
                }
            )
    rows.sort(key=lambda row: sha1_hex(f"{seed}:generated-style-seed:{row['pair_id']}:{row['role']}"))
    if int(max_seeds) > 0:
        rows = rows[: int(max_seeds)]
    for index, row in enumerate(rows):
        row["rewrite_seed_id"] = f"style-seed-{index:05d}"
    return rows


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    pair_path = _resolve(workspace_root, args.pair_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    rows = build_seed_rows(
        read_jsonl(pair_path),
        max_seeds=int(args.max_seeds),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        seed=int(args.seed),
    )
    if not rows:
        raise ValueError("No style rewrite seeds were emitted.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "seeds.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(
        out_dir / "summary.json",
        {
            "stage": "D4-style-rewrite-seed-build",
            "pair_jsonl": str(pair_path),
            "seed_jsonl": str(out_path),
            "n_seeds": int(len(rows)),
            "max_seeds": int(args.max_seeds),
            "min_tokens": int(args.min_tokens),
            "max_tokens": int(args.max_tokens),
            "seed": int(args.seed),
        },
    )
    print(f"seeds={out_path}")
    print(f"n_seeds={len(rows)}")


if __name__ == "__main__":
    main()
