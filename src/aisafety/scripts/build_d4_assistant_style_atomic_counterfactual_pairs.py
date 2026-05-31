"""Build atomic assistant-style marker counterfactuals from paired answer corpora."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text, token_count
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json


DEFAULT_PAIR_JSONL = (
    Path("data") / "derived" / "d4_human_llm_alignment_pairs_matched_lenlex_relaxed_v1" / "pairs.jsonl"
)
DEFAULT_OUT_DIR = Path("data") / "derived" / "d4_assistant_style_atomic_counterfactual_pairs_v1"

MARKERS = (
    ("answer_label", "assistant_answer_label", "Answer:\n", re.compile(r"(?i)^\s*answer\s*:\s*")),
    ("key_points_label", "assistant_answer_label", "Key points:\n", re.compile(r"(?i)^\s*key points\s*:\s*")),
    ("overall_preface", "formal_discourse_marker", "Overall, ", re.compile(r"(?i)^\s*overall,\s*")),
    ("furthermore_preface", "formal_discourse_marker", "Furthermore, ", re.compile(r"(?i)^\s*furthermore,\s*")),
    ("conclusion_preface", "formal_discourse_marker", "In conclusion, ", re.compile(r"(?i)^\s*in conclusion,\s*")),
    ("response_label", "placebo_response_label", "Response:\n", re.compile(r"(?i)^\s*response\s*:\s*")),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--max-length-ratio", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _prompt(row: dict[str, Any]) -> str:
    for key in ("question", "title", "prompt"):
        value = flat_text(str(row.get(key) or ""))
        if value:
            return value
    return ""


def _cap_pairs(rows: list[dict[str, Any]], *, max_pairs: int, seed: int) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if pair_id:
            by_pair.setdefault(pair_id, row)
    ordered = sorted(by_pair.values(), key=lambda row: sha1_hex(f"{seed}:atomic:{row['pair_id']}"))
    if int(max_pairs) > 0:
        ordered = ordered[: int(max_pairs)]
    return ordered


def _metadata(row: dict[str, Any], *, role: str, text: str) -> dict[str, Any]:
    return {
        "pair_id": str(row.get("pair_id") or ""),
        "source_dataset": str(row.get("source_dataset") or ""),
        "subset": str(row.get("subset") or ""),
        "split": str(row.get("split") or ""),
        "item_type": str(row.get("item_type") or ""),
        "role": role,
        "prompt": _prompt(row),
        "base_text": flat_text(text),
        "base_sha1": sha1_hex(flat_text(text)),
        "base_tokens": int(token_count(text)),
        "rewrite_family": "deterministic_atomic",
    }


def _emit_variant(
    *,
    base: dict[str, Any],
    marker_id: str,
    axis: str,
    direction: str,
    variant_text: str,
) -> dict[str, Any] | None:
    variant_text = flat_text(variant_text)
    if not variant_text or variant_text == str(base["base_text"]):
        return None
    variant_tokens = token_count(variant_text)
    counterfactual_id = sha1_hex(
        "|".join(
            [
                str(base["pair_id"]),
                str(base["role"]),
                marker_id,
                direction,
                str(base["base_sha1"]),
                sha1_hex(variant_text),
            ]
        )
    )
    return {
        **base,
        "counterfactual_id": counterfactual_id,
        "axis": axis,
        "direction": direction,
        "transform_id": f"atomic_{marker_id}_{direction}_v1",
        "marker_id": marker_id,
        "variant_text": variant_text,
        "variant_sha1": sha1_hex(variant_text),
        "variant_tokens": int(variant_tokens),
        "length_delta": int(variant_tokens - int(base["base_tokens"])),
        "length_ratio": float(variant_tokens / max(float(base["base_tokens"]), 1.0)),
        "content_preservation_flags": "atomic_marker_edit;single_prefix_edit;semantic_content_unchanged",
    }


def build_atomic_rows(
    pair_rows: list[dict[str, Any]],
    *,
    max_pairs: int,
    min_tokens: int,
    max_length_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    selected = _cap_pairs(pair_rows, max_pairs=max_pairs, seed=seed)
    for row in selected:
        for role, text_key in (("human", "human_text"), ("llm", "llm_text")):
            text = flat_text(str(row.get(text_key) or ""))
            if token_count(text) < int(min_tokens):
                skipped["base_too_short"] += 1
                continue
            base = _metadata(row, role=role, text=text)
            for marker_id, axis, prefix, pattern in MARKERS:
                if not pattern.search(text):
                    variant = _emit_variant(
                        base=base,
                        marker_id=marker_id,
                        axis=axis,
                        direction="increase",
                        variant_text=prefix + text,
                    )
                else:
                    variant = _emit_variant(
                        base=base,
                        marker_id=marker_id,
                        axis=axis,
                        direction="decrease",
                        variant_text=pattern.sub("", text, count=1),
                    )
                if variant is None:
                    skipped[f"{marker_id}::unchanged"] += 1
                    continue
                if float(variant["length_ratio"]) > float(max_length_ratio):
                    skipped[f"{marker_id}::length_ratio"] += 1
                    continue
                rows.append(variant)
    summary = {
        "n_input_pairs": int(len(selected)),
        "n_counterfactuals": int(len(rows)),
        "counts_by_marker": dict(sorted(Counter(str(row["marker_id"]) for row in rows).items())),
        "counts_by_axis": dict(sorted(Counter(str(row["axis"]) for row in rows).items())),
        "counts_by_role": dict(sorted(Counter(str(row["role"]) for row in rows).items())),
        "skipped": dict(sorted(skipped.items())),
    }
    return rows, summary


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    pair_path = _resolve(workspace_root, args.pair_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    rows, summary = build_atomic_rows(
        read_jsonl(pair_path),
        max_pairs=int(args.max_pairs),
        min_tokens=int(args.min_tokens),
        max_length_ratio=float(args.max_length_ratio),
        seed=int(args.seed),
    )
    if not rows:
        raise ValueError("No atomic assistant-style counterfactual rows were emitted.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "counterfactuals.jsonl"
    _write_jsonl(out_path, rows)
    write_json(
        out_dir / "summary.json",
        {
            "stage": "D4-assistant-style-atomic-counterfactual-build",
            "pair_jsonl": str(pair_path),
            "counterfactual_jsonl": str(out_path),
            "out_dir": str(out_dir),
            "max_pairs": int(args.max_pairs),
            "min_tokens": int(args.min_tokens),
            "max_length_ratio": float(args.max_length_ratio),
            "seed": int(args.seed),
            **summary,
        },
    )
    print(f"counterfactuals={out_path}")
    print(f"n_counterfactuals={len(rows)}")


if __name__ == "__main__":
    main()
