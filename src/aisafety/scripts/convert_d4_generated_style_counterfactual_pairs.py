"""Convert generated plain-versus-assistant rewrites into D4 counterfactual rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from aisafety.config import PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text, token_count
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json


DEFAULT_OUT_DIR = Path("data") / "derived" / "d4_assistant_style_generated_counterfactual_pairs_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--style-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--dimension", default="ai_tone")
    parser.add_argument("--assistant-label", default="rlhf_ai_tone")
    parser.add_argument("--plain-label", default="human_plain")
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--min-length-ratio", type=float, default=0.7)
    parser.add_argument("--max-length-ratio", type=float, default=1.3)
    parser.add_argument("--min-content-token-jaccard", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _meta(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("meta")
    return raw if isinstance(raw, dict) else {}


def _content_tokens(text: str) -> set[str]:
    return {item.lower() for item in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", str(text or ""))}


def _jaccard(left: str, right: str) -> float:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return float(len(left_tokens & right_tokens) / len(union))


def build_generated_counterfactual_rows(
    style_rows: list[dict[str, Any]],
    *,
    dimension: str,
    assistant_label: str,
    plain_label: str,
    min_tokens: int,
    min_length_ratio: float,
    max_length_ratio: float,
    min_content_token_jaccard: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    skipped = Counter()
    for row in style_rows:
        if str(row.get("dimension") or "") != str(dimension):
            continue
        meta = _meta(row)
        seed_id = str(meta.get("rewrite_seed_id") or row.get("seed_id") or "")
        label = str(row.get("label") or "")
        if not seed_id or label not in {assistant_label, plain_label}:
            continue
        grouped.setdefault(seed_id, {})[label] = row

    out: list[dict[str, Any]] = []
    for seed_id, labels in sorted(grouped.items()):
        if assistant_label not in labels or plain_label not in labels:
            skipped["missing_style_side"] += 1
            continue
        assistant = labels[assistant_label]
        plain = labels[plain_label]
        meta = _meta(assistant) or _meta(plain)
        cue_plus = flat_text(str(assistant.get("generated_text") or ""))
        cue_minus = flat_text(str(plain.get("generated_text") or ""))
        source_text = flat_text(str(assistant.get("seed_text") or plain.get("seed_text") or ""))
        plus_tokens = token_count(cue_plus)
        minus_tokens = token_count(cue_minus)
        if min(plus_tokens, minus_tokens) < int(min_tokens):
            skipped["rewrite_too_short"] += 1
            continue
        ratio = plus_tokens / max(float(minus_tokens), 1.0)
        if ratio < float(min_length_ratio) or ratio > float(max_length_ratio):
            skipped["length_ratio_out_of_bounds"] += 1
            continue
        if cue_plus == cue_minus:
            skipped["unchanged"] += 1
            continue
        plus_source_jaccard = _jaccard(cue_plus, source_text)
        minus_source_jaccard = _jaccard(cue_minus, source_text)
        if min(plus_source_jaccard, minus_source_jaccard) < float(min_content_token_jaccard):
            skipped["content_token_jaccard_below_threshold"] += 1
            continue
        counterfactual_id = sha1_hex(
            "|".join(
                [
                    seed_id,
                    str(dimension),
                    str(assistant_label),
                    str(plain_label),
                    sha1_hex(cue_plus),
                    sha1_hex(cue_minus),
                ]
            )
        )
        out.append(
            {
                "counterfactual_id": counterfactual_id,
                "pair_id": str(meta.get("pair_id") or seed_id),
                "source_dataset": str(meta.get("source_dataset") or "generated_style_rewrite"),
                "subset": str(meta.get("subset") or ""),
                "split": str(meta.get("split") or ""),
                "item_type": str(meta.get("item_type") or ""),
                "role": str(meta.get("role") or ""),
                "prompt": flat_text(str(meta.get("prompt") or "")),
                "axis": "generated_assistant_style",
                "direction": "increase",
                "transform_id": f"generated_{dimension}_{assistant_label}_vs_{plain_label}_v1",
                "rewrite_family": "generated_paired",
                "rewrite_seed_id": seed_id,
                "source_text": source_text,
                "base_text": cue_minus,
                "variant_text": cue_plus,
                "base_sha1": sha1_hex(cue_minus),
                "variant_sha1": sha1_hex(cue_plus),
                "base_tokens": int(minus_tokens),
                "variant_tokens": int(plus_tokens),
                "length_delta": int(plus_tokens - minus_tokens),
                "length_ratio": float(ratio),
                "generation_model": str(assistant.get("model") or plain.get("model") or ""),
                "assistant_source_token_jaccard": float(plus_source_jaccard),
                "plain_source_token_jaccard": float(minus_source_jaccard),
                "content_preservation_flags": (
                    "generated_pair;same_source_answer;plain_vs_assistant_style;"
                    "semantic_preservation_requires_audit"
                ),
            }
        )
    summary = {
        "n_input_rows": int(len(style_rows)),
        "n_seed_groups": int(len(grouped)),
        "n_counterfactuals": int(len(out)),
        "counts_by_role": dict(sorted(Counter(str(row["role"]) for row in out).items())),
        "counts_by_source": dict(sorted(Counter(str(row["source_dataset"]) for row in out).items())),
        "skipped": dict(sorted(skipped.items())),
    }
    return out, summary


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    input_path = _resolve(workspace_root, args.style_pairs_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    rows, summary = build_generated_counterfactual_rows(
        read_jsonl(input_path),
        dimension=str(args.dimension),
        assistant_label=str(args.assistant_label),
        plain_label=str(args.plain_label),
        min_tokens=int(args.min_tokens),
        min_length_ratio=float(args.min_length_ratio),
        max_length_ratio=float(args.max_length_ratio),
        min_content_token_jaccard=float(args.min_content_token_jaccard),
    )
    if not rows:
        raise ValueError("No generated assistant-style counterfactual rows were emitted.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "counterfactuals.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(
        out_dir / "summary.json",
        {
            "stage": "D4-generated-style-counterfactual-conversion",
            "style_pairs_jsonl": str(input_path),
            "counterfactual_jsonl": str(out_path),
            "out_dir": str(out_dir),
            "dimension": str(args.dimension),
            "assistant_label": str(args.assistant_label),
            "plain_label": str(args.plain_label),
            "min_tokens": int(args.min_tokens),
            "min_length_ratio": float(args.min_length_ratio),
            "max_length_ratio": float(args.max_length_ratio),
            "min_content_token_jaccard": float(args.min_content_token_jaccard),
            **summary,
        },
    )
    print(f"counterfactuals={out_path}")
    print(f"n_counterfactuals={len(rows)}")


if __name__ == "__main__":
    main()
