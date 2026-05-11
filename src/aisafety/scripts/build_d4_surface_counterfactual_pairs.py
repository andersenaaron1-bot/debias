"""Build deterministic D4 surface-cue counterfactual rows from paired corpora."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import (
    DEFAULT_SURFACE_AXES,
    CounterfactualSkip,
    CounterfactualVariant,
    build_counterfactual_variants,
    flat_text,
    token_count,
)
from aisafety.mech.d4_io import resolve_path, sha1_hex, write_json
from aisafety.scripts.build_d4_human_llm_alignment_pairs import _csv_set
from aisafety.scripts.run_d4_candidate_feature_pair_alignment import (
    _cap_pairs,
    _read_pair_file,
)


DEFAULT_PAIR_JSONL = Path("data") / "derived" / "d4_human_llm_alignment_pairs_strat10k_v3" / "pairs.jsonl"
DEFAULT_OUT_DIR = Path("data") / "derived" / "d4_surface_counterfactual_pairs_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--axes", type=str, default=",".join(DEFAULT_SURFACE_AXES))
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--min-length-ratio", type=float, default=0.7)
    parser.add_argument("--max-length-ratio", type=float, default=1.3)
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


def _prompt_for_pair(row: Any) -> str:
    for name in ("question", "title", "prompt"):
        if hasattr(row, name):
            value = str(getattr(row, name) or "").strip()
            if value:
                return value
    return ""


def _side_text(row: Any, role: str) -> str:
    if role == "human":
        return str(getattr(row, "human_text"))
    if role == "llm":
        return str(getattr(row, "llm_text"))
    raise ValueError(f"Unknown role: {role}")


def _base_metadata(row: Any, role: str, base_text: str) -> dict[str, Any]:
    base_tokens = token_count(base_text)
    return {
        "pair_id": str(getattr(row, "pair_id")),
        "source_dataset": str(getattr(row, "source_dataset", "")),
        "bundle_creation_role": str(getattr(row, "bundle_creation_role", "")),
        "group_id": str(getattr(row, "group_id", "")),
        "split": str(getattr(row, "split", "")),
        "item_type": str(getattr(row, "item_type", "")),
        "subset": str(getattr(row, "subset", "")),
        "title": str(getattr(row, "title", "")),
        "question": str(getattr(row, "question", "")),
        "llm_generator": str(getattr(row, "llm_generator", "")),
        "role": role,
        "prompt": _prompt_for_pair(row),
        "base_text": flat_text(base_text),
        "base_sha1": sha1_hex(flat_text(base_text)),
        "base_tokens": int(base_tokens),
    }


def build_counterfactual_rows(
    pair_df,
    *,
    axes: set[str],
    min_tokens: int,
    min_length_ratio: float,
    max_length_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    skipped_by_axis = Counter()
    emitted_by_axis = Counter()
    emitted_by_source = Counter()
    emitted_by_role = Counter()

    for row in pair_df.itertuples(index=False):
        for role in ("human", "llm"):
            base_text = _side_text(row, role)
            base = _base_metadata(row, role, base_text)
            variants = build_counterfactual_variants(
                base_text,
                axes=axes,
                min_tokens=int(min_tokens),
                min_length_ratio=float(min_length_ratio),
                max_length_ratio=float(max_length_ratio),
            )
            for variant in variants:
                if isinstance(variant, CounterfactualSkip):
                    key = f"{variant.axis}::{variant.direction}::{variant.reason}"
                    skipped[key] += 1
                    skipped_by_axis[f"{variant.axis}::{variant.direction}"] += 1
                    continue
                assert isinstance(variant, CounterfactualVariant)
                variant_tokens = token_count(variant.text)
                counterfactual_id = sha1_hex(
                    "|".join(
                        [
                            str(base["pair_id"]),
                            role,
                            variant.axis,
                            variant.direction,
                            variant.transform_id,
                            str(base["base_sha1"]),
                            sha1_hex(flat_text(variant.text)),
                        ]
                    )
                )
                row_out = {
                    **base,
                    "counterfactual_id": counterfactual_id,
                    "axis": variant.axis,
                    "direction": variant.direction,
                    "transform_id": variant.transform_id,
                    "variant_text": flat_text(variant.text),
                    "variant_sha1": sha1_hex(flat_text(variant.text)),
                    "variant_tokens": int(variant_tokens),
                    "length_delta": int(variant_tokens - int(base["base_tokens"])),
                    "length_ratio": float(variant_tokens / max(float(base["base_tokens"]), 1.0)),
                    "content_preservation_flags": ";".join(variant.flags),
                }
                rows.append(row_out)
                emitted_by_axis[f"{variant.axis}::{variant.direction}"] += 1
                emitted_by_source[str(base["source_dataset"])] += 1
                emitted_by_role[role] += 1

    summary = {
        "n_counterfactuals": int(len(rows)),
        "axes": sorted(axes),
        "emitted_by_axis_direction": dict(sorted(emitted_by_axis.items())),
        "emitted_by_source_dataset": dict(sorted(emitted_by_source.items())),
        "emitted_by_role": dict(sorted(emitted_by_role.items())),
        "skipped": dict(sorted(skipped.items())),
        "skipped_by_axis_direction": dict(sorted(skipped_by_axis.items())),
    }
    return rows, summary


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    pair_path = _resolve(workspace_root, args.pair_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    axes = _csv_set(str(args.axes))
    if not axes:
        raise ValueError("--axes produced an empty axis set.")

    pair_df = _cap_pairs(_read_pair_file(pair_path), max_pairs=int(args.max_pairs), seed=int(args.seed))
    rows, summary = build_counterfactual_rows(
        pair_df,
        axes=axes,
        min_tokens=int(args.min_tokens),
        min_length_ratio=float(args.min_length_ratio),
        max_length_ratio=float(args.max_length_ratio),
    )
    if not rows:
        raise ValueError("No surface counterfactuals were emitted.")

    out_dir.mkdir(parents=True, exist_ok=True)
    counterfactual_path = out_dir / "counterfactuals.jsonl"
    summary_path = out_dir / "summary.json"
    _write_jsonl(counterfactual_path, rows)
    write_json(
        summary_path,
        {
            "stage": "D4-surface-counterfactual-pair-build",
            "pair_jsonl": str(pair_path),
            "out_dir": str(out_dir),
            "counterfactual_jsonl": str(counterfactual_path),
            "seed": int(args.seed),
            "max_pairs": int(args.max_pairs),
            "min_tokens": int(args.min_tokens),
            "min_length_ratio": float(args.min_length_ratio),
            "max_length_ratio": float(args.max_length_ratio),
            "n_input_pairs": int(len(pair_df)),
            **summary,
        },
    )
    print(f"counterfactuals={counterfactual_path}")
    print(f"summary={summary_path}")
    print(f"n_counterfactuals={len(rows)}")


if __name__ == "__main__":
    main()
