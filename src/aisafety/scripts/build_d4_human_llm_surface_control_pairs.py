"""Build deterministic surface-controlled human-vs-LLM pair files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from aisafety.config import DATA_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import (
    answer_likeness_decrease,
    flat_text,
    normalize_text,
    structured_decrease,
    token_count,
)
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.scripts.build_d4_human_llm_matched_pairs import _mean_metrics, _pair_prompt, _safe_ratio, _text_metrics
from aisafety.scripts.run_d4_surface_counterfactual_audit import _cap_rows


DEFAULT_PAIR_JSONL = DATA_DIR / "derived" / "d4_human_llm_alignment_pairs_matched_lenlex_v1" / "pairs.jsonl"
DEFAULT_OUT_DIR = DATA_DIR / "derived" / "d4_human_llm_alignment_pairs_surface_control_v1"
SURFACE_CONTROL_MODES = ("flat", "structured_decrease", "answer_decrease", "surface_minimized")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", choices=SURFACE_CONTROL_MODES, default="surface_minimized")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--min-response-tokens", type=int, default=20)
    parser.add_argument(
        "--min-transform-length-ratio",
        type=float,
        default=0.65,
        help="Drop a pair if either transformed side is shorter than this fraction of its source text.",
    )
    parser.add_argument(
        "--require-changed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, keep only pairs where at least one side changed under the selected transform.",
    )
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


def _maybe_apply_decrease(text: str, transform_name: str) -> tuple[str, list[str], bool]:
    if transform_name == "structured_decrease":
        variant, flags, reason = structured_decrease(text)
    elif transform_name == "answer_decrease":
        variant, flags, reason = answer_likeness_decrease(text)
    else:
        raise ValueError(f"Unsupported decrease transform: {transform_name}")
    if reason is not None or not variant:
        return text, [f"{transform_name}:skip:{reason or 'empty'}"], False
    if flat_text(variant) == flat_text(text):
        return text, [f"{transform_name}:skip:unchanged"], False
    return normalize_text(variant), [f"{transform_name}:{flag}" for flag in flags], True


def _transform_text(text: str, *, mode: str) -> tuple[str, list[str], bool]:
    current = normalize_text(text)
    flags: list[str] = []
    changed = False

    if mode == "flat":
        transformed = flat_text(current)
        return transformed, ["flat_text"], transformed != current

    if mode == "structured_decrease":
        current, step_flags, step_changed = _maybe_apply_decrease(current, "structured_decrease")
        flags.extend(step_flags)
        changed = changed or step_changed
    elif mode == "answer_decrease":
        current, step_flags, step_changed = _maybe_apply_decrease(current, "answer_decrease")
        flags.extend(step_flags)
        changed = changed or step_changed
    elif mode == "surface_minimized":
        for transform_name in ("answer_decrease", "structured_decrease"):
            current, step_flags, step_changed = _maybe_apply_decrease(current, transform_name)
            flags.extend(step_flags)
            changed = changed or step_changed
    else:
        raise ValueError(f"Unsupported surface control mode: {mode}")

    flattened = flat_text(current)
    flatten_changed = flattened != current
    if flatten_changed:
        flags.append("flat_text")
    return flattened, flags, changed or flatten_changed


def _annotate_metrics(row: dict[str, Any]) -> dict[str, Any]:
    prompt = _pair_prompt(row)
    human = _text_metrics(str(row.get("human_text") or ""), prompt)
    llm = _text_metrics(str(row.get("llm_text") or ""), prompt)
    human_tokens = int(human["tokens"])
    llm_tokens = int(llm["tokens"])
    token_delta = llm_tokens - human_tokens
    return {
        **row,
        "human_token_count": human_tokens,
        "llm_token_count": llm_tokens,
        "token_delta_llm_minus_human": int(token_delta),
        "abs_token_delta": int(abs(token_delta)),
        "length_ratio_max_over_min": float(_safe_ratio(float(llm_tokens), float(human_tokens))),
        "prompt_overlap_human": human["prompt_overlap"],
        "prompt_overlap_llm": llm["prompt_overlap"],
        "prompt_overlap_delta_llm_minus_human": float(llm["prompt_overlap"] - human["prompt_overlap"]),
        "type_token_ratio_human": human["type_token_ratio"],
        "type_token_ratio_llm": llm["type_token_ratio"],
        "type_token_ratio_delta_llm_minus_human": float(llm["type_token_ratio"] - human["type_token_ratio"]),
        "punct_rate_human": human["punct_rate"],
        "punct_rate_llm": llm["punct_rate"],
        "punct_rate_delta_llm_minus_human": float(llm["punct_rate"] - human["punct_rate"]),
        "avg_word_len_human": human["avg_word_len"],
        "avg_word_len_llm": llm["avg_word_len"],
    }


def build_surface_control_pairs(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    min_response_tokens: int,
    min_transform_length_ratio: float,
    require_changed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    skipped = Counter()
    changed_sides = Counter()

    for row in rows:
        human_original = str(row.get("human_text") or "")
        llm_original = str(row.get("llm_text") or "")
        if not flat_text(human_original) or not flat_text(llm_original):
            skipped["missing_text"] += 1
            continue

        human_text, human_flags, human_changed = _transform_text(human_original, mode=mode)
        llm_text, llm_flags, llm_changed = _transform_text(llm_original, mode=mode)
        if bool(require_changed) and not (human_changed or llm_changed):
            skipped["unchanged_pair"] += 1
            continue

        human_original_tokens = token_count(human_original)
        llm_original_tokens = token_count(llm_original)
        human_tokens = token_count(human_text)
        llm_tokens = token_count(llm_text)
        if human_tokens < int(min_response_tokens) or llm_tokens < int(min_response_tokens):
            skipped["too_short_after_transform"] += 1
            continue
        human_ratio = human_tokens / max(float(human_original_tokens), 1.0)
        llm_ratio = llm_tokens / max(float(llm_original_tokens), 1.0)
        if human_ratio < float(min_transform_length_ratio) or llm_ratio < float(min_transform_length_ratio):
            skipped["transform_too_destructive"] += 1
            continue

        changed_sides["human_changed" if human_changed else "human_unchanged"] += 1
        changed_sides["llm_changed" if llm_changed else "llm_unchanged"] += 1
        annotated = {
            **row,
            "surface_control_mode": str(mode),
            "source_pair_id": str(row.get("pair_id") or ""),
            "surface_control_id": sha1_hex(
                "|".join(
                    [
                        str(row.get("pair_id") or ""),
                        str(mode),
                        sha1_hex(flat_text(human_text)),
                        sha1_hex(flat_text(llm_text)),
                    ]
                )
            ),
            "human_text": human_text,
            "llm_text": llm_text,
            "human_original_sha1": sha1_hex(flat_text(human_original)),
            "llm_original_sha1": sha1_hex(flat_text(llm_original)),
            "human_surface_flags": ";".join(human_flags),
            "llm_surface_flags": ";".join(llm_flags),
            "human_surface_changed": bool(human_changed),
            "llm_surface_changed": bool(llm_changed),
            "human_original_tokens": int(human_original_tokens),
            "llm_original_tokens": int(llm_original_tokens),
            "human_transform_length_ratio": float(human_ratio),
            "llm_transform_length_ratio": float(llm_ratio),
        }
        kept.append(_annotate_metrics(annotated))

    summary = {
        "n_input_pairs": int(len(rows)),
        "n_surface_control_pairs": int(len(kept)),
        "mode": str(mode),
        "skipped": dict(skipped),
        "changed_sides": dict(changed_sides),
        "constraints": {
            "min_response_tokens": int(min_response_tokens),
            "min_transform_length_ratio": float(min_transform_length_ratio),
            "require_changed": bool(require_changed),
        },
        "by_dataset": dict(Counter(str(row.get("source_dataset") or "") for row in kept)),
        "by_dataset_subset": {
            f"{dataset}::{subset}": int(count)
            for (dataset, subset), count in Counter(
                (str(row.get("source_dataset") or ""), str(row.get("subset") or row.get("item_type") or ""))
                for row in kept
            ).items()
        },
        "mean_metrics": _mean_metrics(kept),
        "mean_transform_length_ratio": {
            "human": None
            if not kept
            else float(sum(float(row["human_transform_length_ratio"]) for row in kept) / len(kept)),
            "llm": None if not kept else float(sum(float(row["llm_transform_length_ratio"]) for row in kept) / len(kept)),
        },
    }
    return kept, summary


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    pair_path = _resolve(workspace_root, args.pair_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    rows = _cap_rows(read_jsonl(pair_path), max_rows=int(args.max_pairs), seed=int(args.seed))
    if not rows:
        raise ValueError(f"No input pairs found in {pair_path}")

    controlled, summary = build_surface_control_pairs(
        rows,
        mode=str(args.mode),
        min_response_tokens=int(args.min_response_tokens),
        min_transform_length_ratio=float(args.min_transform_length_ratio),
        require_changed=bool(args.require_changed),
    )
    if not controlled:
        raise ValueError("No pairs survived the surface-control transform.")

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_out = out_dir / "pairs.jsonl"
    summary_out = out_dir / "summary.json"
    _write_jsonl(pair_out, controlled)
    write_json(
        summary_out,
        {
            "stage": "D4-human-LLM-surface-control-pair-build",
            "pair_jsonl": str(pair_path),
            "out_dir": str(out_dir),
            "surface_control_pair_jsonl": str(pair_out),
            "summary_json": str(summary_out),
            "seed": int(args.seed),
            "max_pairs": int(args.max_pairs),
            **summary,
        },
    )
    print(f"pairs={pair_out}")
    print(f"summary={summary_out}")
    print(f"n_surface_control_pairs={len(controlled)}")
    print(f"by_dataset={summary['by_dataset']}")
    print(f"changed_sides={summary['changed_sides']}")
    print(f"mean_metrics={summary['mean_metrics']}")


if __name__ == "__main__":
    main()
