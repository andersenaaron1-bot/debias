"""Build fixed-reference BT controls for LM-judge subspace suppression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text, token_count
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json


DEFAULT_OUT_DIR = Path("data") / "derived" / "d4_fixed_reference_bt_controls_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pair-jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=["human_llm", "preference"], required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--include-order-swaps", action=argparse.BooleanOptionalAction, default=True)
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
    for key in ("prompt", "question", "title"):
        text = flat_text(str(row.get(key) or ""))
        if text:
            return text
    return "Compare the two responses."


def _texts(row: dict[str, Any], *, mode: str) -> tuple[str, str, str]:
    if mode == "human_llm":
        return (
            flat_text(str(row.get("llm_text") or "")),
            flat_text(str(row.get("human_text") or "")),
            "llm",
        )
    return (
        flat_text(str(row.get("chosen") or row.get("chosen_text") or "")),
        flat_text(str(row.get("rejected") or row.get("rejected_text") or "")),
        "chosen",
    )


def build_fixed_reference_rows(
    pair_rows: list[dict[str, Any]],
    *,
    mode: str,
    max_pairs: int,
    include_order_swaps: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed: list[tuple[str, dict[str, Any]]] = []
    for index, row in enumerate(pair_rows):
        pair_id = str(row.get("pair_id") or row.get("pref_pair_id") or "")
        if not pair_id:
            pair_id = sha1_hex(f"{mode}:{index}:{_prompt(row)}")
        indexed.append((pair_id, row))
    indexed.sort(key=lambda item: sha1_hex(f"{seed}:{mode}:{item[0]}"))
    if int(max_pairs) > 0:
        indexed = indexed[: int(max_pairs)]
    rows: list[dict[str, Any]] = []
    for pair_id, row in indexed:
        candidate, reference, role = _texts(row, mode=mode)
        if not candidate or not reference:
            continue
        counterfactual_id = sha1_hex(f"fixed-reference:{mode}:{pair_id}:{sha1_hex(candidate)}:{sha1_hex(reference)}")
        orders = ["candidate_first", "reference_first"] if include_order_swaps else ["candidate_first"]
        for order in orders:
            if order == "candidate_first":
                option_a, option_b, cue_plus_option = candidate, reference, "A"
            else:
                option_a, option_b, cue_plus_option = reference, candidate, "B"
            rows.append(
                {
                    "bt_pair_id": sha1_hex(f"{counterfactual_id}:{order}"),
                    "counterfactual_id": counterfactual_id,
                    "pair_id": pair_id,
                    "source_dataset": str(row.get("source_dataset") or mode),
                    "subset": str(row.get("subset") or ""),
                    "split": str(row.get("split") or ""),
                    "item_type": str(row.get("item_type") or ""),
                    "role": role,
                    "axis": f"fixed_reference_{mode}",
                    "direction": "control",
                    "transform_id": f"fixed_reference_{mode}_v1",
                    "presentation_order": order,
                    "comparison_mode": "fixed_reference_control",
                    "cue_plus_option": cue_plus_option,
                    "cue_plus_source": "candidate",
                    "prompt": _prompt(row),
                    "option_a_text": option_a,
                    "option_b_text": option_b,
                    "cue_plus_text": candidate,
                    "cue_minus_text": candidate,
                    "reference_text": reference,
                    "cue_plus_tokens": int(token_count(candidate)),
                    "cue_minus_tokens": int(token_count(candidate)),
                    "reference_tokens": int(token_count(reference)),
                    "cue_plus_minus_cue_minus_tokens": 0,
                    "content_preservation_flags": "fixed_reference_control;observed_equals_neutral",
                }
            )
    return rows, {
        "n_input_pairs": int(len(indexed)),
        "n_bt_pairs": int(len(rows)),
        "n_pairs_used": int(len({row["pair_id"] for row in rows})),
        "include_order_swaps": bool(include_order_swaps),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    pair_path = _resolve(workspace_root, args.pair_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    rows, summary = build_fixed_reference_rows(
        read_jsonl(pair_path),
        mode=str(args.mode),
        max_pairs=int(args.max_pairs),
        include_order_swaps=bool(args.include_order_swaps),
        seed=int(args.seed),
    )
    if not rows:
        raise ValueError("No fixed-reference BT control rows were emitted.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bt_pairs.jsonl"
    _write_jsonl(out_path, rows)
    write_json(
        out_dir / "summary.json",
        {
            "stage": "D4-fixed-reference-BT-control-build",
            "pair_jsonl": str(pair_path),
            "bt_pairs_jsonl": str(out_path),
            "mode": str(args.mode),
            "out_dir": str(out_dir),
            "max_pairs": int(args.max_pairs),
            "seed": int(args.seed),
            **summary,
        },
    )
    print(f"bt_pairs={out_path}")
    print(f"n_bt_pairs={len(rows)}")
    print(f"n_pairs={summary['n_pairs_used']}")


if __name__ == "__main__":
    main()
