"""Build anchored LM-judge comparisons for D4 style counterfactuals."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text, token_count
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.scripts.build_d4_human_llm_alignment_pairs import _csv_set
from aisafety.scripts.build_d4_bt_stage_contrast_pairs import _cue_texts, _filtered_rows, _prompt
from aisafety.scripts.run_d4_surface_counterfactual_audit import _cap_rows


DEFAULT_PAIR_JSONL = (
    Path("data") / "derived" / "d4_human_llm_alignment_pairs_matched_lenlex_relaxed_v1" / "pairs.jsonl"
)
DEFAULT_OUT_DIR = Path("data") / "derived" / "d4_anchored_style_bt_pairs_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--counterfactual-jsonl", type=Path, required=True)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--axes", default="")
    parser.add_argument("--directions", default="")
    parser.add_argument("--roles", default="")
    parser.add_argument("--max-counterfactuals", type=int, default=0)
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


def _reference(pair: dict[str, Any], *, role: str) -> tuple[str, str]:
    if str(role) == "human":
        return "llm", flat_text(str(pair.get("llm_text") or ""))
    if str(role) == "llm":
        return "human", flat_text(str(pair.get("human_text") or ""))
    raise ValueError(f"Expected counterfactual role human or llm, got: {role!r}")


def _anchored_row(
    row: dict[str, Any],
    *,
    reference_role: str,
    reference_text: str,
    cue_plus_text: str,
    cue_minus_text: str,
    cue_plus_source: str,
    order: str,
) -> dict[str, Any]:
    if order == "candidate_first":
        option_a = cue_plus_text
        option_b = reference_text
        cue_plus_option = "A"
    elif order == "reference_first":
        option_a = reference_text
        option_b = cue_plus_text
        cue_plus_option = "B"
    else:
        raise ValueError(f"Unknown anchored order: {order}")
    bt_pair_id = sha1_hex(
        "|".join(
            [
                "anchored-style",
                str(row.get("counterfactual_id") or ""),
                str(row.get("axis") or ""),
                str(row.get("direction") or ""),
                str(row.get("role") or ""),
                order,
                sha1_hex(option_a),
                sha1_hex(option_b),
                sha1_hex(cue_minus_text),
            ]
        )
    )
    return {
        "bt_pair_id": bt_pair_id,
        "counterfactual_id": str(row.get("counterfactual_id") or ""),
        "pair_id": str(row.get("pair_id") or ""),
        "source_dataset": str(row.get("source_dataset") or ""),
        "subset": str(row.get("subset") or ""),
        "split": str(row.get("split") or ""),
        "item_type": str(row.get("item_type") or ""),
        "role": str(row.get("role") or ""),
        "axis": str(row.get("axis") or ""),
        "direction": str(row.get("direction") or ""),
        "transform_id": str(row.get("transform_id") or ""),
        "presentation_order": order,
        "comparison_mode": "anchored_reference",
        "cue_plus_option": cue_plus_option,
        "cue_plus_source": cue_plus_source,
        "prompt": _prompt(row),
        "option_a_text": option_a,
        "option_b_text": option_b,
        "cue_plus_text": cue_plus_text,
        "cue_minus_text": cue_minus_text,
        "reference_role": str(reference_role),
        "reference_text": reference_text,
        "option_a_sha1": sha1_hex(option_a),
        "option_b_sha1": sha1_hex(option_b),
        "cue_plus_sha1": sha1_hex(cue_plus_text),
        "cue_minus_sha1": sha1_hex(cue_minus_text),
        "reference_sha1": sha1_hex(reference_text),
        "option_a_tokens": int(token_count(option_a)),
        "option_b_tokens": int(token_count(option_b)),
        "cue_plus_tokens": int(token_count(cue_plus_text)),
        "cue_minus_tokens": int(token_count(cue_minus_text)),
        "reference_tokens": int(token_count(reference_text)),
        "cue_plus_minus_cue_minus_tokens": int(token_count(cue_plus_text) - token_count(cue_minus_text)),
        "content_preservation_flags": str(row.get("content_preservation_flags") or ""),
    }


def build_anchored_bt_rows(
    counterfactual_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    *,
    include_order_swaps: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_by_id = {
        str(row.get("pair_id") or ""): row
        for row in pair_rows
        if str(row.get("pair_id") or "")
    }
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for row in counterfactual_rows:
        pair_id = str(row.get("pair_id") or "")
        pair = pair_by_id.get(pair_id)
        if pair is None:
            skipped["missing_pair_id"] += 1
            continue
        try:
            reference_role, reference_text = _reference(pair, role=str(row.get("role") or ""))
            cue_plus, cue_minus, cue_plus_source = _cue_texts(row)
        except ValueError:
            skipped["unsupported_role_or_direction"] += 1
            continue
        if not cue_plus or not cue_minus or not reference_text:
            skipped["empty_text"] += 1
            continue
        orders = ["candidate_first", "reference_first"] if include_order_swaps else ["candidate_first"]
        for order in orders:
            rows.append(
                _anchored_row(
                    row,
                    reference_role=reference_role,
                    reference_text=reference_text,
                    cue_plus_text=cue_plus,
                    cue_minus_text=cue_minus,
                    cue_plus_source=cue_plus_source,
                    order=order,
                )
            )
    summary = {
        "n_bt_pairs": int(len(rows)),
        "n_counterfactuals_used": int(len({row["counterfactual_id"] for row in rows})),
        "include_order_swaps": bool(include_order_swaps),
        "counts_by_axis": dict(sorted(Counter(str(row["axis"]) for row in rows).items())),
        "counts_by_role": dict(sorted(Counter(str(row["role"]) for row in rows).items())),
        "counts_by_reference_role": dict(sorted(Counter(str(row["reference_role"]) for row in rows).items())),
        "skipped": dict(sorted(skipped.items())),
    }
    return rows, summary


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    counterfactual_path = _resolve(workspace_root, args.counterfactual_jsonl)
    pair_path = _resolve(workspace_root, args.pair_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    rows = _filtered_rows(
        read_jsonl(counterfactual_path),
        axes=_csv_set(str(args.axes)),
        directions=_csv_set(str(args.directions)),
        roles=_csv_set(str(args.roles)),
    )
    rows = _cap_rows(rows, max_rows=int(args.max_counterfactuals), seed=int(args.seed))
    bt_rows, summary = build_anchored_bt_rows(
        rows,
        read_jsonl(pair_path),
        include_order_swaps=bool(args.include_order_swaps),
    )
    if not bt_rows:
        raise ValueError("No anchored style comparison rows were emitted.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bt_pairs.jsonl"
    _write_jsonl(out_path, bt_rows)
    write_json(
        out_dir / "summary.json",
        {
            "stage": "D4-anchored-style-BT-pair-build",
            "counterfactual_jsonl": str(counterfactual_path),
            "pair_jsonl": str(pair_path),
            "bt_pairs_jsonl": str(out_path),
            "out_dir": str(out_dir),
            "max_counterfactuals": int(args.max_counterfactuals),
            "seed": int(args.seed),
            **summary,
        },
    )
    print(f"bt_pairs={out_path}")
    print(f"n_bt_pairs={len(bt_rows)}")
    print(f"n_counterfactuals={summary['n_counterfactuals_used']}")


if __name__ == "__main__":
    main()
