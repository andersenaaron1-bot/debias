"""Build Bradley-Terry stage-contrast pairs from D4 surface counterfactuals."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text, token_count
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.scripts.build_d4_human_llm_alignment_pairs import _csv_set
from aisafety.scripts.run_d4_surface_counterfactual_audit import _cap_rows


DEFAULT_COUNTERFACTUAL_JSONL = (
    Path("data") / "derived" / "d4_surface_counterfactual_pairs_v1" / "counterfactuals.jsonl"
)
DEFAULT_OUT_DIR = Path("data") / "derived" / "d4_bt_stage_contrast_pairs_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--counterfactual-jsonl", type=Path, default=DEFAULT_COUNTERFACTUAL_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--axes", type=str, default="")
    parser.add_argument("--directions", type=str, default="")
    parser.add_argument("--roles", type=str, default="")
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


def _write_csv(path: Path, rows: list[dict[str, Any]], *, limit: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = rows[: int(limit)]
    fields: list[str] = []
    for row in preview:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(preview)


def _filtered_rows(
    rows: list[dict[str, Any]],
    *,
    axes: set[str],
    directions: set[str],
    roles: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if axes and str(row.get("axis") or "") not in axes:
            continue
        if directions and str(row.get("direction") or "") not in directions:
            continue
        if roles and str(row.get("role") or "") not in roles:
            continue
        out.append(row)
    return out


def _cue_texts(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return cue_plus, cue_minus, and which side is cue_plus in the source row."""

    direction = str(row.get("direction") or "")
    base = flat_text(str(row.get("base_text") or ""))
    variant = flat_text(str(row.get("variant_text") or ""))
    if direction == "increase":
        return variant, base, "variant"
    if direction == "decrease":
        return base, variant, "base"
    raise ValueError(f"Unsupported counterfactual direction: {direction!r}")


def _prompt(row: dict[str, Any]) -> str:
    for key in ("prompt", "question", "title"):
        value = flat_text(str(row.get(key) or ""))
        if value:
            return value
    return "Compare the two responses."


def _bt_row(
    row: dict[str, Any],
    *,
    order: str,
    cue_plus_text: str,
    cue_minus_text: str,
    cue_plus_source: str,
) -> dict[str, Any]:
    if order == "plus_first":
        option_a = cue_plus_text
        option_b = cue_minus_text
        cue_plus_option = "A"
    elif order == "minus_first":
        option_a = cue_minus_text
        option_b = cue_plus_text
        cue_plus_option = "B"
    else:
        raise ValueError(f"Unknown order: {order}")

    bt_pair_id = sha1_hex(
        "|".join(
            [
                str(row.get("counterfactual_id") or ""),
                str(row.get("axis") or ""),
                str(row.get("direction") or ""),
                str(row.get("role") or ""),
                order,
                sha1_hex(option_a),
                sha1_hex(option_b),
            ]
        )
    )
    cue_plus_tokens = token_count(cue_plus_text)
    cue_minus_tokens = token_count(cue_minus_text)
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
        "cue_plus_option": cue_plus_option,
        "cue_plus_source": cue_plus_source,
        "prompt": _prompt(row),
        "option_a_text": option_a,
        "option_b_text": option_b,
        "cue_plus_text": cue_plus_text,
        "cue_minus_text": cue_minus_text,
        "option_a_sha1": sha1_hex(option_a),
        "option_b_sha1": sha1_hex(option_b),
        "cue_plus_sha1": sha1_hex(cue_plus_text),
        "cue_minus_sha1": sha1_hex(cue_minus_text),
        "option_a_tokens": token_count(option_a),
        "option_b_tokens": token_count(option_b),
        "cue_plus_tokens": int(cue_plus_tokens),
        "cue_minus_tokens": int(cue_minus_tokens),
        "cue_plus_minus_cue_minus_tokens": int(cue_plus_tokens - cue_minus_tokens),
        "length_ratio_plus_over_minus": float(cue_plus_tokens / max(float(cue_minus_tokens), 1.0)),
        "content_preservation_flags": str(row.get("content_preservation_flags") or ""),
    }


def build_bt_rows(
    counterfactual_rows: list[dict[str, Any]],
    *,
    include_order_swaps: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in counterfactual_rows:
        cue_plus, cue_minus, cue_plus_source = _cue_texts(row)
        if not cue_plus or not cue_minus or flat_text(cue_plus) == flat_text(cue_minus):
            continue
        orders = ["plus_first", "minus_first"] if include_order_swaps else ["plus_first"]
        for order in orders:
            rows.append(
                _bt_row(
                    row,
                    order=order,
                    cue_plus_text=cue_plus,
                    cue_minus_text=cue_minus,
                    cue_plus_source=cue_plus_source,
                )
            )

    counts_axis_direction = Counter(f"{row['axis']}::{row['direction']}" for row in rows)
    counts_axis_order = Counter(f"{row['axis']}::{row['presentation_order']}" for row in rows)
    counts_source = Counter(str(row["source_dataset"]) for row in rows)
    counts_role = Counter(str(row["role"]) for row in rows)
    summary = {
        "n_bt_pairs": int(len(rows)),
        "n_counterfactuals_used": int(len({row["counterfactual_id"] for row in rows})),
        "include_order_swaps": bool(include_order_swaps),
        "counts_by_axis_direction": dict(sorted(counts_axis_direction.items())),
        "counts_by_axis_order": dict(sorted(counts_axis_order.items())),
        "counts_by_source_dataset": dict(sorted(counts_source.items())),
        "counts_by_role": dict(sorted(counts_role.items())),
    }
    return rows, summary


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    counterfactual_path = _resolve(workspace_root, args.counterfactual_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)

    rows = read_jsonl(counterfactual_path)
    rows = _filtered_rows(
        rows,
        axes=_csv_set(str(args.axes)),
        directions=_csv_set(str(args.directions)),
        roles=_csv_set(str(args.roles)),
    )
    rows = _cap_rows(rows, max_rows=int(args.max_counterfactuals), seed=int(args.seed))
    if not rows:
        raise ValueError(f"No counterfactual rows selected from {counterfactual_path}")

    bt_rows, summary = build_bt_rows(rows, include_order_swaps=bool(args.include_order_swaps))
    if not bt_rows:
        raise ValueError("No Bradley-Terry comparison rows were emitted.")

    out_dir.mkdir(parents=True, exist_ok=True)
    bt_path = out_dir / "bt_pairs.jsonl"
    preview_path = out_dir / "bt_pairs_preview.csv"
    summary_path = out_dir / "summary.json"
    _write_jsonl(bt_path, bt_rows)
    _write_csv(preview_path, bt_rows)
    write_json(
        summary_path,
        {
            "stage": "D4-BT-stage-contrast-pair-build",
            "counterfactual_jsonl": str(counterfactual_path),
            "out_dir": str(out_dir),
            "bt_pairs_jsonl": str(bt_path),
            "bt_pairs_preview_csv": str(preview_path),
            "seed": int(args.seed),
            "axes_filter": sorted(_csv_set(str(args.axes))),
            "directions_filter": sorted(_csv_set(str(args.directions))),
            "roles_filter": sorted(_csv_set(str(args.roles))),
            "max_counterfactuals": int(args.max_counterfactuals),
            **summary,
        },
    )
    print(f"bt_pairs={bt_path}")
    print(f"summary={summary_path}")
    print(f"n_bt_pairs={len(bt_rows)}")


if __name__ == "__main__":
    main()
