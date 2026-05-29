"""Export source texts for top decision-manifold factor units."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text
from aisafety.mech.d4_io import read_jsonl, resolve_path, write_json


DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_decision_manifold_top_texts_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--counterfactual-jsonl", type=Path, required=True)
    parser.add_argument("--method", default="pca")
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        help="Component to export, e.g. pc2. May be passed multiple times.",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--text-preview-chars", type=int, default=800)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _clip(value: Any, limit: int) -> str:
    text = flat_text(str(value or ""))
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _cue_texts(row: dict[str, Any]) -> tuple[str, str, str]:
    direction = str(row.get("direction") or "")
    base = flat_text(str(row.get("base_text") or ""))
    variant = flat_text(str(row.get("variant_text") or ""))
    if direction == "increase":
        return variant, base, "variant"
    if direction == "decrease":
        return base, variant, "base"
    return variant, base, ""


def export_top_texts(
    top_units: pd.DataFrame,
    counterfactual_rows: list[dict[str, Any]],
    *,
    method: str,
    components: list[str],
    top_k: int,
    preview_chars: int,
) -> pd.DataFrame:
    cf_by_id = {str(row.get("counterfactual_id") or ""): row for row in counterfactual_rows}
    selected = top_units[top_units["method"].astype(str) == str(method)].copy()
    if components:
        selected = selected[selected["component"].astype(str).isin({str(c) for c in components})].copy()
    if selected.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    selected = selected.sort_values(["component", "abs_score"], ascending=[True, False])
    for component, group in selected.groupby("component", sort=True):
        for rank, (_, item) in enumerate(group.head(max(int(top_k), 0)).iterrows(), start=1):
            unit_id = str(item.get("unit_id") or "")
            cf = cf_by_id.get(unit_id, {})
            cue_plus, cue_minus, cue_plus_source = _cue_texts(cf)
            row = {
                "method": str(item.get("method") or ""),
                "component": str(component),
                "rank": int(rank),
                "unit_id": unit_id,
                "score": float(item.get("score")),
                "abs_score": float(item.get("abs_score")),
                "source_dataset": str(item.get("source_dataset") or cf.get("source_dataset") or ""),
                "subset": str(cf.get("subset") or ""),
                "item_type": str(item.get("item_type") or cf.get("item_type") or ""),
                "axis": str(item.get("axis") or cf.get("axis") or ""),
                "direction": str(cf.get("direction") or ""),
                "role": str(item.get("role") or cf.get("role") or ""),
                "transform_id": str(cf.get("transform_id") or ""),
                "pair_id": str(cf.get("pair_id") or ""),
                "title": str(cf.get("title") or ""),
                "prompt": _clip(cf.get("prompt") or cf.get("question") or "", preview_chars),
                "base_text": _clip(cf.get("base_text") or "", preview_chars),
                "variant_text": _clip(cf.get("variant_text") or "", preview_chars),
                "cue_plus_text": _clip(cue_plus, preview_chars),
                "cue_minus_text": _clip(cue_minus, preview_chars),
                "cue_plus_source": cue_plus_source,
                "content_preservation_flags": str(cf.get("content_preservation_flags") or ""),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    factor_dir = _resolve(workspace_root, args.factor_dir)
    counterfactual_jsonl = _resolve(workspace_root, args.counterfactual_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)

    top_units = pd.read_csv(factor_dir / "top_units.csv")
    counterfactual_rows = read_jsonl(counterfactual_jsonl)
    out = export_top_texts(
        top_units,
        counterfactual_rows,
        method=str(args.method),
        components=[str(item) for item in args.component],
        top_k=int(args.top_k),
        preview_chars=int(args.text_preview_chars),
    )
    if out.empty:
        raise ValueError("No matching top units were selected.")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "top_unit_texts.csv"
    jsonl_path = out_dir / "top_unit_texts.jsonl"
    out.to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in out.to_dict("records"):
            import json

            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(
        out_dir / "summary.json",
        {
            "stage": "D4-decision-manifold-top-text-export",
            "factor_dir": str(factor_dir),
            "counterfactual_jsonl": str(counterfactual_jsonl),
            "method": str(args.method),
            "components": [str(item) for item in args.component],
            "top_k": int(args.top_k),
            "text_preview_chars": int(args.text_preview_chars),
            "n_rows": int(len(out)),
            "outputs": {
                "top_unit_texts_csv": str(csv_path),
                "top_unit_texts_jsonl": str(jsonl_path),
                "summary_json": str(out_dir / "summary.json"),
            },
        },
    )
    print(f"out_dir={out_dir}")
    print(f"n_rows={len(out)}")
    print(f"csv={csv_path}")
    print(f"jsonl={jsonl_path}")


if __name__ == "__main__":
    main()
