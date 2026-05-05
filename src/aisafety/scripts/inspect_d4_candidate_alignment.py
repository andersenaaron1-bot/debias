"""Inspect or package a D4 human-vs-LLM candidate-feature alignment run.

This script is intentionally local-friendly: the input may be either the run
directory or a downloaded ``.tar.gz`` archive. When given an archive it extracts
it safely, locates ``alignment_manifest.json``, and writes a compact markdown
readout next to the extracted run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import tarfile
from typing import Any


DEFAULT_RUN_DIR = Path("artifacts") / "mechanistic" / "d4_j0_human_llm_candidate_alignment_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Alignment run directory or .tar.gz archive.",
    )
    parser.add_argument(
        "--extract-to",
        type=Path,
        default=None,
        help="Extraction directory when --input is an archive. Defaults to the archive parent.",
    )
    parser.add_argument("--out-md", type=Path, default=None, help="Markdown readout path.")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--source-top-k", type=int, default=40)
    parser.add_argument("--archive", action="store_true", help="Write a .tar.gz archive for directory input.")
    parser.add_argument("--no-print", action="store_true", help="Write files but do not print the readout.")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).is_file():
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, ndigits: int = 4) -> str:
    val = _float_or_none(value)
    if val is None:
        return "NA"
    return str(round(val, ndigits))


def _median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[int(0.9 * (len(ordered) - 1))])


def _count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _abs_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = _float_or_none(row.get(key))
    return default if value is None else abs(value)


def _float_sort(row: dict[str, str], key: str, default: float = 9.0) -> float:
    value = _float_or_none(row.get(key))
    return default if value is None else float(value)


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            text = _fmt(value) if isinstance(value, float) else str(value)
            vals.append(text.replace("\n", " ").replace("|", "\\|")[:400])
        out.append("| " + " | ".join(vals) + " |")
    return out


def _safe_extract_tar(archive: Path, extract_to: Path) -> Path:
    archive = Path(archive).resolve()
    extract_to = Path(extract_to).resolve()
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (extract_to / member.name).resolve()
            if not str(target).startswith(str(extract_to)):
                raise ValueError(f"Unsafe archive member path: {member.name}")
        tar.extractall(extract_to)

    manifests = sorted(extract_to.rglob("alignment_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No alignment_manifest.json found after extracting {archive}")
    return manifests[0].parent


def _archive_run(run_dir: Path) -> Path:
    run_dir = Path(run_dir).resolve()
    archive_path = run_dir.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    return archive_path


def _resolve_input(input_path: Path, extract_to: Path | None) -> Path:
    path = Path(input_path).resolve()
    if path.is_dir():
        return path
    if path.name.endswith(".tar.gz"):
        target = Path(extract_to).resolve() if extract_to is not None else path.parent
        return _safe_extract_tar(path, target)
    raise FileNotFoundError(f"Input must be a run directory or .tar.gz archive: {input_path}")


def _control_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in ("discovered", "random_control"):
        subset = [
            row
            for row in rows
            if (row.get("candidate_kind") == "random_control") == (label == "random_control")
        ]
        controlled = [
            _abs_float(row, "length_controlled_spearman_delta_with_j0_margin")
            for row in subset
            if _float_or_none(row.get("length_controlled_spearman_delta_with_j0_margin")) is not None
        ]
        auc_delta = [
            abs(float(row["auc_j0_llm_choice_from_activation_delta"]) - 0.5)
            for row in subset
            if _float_or_none(row.get("auc_j0_llm_choice_from_activation_delta")) is not None
        ]
        out.append(
            {
                "kind": label,
                "n": len(subset),
                "median_abs_controlled_rho": _median(controlled),
                "p90_abs_controlled_rho": _p90(controlled),
                "median_abs_choice_auc_delta": _median(auc_delta),
                "p90_abs_choice_auc_delta": _p90(auc_delta),
            }
        )
    return out


def _top_alignment(rows: list[dict[str, str]], *, top_k: int) -> list[dict[str, Any]]:
    discovered = [row for row in rows if row.get("candidate_kind") != "random_control"]
    ranked = sorted(
        discovered,
        key=lambda row: (
            _float_sort(row, "length_controlled_spearman_q"),
            -_abs_float(row, "length_controlled_spearman_delta_with_j0_margin"),
            -abs((_float_or_none(row.get("auc_j0_llm_choice_from_activation_delta")) or 0.5) - 0.5),
        ),
    )
    columns = [
        "hidden_layer",
        "feature_idx",
        "atoms",
        "bundles",
        "n_pairs",
        "mean_llm_minus_human_activation",
        "activation_auc_llm_vs_human",
        "auc_j0_llm_choice_from_activation_delta",
        "length_controlled_spearman_delta_with_j0_margin",
        "length_controlled_spearman_q",
        "source_sign_consistency",
        "n_sources_with_min_pairs",
        "candidate_reasons",
    ]
    return [{col: row.get(col, "") for col in columns} for row in ranked[: int(top_k)]]


def _top_sources(rows: list[dict[str, str]], *, top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -_abs_float(row, "spearman_delta_with_j0_margin"),
            -abs((_float_or_none(row.get("auc_j0_llm_choice_from_activation_delta")) or 0.5) - 0.5),
        ),
    )
    columns = [
        "source_dataset",
        "item_type",
        "hidden_layer",
        "feature_idx",
        "atoms",
        "bundles",
        "n_pairs",
        "mean_llm_minus_human_activation",
        "activation_auc_llm_vs_human",
        "auc_j0_llm_choice_from_activation_delta",
        "spearman_delta_with_j0_margin",
    ]
    return [{col: row.get(col, "") for col in columns} for row in ranked[: int(top_k)]]


def build_readout(run_dir: Path, *, top_k: int, source_top_k: int) -> str:
    """Return a compact markdown readout for an alignment run."""

    manifest = _read_json(run_dir / "alignment_manifest.json")
    align_rows = _read_csv(run_dir / "candidate_feature_human_llm_alignment.csv")
    source_rows = _read_csv(run_dir / "candidate_feature_source_alignment.csv")
    pair_rows = _read_csv(run_dir / "pair_scores.csv")

    discovered = [row for row in align_rows if row.get("candidate_kind") != "random_control"]
    controls = [row for row in align_rows if row.get("candidate_kind") == "random_control"]
    significant = [
        row
        for row in discovered
        if (_float_or_none(row.get("length_controlled_spearman_q")) or 1.0) <= 0.1
    ]

    lines: list[str] = []
    lines.append("# D4 Human-vs-LLM Candidate Alignment Readout")
    lines.append("")
    lines.append("## Manifest")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- n_pairs: `{manifest.get('n_pairs', len(pair_rows))}`")
    lines.append(f"- n_unique_texts: `{manifest.get('n_unique_texts', 'NA')}`")
    lines.append(f"- y_llm_chosen_rate: `{_fmt(manifest.get('y_llm_chosen_rate'))}`")
    lines.append(f"- mean_llm_margin_pair: `{_fmt(manifest.get('mean_llm_margin_pair'))}`")
    lines.append(f"- completed_layers: `{manifest.get('completed_hidden_layers', [])}`")
    lines.append(f"- skipped_layers: `{manifest.get('skipped_layers', [])}`")
    lines.append(f"- discovered_rows: `{len(discovered)}`")
    lines.append(f"- random_control_rows: `{len(controls)}`")
    lines.append(f"- discovered_q_le_0.10: `{len(significant)}`")
    lines.append("")
    lines.append("## Pair Composition")
    for key in ("source_dataset", "item_type", "split"):
        lines.append(f"- {key}: `{_count_by(pair_rows, key)}`")
    lines.append("")
    lines.append("## Control Baseline")
    lines.extend(
        _md_table(
            _control_summary(align_rows),
            [
                "kind",
                "n",
                "median_abs_controlled_rho",
                "p90_abs_controlled_rho",
                "median_abs_choice_auc_delta",
                "p90_abs_choice_auc_delta",
            ],
        )
    )
    lines.append("")
    lines.append("## Top Controlled Judge-Aligned Discovered Features")
    top_rows = _top_alignment(align_rows, top_k=top_k)
    lines.extend(
        _md_table(
            top_rows,
            [
                "hidden_layer",
                "feature_idx",
                "atoms",
                "bundles",
                "n_pairs",
                "mean_llm_minus_human_activation",
                "activation_auc_llm_vs_human",
                "auc_j0_llm_choice_from_activation_delta",
                "length_controlled_spearman_delta_with_j0_margin",
                "length_controlled_spearman_q",
                "source_sign_consistency",
                "n_sources_with_min_pairs",
                "candidate_reasons",
            ],
        )
    )
    lines.append("")
    lines.append("## Top Source/Domain Rows")
    source_top = _top_sources(source_rows, top_k=source_top_k)
    lines.extend(
        _md_table(
            source_top,
            [
                "source_dataset",
                "item_type",
                "hidden_layer",
                "feature_idx",
                "atoms",
                "bundles",
                "n_pairs",
                "mean_llm_minus_human_activation",
                "activation_auc_llm_vs_human",
                "auc_j0_llm_choice_from_activation_delta",
                "spearman_delta_with_j0_margin",
            ],
        )
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    run_dir = _resolve_input(args.input, args.extract_to)
    if not (run_dir / "alignment_manifest.json").is_file():
        raise FileNotFoundError(f"Missing alignment_manifest.json in {run_dir}")

    archive_path: Path | None = None
    if bool(args.archive):
        archive_path = _archive_run(run_dir)

    readout = build_readout(run_dir, top_k=int(args.top_k), source_top_k=int(args.source_top_k))
    out_md = args.out_md if args.out_md is not None else run_dir / "alignment_readout.md"
    out_md = Path(out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(readout, encoding="utf-8")

    print(f"run_dir={run_dir}")
    print(f"readout={out_md}")
    if archive_path is not None:
        print(f"archive={archive_path}")
    if not bool(args.no_print):
        print("")
        print(readout)


if __name__ == "__main__":
    main()
