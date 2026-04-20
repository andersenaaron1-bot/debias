"""Build the first canonical D4 dataset pack for mechanistic tracing.

This script materializes the first D4 pack from already-derived artifacts:

- broad atom-probe texts from the bundle-creation corpus
- Laurito ecological pair/run comparisons from the first-wave D3 outputs
- held-out SHP preference pairs as a content-anchor set
- optional rewrite-control pairs when they are available

It does not run SAEs itself. The purpose is to freeze the exact text sets,
atom labels, bundle labels, and run comparisons that the first mechanistic
pass should use.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd

from aisafety.config import PROJECT_ROOT
from aisafety.ontology.validation import score_records_with_atoms


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--config-json",
        type=Path,
        default=Path("configs") / "datasets" / "d4_dataset_pack_v1.json",
        help="Dataset-pack configuration JSON.",
    )
    p.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--out-dir", type=Path, default=None, help="Optional override for the output directory.")
    return p.parse_args()


def _load_config(config_json: Path, *, workspace_root: Path) -> dict[str, Any]:
    config_path = config_json if config_json.is_absolute() else (workspace_root / config_json)
    payload = _read_json(config_path)
    payload["_config_path"] = str(config_path.resolve())
    return payload


def _load_d4_panel(path: Path) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    payload = _read_json(path)
    bundles: dict[str, dict[str, Any]] = {}
    atom_set: set[str] = set(str(atom) for atom in payload.get("priority_atoms", []))
    for row in payload.get("trace_bundles", []):
        if not isinstance(row, dict):
            continue
        bundle_id = str(row.get("bundle_id") or "").strip()
        if not bundle_id:
            continue
        member_atoms = [str(atom).strip() for atom in row.get("member_atoms", []) if str(atom).strip()]
        readout_bundles = [str(x).strip() for x in row.get("readout_bundles", []) if str(x).strip()]
        bundles[bundle_id] = {
            "status": str(row.get("status") or ""),
            "member_atoms": member_atoms,
            "readout_bundles": readout_bundles,
            "motivation": str(row.get("motivation") or ""),
        }
        atom_set.update(member_atoms)
    return bundles, sorted(atom_set), [str(x) for x in payload.get("priority_atoms", [])]


def _bundle_scores_from_atoms(atom_scores: dict[str, float], bundles: dict[str, dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for bundle_id, payload in bundles.items():
        atoms = [atom for atom in payload["member_atoms"] if atom in atom_scores]
        if not atoms:
            out[bundle_id] = 0.0
            continue
        out[bundle_id] = float(sum(float(atom_scores[atom]) for atom in atoms) / float(len(atoms)))
    return out


def _build_atom_probe_set(
    *,
    atom_probe_jsonl: Path,
    bundles: dict[str, dict[str, Any]],
    d4_atoms: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = _read_jsonl(atom_probe_jsonl)
    if not raw_rows:
        raise ValueError(f"No rows found in atom-probe source: {atom_probe_jsonl}")

    scored = score_records_with_atoms(raw_rows)
    rows: list[dict[str, Any]] = []
    for row in scored.to_dict(orient="records"):
        meta = dict(row.get("meta") or {})
        atom_scores = {atom: float(row.get(atom, 0.0)) for atom in d4_atoms}
        rows.append(
            {
                "example_id": row.get("example_id"),
                "group_id": row.get("group_id"),
                "split": row.get("split"),
                "item_type": row.get("item_type"),
                "dataset": row.get("dataset"),
                "subset": row.get("subset"),
                "source": row.get("source"),
                "title": row.get("title"),
                "text": row.get("text"),
                "generator": row.get("generator"),
                "prompt_name": row.get("prompt_name"),
                "question": row.get("question"),
                "word_count": int(row.get("word_count", 0) or 0),
                "bundle_creation_role": meta.get("bundle_creation_role"),
                "bundle_creation_dataset_id": meta.get("bundle_creation_dataset_id"),
                "bundle_creation_stratum_id": meta.get("bundle_creation_stratum_id"),
                "atom_scores": atom_scores,
                "bundle_scores": _bundle_scores_from_atoms(atom_scores, bundles),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "n_rows": int(len(df)),
        "by_split": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()},
        "by_item_type": {str(k): int(v) for k, v in df["item_type"].value_counts().sort_index().items()},
        "by_dataset": {str(k): int(v) for k, v in df["dataset"].value_counts().sort_index().items()},
        "by_bundle_creation_role": {
            str(k): int(v)
            for k, v in df["bundle_creation_role"].fillna("unknown").value_counts().sort_index().items()
        },
    }
    return rows, summary


def _score_single_text(text: str, *, bundles: dict[str, dict[str, Any]], d4_atoms: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    scored = score_records_with_atoms([{"text": str(text or "")}]).to_dict(orient="records")[0]
    atom_scores = {atom: float(scored.get(atom, 0.0)) for atom in d4_atoms}
    return atom_scores, _bundle_scores_from_atoms(atom_scores, bundles)


def _build_content_anchor_set(
    *,
    content_anchor_jsonl: Path,
    bundles: dict[str, dict[str, Any]],
    d4_atoms: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = _read_jsonl(content_anchor_jsonl)
    if not raw_rows:
        raise ValueError(f"No rows found in content-anchor source: {content_anchor_jsonl}")

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        chosen = str(row.get("chosen") or "")
        rejected = str(row.get("rejected") or "")
        chosen_atoms, chosen_bundles = _score_single_text(chosen, bundles=bundles, d4_atoms=d4_atoms)
        rejected_atoms, rejected_bundles = _score_single_text(rejected, bundles=bundles, d4_atoms=d4_atoms)
        rows.append(
            {
                "pair_id": row.get("pair_id"),
                "source_dataset": row.get("source_dataset"),
                "domain": row.get("domain"),
                "prompt": row.get("prompt"),
                "chosen_text": chosen,
                "rejected_text": rejected,
                "chosen_atom_scores": chosen_atoms,
                "rejected_atom_scores": rejected_atoms,
                "chosen_bundle_scores": chosen_bundles,
                "rejected_bundle_scores": rejected_bundles,
                "meta": row.get("meta") or {},
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        "n_rows": int(len(df)),
        "by_domain": {str(k): int(v) for k, v in df["domain"].fillna("unknown").value_counts().sort_index().items()},
        "by_source_dataset": {
            str(k): int(v) for k, v in df["source_dataset"].fillna("unknown").value_counts().sort_index().items()
        },
    }
    return rows, summary


def _build_rewrite_control_set(
    *,
    base_dir: Path,
    rewrite_glob: str | None,
    bundles: dict[str, dict[str, Any]],
    d4_atoms: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rewrite_glob:
        return [], {"n_rows": 0, "status": "not_requested"}

    pattern = str((_resolve_path(base_dir, rewrite_glob) if not Path(rewrite_glob).is_absolute() else Path(rewrite_glob)))
    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        return [], {"n_rows": 0, "status": "missing"}

    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            seed_text = str(row.get("seed_text") or "")
            generated_text = str(row.get("generated_text") or "")
            seed_atoms, seed_bundles = _score_single_text(seed_text, bundles=bundles, d4_atoms=d4_atoms)
            gen_atoms, gen_bundles = _score_single_text(generated_text, bundles=bundles, d4_atoms=d4_atoms)
            rows.append(
                {
                    "dimension": row.get("dimension"),
                    "label": row.get("label"),
                    "seed_source": row.get("seed_source"),
                    "seed_id": row.get("seed_id"),
                    "model": row.get("model"),
                    "item_type": (row.get("meta") or {}).get("item_type"),
                    "source": (row.get("meta") or {}).get("source"),
                    "title": (row.get("meta") or {}).get("title"),
                    "seed_text": seed_text,
                    "generated_text": generated_text,
                    "seed_atom_scores": seed_atoms,
                    "generated_atom_scores": gen_atoms,
                    "seed_bundle_scores": seed_bundles,
                    "generated_bundle_scores": gen_bundles,
                }
            )
    df = pd.DataFrame(rows)
    summary = {
        "n_rows": int(len(df)),
        "status": "ok",
        "by_dimension": {str(k): int(v) for k, v in df["dimension"].fillna("unknown").value_counts().sort_index().items()},
    }
    return rows, summary


def _read_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _build_laurito_ecology_set(
    *,
    d3_root: Path,
    run_ids: list[str],
    canonical_text_score_run: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pair_frames: list[pd.DataFrame] = []
    atom_effect_frames: list[pd.DataFrame] = []
    bundle_effect_frames: list[pd.DataFrame] = []
    run_summaries: dict[str, Any] = {}
    text_atom_scores: pd.DataFrame | None = None

    for run_id in run_ids:
        run_dir = d3_root / f"d3_{run_id}"
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing D3 directory for run {run_id}: {run_dir}")

        pair_df = pd.read_csv(run_dir / "pair_level_inputs.csv")
        pair_df.insert(0, "run_id", run_id)
        pair_frames.append(pair_df)

        atom_df = pd.DataFrame(_read_tsv(run_dir / "atom_effects.tsv"))
        atom_df.insert(0, "run_id", run_id)
        atom_effect_frames.append(atom_df)

        bundle_df = pd.DataFrame(_read_tsv(run_dir / "bundle_effects.tsv"))
        bundle_df.insert(0, "run_id", run_id)
        bundle_effect_frames.append(bundle_df)

        run_summaries[run_id] = _read_json(run_dir / "summary.json")

        if run_id == canonical_text_score_run:
            text_atom_scores = pd.read_csv(run_dir / "text_atom_scores.csv")

    if text_atom_scores is None:
        raise ValueError(f"canonical_text_score_run {canonical_text_score_run!r} not present in run_ids")

    summary = {
        "run_ids": run_ids,
        "canonical_text_score_run": canonical_text_score_run,
        "pair_rows_by_run": {run_id: int(run_summaries[run_id]["n_pair_rows"]) for run_id in run_ids},
        "llm_choice_rate_by_run": {
            run_id: float(run_summaries[run_id]["by_item_type"]["paper"]["llm_choice_rate"])
            if "paper" in run_summaries[run_id].get("by_item_type", {})
            else None
            for run_id in run_ids
        },
        "n_unique_texts": int(len(text_atom_scores)),
    }
    return (
        pd.concat(pair_frames, ignore_index=True),
        pd.concat(atom_effect_frames, ignore_index=True),
        pd.concat(bundle_effect_frames, ignore_index=True),
        text_atom_scores,
        summary,
    )


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    config = _load_config(args.config_json, workspace_root=workspace_root)

    out_dir = Path(args.out_dir).resolve() if args.out_dir is not None else _resolve_path(workspace_root, str(config["out_dir"]))
    if out_dir is None:
        raise ValueError("Could not resolve output directory.")
    out_dir.mkdir(parents=True, exist_ok=True)

    d4_ontology_json = _resolve_path(workspace_root, str(config["d4_ontology_json"]))
    atom_probe_jsonl = _resolve_path(workspace_root, str(config["atom_probe_jsonl"]))
    content_anchor_jsonl = _resolve_path(workspace_root, str(config["content_anchor_jsonl"]))
    d3_root = _resolve_path(workspace_root, str(config["d3_root"]))
    run_ids = [str(x) for x in config.get("run_ids", [])]
    canonical_text_score_run = str(config.get("canonical_text_score_run") or "")

    if d4_ontology_json is None or atom_probe_jsonl is None or content_anchor_jsonl is None or d3_root is None:
        raise ValueError("Missing required resolved paths.")

    bundles, d4_atoms, priority_atoms = _load_d4_panel(d4_ontology_json)

    atom_probe_rows, atom_probe_summary = _build_atom_probe_set(
        atom_probe_jsonl=atom_probe_jsonl,
        bundles=bundles,
        d4_atoms=d4_atoms,
    )
    _write_jsonl(out_dir / "atom_probe_set.jsonl", atom_probe_rows)

    content_anchor_rows, content_anchor_summary = _build_content_anchor_set(
        content_anchor_jsonl=content_anchor_jsonl,
        bundles=bundles,
        d4_atoms=d4_atoms,
    )
    _write_jsonl(out_dir / "content_anchor_set.jsonl", content_anchor_rows)

    rewrite_rows, rewrite_summary = _build_rewrite_control_set(
        base_dir=workspace_root,
        rewrite_glob=config.get("rewrite_glob"),
        bundles=bundles,
        d4_atoms=d4_atoms,
    )
    if rewrite_rows:
        _write_jsonl(out_dir / "rewrite_control_set.jsonl", rewrite_rows)

    laurito_pairs, atom_effects, bundle_effects, text_atom_scores, laurito_summary = _build_laurito_ecology_set(
        d3_root=d3_root,
        run_ids=run_ids,
        canonical_text_score_run=canonical_text_score_run,
    )
    laurito_pairs.to_csv(out_dir / "laurito_pair_runs.csv", index=False)
    atom_effects.to_csv(out_dir / "laurito_atom_effects.csv", index=False)
    bundle_effects.to_csv(out_dir / "laurito_bundle_effects.csv", index=False)
    text_atom_scores.to_csv(out_dir / "laurito_text_atom_scores.csv", index=False)

    manifest = {
        "name": str(config.get("name") or "d4_dataset_pack"),
        "description": str(config.get("description") or ""),
        "config_json": str(config["_config_path"]),
        "workspace_root": str(workspace_root),
        "d4_ontology_json": str(d4_ontology_json),
        "atom_probe_jsonl": str(atom_probe_jsonl),
        "content_anchor_jsonl": str(content_anchor_jsonl),
        "d3_root": str(d3_root),
        "run_ids": run_ids,
        "canonical_text_score_run": canonical_text_score_run,
        "trace_bundles": bundles,
        "d4_atoms": d4_atoms,
        "priority_atoms": priority_atoms,
        "outputs": {
            "atom_probe_set": str(out_dir / "atom_probe_set.jsonl"),
            "content_anchor_set": str(out_dir / "content_anchor_set.jsonl"),
            "rewrite_control_set": None if not rewrite_rows else str(out_dir / "rewrite_control_set.jsonl"),
            "laurito_pair_runs": str(out_dir / "laurito_pair_runs.csv"),
            "laurito_atom_effects": str(out_dir / "laurito_atom_effects.csv"),
            "laurito_bundle_effects": str(out_dir / "laurito_bundle_effects.csv"),
            "laurito_text_atom_scores": str(out_dir / "laurito_text_atom_scores.csv"),
        },
        "summaries": {
            "atom_probe_set": atom_probe_summary,
            "content_anchor_set": content_anchor_summary,
            "rewrite_control_set": rewrite_summary,
            "laurito_ecology_set": laurito_summary,
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "summary.json", manifest["summaries"])

    print(f"Wrote D4 dataset pack to {out_dir}")
    print(f"Wrote manifest to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
