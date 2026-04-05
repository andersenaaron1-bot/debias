"""Build D2 bundle-validation artifacts from an atom-scored discovery corpus.

Example:
  python -m aisafety.scripts.build_bundle_validation_d2 ^
    --input-jsonl data\\derived\\cue_discovery_v2\\corpus.jsonl ^
    --out-dir data\\derived\\style_groups\\d2_validation_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aisafety.config import DATA_DIR, DEFAULT_SEED
from aisafety.ontology.atoms import ATOM_SPEC_NAMES, get_atom_specs
from aisafety.ontology.validation import (
    build_bundle_validation,
    compute_atom_summaries,
    compute_pairwise_cooccurrence,
    read_jsonl,
    score_records_with_atoms,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-jsonl",
        type=Path,
        default=DATA_DIR / "derived" / "cue_discovery_v2" / "corpus.jsonl",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "derived" / "style_groups" / "d2_validation_v1",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--bootstrap-samples", type=int, default=200)
    p.add_argument("--null-samples", type=int, default=500)
    p.add_argument(
        "--write-scored-jsonl",
        action="store_true",
        help="Also write a per-record atom-scored JSONL.",
    )
    return p.parse_args()


def _bundle_rows_to_tsv_rows(bundle_validation: dict[str, dict]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for bundle, payload in bundle_validation.items():
        rows.append(
            {
                "bundle": bundle,
                "status": str(payload["status"]),
                "n_atoms_validation": str(payload["n_atoms_validation"]),
                "observed_mean_pairwise_r": f"{float(payload['observed_mean_pairwise_r']):.6f}",
                "bootstrap_ci_95_low": f"{float(payload['bootstrap_ci_95_low']):.6f}",
                "bootstrap_ci_95_high": f"{float(payload['bootstrap_ci_95_high']):.6f}",
                "null_mean_pairwise_r": f"{float(payload['null_mean_pairwise_r']):.6f}",
                "empirical_p_value": f"{float(payload['empirical_p_value']):.6f}",
                "mean_coactivation_jaccard": f"{float(payload['mean_coactivation_jaccard']):.6f}",
                "mean_co_clustering_probability": f"{float(payload['mean_co_clustering_probability']):.6f}",
                "first_pc_explained_variance": f"{float(payload['first_pc_explained_variance']):.6f}",
                "source_auc_human_vs_llm": "" if payload["source_auc_human_vs_llm"] is None else f"{float(payload['source_auc_human_vs_llm']):.6f}",
                "best_derived_cluster_id": str(payload["best_derived_cluster_id"]),
                "best_derived_cluster_jaccard": f"{float(payload['best_derived_cluster_jaccard']):.6f}",
                "member_atoms_validation": ";".join(payload["member_atoms_validation"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.input_jsonl)
    if not records:
        raise ValueError(f"No rows found in {args.input_jsonl}")

    df = score_records_with_atoms(records)
    atom_cols = list(ATOM_SPEC_NAMES)

    atom_summary = compute_atom_summaries(
        df,
        atom_cols,
        n_bootstrap=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    pairwise = compute_pairwise_cooccurrence(df, atom_cols)
    bundle_validation = build_bundle_validation(
        df,
        n_bootstrap=int(args.bootstrap_samples),
        n_null_samples=int(args.null_samples),
        seed=int(args.seed),
    )

    operationalization = {
        "source_inventory": str(DATA_DIR / "derived" / "style_groups" / "candidate_atom_inventory_d1.tsv"),
        "n_atoms": int(len(atom_cols)),
        "atoms": {atom_id: spec.to_dict() for atom_id, spec in get_atom_specs().items()},
    }
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "n_records": int(len(df)),
        "seed": int(args.seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "null_samples": int(args.null_samples),
        "n_atom_features": int(len(atom_cols)),
        "by_item_type_source": {
            str(item_type): {
                str(source): int(count)
                for source, count in grp["source"].value_counts().sort_index().items()
            }
            for item_type, grp in df.groupby("item_type")
        },
    }

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    with (args.out_dir / "atom_operationalization.json").open("w", encoding="utf-8") as f:
        json.dump(operationalization, f, ensure_ascii=False, indent=2, sort_keys=True)
    with (args.out_dir / "atom_summary.json").open("w", encoding="utf-8") as f:
        json.dump(atom_summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    with (args.out_dir / "bundle_validation.json").open("w", encoding="utf-8") as f:
        json.dump(bundle_validation, f, ensure_ascii=False, indent=2, sort_keys=True)

    pairwise.to_csv(args.out_dir / "pairwise_cooccurrence.csv", index=False)

    tsv_rows = _bundle_rows_to_tsv_rows(bundle_validation["bundle_validation"])
    if tsv_rows:
        import csv

        with (args.out_dir / "bundle_validation.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(tsv_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(tsv_rows)

    if args.write_scored_jsonl:
        keep_meta = [
            "example_id",
            "group_id",
            "split",
            "item_type",
            "dataset",
            "subset",
            "source",
            "title",
            "generator",
            "prompt_name",
            "question",
        ]
        with (args.out_dir / "atom_scores.jsonl").open("w", encoding="utf-8") as f:
            for row in df.to_dict(orient="records"):
                out = {key: row[key] for key in keep_meta if key in row}
                out["atom_scores"] = {atom: float(row[atom]) for atom in atom_cols}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote D2 artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
