"""Materialize a stratified corpus for bundle creation from the corpus-spec config.

Example:
  python -m aisafety.scripts.build_bundle_creation_corpus ^
    --spec-json configs\\datasets\\bundle_creation_corpus_spec_v1.json ^
    --hc3-dir data\\HC3 ^
    --hape-jsonl data\\external\\hape_excerpt.jsonl ^
    --pubmed-jsonl data\\external\\pubmed_abstracts_excerpt.jsonl ^
    --movie-summary-jsonl data\\external\\movie_summaries_excerpt.jsonl ^
    --product-jsonl data\\external\\amazon_product_descriptions_excerpt.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aisafety.config import DATA_DIR, DEFAULT_SEED
from aisafety.data.bundle_corpus import load_bundle_creation_spec, materialize_bundle_creation_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--spec-json",
        type=Path,
        default=Path("configs") / "datasets" / "bundle_creation_corpus_spec_v1.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "derived" / "bundle_creation_corpus_v1",
    )
    p.add_argument("--hc3-dir", type=Path, default=DATA_DIR / "HC3")
    p.add_argument("--remote-hllmc2-sources", type=str, default="")
    p.add_argument("--remote-hllmc2-max-groups-per-source", type=int, default=0)
    p.add_argument("--remote-hllmc2-cache-dir", type=Path, default=None)
    p.add_argument("--hape-jsonl", type=Path, default=None)
    p.add_argument("--pubmed-jsonl", type=Path, default=None)
    p.add_argument("--movie-summary-jsonl", type=Path, default=None)
    p.add_argument("--product-jsonl", type=Path, default=None)
    p.add_argument("--paper-llm-jsonl", type=Path, default=None)
    p.add_argument("--movie-llm-jsonl", type=Path, default=None)
    p.add_argument("--product-llm-jsonl", type=Path, default=None)
    p.add_argument("--hc3-plus-jsonl", type=Path, default=None)
    p.add_argument("--rewrite-jsonl", type=Path, default=None)
    p.add_argument("--no-laurito-ecology", dest="include_laurito_ecology", action="store_false")
    p.set_defaults(include_laurito_ecology=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--max-local-variants-per-group-source", type=int, default=4)
    return p.parse_args()


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spec = load_bundle_creation_spec(args.spec_json)
    records, summary = materialize_bundle_creation_records(
        spec,
        hc3_dir=args.hc3_dir,
        remote_hllmc2_sources=_parse_csv(str(args.remote_hllmc2_sources)),
        remote_hllmc2_max_groups_per_source=int(args.remote_hllmc2_max_groups_per_source),
        remote_hllmc2_cache_dir=None if args.remote_hllmc2_cache_dir is None else Path(args.remote_hllmc2_cache_dir),
        hape_jsonl=args.hape_jsonl,
        pubmed_jsonl=args.pubmed_jsonl,
        movie_summary_jsonl=args.movie_summary_jsonl,
        product_jsonl=args.product_jsonl,
        paper_llm_jsonl=args.paper_llm_jsonl,
        movie_llm_jsonl=args.movie_llm_jsonl,
        product_llm_jsonl=args.product_llm_jsonl,
        hc3_plus_jsonl=args.hc3_plus_jsonl,
        rewrite_jsonl=args.rewrite_jsonl,
        include_laurito_ecology=bool(args.include_laurito_ecology),
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        max_local_variants_per_group_source=int(args.max_local_variants_per_group_source),
    )

    record_dicts = [rec.to_dict() for rec in records]
    _write_jsonl(args.out_dir / "all_records.jsonl", record_dicts)

    roles = sorted({str((rec.meta or {}).get("bundle_creation_role") or "unknown") for rec in records})
    role_outputs: dict[str, str] = {}
    for role in roles:
        subset = [rec.to_dict() for rec in records if str((rec.meta or {}).get("bundle_creation_role") or "unknown") == role]
        out_path = args.out_dir / f"{role}.jsonl"
        _write_jsonl(out_path, subset)
        role_outputs[role] = str(out_path)

    summary_payload = {
        "spec_json": str(args.spec_json),
        "out_dir": str(args.out_dir),
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "include_laurito_ecology": bool(args.include_laurito_ecology),
        "role_outputs": role_outputs,
        **summary,
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote {len(records)} records to {args.out_dir}")
    print(f"Wrote summary to {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
