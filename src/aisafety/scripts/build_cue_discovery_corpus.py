"""Build a unified cue-discovery corpus from local domain data and HC3.

Example:
  python -m aisafety.scripts.build_cue_discovery_corpus ^
    --out-jsonl data\\derived\\cue_discovery\\corpus.jsonl ^
    --summary-json data\\derived\\cue_discovery\\summary.json ^
    --max-groups-per-item-type paper=200,product=200,movie=200,hc3=5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aisafety.config import DATA_DIR, DEFAULT_SEED
from aisafety.data.cue_corpus import (
    collect_cue_corpus_records,
    limit_records_by_item_type,
    summarize_cue_corpus,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-jsonl", type=Path, default=DATA_DIR / "derived" / "cue_discovery" / "corpus.jsonl")
    p.add_argument(
        "--summary-json",
        type=Path,
        default=DATA_DIR / "derived" / "cue_discovery" / "summary.json",
    )
    p.add_argument("--hc3-dir", type=Path, default=DATA_DIR / "HC3")
    p.add_argument("--no-hc3", dest="include_hc3", action="store_false")
    p.set_defaults(include_hc3=True)
    p.add_argument(
        "--remote-hc3-configs",
        type=str,
        default="",
        help="Comma-separated remote HC3 configs to sample, e.g. reddit_eli5",
    )
    p.add_argument(
        "--remote-hc3-max-groups-per-config",
        type=int,
        default=0,
        help="Max question groups to import per remote HC3 config.",
    )
    p.add_argument("--remote-cache-dir", type=Path, default=None)
    p.add_argument(
        "--remote-hllmc2-sources",
        type=str,
        default="",
        help="Comma-separated H-LLMC2 sources to sample, e.g. finance,medicine,open_qa,reddit_eli5",
    )
    p.add_argument(
        "--remote-hllmc2-max-groups-per-source",
        type=int,
        default=0,
        help="Max prompt groups to import per H-LLMC2 source.",
    )
    p.add_argument("--remote-hllmc2-cache-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument(
        "--max-local-variants-per-group-source",
        type=int,
        default=4,
        help="Cap prompt variants kept per local title/source group to limit prompt duplication.",
    )
    p.add_argument(
        "--max-groups-per-item-type",
        type=str,
        default="",
        help="Comma-separated caps like paper=200,product=200,movie=200,hc3=5000",
    )
    return p.parse_args()


def _parse_group_caps(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid cap spec {part!r}; expected item_type=n")
        key, val = part.split("=", 1)
        out[key.strip()] = int(val.strip())
    return out


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    records = collect_cue_corpus_records(
        include_hc3=bool(args.include_hc3),
        hc3_dir=Path(args.hc3_dir),
        remote_hc3_configs=_parse_csv(str(args.remote_hc3_configs)),
        remote_hc3_max_groups_per_config=int(args.remote_hc3_max_groups_per_config),
        remote_cache_dir=None if args.remote_cache_dir is None else Path(args.remote_cache_dir),
        remote_hllmc2_sources=_parse_csv(str(args.remote_hllmc2_sources)),
        remote_hllmc2_max_groups_per_source=int(args.remote_hllmc2_max_groups_per_source),
        remote_hllmc2_cache_dir=None if args.remote_hllmc2_cache_dir is None else Path(args.remote_hllmc2_cache_dir),
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        max_variants_per_group_source=int(args.max_local_variants_per_group_source),
    )
    caps = _parse_group_caps(str(args.max_groups_per_item_type))
    if caps:
        records = limit_records_by_item_type(records, max_groups_by_item_type=caps, seed=int(args.seed))

    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    summary = {
        "out_jsonl": str(args.out_jsonl),
        "include_hc3": bool(args.include_hc3),
        "hc3_dir": str(args.hc3_dir),
        "remote_hc3_configs": _parse_csv(str(args.remote_hc3_configs)),
        "remote_hc3_max_groups_per_config": int(args.remote_hc3_max_groups_per_config),
        "remote_cache_dir": None if args.remote_cache_dir is None else str(args.remote_cache_dir),
        "remote_hllmc2_sources": _parse_csv(str(args.remote_hllmc2_sources)),
        "remote_hllmc2_max_groups_per_source": int(args.remote_hllmc2_max_groups_per_source),
        "remote_hllmc2_cache_dir": None if args.remote_hllmc2_cache_dir is None else str(args.remote_hllmc2_cache_dir),
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "max_local_variants_per_group_source": int(args.max_local_variants_per_group_source),
        "max_groups_per_item_type": caps,
        **summarize_cue_corpus(records),
    }
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote {len(records)} records to {args.out_jsonl}")
    print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
