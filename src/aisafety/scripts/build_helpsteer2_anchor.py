"""Build a pointwise multi-attribute anchor dataset from HelpSteer2.

Outputs JSONL rows with:
  - prompt
  - response
  - utility_target
  - attribute_targets

Targets are normalized to [0, 1] by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

from aisafety.config import DATA_DIR
from aisafety.reward.io_jsonl import write_jsonl


ATTRIBUTE_NAMES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-id", type=str, default="nvidia/HelpSteer2")
    p.add_argument("--out-dir", type=Path, default=DATA_DIR / "derived" / "helpsteer2_anchor")
    p.add_argument("--max-train", type=int, default=0, help="Optional cap; 0 keeps all rows.")
    p.add_argument("--max-val", type=int, default=0, help="Optional cap; 0 keeps all rows.")
    p.add_argument("--normalize-targets", action="store_true")
    p.add_argument("--no-normalize-targets", dest="normalize_targets", action="store_false")
    p.set_defaults(normalize_targets=True)
    p.add_argument(
        "--attribute-weights",
        type=str,
        default="helpfulness=0.30,correctness=0.30,coherence=0.20,complexity=0.10,verbosity=0.10",
        help="Comma-separated weights used to build the scalar utility target.",
    )
    return p.parse_args()


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _parse_weights(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid weight spec {part!r}; expected name=value")
        name, value = part.split("=", 1)
        out[str(name).strip()] = float(value)
    missing = [name for name in ATTRIBUTE_NAMES if name not in out]
    if missing:
        raise ValueError(f"Missing weights for attributes: {missing}")
    total = sum(float(out[name]) for name in ATTRIBUTE_NAMES)
    if total <= 0:
        raise ValueError("Attribute weights must sum to a positive value.")
    return {name: float(out[name]) / float(total) for name in ATTRIBUTE_NAMES}


def _normalize_score(score: float, *, normalize_targets: bool) -> float:
    if not normalize_targets:
        return float(score)
    return float(score) / 4.0


def _convert_row(
    row: dict,
    *,
    dataset_id: str,
    split: str,
    normalize_targets: bool,
    weights: dict[str, float],
) -> dict | None:
    prompt = str(row.get("prompt") or "").strip()
    response = str(row.get("response") or "").strip()
    if not prompt or not response:
        return None

    attr_targets: dict[str, float] = {}
    raw_targets: dict[str, float] = {}
    for name in ATTRIBUTE_NAMES:
        value = row.get(name)
        if value is None:
            return None
        val_f = float(value)
        raw_targets[name] = val_f
        attr_targets[name] = _normalize_score(val_f, normalize_targets=normalize_targets)

    utility_raw = sum(float(weights[name]) * float(raw_targets[name]) for name in ATTRIBUTE_NAMES)
    utility_target = _normalize_score(utility_raw, normalize_targets=normalize_targets)
    example_id = _sha1_hex(
        json.dumps(
            {
                "dataset_id": str(dataset_id),
                "split": str(split),
                "prompt_sha1": _sha1_hex(prompt),
                "response_sha1": _sha1_hex(response),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return {
        "example_id": example_id,
        "source_dataset": str(dataset_id),
        "prompt": prompt,
        "response": response,
        "utility_target": float(utility_target),
        "attribute_targets": {name: float(attr_targets[name]) for name in ATTRIBUTE_NAMES},
        "meta": {"split": str(split)},
    }


def _convert_split(
    *,
    dataset_id: str,
    split: str,
    max_rows: int,
    normalize_targets: bool,
    weights: dict[str, float],
) -> list[dict]:
    ds = load_dataset(str(dataset_id), split=str(split))
    rows: list[dict] = []
    for row in ds:
        out = _convert_row(
            row,
            dataset_id=dataset_id,
            split=split,
            normalize_targets=normalize_targets,
            weights=weights,
        )
        if out is None:
            continue
        rows.append(out)
        if int(max_rows) > 0 and len(rows) >= int(max_rows):
            break
    return rows


def main() -> None:
    args = parse_args()
    weights = _parse_weights(str(args.attribute_weights))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _convert_split(
        dataset_id=str(args.dataset_id),
        split="train",
        max_rows=int(args.max_train),
        normalize_targets=bool(args.normalize_targets),
        weights=weights,
    )
    val_rows = _convert_split(
        dataset_id=str(args.dataset_id),
        split="validation",
        max_rows=int(args.max_val),
        normalize_targets=bool(args.normalize_targets),
        weights=weights,
    )

    write_jsonl(args.out_dir / "anchor_train.jsonl", train_rows)
    write_jsonl(args.out_dir / "anchor_val.jsonl", val_rows)
    stats = {
        "dataset_id": str(args.dataset_id),
        "normalize_targets": bool(args.normalize_targets),
        "attribute_names": list(ATTRIBUTE_NAMES),
        "attribute_weights": weights,
        "n_train": int(len(train_rows)),
        "n_val": int(len(val_rows)),
    }
    with (args.out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote {len(train_rows)} train rows to {args.out_dir / 'anchor_train.jsonl'}")
    print(f"Wrote {len(val_rows)} val rows to {args.out_dir / 'anchor_val.jsonl'}")
    print(f"Wrote stats to {args.out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
