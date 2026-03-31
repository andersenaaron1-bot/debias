"""Build pairwise preference JSONL from SHP-2 (Stanford Human Preferences).

Outputs a unified preference-pair schema:
  - prompt (history)
  - chosen / rejected (A/B comments according to label)

Example:
  python -m aisafety.scripts.build_pref_pairs_shp2 ^
    --out-dir data\\derived\\pref_pairs_shp2 ^
    --max-train 300000 ^
    --max-val 20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from aisafety.config import DATA_DIR, DEFAULT_SEED
from aisafety.reward.io_jsonl import write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DATA_DIR / "derived" / "pref_pairs_shp2")
    p.add_argument("--dataset-id", type=str, default="stanfordnlp/SHP-2")
    p.add_argument("--split", type=str, default="train", help="HF split to sample from (usually 'train').")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--streaming", action="store_true", help="Use HF streaming mode.")
    p.add_argument("--no-streaming", dest="streaming", action="store_false")
    p.set_defaults(streaming=True)
    p.add_argument("--shuffle", action="store_true", help="Shuffle stream before sampling.")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    p.set_defaults(shuffle=True)
    p.add_argument("--shuffle-buffer-size", type=int, default=50_000)
    p.add_argument("--max-train", type=int, default=300_000)
    p.add_argument("--max-val", type=int, default=20_000)
    p.add_argument("--val-frac", type=float, default=0.05, help="Hash-based split fraction for validation.")
    p.add_argument("--max-history-chars", type=int, default=2000)
    p.add_argument("--max-response-chars", type=int, default=2000)
    return p.parse_args()


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _assign_split(pair_id: str, *, seed: int, val_frac: float) -> str:
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be in [0,1)")
    h = _sha1_hex(f"{seed}:{pair_id}")
    r = int(h[:8], 16) / float(2**32)
    return "val" if r < float(val_frac) else "train"


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _iter_rows(ds) -> Iterable[dict]:
    for row in ds:
        if not isinstance(row, dict):
            continue
        yield row


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        str(args.dataset_id),
        split=str(args.split),
        streaming=bool(args.streaming),
    )
    if bool(args.shuffle):
        if bool(args.streaming):
            ds = ds.shuffle(seed=int(args.seed), buffer_size=int(args.shuffle_buffer_size))
        else:
            ds = ds.shuffle(seed=int(args.seed))

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    need_train = int(args.max_train) if args.max_train else 0
    need_val = int(args.max_val) if args.max_val else 0
    if need_train <= 0 and need_val <= 0:
        raise ValueError("Set --max-train and/or --max-val to a positive value.")

    seen = 0
    skipped = 0
    for row in _iter_rows(ds):
        seen += 1
        history = row.get("history")
        a = row.get("human_ref_A")
        b = row.get("human_ref_B")
        label = row.get("labels")
        if not isinstance(history, str) or not isinstance(a, str) or not isinstance(b, str):
            skipped += 1
            continue
        if label not in {0, 1}:
            skipped += 1
            continue

        chosen = a if int(label) == 1 else b
        rejected = b if int(label) == 1 else a
        prompt = _truncate(history.strip(), int(args.max_history_chars))
        chosen = _truncate(chosen.strip(), int(args.max_response_chars))
        rejected = _truncate(rejected.strip(), int(args.max_response_chars))
        if not prompt or not chosen or not rejected:
            skipped += 1
            continue

        domain = row.get("domain")
        post_id = row.get("post_id") or row.get("id") or ""
        c_a = row.get("c_root_id_A") or row.get("comment_id_A") or ""
        c_b = row.get("c_root_id_B") or row.get("comment_id_B") or ""
        pair_id = _sha1_hex(
            json.dumps(
                {
                    "dataset": str(args.dataset_id),
                    "domain": str(domain or ""),
                    "post_id": str(post_id),
                    "c_root_id_A": str(c_a),
                    "c_root_id_B": str(c_b),
                    "history_sha1": _sha1_hex(prompt),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        split = _assign_split(pair_id, seed=int(args.seed), val_frac=float(args.val_frac))
        out = {
            "pair_id": pair_id,
            "source_dataset": str(args.dataset_id),
            "domain": "" if domain is None else str(domain),
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "meta": {
                "split": str(args.split),
                "post_id": row.get("post_id"),
                "score_A": row.get("score_A"),
                "score_B": row.get("score_B"),
            },
        }
        if split == "val":
            if need_val > 0 and len(val_rows) < need_val:
                val_rows.append(out)
        else:
            if need_train > 0 and len(train_rows) < need_train:
                train_rows.append(out)

        if (need_train <= 0 or len(train_rows) >= need_train) and (need_val <= 0 or len(val_rows) >= need_val):
            break

    write_jsonl(out_dir / "pref_pairs_train.jsonl", train_rows)
    write_jsonl(out_dir / "pref_pairs_val.jsonl", val_rows)
    stats = {
        "dataset_id": str(args.dataset_id),
        "split": str(args.split),
        "streaming": bool(args.streaming),
        "shuffle": bool(args.shuffle),
        "seen_rows": int(seen),
        "skipped_rows": int(skipped),
        "n_train": int(len(train_rows)),
        "n_val": int(len(val_rows)),
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote {len(train_rows)} train pairs to {out_dir / 'pref_pairs_train.jsonl'}")
    print(f"Wrote {len(val_rows)} val pairs to {out_dir / 'pref_pairs_val.jsonl'}")
    print(f"Wrote stats to {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
