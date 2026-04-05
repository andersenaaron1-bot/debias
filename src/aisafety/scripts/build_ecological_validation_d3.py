"""Build D3 ecological atom/bundle validation tables on Laurito-style A/B trials.

This script scores a reward model on local Laurito trials, collapses balanced A/B
swaps to one de-biased pair decision, and measures which atom and bundle deltas
track LLM-vs-human judge preference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aisafety.config import DATA_DIR, DEFAULT_CACHE_DIR, DEFAULT_SEED
from aisafety.data import DOMAINS, build_all_trials
from aisafety.ontology.ecology import (
    attach_atom_and_bundle_deltas,
    build_ecological_effect_tables,
    build_pair_text_atom_scores,
    collapse_scored_trials_to_pairs,
    ensure_choice_columns,
    load_bundle_members,
)
from aisafety.reward.model import load_reward_scorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials-csv", type=Path, default=None)
    p.add_argument("--scored-trials-csv", type=Path, default=None)
    p.add_argument("--out-scored-trials-csv", type=Path, default=None)
    p.add_argument(
        "--bundle-validation-json",
        type=Path,
        default=DATA_DIR / "derived" / "style_groups" / "d2_validation_bundle_creation_v1" / "bundle_validation.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "derived" / "style_groups" / "d3_ecological_validation_v1",
    )
    p.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--lora-adapter-dir", type=Path, default=None)
    p.add_argument("--value-head", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--min-bundle-atoms", type=int, default=2)
    return p.parse_args()


@torch.no_grad()
def _score_texts(model, tok, texts: list[str], *, max_length: int, batch_size: int, device) -> np.ndarray:
    scores: list[float] = []
    model.eval()
    for i in range(0, len(texts), int(batch_size)):
        batch = texts[i : i + int(batch_size)]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        s = model(enc["input_ids"], enc["attention_mask"]).detach().float().cpu().numpy()
        scores.extend(s.tolist())
    return np.asarray(scores, dtype=np.float32)


def _prepare_trials_csv(path: Path, *, seed: int) -> Path:
    df = build_all_trials(DOMAINS, seed=int(seed), balance_order=True)
    if df.empty:
        raise RuntimeError("Could not build Laurito trials from local domain data.")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _load_or_score_trials(args: argparse.Namespace) -> pd.DataFrame:
    if args.scored_trials_csv is not None:
        return pd.read_csv(args.scored_trials_csv)

    trials_csv = (
        Path(args.trials_csv)
        if args.trials_csv is not None
        else _prepare_trials_csv(args.out_dir / "inputs" / "laurito_trials.csv", seed=int(args.seed))
    )
    df = pd.read_csv(trials_csv)
    required = {"A_text", "B_text", "A_source", "B_source", "item_type", "title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required trial columns: {sorted(missing)}")

    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    scorer, tok = load_reward_scorer(
        model_id=str(args.model_id),
        cache_dir=Path(args.cache_dir),
        lora_adapter_dir=args.lora_adapter_dir,
        value_head_path=args.value_head,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    device = next(p for p in scorer.parameters() if p.device.type != "meta").device
    scores = _score_texts(
        scorer,
        tok,
        df["A_text"].astype(str).tolist() + df["B_text"].astype(str).tolist(),
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        device=device,
    )
    n = len(df)
    out = df.copy()
    out["score_A"] = scores[:n]
    out["score_B"] = scores[n:]
    out = ensure_choice_columns(out, seed=int(args.seed))
    out_scored_trials_csv = (
        Path(args.out_scored_trials_csv)
        if args.out_scored_trials_csv is not None
        else args.out_dir / "inputs" / "laurito_scored_trials.csv"
    )
    out_scored_trials_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_scored_trials_csv, index=False)
    return out


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _effect_rows(table: dict[str, dict], *, include_d2: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, payload in table.items():
        row: dict[str, object] = {
            "name": name,
            "status": payload.get("status"),
            "n_pairs": payload.get("n_pairs"),
            "mean_llm_minus_human_delta": payload.get("mean_llm_minus_human_delta"),
            "signed_effect_z": payload.get("signed_effect_z"),
            "signed_effect_ci_95_low": payload.get("signed_effect_ci_95_low"),
            "signed_effect_ci_95_high": payload.get("signed_effect_ci_95_high"),
            "auc_llm_choice": payload.get("auc_llm_choice"),
            "spearman_with_llm_margin": payload.get("spearman_with_llm_margin"),
        }
        for item_type, item_payload in sorted((payload.get("by_item_type") or {}).items()):
            row[f"{item_type}__signed_effect_z"] = item_payload.get("signed_effect_z")
            row[f"{item_type}__auc_llm_choice"] = item_payload.get("auc_llm_choice")
        if include_d2:
            row["d2_status"] = payload.get("d2_status")
            row["d2_n_atoms_validation"] = payload.get("d2_n_atoms_validation")
            row["d2_member_atoms_validation"] = ";".join(payload.get("d2_member_atoms_validation") or [])
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    resolved_out_scored_trials_csv = (
        None
        if args.scored_trials_csv is not None
        else (
            Path(args.out_scored_trials_csv)
            if args.out_scored_trials_csv is not None
            else args.out_dir / "inputs" / "laurito_scored_trials.csv"
        )
    )

    bundle_members = load_bundle_members(args.bundle_validation_json, min_bundle_atoms=int(args.min_bundle_atoms))
    with Path(args.bundle_validation_json).open("r", encoding="utf-8") as f:
        bundle_validation_payload = json.load(f)

    scored_trials = _load_or_score_trials(args)
    pair_df = collapse_scored_trials_to_pairs(scored_trials, seed=int(args.seed))
    text_atom_scores = build_pair_text_atom_scores(pair_df)
    pair_with_deltas = attach_atom_and_bundle_deltas(
        pair_df,
        text_atom_scores=text_atom_scores,
        bundle_members=bundle_members,
    )

    atom_effects, bundle_effects = build_ecological_effect_tables(
        pair_with_deltas,
        n_bootstrap=int(args.bootstrap),
        seed=int(args.seed),
        bundle_metadata=bundle_validation_payload,
    )

    summary = {
        "n_trial_rows": int(len(scored_trials)),
        "n_pair_rows": int(len(pair_df)),
        "n_unique_texts": int(len(text_atom_scores)),
        "n_atom_effects": int(len(atom_effects)),
        "n_bundle_effects": int(len(bundle_effects)),
        "bundle_validation_json": str(args.bundle_validation_json),
        "scored_trials_csv": None if args.scored_trials_csv is None else str(args.scored_trials_csv),
        "out_scored_trials_csv": None if resolved_out_scored_trials_csv is None else str(resolved_out_scored_trials_csv),
        "trials_csv": None if args.trials_csv is None else str(args.trials_csv),
        "model_id": None if args.scored_trials_csv is not None else str(args.model_id),
        "lora_adapter_dir": None if args.lora_adapter_dir is None else str(args.lora_adapter_dir),
        "value_head": None if args.value_head is None else str(args.value_head),
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "by_item_type": {
            str(item_type): {
                "n_pairs": int(len(grp)),
                "llm_choice_rate": float(grp["y_llm_chosen"].mean()),
            }
            for item_type, grp in pair_df.groupby("item_type")
        },
    }

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    with (args.out_dir / "atom_effects.json").open("w", encoding="utf-8") as f:
        json.dump(atom_effects, f, ensure_ascii=False, indent=2, sort_keys=True)
    with (args.out_dir / "bundle_effects.json").open("w", encoding="utf-8") as f:
        json.dump(bundle_effects, f, ensure_ascii=False, indent=2, sort_keys=True)

    _write_tsv(args.out_dir / "atom_effects.tsv", _effect_rows(atom_effects, include_d2=False))
    _write_tsv(args.out_dir / "bundle_effects.tsv", _effect_rows(bundle_effects, include_d2=True))
    pair_with_deltas.to_csv(args.out_dir / "pair_level_inputs.csv", index=False)
    text_atom_scores.to_csv(args.out_dir / "text_atom_scores.csv", index=False)

    print(f"Wrote D3 ecological validation artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
