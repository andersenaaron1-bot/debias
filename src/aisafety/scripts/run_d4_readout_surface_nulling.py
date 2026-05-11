"""Fit and evaluate readout-space nulling directions for D4 surface cues."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import DEFAULT_SURFACE_AXES
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.mech.readout_nulling import encode_pooled_texts, orthonormal_basis, score_pooled
from aisafety.reward.io_jsonl import iter_jsonl
from aisafety.reward.text_format import format_prompt_response
from aisafety.scripts.build_d4_human_llm_alignment_pairs import _csv_set
from aisafety.scripts.run_d4_candidate_feature_pair_alignment import (
    _cap_pairs,
    _load_scorer_and_tokenizer,
    _read_pair_file,
    _scorer_device,
    _write_csv,
)


DEFAULT_COUNTERFACTUAL_JSONL = (
    Path("data") / "derived" / "d4_surface_counterfactual_pairs_v1" / "counterfactuals.jsonl"
)
DEFAULT_PAIR_JSONL = Path("data") / "derived" / "d4_human_llm_alignment_pairs_strat10k_v3" / "pairs.jsonl"
DEFAULT_PREF_VAL_JSONL = Path("data") / "derived" / "pref_pairs_shp2" / "pref_pairs_val.jsonl"
DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_j0_readout_surface_nulling_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--counterfactual-jsonl", type=Path, default=DEFAULT_COUNTERFACTUAL_JSONL)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--pref-val-jsonl", type=Path, default=DEFAULT_PREF_VAL_JSONL)
    parser.add_argument("--axes", type=str, default=",".join(DEFAULT_SURFACE_AXES))
    parser.add_argument("--reward-run-dir", type=Path, default=Path("artifacts") / "reward" / "j0_anchor_v1_h100compact")
    parser.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--fit-frac", type=float, default=0.5)
    parser.add_argument("--min-direction-rows", type=int, default=20)
    parser.add_argument("--max-counterfactuals", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=3000)
    parser.add_argument("--max-pref-pairs", type=int, default=1000)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--skip-pref-retention", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _cap_rows(rows: list[dict[str, Any]], *, max_rows: int, seed: int, salt: str) -> list[dict[str, Any]]:
    if int(max_rows) <= 0 or len(rows) <= int(max_rows):
        return list(rows)
    ordered = sorted(rows, key=lambda row: sha1_hex(f"{seed}:{salt}:{row.get('counterfactual_id')}"))
    chosen = {str(row.get("counterfactual_id")) for row in ordered[: int(max_rows)]}
    return [row for row in rows if str(row.get("counterfactual_id")) in chosen]


def _split_counterfactuals(df: pd.DataFrame, *, fit_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frac = min(max(float(fit_frac), 0.05), 0.95)
    order = sorted(
        range(len(df)),
        key=lambda idx: sha1_hex(f"{seed}:readout-fit:{df.iloc[idx]['counterfactual_id']}"),
    )
    n_fit = max(1, int(round(len(order) * frac)))
    fit_idx = set(order[:n_fit])
    split = np.asarray(["fit" if idx in fit_idx else "eval" for idx in range(len(df))], dtype=object)
    out = df.copy()
    out["nulling_split"] = split
    return out[out["nulling_split"] == "fit"].reset_index(drop=True), out[out["nulling_split"] == "eval"].reset_index(drop=True)


def _encode_unique_texts(
    *,
    scorer: Any,
    tokenizer: Any,
    texts: list[str],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    text_by_id: dict[str, str] = {}
    for text in texts:
        text_by_id.setdefault(sha1_hex(str(text)), str(text))
    text_ids = sorted(text_by_id)
    pooled = encode_pooled_texts(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=[text_by_id[text_id] for text_id in text_ids],
        batch_size=int(args.encode_batch_size),
        max_length=int(args.max_length),
        device=_scorer_device(scorer),
    )
    return {text_id: pooled[idx] for idx, text_id in enumerate(text_ids)}


def _row_pooled(
    pooled_by_id: dict[str, torch.Tensor],
    df: pd.DataFrame,
    column: str,
) -> torch.Tensor:
    rows = [pooled_by_id[sha1_hex(str(text))] for text in df[column].astype(str)]
    if not rows:
        return torch.zeros((0, 0), dtype=torch.float32)
    return torch.stack(rows, dim=0)


def _fit_axis_directions(
    fit_df: pd.DataFrame,
    pooled_by_id: dict[str, torch.Tensor],
    *,
    axes: set[str],
    min_rows: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    directions: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for axis in sorted(axes):
        group = fit_df[fit_df["axis"].astype(str) == axis]
        deltas: list[torch.Tensor] = []
        n_increase = 0
        n_decrease = 0
        for row in group.itertuples(index=False):
            base = pooled_by_id[sha1_hex(str(getattr(row, "base_text")))]
            variant = pooled_by_id[sha1_hex(str(getattr(row, "variant_text")))]
            direction = str(getattr(row, "direction"))
            if direction == "increase":
                deltas.append(variant - base)
                n_increase += 1
            elif direction == "decrease":
                deltas.append(base - variant)
                n_decrease += 1
        if len(deltas) < int(min_rows):
            rows.append(
                {
                    "axis": axis,
                    "status": "skipped",
                    "reason": "too_few_rows",
                    "n_rows": int(len(deltas)),
                    "n_increase": int(n_increase),
                    "n_decrease": int(n_decrease),
                    "direction_norm": None,
                }
            )
            continue
        mat = torch.stack(deltas, dim=0).float()
        direction_vec = mat.mean(dim=0)
        norm = float(torch.linalg.vector_norm(direction_vec).item())
        if norm <= 1e-8:
            rows.append(
                {
                    "axis": axis,
                    "status": "skipped",
                    "reason": "zero_norm",
                    "n_rows": int(len(deltas)),
                    "n_increase": int(n_increase),
                    "n_decrease": int(n_decrease),
                    "direction_norm": norm,
                }
            )
            continue
        directions[axis] = direction_vec
        rows.append(
            {
                "axis": axis,
                "status": "ok",
                "reason": "",
                "n_rows": int(len(deltas)),
                "n_increase": int(n_increase),
                "n_decrease": int(n_decrease),
                "direction_norm": norm,
            }
        )
    return directions, rows


def _score_counterfactual_eval(
    eval_df: pd.DataFrame,
    pooled_by_id: dict[str, torch.Tensor],
    *,
    scorer: Any,
    basis: torch.Tensor,
    args: argparse.Namespace,
) -> pd.DataFrame:
    base_pooled = _row_pooled(pooled_by_id, eval_df, "base_text")
    variant_pooled = _row_pooled(pooled_by_id, eval_df, "variant_text")
    base_original, base_null = score_pooled(
        scorer=scorer,
        pooled=base_pooled,
        basis=basis,
        batch_size=int(args.score_batch_size),
        device=_scorer_device(scorer),
    )
    variant_original, variant_null = score_pooled(
        scorer=scorer,
        pooled=variant_pooled,
        basis=basis,
        batch_size=int(args.score_batch_size),
        device=_scorer_device(scorer),
    )
    out = eval_df.copy()
    out["base_reward_original"] = base_original.numpy().astype(float)
    out["variant_reward_original"] = variant_original.numpy().astype(float)
    out["reward_delta_original"] = out["variant_reward_original"] - out["base_reward_original"]
    out["base_reward_null"] = base_null.numpy().astype(float)
    out["variant_reward_null"] = variant_null.numpy().astype(float)
    out["reward_delta_null"] = out["variant_reward_null"] - out["base_reward_null"]
    out["reward_delta_change_null_minus_original"] = out["reward_delta_null"] - out["reward_delta_original"]
    return out


def _score_pair_nulling(
    pair_df: pd.DataFrame,
    *,
    scorer: Any,
    tokenizer: Any,
    basis: torch.Tensor,
    args: argparse.Namespace,
) -> pd.DataFrame:
    pooled_by_id = _encode_unique_texts(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=pair_df["human_text"].astype(str).tolist() + pair_df["llm_text"].astype(str).tolist(),
        args=args,
    )
    human_pooled = _row_pooled(pooled_by_id, pair_df, "human_text")
    llm_pooled = _row_pooled(pooled_by_id, pair_df, "llm_text")
    human_original, human_null = score_pooled(
        scorer=scorer,
        pooled=human_pooled,
        basis=basis,
        batch_size=int(args.score_batch_size),
        device=_scorer_device(scorer),
    )
    llm_original, llm_null = score_pooled(
        scorer=scorer,
        pooled=llm_pooled,
        basis=basis,
        batch_size=int(args.score_batch_size),
        device=_scorer_device(scorer),
    )
    out = pair_df.copy()
    out["human_reward_original"] = human_original.numpy().astype(float)
    out["llm_reward_original"] = llm_original.numpy().astype(float)
    out["llm_margin_original"] = out["llm_reward_original"] - out["human_reward_original"]
    out["y_llm_chosen_original"] = (out["llm_margin_original"] > 0.0).astype(int)
    out["human_reward_null"] = human_null.numpy().astype(float)
    out["llm_reward_null"] = llm_null.numpy().astype(float)
    out["llm_margin_null"] = out["llm_reward_null"] - out["human_reward_null"]
    out["y_llm_chosen_null"] = (out["llm_margin_null"] > 0.0).astype(int)
    out["llm_margin_change_null_minus_original"] = out["llm_margin_null"] - out["llm_margin_original"]
    return out


def _read_pref_rows(path: Path, *, max_pairs: int) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(iter_jsonl(path)):
        if int(max_pairs) > 0 and idx >= int(max_pairs):
            break
        prompt = str(row.get("prompt") or "")
        chosen = str(row.get("chosen") or "")
        rejected = str(row.get("rejected") or "")
        rows.append(
            {
                "pref_pair_id": sha1_hex(f"{idx}:{prompt[:200]}:{chosen[:120]}:{rejected[:120]}"),
                "prompt": prompt,
                "chosen_text": format_prompt_response(prompt, chosen),
                "rejected_text": format_prompt_response(prompt, rejected),
            }
        )
    return pd.DataFrame(rows)


def _score_pref_nulling(
    pref_df: pd.DataFrame,
    *,
    scorer: Any,
    tokenizer: Any,
    basis: torch.Tensor,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if pref_df.empty:
        return pref_df
    pooled_by_id = _encode_unique_texts(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=pref_df["chosen_text"].astype(str).tolist() + pref_df["rejected_text"].astype(str).tolist(),
        args=args,
    )
    chosen_pooled = _row_pooled(pooled_by_id, pref_df, "chosen_text")
    rejected_pooled = _row_pooled(pooled_by_id, pref_df, "rejected_text")
    chosen_original, chosen_null = score_pooled(
        scorer=scorer,
        pooled=chosen_pooled,
        basis=basis,
        batch_size=int(args.score_batch_size),
        device=_scorer_device(scorer),
    )
    rejected_original, rejected_null = score_pooled(
        scorer=scorer,
        pooled=rejected_pooled,
        basis=basis,
        batch_size=int(args.score_batch_size),
        device=_scorer_device(scorer),
    )
    out = pref_df.copy()
    out["chosen_reward_original"] = chosen_original.numpy().astype(float)
    out["rejected_reward_original"] = rejected_original.numpy().astype(float)
    out["pref_margin_original"] = out["chosen_reward_original"] - out["rejected_reward_original"]
    out["chosen_reward_null"] = chosen_null.numpy().astype(float)
    out["rejected_reward_null"] = rejected_null.numpy().astype(float)
    out["pref_margin_null"] = out["chosen_reward_null"] - out["rejected_reward_null"]
    out["pref_margin_change_null_minus_original"] = out["pref_margin_null"] - out["pref_margin_original"]
    out["chosen_correct_original"] = (out["pref_margin_original"] > 0.0).astype(int)
    out["chosen_correct_null"] = (out["pref_margin_null"] > 0.0).astype(int)
    return out


def _mean(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean())


def _summary_rows(
    *,
    cf_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    pref_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "section": "counterfactual",
            "group": "all",
            "n": int(len(cf_df)),
            "original_mean_reward_delta": _mean(cf_df["reward_delta_original"]),
            "null_mean_reward_delta": _mean(cf_df["reward_delta_null"]),
            "mean_delta_change_null_minus_original": _mean(cf_df["reward_delta_change_null_minus_original"]),
            "mean_abs_delta_original": _mean(cf_df["reward_delta_original"].abs()),
            "mean_abs_delta_null": _mean(cf_df["reward_delta_null"].abs()),
        }
    )
    for (axis, direction), group in cf_df.groupby(["axis", "direction"], sort=True):
        rows.append(
            {
                "section": "counterfactual_axis_direction",
                "group": f"{axis}::{direction}",
                "n": int(len(group)),
                "original_mean_reward_delta": _mean(group["reward_delta_original"]),
                "null_mean_reward_delta": _mean(group["reward_delta_null"]),
                "mean_delta_change_null_minus_original": _mean(group["reward_delta_change_null_minus_original"]),
                "mean_abs_delta_original": _mean(group["reward_delta_original"].abs()),
                "mean_abs_delta_null": _mean(group["reward_delta_null"].abs()),
            }
        )
    rows.append(
        {
            "section": "human_llm_pairs",
            "group": "all",
            "n": int(len(pair_df)),
            "original_mean_llm_margin": _mean(pair_df["llm_margin_original"]),
            "null_mean_llm_margin": _mean(pair_df["llm_margin_null"]),
            "mean_margin_change_null_minus_original": _mean(pair_df["llm_margin_change_null_minus_original"]),
            "original_llm_chosen_rate": _mean(pair_df["y_llm_chosen_original"]),
            "null_llm_chosen_rate": _mean(pair_df["y_llm_chosen_null"]),
        }
    )
    for source, group in pair_df.groupby("source_dataset", sort=True):
        rows.append(
            {
                "section": "human_llm_pairs_source",
                "group": str(source),
                "n": int(len(group)),
                "original_mean_llm_margin": _mean(group["llm_margin_original"]),
                "null_mean_llm_margin": _mean(group["llm_margin_null"]),
                "mean_margin_change_null_minus_original": _mean(group["llm_margin_change_null_minus_original"]),
                "original_llm_chosen_rate": _mean(group["y_llm_chosen_original"]),
                "null_llm_chosen_rate": _mean(group["y_llm_chosen_null"]),
            }
        )
    if not pref_df.empty:
        rows.append(
            {
                "section": "preference_retention",
                "group": "all",
                "n": int(len(pref_df)),
                "original_pairwise_acc": _mean(pref_df["chosen_correct_original"]),
                "null_pairwise_acc": _mean(pref_df["chosen_correct_null"]),
                "original_mean_pref_margin": _mean(pref_df["pref_margin_original"]),
                "null_mean_pref_margin": _mean(pref_df["pref_margin_null"]),
                "mean_pref_margin_change_null_minus_original": _mean(pref_df["pref_margin_change_null_minus_original"]),
            }
        )
    return rows


def _write_outputs(
    *,
    out_dir: Path,
    direction_rows: list[dict[str, Any]],
    basis: torch.Tensor,
    directions: dict[str, torch.Tensor],
    cf_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    pref_df: pd.DataFrame,
    summary_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "basis": basis.cpu(),
            "directions": {axis: vec.cpu() for axis, vec in directions.items()},
            "direction_rows": direction_rows,
        },
        out_dir / "surface_directions.pt",
    )
    _write_csv(out_dir / "surface_direction_summary.csv", direction_rows)
    cf_df.to_csv(out_dir / "counterfactual_nulling_scores.csv", index=False)
    pair_df.to_csv(out_dir / "pair_nulling_scores.csv", index=False)
    if not pref_df.empty:
        pref_df.to_csv(out_dir / "preference_retention_scores.csv", index=False)
    _write_csv(out_dir / "nulling_summary.csv", summary_rows)
    write_json(out_dir / "manifest.json", manifest)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    counterfactual_path = _resolve(workspace_root, args.counterfactual_jsonl)
    pair_path = _resolve(workspace_root, args.pair_jsonl)
    pref_path = _resolve(workspace_root, args.pref_val_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    axes = _csv_set(str(args.axes))
    if not axes:
        raise ValueError("--axes produced an empty axis set.")

    cf_rows = _cap_rows(
        read_jsonl(counterfactual_path),
        max_rows=int(args.max_counterfactuals),
        seed=int(args.seed),
        salt="readout-counterfactual-cap",
    )
    if not cf_rows:
        raise ValueError(f"No counterfactual rows found in {counterfactual_path}")
    cf_df_all = pd.DataFrame([row for row in cf_rows if str(row.get("axis")) in axes])
    if cf_df_all.empty:
        raise ValueError(f"No counterfactual rows matched axes: {sorted(axes)}")
    fit_df, eval_df = _split_counterfactuals(cf_df_all, fit_frac=float(args.fit_frac), seed=int(args.seed))
    if eval_df.empty:
        raise ValueError("Counterfactual eval split is empty.")

    scorer, tokenizer = _load_scorer_and_tokenizer(args, workspace_root)
    fit_pooled = _encode_unique_texts(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=fit_df["base_text"].astype(str).tolist() + fit_df["variant_text"].astype(str).tolist(),
        args=args,
    )
    directions, direction_rows = _fit_axis_directions(
        fit_df,
        fit_pooled,
        axes=axes,
        min_rows=int(args.min_direction_rows),
    )
    if not directions:
        raise ValueError("No readout directions were fit.")
    direction_mat = torch.stack([directions[axis] for axis in sorted(directions)], dim=0)
    basis = orthonormal_basis(direction_mat)
    if basis.shape[1] <= 0:
        raise ValueError("Readout nulling basis is empty after orthogonalization.")

    eval_pooled = _encode_unique_texts(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=eval_df["base_text"].astype(str).tolist() + eval_df["variant_text"].astype(str).tolist(),
        args=args,
    )
    cf_scored = _score_counterfactual_eval(
        eval_df,
        eval_pooled,
        scorer=scorer,
        basis=basis,
        args=args,
    )

    pair_df = _cap_pairs(_read_pair_file(pair_path), max_pairs=int(args.max_pairs), seed=int(args.seed))
    pair_scored = _score_pair_nulling(
        pair_df,
        scorer=scorer,
        tokenizer=tokenizer,
        basis=basis,
        args=args,
    )

    pref_scored = pd.DataFrame()
    if not bool(args.skip_pref_retention):
        if not pref_path.is_file():
            raise FileNotFoundError(f"Preference validation file not found: {pref_path}")
        pref_df = _read_pref_rows(pref_path, max_pairs=int(args.max_pref_pairs))
        pref_scored = _score_pref_nulling(
            pref_df,
            scorer=scorer,
            tokenizer=tokenizer,
            basis=basis,
            args=args,
        )

    summary_rows = _summary_rows(cf_df=cf_scored, pair_df=pair_scored, pref_df=pref_scored)
    manifest = {
        "stage": "D4-readout-surface-nulling",
        "counterfactual_jsonl": str(counterfactual_path),
        "pair_jsonl": str(pair_path),
        "pref_val_jsonl": str(pref_path),
        "reward_run_dir": str(_resolve(workspace_root, args.reward_run_dir)),
        "model_id": str(args.model_id),
        "axes_requested": sorted(axes),
        "axes_fit": sorted(directions),
        "fit_frac": float(args.fit_frac),
        "n_counterfactuals_total": int(len(cf_df_all)),
        "n_counterfactuals_fit": int(len(fit_df)),
        "n_counterfactuals_eval": int(len(cf_scored)),
        "n_pairs": int(len(pair_scored)),
        "n_pref_pairs": int(len(pref_scored)),
        "basis_rank": int(basis.shape[1]),
        "counterfactual_counts_by_axis_direction": {
            f"{axis}::{direction}": int(count)
            for (axis, direction), count in Counter(
                zip(cf_scored["axis"].astype(str), cf_scored["direction"].astype(str))
            ).items()
        },
        "outputs": {
            "surface_directions_pt": str(out_dir / "surface_directions.pt"),
            "surface_direction_summary_csv": str(out_dir / "surface_direction_summary.csv"),
            "counterfactual_nulling_scores_csv": str(out_dir / "counterfactual_nulling_scores.csv"),
            "pair_nulling_scores_csv": str(out_dir / "pair_nulling_scores.csv"),
            "preference_retention_scores_csv": str(out_dir / "preference_retention_scores.csv"),
            "nulling_summary_csv": str(out_dir / "nulling_summary.csv"),
            "manifest_json": str(out_dir / "manifest.json"),
        },
    }
    _write_outputs(
        out_dir=out_dir,
        direction_rows=direction_rows,
        basis=basis,
        directions=directions,
        cf_df=cf_scored,
        pair_df=pair_scored,
        pref_df=pref_scored,
        summary_rows=summary_rows,
        manifest=manifest,
    )
    print(f"out_dir={out_dir}")
    print(f"summary={out_dir / 'nulling_summary.csv'}")
    print(f"manifest={out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
