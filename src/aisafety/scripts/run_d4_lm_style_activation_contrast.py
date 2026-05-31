"""Measure layerwise answer-representation shifts for D4 style counterfactuals."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.mech.labels import parse_int_list, select_hidden_layers
from aisafety.scripts.build_d4_human_llm_alignment_pairs import _csv_set
from aisafety.scripts.run_d4_bt_stage_contrast import _load_lm


DEFAULT_COUNTERFACTUAL_JSONL = (
    Path("data") / "derived" / "d4_assistant_style_atomic_counterfactual_pairs_v1" / "counterfactuals.jsonl"
)
DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_lm_style_activation_contrast_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--counterfactual-jsonl", type=Path, default=DEFAULT_COUNTERFACTUAL_JSONL)
    parser.add_argument("--bt-scores", type=Path, default=None)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--axes", default="")
    parser.add_argument("--max-counterfactuals", type=int, default=1500)
    parser.add_argument("--selected-layers", default="")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--tail-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _score_csv(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "bt_stage_scores.csv"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find bt_stage_scores.csv in {path}")


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return None
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _fold_id(value: str, *, folds: int, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}|style-activation|{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % max(int(folds), 1)


def _cap_rows(
    rows: list[dict[str, Any]],
    *,
    axes: set[str],
    max_counterfactuals: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = [row for row in rows if not axes or str(row.get("axis") or "") in axes]
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        counterfactual_id = str(row.get("counterfactual_id") or "")
        if counterfactual_id:
            by_id.setdefault(counterfactual_id, row)
    ordered = sorted(by_id.values(), key=lambda row: sha1_hex(f"{seed}:activation:{row['counterfactual_id']}"))
    if int(max_counterfactuals) > 0:
        ordered = ordered[: int(max_counterfactuals)]
    return ordered


def _cue_texts(row: dict[str, Any]) -> tuple[str, str]:
    base = flat_text(str(row.get("base_text") or ""))
    variant = flat_text(str(row.get("variant_text") or ""))
    direction = str(row.get("direction") or "")
    if direction == "increase":
        return variant, base
    if direction == "decrease":
        return base, variant
    raise ValueError(f"Unsupported direction: {direction!r}")


def _selected_layers(model: Any, *, raw: str, stride: int, tail_layers: int) -> list[int]:
    num_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if num_layers <= 0:
        raise ValueError("Model config does not expose num_hidden_layers.")
    layers = parse_int_list(raw) if str(raw).strip() else select_hidden_layers(
        num_layers,
        stride=int(stride),
        tail_layers=int(tail_layers),
    )
    layers = [int(layer) for layer in layers if 1 <= int(layer) <= num_layers]
    if not layers:
        raise ValueError("No selected layers remain after model-layer filtering.")
    return layers


def _extract_mean_pooled_hidden(
    *,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    selected_layers: list[int],
    batch_size: int,
    max_length: int,
) -> dict[int, np.ndarray]:
    import torch

    device = next(param for param in model.parameters() if param.device.type != "meta").device
    outputs: dict[int, list[np.ndarray]] = {int(layer): [] for layer in selected_layers}
    for start in range(0, len(texts), max(int(batch_size), 1)):
        batch = texts[start : start + max(int(batch_size), 1)]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        if result.hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        mask = attention_mask.unsqueeze(-1).to(dtype=torch.float32)
        denom = mask.sum(dim=1).clamp_min(1.0)
        for layer in selected_layers:
            hidden = result.hidden_states[int(layer)].float()
            pooled = (hidden * mask).sum(dim=1) / denom
            outputs[int(layer)].append(pooled.detach().cpu().numpy().astype(np.float32))
        del encoded, input_ids, attention_mask, result, mask, denom
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in outputs.items()}


def _load_binary_margin(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    df = pd.read_csv(_score_csv(path))
    if "counterfactual_id" not in df.columns or "cue_plus_margin" not in df.columns:
        raise ValueError(f"Binary score file has no counterfactual_id/cue_plus_margin columns: {path}")
    df["cue_plus_margin"] = pd.to_numeric(df["cue_plus_margin"], errors="coerce")
    return (
        df.dropna(subset=["cue_plus_margin"])
        .groupby("counterfactual_id", sort=True)["cue_plus_margin"]
        .mean()
        .astype(float)
        .to_dict()
    )


def cross_validated_direction_scores(
    deltas: np.ndarray,
    counterfactual_ids: list[str],
    *,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fold_ids = np.asarray([_fold_id(item, folds=folds, seed=seed) for item in counterfactual_ids], dtype=int)
    projections = np.full((len(counterfactual_ids),), np.nan, dtype=float)
    cosines = np.full((len(counterfactual_ids),), np.nan, dtype=float)
    for fold in sorted(set(fold_ids)):
        train = fold_ids != fold
        test = fold_ids == fold
        if int(train.sum()) < 2 or int(test.sum()) < 1:
            continue
        direction = np.mean(deltas[train], axis=0)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-12:
            continue
        direction = direction / direction_norm
        projections[test] = deltas[test] @ direction
        delta_norms = np.linalg.norm(deltas[test], axis=1)
        cosines[test] = np.divide(
            projections[test],
            delta_norms,
            out=np.zeros_like(projections[test]),
            where=delta_norms > 1e-12,
        )
    return projections, cosines, fold_ids


def analyze_activation_deltas(
    *,
    counterfactual_rows: list[dict[str, Any]],
    pooled_by_layer: dict[int, np.ndarray],
    text_index: dict[str, int],
    binary_margin: dict[str, float],
    run_label: str,
    model_id: str,
    folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    ids = [str(row["counterfactual_id"]) for row in counterfactual_rows]
    plus_indices: list[int] = []
    minus_indices: list[int] = []
    for row in counterfactual_rows:
        cue_plus, cue_minus = _cue_texts(row)
        plus_indices.append(text_index[sha1_hex(cue_plus)])
        minus_indices.append(text_index[sha1_hex(cue_minus)])
    plus_idx = np.asarray(plus_indices, dtype=int)
    minus_idx = np.asarray(minus_indices, dtype=int)
    margins = np.asarray([binary_margin.get(item, math.nan) for item in ids], dtype=float)
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    directions: dict[str, np.ndarray] = {}

    def append_summary(
        *,
        hidden_layer: int,
        group_type: str,
        group_value: str,
        group_indices: np.ndarray,
        deltas: np.ndarray,
        margins: np.ndarray,
    ) -> None:
        group_deltas = deltas[group_indices]
        group_ids = [ids[index] for index in group_indices]
        group_margins = margins[group_indices]
        delta_norms = np.linalg.norm(group_deltas, axis=1)
        mean_direction = np.mean(group_deltas, axis=0)
        projections, cosines, _ = cross_validated_direction_scores(
            group_deltas,
            group_ids,
            folds=int(folds),
            seed=int(seed),
        )
        valid_cosines = cosines[np.isfinite(cosines)]
        valid_projections = projections[np.isfinite(projections)]
        summary_rows.append(
            {
                "run_label": str(run_label),
                "model_id": str(model_id),
                "hidden_layer": int(hidden_layer),
                "group_type": str(group_type),
                "group_value": str(group_value),
                "n_counterfactuals": int(len(group_deltas)),
                "n_cv_scored": int(len(valid_cosines)),
                "mean_delta_norm": float(np.mean(delta_norms)),
                "mean_direction_norm": float(np.linalg.norm(mean_direction)),
                "direction_concentration": float(
                    np.linalg.norm(mean_direction) / max(float(np.mean(delta_norms)), 1e-12)
                ),
                "mean_cv_style_projection": None
                if not len(valid_projections)
                else float(np.mean(valid_projections)),
                "mean_cv_cosine_to_style_direction": None
                if not len(valid_cosines)
                else float(np.mean(valid_cosines)),
                "positive_cv_cosine_rate": None
                if not len(valid_cosines)
                else float(np.mean(valid_cosines > 0.0)),
                "pearson_projection_with_binary_margin": _pearson(projections, group_margins),
                "n_binary_margins": int(np.isfinite(group_margins).sum()),
            }
        )

    metadata_cols = (
        "counterfactual_id",
        "pair_id",
        "source_dataset",
        "subset",
        "item_type",
        "role",
        "axis",
        "direction",
        "transform_id",
        "rewrite_family",
        "marker_id",
    )
    metadata = pd.DataFrame(
        [{col: str(row.get(col) or "") for col in metadata_cols} for row in counterfactual_rows]
    )
    for hidden_layer, pooled in sorted(pooled_by_layer.items()):
        deltas = pooled[plus_idx] - pooled[minus_idx]
        delta_norms = np.linalg.norm(deltas, axis=1)
        mean_direction = np.mean(deltas, axis=0)
        directions[f"hidden_{int(hidden_layer)}"] = mean_direction.astype(np.float32)
        projections, cosines, fold_ids = cross_validated_direction_scores(
            deltas,
            ids,
            folds=int(folds),
            seed=int(seed),
        )
        layer_df = metadata.copy()
        layer_df["run_label"] = str(run_label)
        layer_df["model_id"] = str(model_id)
        layer_df["hidden_layer"] = int(hidden_layer)
        layer_df["cv_fold"] = fold_ids
        layer_df["delta_norm"] = delta_norms
        layer_df["cv_style_projection"] = projections
        layer_df["cv_cosine_to_style_direction"] = cosines
        layer_df["binary_cue_plus_margin"] = margins
        detail_frames.append(layer_df)
        append_summary(
            hidden_layer=int(hidden_layer),
            group_type="all",
            group_value="all",
            group_indices=np.arange(len(deltas), dtype=int),
            deltas=deltas,
            margins=margins,
        )
        for group_type, cols in (
            ("axis", ["axis"]),
            ("transform_id", ["transform_id"]),
            ("axis_role", ["axis", "role"]),
        ):
            for group_value, group in layer_df.groupby(cols, sort=True):
                if not isinstance(group_value, tuple):
                    group_value = (group_value,)
                group_indices = group.index.to_numpy(dtype=int)
                if len(group_indices) < max(int(folds), 3):
                    continue
                append_summary(
                    hidden_layer=int(hidden_layer),
                    group_type=group_type,
                    group_value="::".join(map(str, group_value)),
                    group_indices=group_indices,
                    deltas=deltas,
                    margins=margins,
                )
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return details, pd.DataFrame(summary_rows), directions


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    counterfactual_path = _resolve(workspace_root, args.counterfactual_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    rows = _cap_rows(
        read_jsonl(counterfactual_path),
        axes=_csv_set(str(args.axes)),
        max_counterfactuals=int(args.max_counterfactuals),
        seed=int(args.seed),
    )
    if not rows:
        raise ValueError(f"No counterfactual rows selected from {counterfactual_path}")
    model, tokenizer = _load_lm(args)
    layers = _selected_layers(
        model,
        raw=str(args.selected_layers),
        stride=int(args.layer_stride),
        tail_layers=int(args.tail_layers),
    )
    text_by_id: dict[str, str] = {}
    for row in rows:
        cue_plus, cue_minus = _cue_texts(row)
        text_by_id.setdefault(sha1_hex(cue_plus), cue_plus)
        text_by_id.setdefault(sha1_hex(cue_minus), cue_minus)
    text_ids = sorted(text_by_id)
    text_index = {text_id: index for index, text_id in enumerate(text_ids)}
    pooled = _extract_mean_pooled_hidden(
        model=model,
        tokenizer=tokenizer,
        texts=[text_by_id[text_id] for text_id in text_ids],
        selected_layers=layers,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    margin_path = None if args.bt_scores is None else _resolve(workspace_root, args.bt_scores)
    details, summary, directions = analyze_activation_deltas(
        counterfactual_rows=rows,
        pooled_by_layer=pooled,
        text_index=text_index,
        binary_margin=_load_binary_margin(margin_path),
        run_label=str(args.run_label or args.model_id),
        model_id=str(args.model_id),
        folds=int(args.cv_folds),
        seed=int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(out_dir / "counterfactual_layer_scores.csv", index=False)
    summary.to_csv(out_dir / "layer_summary.csv", index=False)
    np.savez_compressed(out_dir / "layer_mean_directions.npz", **directions)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "D4-LM-style-activation-contrast",
            "counterfactual_jsonl": str(counterfactual_path),
            "bt_scores": None if margin_path is None else str(margin_path),
            "out_dir": str(out_dir),
            "run_label": str(args.run_label or args.model_id),
            "model_id": str(args.model_id),
            "selected_hidden_layers": layers,
            "n_counterfactuals": int(len(rows)),
            "n_unique_texts": int(len(text_ids)),
            "pooling": "mean_answer_token_residual",
            "max_counterfactuals": int(args.max_counterfactuals),
            "max_length": int(args.max_length),
            "cv_folds": int(args.cv_folds),
            "seed": int(args.seed),
        },
    )
    print(f"out_dir={out_dir}")
    print(f"layers={','.join(map(str, layers))}")
    print(f"n_counterfactuals={len(rows)}")


if __name__ == "__main__":
    main()
