"""Score D4 surface-cue counterfactuals with J0 and SAE bundle detectors."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import DEFAULT_SURFACE_AXES
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.mech.interventions import FeatureSpec, group_features_by_layer, load_bundle_feature_specs
from aisafety.mech.sae import format_sae_id, load_sae
from aisafety.scripts.build_d4_human_llm_alignment_pairs import _csv_set
from aisafety.scripts.run_d4_candidate_feature_pair_alignment import (
    _encode_texts_candidate_sae,
    _load_scorer_and_tokenizer,
    _score_texts,
    _scorer_device,
    _write_csv,
)
from aisafety.scripts.run_d4_feature_perturbation import _fill_sae_defaults, _standardize


DEFAULT_COUNTERFACTUAL_JSONL = (
    Path("data") / "derived" / "d4_surface_counterfactual_pairs_v1" / "counterfactuals.jsonl"
)
DEFAULT_REGISTRY_DIR = Path("artifacts") / "mechanistic" / "d4_j0_bundle_candidate_registry_v1"
DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_j0_surface_counterfactual_audit_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--counterfactual-jsonl", type=Path, default=DEFAULT_COUNTERFACTUAL_JSONL)
    parser.add_argument("--bundle-registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--bundle-ids", type=str, default=",".join(DEFAULT_SURFACE_AXES))
    parser.add_argument("--reward-run-dir", type=Path, default=Path("artifacts") / "reward" / "j0_anchor_v1_h100compact")
    parser.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--sae-release", type=str, default="gemma-scope-9b-pt-res-canonical")
    parser.add_argument("--sae-id-template", type=str, default="layer_{sae_layer}/width_16k/canonical")
    parser.add_argument("--max-counterfactuals", type=int, default=0)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--sae-batch-size", type=int, default=4)
    parser.add_argument("--sae-token-chunk-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--skip-sae", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _cap_rows(rows: list[dict[str, Any]], *, max_rows: int, seed: int) -> list[dict[str, Any]]:
    if int(max_rows) <= 0 or len(rows) <= int(max_rows):
        return list(rows)
    ordered = sorted(
        rows,
        key=lambda row: sha1_hex(f"{seed}:counterfactual-audit:{row.get('counterfactual_id')}"),
    )
    chosen = {str(row.get("counterfactual_id")) for row in ordered[: int(max_rows)]}
    return [row for row in rows if str(row.get("counterfactual_id")) in chosen]


def _score_unique_texts(
    *,
    df: pd.DataFrame,
    scorer: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    text_by_id: dict[str, str] = {}
    for text in df["base_text"].astype(str).tolist() + df["variant_text"].astype(str).tolist():
        text_by_id.setdefault(sha1_hex(text), text)
    text_ids = sorted(text_by_id)
    texts = [text_by_id[text_id] for text_id in text_ids]
    scores = _score_texts(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=int(args.score_batch_size),
        max_length=int(args.max_length),
    )
    score_by_id = {text_id: float(score) for text_id, score in zip(text_ids, scores, strict=True)}
    base = np.asarray([score_by_id[sha1_hex(text)] for text in df["base_text"].astype(str)], dtype=float)
    variant = np.asarray([score_by_id[sha1_hex(text)] for text in df["variant_text"].astype(str)], dtype=float)
    return base, variant


def _load_bundle_features(
    registry_dir: Path,
    *,
    bundle_ids: set[str],
    default_release: str,
    sae_id_template: str,
) -> dict[str, list[FeatureSpec]]:
    out: dict[str, list[FeatureSpec]] = {}
    for bundle_id in sorted(bundle_ids):
        specs = load_bundle_feature_specs(registry_dir, bundle_id=bundle_id)
        if specs:
            out[bundle_id] = _fill_sae_defaults(
                specs,
                default_release=default_release,
                sae_id_template=sae_id_template,
            )
    return out


def _unique_specs(bundle_features: dict[str, list[FeatureSpec]]) -> list[FeatureSpec]:
    seen: set[tuple[int, int]] = set()
    specs: list[FeatureSpec] = []
    for rows in bundle_features.values():
        for spec in rows:
            key = (int(spec.hidden_layer), int(spec.feature_idx))
            if key in seen:
                continue
            seen.add(key)
            specs.append(spec)
    return specs


def _attach_bundle_activation_delta(
    *,
    df: pd.DataFrame,
    scorer: Any,
    tokenizer: Any,
    bundle_features: dict[str, list[FeatureSpec]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not bundle_features:
        return df, []

    text_by_id: dict[str, str] = {}
    for text in df["base_text"].astype(str).tolist() + df["variant_text"].astype(str).tolist():
        text_by_id.setdefault(sha1_hex(text), text)
    text_ids = sorted(text_by_id)
    text_id_to_row = {text_id: idx for idx, text_id in enumerate(text_ids)}
    texts = [text_by_id[text_id] for text_id in text_ids]

    feature_values: dict[tuple[int, int], np.ndarray] = {}
    sae_by_layer: dict[int, Any] = {}
    for hidden_layer, layer_specs in sorted(group_features_by_layer(_unique_specs(bundle_features)).items()):
        first = layer_specs[0]
        sae_by_layer[int(hidden_layer)] = load_sae(
            release=str(first.sae_release),
            sae_id=str(first.sae_id),
            device=_scorer_device(scorer),
        )
        feature_indices = [int(spec.feature_idx) for spec in layer_specs]
        feats = _encode_texts_candidate_sae(
            scorer=scorer,
            tokenizer=tokenizer,
            sae=sae_by_layer[int(hidden_layer)],
            texts=texts,
            hidden_layer=int(hidden_layer),
            feature_indices=feature_indices,
            batch_size=int(args.sae_batch_size),
            max_length=int(args.max_length),
            aggregation="max",
            token_chunk_size=int(args.sae_token_chunk_size),
        )
        for col, spec in enumerate(layer_specs):
            feature_values[(int(hidden_layer), int(spec.feature_idx))] = _standardize(feats[:, col])

    bundle_scores: dict[str, np.ndarray] = {}
    feature_summary_rows: list[dict[str, Any]] = []
    for bundle_id, specs in sorted(bundle_features.items()):
        cols: list[np.ndarray] = []
        for spec in specs:
            values = feature_values[(int(spec.hidden_layer), int(spec.feature_idx))]
            cols.append(float(spec.direction) * values)
            feature_summary_rows.append(
                {
                    "bundle_id": bundle_id,
                    "hidden_layer": int(spec.hidden_layer),
                    "feature_idx": int(spec.feature_idx),
                    "feature_id": spec.feature_id,
                    "direction": int(spec.direction),
                    "signed_alignment": float(spec.signed_alignment),
                    "mean_signed_activation": float(np.mean(float(spec.direction) * values)),
                    "std_signed_activation": float(np.std(float(spec.direction) * values)),
                }
            )
        bundle_scores[bundle_id] = np.mean(np.stack(cols, axis=1), axis=1)

    out = df.copy()
    base_scores: list[float | None] = []
    variant_scores: list[float | None] = []
    deltas: list[float | None] = []
    for row in out.itertuples(index=False):
        axis = str(getattr(row, "axis"))
        scores = bundle_scores.get(axis)
        if scores is None:
            base_scores.append(None)
            variant_scores.append(None)
            deltas.append(None)
            continue
        base_value = float(scores[text_id_to_row[sha1_hex(str(getattr(row, "base_text")))]] )
        variant_value = float(scores[text_id_to_row[sha1_hex(str(getattr(row, "variant_text")))]] )
        base_scores.append(base_value)
        variant_scores.append(variant_value)
        deltas.append(variant_value - base_value)
    out["matching_bundle_signed_base"] = base_scores
    out["matching_bundle_signed_variant"] = variant_scores
    out["matching_bundle_signed_delta"] = deltas
    return out, feature_summary_rows


def _controlled_delta(values: np.ndarray, length_delta: np.ndarray, length_ratio: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if len(y) == 0:
        return y
    cols = [np.ones(len(y), dtype=float)]
    for raw in (length_delta, length_ratio):
        vals = np.asarray(raw, dtype=float)
        std = float(np.std(vals))
        cols.append((vals - float(np.mean(vals))) / std if std > 1e-12 else vals * 0.0)
    design = np.stack(cols, axis=1)
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return y - design @ beta + float(np.mean(y))
    except np.linalg.LinAlgError:
        return y


def _mean_or_none(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _summary_row(df: pd.DataFrame, *, group_type: str, group_value: str) -> dict[str, Any]:
    reward_delta = pd.to_numeric(df["reward_delta"], errors="coerce")
    return {
        "group_type": group_type,
        "group_value": group_value,
        "n_counterfactuals": int(len(df)),
        "mean_reward_delta": float(reward_delta.mean()),
        "median_reward_delta": float(reward_delta.median()),
        "mean_abs_reward_delta": float(reward_delta.abs().mean()),
        "mean_length_delta": _mean_or_none(df["length_delta"]),
        "mean_length_ratio": _mean_or_none(df["length_ratio"]),
        "length_controlled_mean_reward_delta": _mean_or_none(df["length_controlled_reward_delta"]),
        "mean_matching_bundle_signed_delta": _mean_or_none(df.get("matching_bundle_signed_delta", pd.Series(dtype=float))),
    }


def _build_summaries(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    axis_rows: list[dict[str, Any]] = [_summary_row(df, group_type="all", group_value="all")]
    for (axis, direction), group in df.groupby(["axis", "direction"], sort=True):
        axis_rows.append(_summary_row(group, group_type="axis_direction", group_value=f"{axis}::{direction}"))
    for (axis, direction, role), group in df.groupby(["axis", "direction", "role"], sort=True):
        axis_rows.append(_summary_row(group, group_type="axis_direction_role", group_value=f"{axis}::{direction}::{role}"))

    source_rows: list[dict[str, Any]] = []
    for (source, axis, direction), group in df.groupby(["source_dataset", "axis", "direction"], sort=True):
        source_rows.append(
            _summary_row(
                group,
                group_type="source_axis_direction",
                group_value=f"{source}::{axis}::{direction}",
            )
        )
    return axis_rows, source_rows


def _score_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "counterfactual_id",
        "pair_id",
        "source_dataset",
        "subset",
        "split",
        "item_type",
        "role",
        "axis",
        "direction",
        "transform_id",
        "base_tokens",
        "variant_tokens",
        "length_delta",
        "length_ratio",
        "base_reward",
        "variant_reward",
        "reward_delta",
        "length_controlled_reward_delta",
        "matching_bundle_signed_base",
        "matching_bundle_signed_variant",
        "matching_bundle_signed_delta",
        "content_preservation_flags",
    ]
    return [col for col in preferred if col in df.columns]


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    counterfactual_path = _resolve(workspace_root, args.counterfactual_jsonl)
    registry_dir = _resolve(workspace_root, args.bundle_registry_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    bundle_ids = _csv_set(str(args.bundle_ids))

    rows = _cap_rows(read_jsonl(counterfactual_path), max_rows=int(args.max_counterfactuals), seed=int(args.seed))
    if not rows:
        raise ValueError(f"No counterfactual rows found in {counterfactual_path}")
    df = pd.DataFrame(rows)
    scorer, tokenizer = _load_scorer_and_tokenizer(args, workspace_root)
    base_scores, variant_scores = _score_unique_texts(df=df, scorer=scorer, tokenizer=tokenizer, args=args)
    df["base_reward"] = base_scores
    df["variant_reward"] = variant_scores
    df["reward_delta"] = df["variant_reward"] - df["base_reward"]
    df["length_controlled_reward_delta"] = _controlled_delta(
        df["reward_delta"].to_numpy(dtype=float),
        pd.to_numeric(df["length_delta"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(df["length_ratio"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
    )

    feature_rows: list[dict[str, Any]] = []
    if not bool(args.skip_sae):
        bundle_features = _load_bundle_features(
            registry_dir,
            bundle_ids=bundle_ids,
            default_release=str(args.sae_release),
            sae_id_template=str(args.sae_id_template),
        )
        df, feature_rows = _attach_bundle_activation_delta(
            df=df,
            scorer=scorer,
            tokenizer=tokenizer,
            bundle_features=bundle_features,
            args=args,
        )

    axis_rows, source_rows = _build_summaries(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    df[_score_columns(df)].to_csv(out_dir / "surface_counterfactual_scores.csv", index=False)
    _write_csv(out_dir / "axis_summary.csv", axis_rows)
    _write_csv(out_dir / "source_summary.csv", source_rows)
    _write_csv(out_dir / "bundle_activation_delta.csv", feature_rows)
    manifest = {
        "stage": "D4-surface-counterfactual-audit",
        "counterfactual_jsonl": str(counterfactual_path),
        "bundle_registry_dir": str(registry_dir),
        "bundle_ids": sorted(bundle_ids),
        "reward_run_dir": str(_resolve(workspace_root, args.reward_run_dir)),
        "model_id": str(args.model_id),
        "n_counterfactuals": int(len(df)),
        "n_feature_rows": int(len(feature_rows)),
        "skip_sae": bool(args.skip_sae),
        "counts_by_axis_direction": {
            f"{axis}::{direction}": int(count)
            for (axis, direction), count in Counter(zip(df["axis"].astype(str), df["direction"].astype(str))).items()
        },
        "outputs": {
            "surface_counterfactual_scores_csv": str(out_dir / "surface_counterfactual_scores.csv"),
            "axis_summary_csv": str(out_dir / "axis_summary.csv"),
            "source_summary_csv": str(out_dir / "source_summary.csv"),
            "bundle_activation_delta_csv": str(out_dir / "bundle_activation_delta.csv"),
            "manifest_json": str(out_dir / "manifest.json"),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"out_dir={out_dir}")
    print(f"scores={out_dir / 'surface_counterfactual_scores.csv'}")
    print(f"axis_summary={out_dir / 'axis_summary.csv'}")
    print(f"manifest={out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
