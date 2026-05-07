"""Run D4 SAE feature perturbations on high/low bundle-occurrence pairs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import resolve_path, sha1_hex, write_json
from aisafety.mech.interventions import (
    FeatureSpec,
    assign_quantile_bins,
    group_features_by_layer,
    iter_feature_rows,
    load_bundle_feature_specs,
    load_matched_random_control_specs,
    register_feature_damping_hooks,
    remove_hooks,
)
from aisafety.mech.sae import format_sae_id, load_sae
from aisafety.scripts.run_d4_candidate_feature_pair_alignment import (
    _cap_pairs,
    _encode_texts_candidate_sae,
    _load_scorer_and_tokenizer,
    _read_pair_file,
    _score_texts,
    _scorer_device,
    _write_csv,
)


DEFAULT_PAIR_JSONL = Path("data") / "derived" / "d4_human_llm_alignment_pairs_strat10k_v3" / "pairs.jsonl"
DEFAULT_REGISTRY_DIR = Path("artifacts") / "mechanistic" / "d4_j0_bundle_candidate_registry_v1"
DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_j0_formal_bundle_feature_perturbation_scout_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--bundle-registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--bundle-id", type=str, default="formal_institutional_packaging")
    parser.add_argument("--reward-run-dir", type=Path, default=Path("artifacts") / "reward" / "j0_anchor_v1_h100compact")
    parser.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--sae-release", type=str, default="gemma-scope-9b-pt-res-canonical")
    parser.add_argument("--sae-id-template", type=str, default="layer_{sae_layer}/width_16k/canonical")
    parser.add_argument("--max-pairs", type=int, default=1200)
    parser.add_argument("--max-features", type=int, default=0)
    parser.add_argument("--include-source-sensitive", action="store_true")
    parser.add_argument("--damping-strength", type=float, default=0.5)
    parser.add_argument("--high-low-frac", type=float, default=0.25)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--sae-batch-size", type=int, default=4)
    parser.add_argument("--sae-token-chunk-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--skip-random-controls", action="store_true")
    parser.add_argument("--random-control-rank", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _feature_key(spec: FeatureSpec) -> tuple[int, int]:
    return int(spec.hidden_layer), int(spec.feature_idx)


def _fill_sae_defaults(
    features: list[FeatureSpec],
    *,
    default_release: str,
    sae_id_template: str,
) -> list[FeatureSpec]:
    out: list[FeatureSpec] = []
    for spec in features:
        sae_id = spec.sae_id or format_sae_id(str(sae_id_template), hidden_layer=int(spec.hidden_layer))
        out.append(
            FeatureSpec(
                bundle_id=spec.bundle_id,
                hidden_layer=int(spec.hidden_layer),
                feature_idx=int(spec.feature_idx),
                sae_release=spec.sae_release or str(default_release),
                sae_id=sae_id,
                aggregation=spec.aggregation,
                feature_role=spec.feature_role,
                freeze_status=spec.freeze_status,
                atoms=spec.atoms,
                signed_alignment=float(spec.signed_alignment),
                source_row=spec.source_row,
            )
        )
    return out


def _load_sae_by_layer(
    *,
    scorer: Any,
    features: list[FeatureSpec],
) -> dict[int, Any]:
    device = _scorer_device(scorer)
    sae_by_layer: dict[int, Any] = {}
    by_layer = group_features_by_layer(features)
    for hidden_layer, layer_features in sorted(by_layer.items()):
        first = layer_features[0]
        sae_by_layer[int(hidden_layer)] = load_sae(
            release=str(first.sae_release),
            sae_id=str(first.sae_id),
            device=device,
        )
    return sae_by_layer


def _unique_text_frame(pair_df: pd.DataFrame) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray]:
    text_by_id: dict[str, str] = {}
    for text in pair_df["human_text"].astype(str).tolist() + pair_df["llm_text"].astype(str).tolist():
        text_by_id.setdefault(sha1_hex(text), text)
    unique_ids = sorted(text_by_id)
    unique_texts = [text_by_id[text_id] for text_id in unique_ids]
    text_id_to_row = {text_id: idx for idx, text_id in enumerate(unique_ids)}
    human_rows = np.asarray(
        [text_id_to_row[sha1_hex(text)] for text in pair_df["human_text"].astype(str)],
        dtype=int,
    )
    llm_rows = np.asarray(
        [text_id_to_row[sha1_hex(text)] for text in pair_df["llm_text"].astype(str)],
        dtype=int,
    )
    return unique_texts, text_id_to_row, human_rows, llm_rows


def _standardize(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    std = float(np.std(vals))
    if std <= 1e-8:
        return vals * 0.0
    return (vals - float(np.mean(vals))) / std


def _attach_bundle_occurrence(
    *,
    pair_df: pd.DataFrame,
    scorer: Any,
    tokenizer: Any,
    sae_by_layer: dict[int, Any],
    target_features: list[FeatureSpec],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    unique_texts, _text_id_to_row, human_rows, llm_rows = _unique_text_frame(pair_df)
    out = pair_df.copy()
    signed_delta_cols: list[np.ndarray] = []
    presence_cols: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []

    for hidden_layer, layer_features in sorted(group_features_by_layer(target_features).items()):
        feature_indices = [int(spec.feature_idx) for spec in layer_features]
        feats = _encode_texts_candidate_sae(
            scorer=scorer,
            tokenizer=tokenizer,
            sae=sae_by_layer[int(hidden_layer)],
            texts=unique_texts,
            hidden_layer=int(hidden_layer),
            feature_indices=feature_indices,
            batch_size=int(args.sae_batch_size),
            max_length=int(args.max_length),
            aggregation="max",
            token_chunk_size=int(args.sae_token_chunk_size),
        )
        for col, spec in enumerate(layer_features):
            all_vals = _standardize(feats[:, col])
            human_vals = all_vals[human_rows]
            llm_vals = all_vals[llm_rows]
            signed_delta = float(spec.direction) * (llm_vals - human_vals)
            signed_delta_cols.append(signed_delta)
            presence_cols.append(np.maximum(human_vals, llm_vals))
            feature_rows.append(
                {
                    "hidden_layer": int(spec.hidden_layer),
                    "feature_idx": int(spec.feature_idx),
                    "feature_id": spec.feature_id,
                    "direction": int(spec.direction),
                    "signed_alignment": float(spec.signed_alignment),
                    "mean_human_z_activation": float(np.mean(human_vals)),
                    "mean_llm_z_activation": float(np.mean(llm_vals)),
                    "mean_signed_pair_delta": float(np.mean(signed_delta)),
                    "p90_abs_signed_pair_delta": float(np.quantile(np.abs(signed_delta), 0.9)),
                }
            )

    if not signed_delta_cols:
        raise ValueError("No bundle occurrence columns were computed.")
    signed_delta_mat = np.stack(signed_delta_cols, axis=1)
    presence_mat = np.stack(presence_cols, axis=1)
    out["bundle_signed_delta"] = np.mean(signed_delta_mat, axis=1)
    out["bundle_presence_score"] = np.mean(presence_mat, axis=1)
    out["bundle_occurrence_bin"] = assign_quantile_bins(
        out["bundle_signed_delta"].to_numpy(dtype=float),
        high_low_frac=float(args.high_low_frac),
    )
    return out, feature_rows


def _score_pair_sides(
    *,
    pair_df: pd.DataFrame,
    scorer: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    human_scores = _score_texts(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=pair_df["human_text"].astype(str).tolist(),
        batch_size=int(args.score_batch_size),
        max_length=int(args.max_length),
    )
    llm_scores = _score_texts(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=pair_df["llm_text"].astype(str).tolist(),
        batch_size=int(args.score_batch_size),
        max_length=int(args.max_length),
    )
    return human_scores.astype(float), llm_scores.astype(float)


def _score_pair_sides_with_damping(
    *,
    pair_df: pd.DataFrame,
    scorer: Any,
    tokenizer: Any,
    features: list[FeatureSpec],
    sae_by_layer: dict[int, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    handles = register_feature_damping_hooks(
        scorer.backbone,
        features_by_layer=group_features_by_layer(features),
        sae_by_layer=sae_by_layer,
        strength=float(args.damping_strength),
    )
    try:
        return _score_pair_sides(pair_df=pair_df, scorer=scorer, tokenizer=tokenizer, args=args)
    finally:
        remove_hooks(handles)


def _attach_scores(
    pair_df: pd.DataFrame,
    *,
    prefix: str,
    human_scores: np.ndarray,
    llm_scores: np.ndarray,
) -> pd.DataFrame:
    out = pair_df.copy()
    out[f"{prefix}_human_reward"] = human_scores.astype(float)
    out[f"{prefix}_llm_reward"] = llm_scores.astype(float)
    out[f"{prefix}_llm_margin_pair"] = out[f"{prefix}_llm_reward"] - out[f"{prefix}_human_reward"]
    out[f"{prefix}_y_llm_chosen"] = (out[f"{prefix}_llm_margin_pair"] > 0.0).astype(int)
    return out


def _mean_or_none(values: pd.Series) -> float | None:
    if len(values) == 0:
        return None
    return float(pd.to_numeric(values, errors="coerce").mean())


def _summary_for_group(
    df: pd.DataFrame,
    *,
    intervention: str,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    baseline_margin = pd.to_numeric(df["baseline_llm_margin_pair"], errors="coerce")
    perturbed_margin = pd.to_numeric(df[f"{intervention}_llm_margin_pair"], errors="coerce")
    baseline_choice = pd.to_numeric(df["baseline_y_llm_chosen"], errors="coerce")
    perturbed_choice = pd.to_numeric(df[f"{intervention}_y_llm_chosen"], errors="coerce")
    margin_change = perturbed_margin - baseline_margin
    choice_change = perturbed_choice - baseline_choice
    row = {
        "intervention": intervention,
        "group_type": group_type,
        "group_value": group_value,
        "n_pairs": int(len(df)),
        "mean_bundle_signed_delta": _mean_or_none(df["bundle_signed_delta"]),
        "mean_bundle_presence_score": _mean_or_none(df["bundle_presence_score"]),
        "baseline_mean_llm_margin": float(baseline_margin.mean()),
        "perturbed_mean_llm_margin": float(perturbed_margin.mean()),
        "mean_margin_change": float(margin_change.mean()),
        "median_margin_change": float(margin_change.median()),
        "baseline_llm_chosen_rate": float(baseline_choice.mean()),
        "perturbed_llm_chosen_rate": float(perturbed_choice.mean()),
        "llm_chosen_rate_change": float(choice_change.mean()),
    }
    if f"control_llm_margin_pair" in df.columns and intervention == "target":
        control_change = pd.to_numeric(df["control_llm_margin_pair"], errors="coerce") - baseline_margin
        row["mean_target_minus_control_margin_change"] = float(
            margin_change.mean() - control_change.mean()
        )
    return row


def _build_summary(pair_df: pd.DataFrame, *, interventions: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for intervention in interventions:
        if f"{intervention}_llm_margin_pair" not in pair_df.columns:
            continue
        rows.append(
            _summary_for_group(
                pair_df,
                intervention=intervention,
                group_type="all",
                group_value="all",
            )
        )
        for bin_name, group in pair_df.groupby("bundle_occurrence_bin", sort=True):
            rows.append(
                _summary_for_group(
                    group,
                    intervention=intervention,
                    group_type="bundle_occurrence_bin",
                    group_value=str(bin_name),
                )
            )
        for source, group in pair_df.groupby("source_dataset", sort=True):
            rows.append(
                _summary_for_group(
                    group,
                    intervention=intervention,
                    group_type="source_dataset",
                    group_value=str(source),
                )
            )
        for (source, bin_name), group in pair_df.groupby(
            ["source_dataset", "bundle_occurrence_bin"],
            sort=True,
        ):
            rows.append(
                _summary_for_group(
                    group,
                    intervention=intervention,
                    group_type="source_dataset_x_bin",
                    group_value=f"{source}::{bin_name}",
                )
            )
    return rows


def _pair_score_columns(pair_df: pd.DataFrame) -> list[str]:
    preferred = [
        "pair_id",
        "source_dataset",
        "bundle_creation_role",
        "group_id",
        "split",
        "item_type",
        "subset",
        "title",
        "question",
        "llm_generator",
        "human_token_count",
        "llm_token_count",
        "token_delta_llm_minus_human",
        "char_delta_llm_minus_human",
        "bundle_signed_delta",
        "bundle_presence_score",
        "bundle_occurrence_bin",
        "baseline_human_reward",
        "baseline_llm_reward",
        "baseline_llm_margin_pair",
        "baseline_y_llm_chosen",
        "target_human_reward",
        "target_llm_reward",
        "target_llm_margin_pair",
        "target_y_llm_chosen",
        "control_human_reward",
        "control_llm_reward",
        "control_llm_margin_pair",
        "control_y_llm_chosen",
    ]
    return [col for col in preferred if col in pair_df.columns]


def _write_outputs(
    *,
    out_dir: Path,
    pair_df: pd.DataFrame,
    feature_rows: list[dict[str, Any]],
    occurrence_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "intervention_feature_set.csv", feature_rows)
    _write_csv(out_dir / "feature_occurrence_summary.csv", occurrence_rows)
    _write_csv(out_dir / "intervention_bin_summary.csv", summary_rows)
    pair_df[_pair_score_columns(pair_df)].to_csv(out_dir / "pair_perturbation_scores.csv", index=False)
    write_json(out_dir / "intervention_manifest.json", manifest)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    pair_path = _resolve(workspace_root, args.pair_jsonl)
    registry_dir = _resolve(workspace_root, args.bundle_registry_dir)
    out_dir = _resolve(workspace_root, args.out_dir)

    target_features = load_bundle_feature_specs(
        registry_dir,
        bundle_id=str(args.bundle_id),
        include_source_sensitive=bool(args.include_source_sensitive),
        max_features=int(args.max_features),
    )
    if not target_features:
        raise ValueError(f"No target features found for bundle_id={args.bundle_id!r}.")
    target_features = _fill_sae_defaults(
        target_features,
        default_release=str(args.sae_release),
        sae_id_template=str(args.sae_id_template),
    )
    control_features: list[FeatureSpec] = []
    if not bool(args.skip_random_controls):
        control_features = load_matched_random_control_specs(
            registry_dir,
            bundle_id=str(args.bundle_id),
            allowed_target_keys={_feature_key(spec) for spec in target_features},
            control_rank=int(args.random_control_rank),
            max_features=int(args.max_features),
        )
        control_features = _fill_sae_defaults(
            control_features,
            default_release=str(args.sae_release),
            sae_id_template=str(args.sae_id_template),
        )

    pair_df = _cap_pairs(_read_pair_file(pair_path), max_pairs=int(args.max_pairs), seed=int(args.seed))
    scorer, tokenizer = _load_scorer_and_tokenizer(args, workspace_root)
    all_features = target_features + [
        spec for spec in control_features if _feature_key(spec) not in {_feature_key(item) for item in target_features}
    ]
    sae_by_layer = _load_sae_by_layer(scorer=scorer, features=all_features)
    pair_df, occurrence_rows = _attach_bundle_occurrence(
        pair_df=pair_df,
        scorer=scorer,
        tokenizer=tokenizer,
        sae_by_layer=sae_by_layer,
        target_features=target_features,
        args=args,
    )

    baseline_human, baseline_llm = _score_pair_sides(
        pair_df=pair_df,
        scorer=scorer,
        tokenizer=tokenizer,
        args=args,
    )
    pair_df = _attach_scores(
        pair_df,
        prefix="baseline",
        human_scores=baseline_human,
        llm_scores=baseline_llm,
    )

    target_human, target_llm = _score_pair_sides_with_damping(
        pair_df=pair_df,
        scorer=scorer,
        tokenizer=tokenizer,
        features=target_features,
        sae_by_layer=sae_by_layer,
        args=args,
    )
    pair_df = _attach_scores(
        pair_df,
        prefix="target",
        human_scores=target_human,
        llm_scores=target_llm,
    )

    interventions = ["target"]
    if control_features:
        control_human, control_llm = _score_pair_sides_with_damping(
            pair_df=pair_df,
            scorer=scorer,
            tokenizer=tokenizer,
            features=control_features,
            sae_by_layer=sae_by_layer,
            args=args,
        )
        pair_df = _attach_scores(
            pair_df,
            prefix="control",
            human_scores=control_human,
            llm_scores=control_llm,
        )
        interventions.append("control")

    summary_rows = _build_summary(pair_df, interventions=interventions)
    feature_rows = [
        *iter_feature_rows(target_features),
        *iter_feature_rows(control_features),
    ]
    manifest = {
        "stage": "D4-SAE-feature-perturbation-scout",
        "pair_jsonl": str(pair_path),
        "bundle_registry_dir": str(registry_dir),
        "bundle_id": str(args.bundle_id),
        "reward_run_dir": str(_resolve(workspace_root, args.reward_run_dir)),
        "model_id": str(args.model_id),
        "sae_release": str(args.sae_release),
        "sae_id_template": str(args.sae_id_template),
        "damping_strength": float(args.damping_strength),
        "high_low_frac": float(args.high_low_frac),
        "include_source_sensitive": bool(args.include_source_sensitive),
        "n_pairs": int(len(pair_df)),
        "n_target_features": int(len(target_features)),
        "n_control_features": int(len(control_features)),
        "target_features": [row for row in iter_feature_rows(target_features)],
        "control_features": [row for row in iter_feature_rows(control_features)],
        "pair_counts_by_source_dataset": {
            str(k): int(v) for k, v in Counter(pair_df["source_dataset"].astype(str)).items()
        },
        "pair_counts_by_occurrence_bin": {
            str(k): int(v) for k, v in Counter(pair_df["bundle_occurrence_bin"].astype(str)).items()
        },
        "outputs": {
            "pair_perturbation_scores_csv": str(out_dir / "pair_perturbation_scores.csv"),
            "intervention_bin_summary_csv": str(out_dir / "intervention_bin_summary.csv"),
            "intervention_feature_set_csv": str(out_dir / "intervention_feature_set.csv"),
            "feature_occurrence_summary_csv": str(out_dir / "feature_occurrence_summary.csv"),
            "intervention_manifest_json": str(out_dir / "intervention_manifest.json"),
        },
    }
    _write_outputs(
        out_dir=out_dir,
        pair_df=pair_df,
        feature_rows=feature_rows,
        occurrence_rows=occurrence_rows,
        summary_rows=summary_rows,
        manifest=manifest,
    )
    print(f"out_dir={out_dir}")
    print(f"pair_scores={out_dir / 'pair_perturbation_scores.csv'}")
    print(f"summary={out_dir / 'intervention_bin_summary.csv'}")
    print(f"manifest={out_dir / 'intervention_manifest.json'}")


if __name__ == "__main__":
    main()
