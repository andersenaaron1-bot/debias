"""Run the D4 SAE feature-localization pass for judge cue atoms.

This is the first SAE-level stage after residual atom recovery. It does not
claim a full circuit. It ranks Gemma Scope SAE features by:

- atom-label separation on the D4 atom-probe set
- transfer to Laurito text-side atom scores
- pair-side alignment with Laurito judge decisions
- overlap with a content-anchor chosen/rejected utility signal

The script processes one residual layer at a time and rewrites intermediate
CSVs after each layer so long runs leave usable partial output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.features.token_positions import take_last_token
from aisafety.reward.model import load_reward_scorer
from aisafety.scripts.run_d4_atom_recovery import (
    _build_atom_label_frame,
    _flatten_content_pairs,
    _read_json,
    _read_jsonl,
    _resolve_path,
    _sample_atom_probe_rows,
    _write_json,
)


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--manifest-json",
        type=Path,
        default=Path("data") / "derived" / "d4_dataset_pack_v1" / "manifest.json",
    )
    p.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--reward-run-dir", type=Path, default=Path("artifacts") / "reward" / "j0_anchor_v1_h100compact")
    p.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--use-4bit", action="store_true")

    p.add_argument("--sae-release", type=str, default="gemma-scope-9b-pt-res-canonical")
    p.add_argument("--sae-id-template", type=str, default="layer_{sae_layer}/width_16k/canonical")
    p.add_argument(
        "--selected-layers",
        type=str,
        default="1,4,8,12,16,20,24,28,32,36,39,40,41,42",
        help="Comma-separated Hugging Face hidden-state indices. SAE layer is hidden_layer - 1.",
    )
    p.add_argument("--skip-missing-sae", action="store_true")
    p.add_argument("--aggregation", choices=["last", "max"], default="max")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sae-token-chunk-size", type=int, default=1024)

    p.add_argument("--max-train-per-item-type", type=int, default=800)
    p.add_argument("--max-val-per-item-type", type=int, default=160)
    p.add_argument("--max-test-per-item-type", type=int, default=160)
    p.add_argument("--label-quantile", type=float, default=0.8)
    p.add_argument("--min-train-examples", type=int, default=60)
    p.add_argument("--min-eval-examples", type=int, default=20)
    p.add_argument("--content-max-pairs", type=int, default=1000)
    p.add_argument("--disable-content-control", action="store_true")
    p.add_argument("--atoms", type=str, default=None, help="Optional comma-separated atom subset.")

    p.add_argument("--feature-candidate-k", type=int, default=250)
    p.add_argument("--top-k-features-per-atom", type=int, default=20)
    p.add_argument("--example-features-per-atom-layer", type=int, default=3)
    p.add_argument("--top-examples-per-feature", type=int, default=5)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts") / "mechanistic" / "d4_j0_sae_feature_analysis_v1")
    return p.parse_args()


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Expected at least one selected layer.")
    return sorted(dict.fromkeys(out))


def _parse_str_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    return out or None


def hidden_layer_to_sae_layer(hidden_layer: int) -> int:
    """Map HF hidden_states index to Gemma Scope residual block index."""

    hidden_layer = int(hidden_layer)
    if hidden_layer <= 0:
        raise ValueError("hidden_layer must be >= 1 because hidden_states[0] is embeddings.")
    return hidden_layer - 1


def format_sae_id(template: str, *, hidden_layer: int) -> str:
    sae_layer = hidden_layer_to_sae_layer(hidden_layer)
    return str(template).format(hidden_layer=int(hidden_layer), sae_layer=int(sae_layer))


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) == 0 or len(y) == 0:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    value = pd.Series(x).corr(pd.Series(y), method="spearman")
    return None if pd.isna(value) else float(value)


def _load_sae(*, release: str, sae_id: str, device: torch.device | str):
    try:
        from sae_lens import SAE
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sae-lens is required for D4 SAE analysis. Install it with "
            "`pip install sae-lens` or add the repo optional dependency `.[mech]`."
        ) from exc

    loaded = SAE.from_pretrained(release=str(release), sae_id=str(sae_id), device=str(device))
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae.eval()
    return sae


def _sae_d_sae(sae: Any) -> int:
    cfg = getattr(sae, "cfg", None)
    if cfg is not None and hasattr(cfg, "d_sae"):
        return int(getattr(cfg, "d_sae"))
    if hasattr(sae, "W_dec"):
        return int(getattr(sae, "W_dec").shape[0])
    raise RuntimeError("Could not infer SAE width.")


def _sae_encode(sae: Any, x: torch.Tensor) -> torch.Tensor:
    acts = sae.encode(x)
    if isinstance(acts, tuple):
        acts = acts[0]
    return acts.to(dtype=torch.float32)


@torch.inference_mode()
def _aggregate_sae_features(
    *,
    sae: Any,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    padding_side: str,
    aggregation: str,
    token_chunk_size: int,
) -> torch.Tensor:
    hidden = hidden.detach().to(dtype=torch.float32)
    aggregation = str(aggregation)
    if aggregation == "last":
        pooled = take_last_token(hidden, attention_mask, padding_side=padding_side)
        return _sae_encode(sae, pooled)

    if aggregation != "max":
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    batch, seq, dim = hidden.shape
    del dim
    d_sae = _sae_d_sae(sae)
    valid = attention_mask.to(dtype=torch.bool)
    flat_hidden = hidden.reshape(batch * seq, hidden.shape[-1])[valid.reshape(-1)]
    flat_ids = (
        torch.arange(batch, device=hidden.device).unsqueeze(1).expand(batch, seq).reshape(-1)[valid.reshape(-1)]
    )
    out = torch.full((batch, d_sae), -torch.inf, device=hidden.device, dtype=torch.float32)
    chunk_size = max(1, int(token_chunk_size))
    for start in range(0, flat_hidden.shape[0], chunk_size):
        end = min(flat_hidden.shape[0], start + chunk_size)
        acts = _sae_encode(sae, flat_hidden[start:end])
        ids = flat_ids[start:end]
        for item_id in torch.unique(ids).tolist():
            mask = ids == int(item_id)
            out[int(item_id)] = torch.maximum(out[int(item_id)], acts[mask].amax(dim=0))
    out[~torch.isfinite(out)] = 0.0
    return out


@torch.inference_mode()
def _encode_texts_sae(
    *,
    scorer,
    tokenizer,
    sae: Any,
    texts: list[str],
    hidden_layer: int,
    batch_size: int,
    max_length: int,
    aggregation: str,
    token_chunk_size: int,
) -> np.ndarray:
    device = next(p for p in scorer.parameters() if p.device.type != "meta").device
    padding_side = getattr(tokenizer, "padding_side", "right")
    outputs: list[np.ndarray] = []
    scorer.eval()

    for start in range(0, len(texts), int(batch_size)):
        batch = texts[start : start + int(batch_size)]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = scorer.backbone(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("Backbone did not return hidden_states.")
        hidden_layer = int(hidden_layer)
        if hidden_layer >= len(hs):
            raise ValueError(f"hidden_layer {hidden_layer} out of range for {len(hs)} hidden states.")
        feats = _aggregate_sae_features(
            sae=sae,
            hidden=hs[hidden_layer],
            attention_mask=enc["attention_mask"],
            padding_side=padding_side,
            aggregation=aggregation,
            token_chunk_size=int(token_chunk_size),
        )
        outputs.append(feats.detach().cpu().numpy().astype(np.float32, copy=False))
        del out, enc, feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not outputs:
        return np.zeros((0, _sae_d_sae(sae)), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def _rank_atom_features_for_layer(
    *,
    features: np.ndarray,
    probe_df: pd.DataFrame,
    atoms: list[str],
    hidden_layer: int,
    sae_release: str,
    sae_id: str,
    aggregation: str,
    min_train_examples: int,
    min_eval_examples: int,
    feature_candidate_k: int,
    top_k_features_per_atom: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train_mask = (probe_df["split"] == "train").to_numpy()
    val_mask = (probe_df["split"] == "val").to_numpy()
    test_mask = (probe_df["split"] == "test").to_numpy()
    eps = 1e-8

    for atom in atoms:
        label_col = f"{atom}__label"
        if label_col not in probe_df.columns:
            continue
        labels = probe_df[label_col].to_numpy(dtype=int)
        train_idx = np.where(train_mask & (labels >= 0))[0]
        val_idx = np.where(val_mask & (labels >= 0))[0]
        test_idx = np.where(test_mask & (labels >= 0))[0]
        if (
            len(train_idx) < int(min_train_examples)
            or len(val_idx) < int(min_eval_examples)
            or len(np.unique(labels[train_idx])) < 2
            or len(np.unique(labels[val_idx])) < 2
        ):
            rows.append(
                {
                    "atom": atom,
                    "hidden_layer": int(hidden_layer),
                    "sae_layer": hidden_layer_to_sae_layer(hidden_layer),
                    "sae_release": sae_release,
                    "sae_id": sae_id,
                    "aggregation": aggregation,
                    "feature_idx": None,
                    "status": "insufficient",
                    "train_n": int(len(train_idx)),
                    "val_n": int(len(val_idx)),
                    "test_n": int(len(test_idx)),
                }
            )
            continue

        x_train = features[train_idx]
        y_train = labels[train_idx]
        pos = x_train[y_train == 1]
        neg = x_train[y_train == 0]
        pos_mean = pos.mean(axis=0)
        neg_mean = neg.mean(axis=0)
        pos_var = pos.var(axis=0)
        neg_var = neg.var(axis=0)
        pooled = np.sqrt(0.5 * (pos_var + neg_var) + eps)
        signed_d = (pos_mean - neg_mean) / pooled
        n_candidates = min(int(feature_candidate_k), features.shape[1])
        candidate_idx = np.argsort(np.abs(signed_d))[-n_candidates:][::-1]

        ranked: list[dict[str, Any]] = []
        for feature_idx in candidate_idx.tolist():
            direction = 1.0 if signed_d[feature_idx] >= 0.0 else -1.0
            val_scores = direction * features[val_idx, feature_idx]
            test_scores = direction * features[test_idx, feature_idx] if len(test_idx) else np.asarray([])
            ranked.append(
                {
                    "atom": atom,
                    "hidden_layer": int(hidden_layer),
                    "sae_layer": hidden_layer_to_sae_layer(hidden_layer),
                    "sae_release": sae_release,
                    "sae_id": sae_id,
                    "aggregation": aggregation,
                    "feature_idx": int(feature_idx),
                    "feature_rank": 0,
                    "status": "ok",
                    "train_n": int(len(train_idx)),
                    "val_n": int(len(val_idx)),
                    "test_n": int(len(test_idx)),
                    "train_pos_mean": float(pos_mean[feature_idx]),
                    "train_neg_mean": float(neg_mean[feature_idx]),
                    "signed_cohen_d": float(signed_d[feature_idx]),
                    "abs_cohen_d": float(abs(signed_d[feature_idx])),
                    "direction": int(direction),
                    "val_auc": _safe_auc(labels[val_idx], val_scores),
                    "test_auc": _safe_auc(labels[test_idx], test_scores) if len(test_idx) else None,
                    "train_pos_activation_rate": float(np.mean(pos[:, feature_idx] > 0.0)),
                    "train_neg_activation_rate": float(np.mean(neg[:, feature_idx] > 0.0)),
                }
            )

        ranked.sort(
            key=lambda r: (
                -1.0 if r["val_auc"] is None else float(r["val_auc"]),
                -1.0 if r["test_auc"] is None else float(r["test_auc"]),
                float(r["abs_cohen_d"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked[: int(top_k_features_per_atom)], start=1):
            row["feature_rank"] = int(rank)
            rows.append(row)

    return pd.DataFrame(rows)


def _attach_laurito_transfer(
    *,
    feature_rows: pd.DataFrame,
    laurito_features: np.ndarray,
    laurito_df: pd.DataFrame,
) -> pd.DataFrame:
    if feature_rows.empty:
        return feature_rows.copy()
    out = feature_rows.copy()
    rhos: list[float | None] = []
    means: list[float | None] = []
    for row in out.itertuples(index=False):
        feature_idx = getattr(row, "feature_idx")
        atom = str(getattr(row, "atom"))
        if pd.isna(feature_idx) or atom not in laurito_df.columns:
            rhos.append(None)
            means.append(None)
            continue
        vals = laurito_features[:, int(feature_idx)]
        gold = laurito_df[atom].to_numpy(dtype=float)
        rhos.append(_safe_spearman(vals, gold))
        means.append(float(np.mean(vals)))
    out["laurito_spearman_with_atom_score"] = rhos
    out["laurito_mean_feature_activation"] = means
    return out


def _laurito_decision_alignment(
    *,
    feature_rows: pd.DataFrame,
    laurito_features: np.ndarray,
    laurito_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    if feature_rows.empty:
        return pd.DataFrame()
    if "run_id" in pair_df.columns:
        pairs = pair_df[pair_df["run_id"].astype(str) == str(run_id)].copy()
    else:
        pairs = pair_df.copy()
    if pairs.empty:
        return pd.DataFrame()

    feature_by_text_id = {
        str(text_id): laurito_features[idx]
        for idx, text_id in enumerate(laurito_df["text_id"].astype(str).tolist())
    }
    human_ids = pairs["human_text"].astype(str).map(_sha1_hex).tolist()
    llm_ids = pairs["llm_text"].astype(str).map(_sha1_hex).tolist()
    valid_pair_idx = [
        idx
        for idx, (human_id, llm_id) in enumerate(zip(human_ids, llm_ids, strict=True))
        if human_id in feature_by_text_id and llm_id in feature_by_text_id
    ]
    if not valid_pair_idx:
        return pd.DataFrame()

    human_mat = np.stack([feature_by_text_id[human_ids[idx]] for idx in valid_pair_idx], axis=0)
    llm_mat = np.stack([feature_by_text_id[llm_ids[idx]] for idx in valid_pair_idx], axis=0)
    pair_subset = pairs.iloc[valid_pair_idx]
    y = pair_subset["y_llm_chosen"].to_numpy(dtype=int)
    margin = pair_subset["llm_margin_pair"].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for row in feature_rows.itertuples(index=False):
        feature_idx = getattr(row, "feature_idx")
        if pd.isna(feature_idx):
            continue
        key = (int(getattr(row, "hidden_layer")), int(feature_idx), str(getattr(row, "aggregation")))
        if key in seen:
            continue
        seen.add(key)
        delta = llm_mat[:, int(feature_idx)] - human_mat[:, int(feature_idx)]
        rows.append(
            {
                "hidden_layer": int(getattr(row, "hidden_layer")),
                "sae_layer": int(getattr(row, "sae_layer")),
                "sae_release": str(getattr(row, "sae_release")),
                "sae_id": str(getattr(row, "sae_id")),
                "aggregation": str(getattr(row, "aggregation")),
                "feature_idx": int(feature_idx),
                "run_id": str(run_id),
                "n_pairs": int(len(delta)),
                "mean_llm_minus_human_activation": float(np.mean(delta)),
                "auc_llm_choice": _safe_auc(y, delta),
                "spearman_with_llm_margin": _safe_spearman(delta, margin),
            }
        )
    return pd.DataFrame(rows)


def _content_utility_overlap(
    *,
    feature_rows: pd.DataFrame,
    content_features: np.ndarray | None,
    content_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if feature_rows.empty or content_features is None or content_df is None or content_df.empty:
        return pd.DataFrame()
    labels = content_df["label"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for row in feature_rows.itertuples(index=False):
        feature_idx = getattr(row, "feature_idx")
        if pd.isna(feature_idx):
            continue
        key = (int(getattr(row, "hidden_layer")), int(feature_idx), str(getattr(row, "aggregation")))
        if key in seen:
            continue
        seen.add(key)
        vals = content_features[:, int(feature_idx)]
        rows.append(
            {
                "hidden_layer": int(getattr(row, "hidden_layer")),
                "sae_layer": int(getattr(row, "sae_layer")),
                "sae_release": str(getattr(row, "sae_release")),
                "sae_id": str(getattr(row, "sae_id")),
                "aggregation": str(getattr(row, "aggregation")),
                "feature_idx": int(feature_idx),
                "content_n_texts": int(len(vals)),
                "content_auc_chosen_vs_rejected": _safe_auc(labels, vals),
                "content_mean_chosen_activation": float(np.mean(vals[labels == 1])) if np.any(labels == 1) else None,
                "content_mean_rejected_activation": float(np.mean(vals[labels == 0])) if np.any(labels == 0) else None,
            }
        )
    return pd.DataFrame(rows)


def _feature_examples_for_layer(
    *,
    layer_rows: pd.DataFrame,
    probe_features: np.ndarray,
    probe_df: pd.DataFrame,
    laurito_features: np.ndarray,
    laurito_df: pd.DataFrame,
    max_features_per_atom_layer: int,
    top_examples: int,
) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    if layer_rows.empty:
        return examples
    ok = layer_rows[layer_rows["status"] == "ok"].copy()
    if ok.empty:
        return examples
    keep = ok.sort_values(["atom", "hidden_layer", "feature_rank"]).groupby(
        ["atom", "hidden_layer"], as_index=False
    ).head(int(max_features_per_atom_layer))
    for row in keep.itertuples(index=False):
        feature_idx = int(getattr(row, "feature_idx"))
        key = (
            f"{getattr(row, 'aggregation')}|hidden_{int(getattr(row, 'hidden_layer'))}|"
            f"feature_{feature_idx}|atom_{getattr(row, 'atom')}"
        )
        probe_order = np.argsort(probe_features[:, feature_idx])[::-1][: int(top_examples)]
        laurito_order = np.argsort(laurito_features[:, feature_idx])[::-1][: int(top_examples)]
        examples[key] = {
            "atom": str(getattr(row, "atom")),
            "hidden_layer": int(getattr(row, "hidden_layer")),
            "sae_layer": int(getattr(row, "sae_layer")),
            "sae_id": str(getattr(row, "sae_id")),
            "aggregation": str(getattr(row, "aggregation")),
            "feature_idx": feature_idx,
            "atom_probe": [
                {
                    "rank": int(rank + 1),
                    "activation": float(probe_features[idx, feature_idx]),
                    "item_type": str(probe_df.iloc[idx].get("item_type", "")),
                    "split": str(probe_df.iloc[idx].get("split", "")),
                    "source_dataset": str(probe_df.iloc[idx].get("source_dataset", "")),
                    "text_preview": str(probe_df.iloc[idx].get("text", ""))[:300],
                }
                for rank, idx in enumerate(probe_order.tolist())
            ],
            "laurito": [
                {
                    "rank": int(rank + 1),
                    "activation": float(laurito_features[idx, feature_idx]),
                    "item_type": str(laurito_df.iloc[idx].get("item_type", "")),
                    "source": str(laurito_df.iloc[idx].get("source", "")),
                    "title": str(laurito_df.iloc[idx].get("title", "")),
                    "text_preview": str(laurito_df.iloc[idx].get("text", ""))[:300],
                }
                for rank, idx in enumerate(laurito_order.tolist())
            ],
        }
    return examples


def _build_bundle_feature_scores(feature_rows: pd.DataFrame, trace_bundles: dict[str, Any]) -> pd.DataFrame:
    if feature_rows.empty:
        return pd.DataFrame()
    ok = feature_rows[feature_rows["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for bundle_id, payload in sorted(trace_bundles.items()):
        member_atoms = [str(atom) for atom in (payload.get("member_atoms") or [])]
        if not member_atoms:
            continue
        sub = ok[ok["atom"].isin(member_atoms)]
        if sub.empty:
            continue
        group_cols = ["hidden_layer", "sae_layer", "sae_release", "sae_id", "aggregation", "feature_idx"]
        for key, grp in sub.groupby(group_cols):
            hidden_layer, sae_layer, sae_release, sae_id, aggregation, feature_idx = key
            atoms = sorted(set(str(atom) for atom in grp["atom"].tolist()))
            rows.append(
                {
                    "bundle_id": str(bundle_id),
                    "status": str(payload.get("status") or ""),
                    "hidden_layer": int(hidden_layer),
                    "sae_layer": int(sae_layer),
                    "sae_release": str(sae_release),
                    "sae_id": str(sae_id),
                    "aggregation": str(aggregation),
                    "feature_idx": int(feature_idx),
                    "n_member_atoms_hit": int(len(atoms)),
                    "member_atoms_hit": ";".join(atoms),
                    "mean_abs_cohen_d": float(grp["abs_cohen_d"].astype(float).mean()),
                    "max_val_auc": float(grp["val_auc"].astype(float).max()),
                    "mean_laurito_abs_spearman": float(
                        grp["laurito_spearman_with_atom_score"].astype(float).abs().mean()
                    )
                    if "laurito_spearman_with_atom_score" in grp.columns
                    else None,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["n_member_atoms_hit", "mean_abs_cohen_d", "max_val_auc"], ascending=False
    )


def _feature_set_manifest(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    run_dir: Path,
    selected_layers: list[int],
    atom_feature_df: pd.DataFrame,
    bundle_feature_df: pd.DataFrame,
) -> dict[str, Any]:
    ok = atom_feature_df[atom_feature_df["status"] == "ok"].copy() if not atom_feature_df.empty else pd.DataFrame()
    top_by_atom: dict[str, list[dict[str, Any]]] = {}
    if not ok.empty:
        sort_cols = ["atom", "val_auc", "test_auc", "laurito_spearman_with_atom_score", "abs_cohen_d"]
        available_sort_cols = [col for col in sort_cols if col in ok.columns]
        sorted_ok = ok.sort_values(available_sort_cols, ascending=[True] + [False] * (len(available_sort_cols) - 1))
        for atom, grp in sorted_ok.groupby("atom"):
            top_by_atom[str(atom)] = grp.head(5).to_dict(orient="records")

    top_bundles = (
        bundle_feature_df.head(50).to_dict(orient="records") if not bundle_feature_df.empty else []
    )
    return {
        "stage": "D4-SAE-1",
        "manifest_json": str(manifest_path),
        "reward_run_dir": str(run_dir),
        "model_id": str(args.model_id),
        "sae_release": str(args.sae_release),
        "sae_id_template": str(args.sae_id_template),
        "aggregation": str(args.aggregation),
        "selected_hidden_layers": selected_layers,
        "selected_sae_layers": [hidden_layer_to_sae_layer(layer) for layer in selected_layers],
        "top_features_by_atom": top_by_atom,
        "top_bundle_feature_sets": top_bundles,
    }


def _write_tables(
    *,
    out_dir: Path,
    atom_feature_rows: list[pd.DataFrame],
    decision_rows: list[pd.DataFrame],
    content_rows: list[pd.DataFrame],
    examples: dict[str, Any],
    trace_bundles: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    atom_df = pd.concat(atom_feature_rows, ignore_index=True) if atom_feature_rows else pd.DataFrame()
    decision_df = pd.concat(decision_rows, ignore_index=True) if decision_rows else pd.DataFrame()
    content_df = pd.concat(content_rows, ignore_index=True) if content_rows else pd.DataFrame()

    if not decision_df.empty and not atom_df.empty:
        atom_df = atom_df.merge(
            decision_df,
            on=["hidden_layer", "sae_layer", "sae_release", "sae_id", "aggregation", "feature_idx"],
            how="left",
        )
    if not content_df.empty and not atom_df.empty:
        atom_df = atom_df.merge(
            content_df,
            on=["hidden_layer", "sae_layer", "sae_release", "sae_id", "aggregation", "feature_idx"],
            how="left",
        )

    bundle_df = _build_bundle_feature_scores(atom_df, trace_bundles=trace_bundles)
    atom_df.to_csv(out_dir / "sae_atom_feature_scores.csv", index=False)
    bundle_df.to_csv(out_dir / "sae_bundle_feature_scores.csv", index=False)
    decision_df.to_csv(out_dir / "sae_laurito_decision_alignment.csv", index=False)
    content_df.to_csv(out_dir / "sae_content_utility_overlap.csv", index=False)
    _write_json(out_dir / "sae_feature_examples.json", examples)
    return atom_df, bundle_df


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    manifest_path = _resolve_path(workspace_root, str(args.manifest_json))
    run_dir = _resolve_path(workspace_root, str(args.reward_run_dir))
    out_dir = _resolve_path(workspace_root, str(args.out_dir))
    if manifest_path is None or run_dir is None or out_dir is None:
        raise ValueError("Could not resolve key paths.")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(manifest_path)
    atom_probe_rows = _read_jsonl(Path(str(manifest["outputs"]["atom_probe_set"])))
    atom_probe_df = _sample_atom_probe_rows(
        atom_probe_rows,
        max_train_per_item_type=int(args.max_train_per_item_type),
        max_val_per_item_type=int(args.max_val_per_item_type),
        max_test_per_item_type=int(args.max_test_per_item_type),
    )
    manifest_atoms = [str(x) for x in manifest.get("d4_atoms", [])]
    atoms = _parse_str_list(args.atoms) or manifest_atoms
    missing_atoms = sorted(set(atoms) - set(manifest_atoms))
    if missing_atoms:
        raise ValueError(f"Requested atoms not present in D4 manifest: {missing_atoms}")
    atom_probe_df = _build_atom_label_frame(atom_probe_df, atoms=atoms, q=float(args.label_quantile))

    laurito_df = pd.read_csv(Path(str(manifest["outputs"]["laurito_text_atom_scores"])))
    pair_df = pd.read_csv(Path(str(manifest["outputs"]["laurito_pair_runs"])))
    canonical_run_id = str(manifest.get("canonical_text_score_run") or "")

    content_anchor_df: pd.DataFrame | None = None
    if not bool(args.disable_content_control):
        content_anchor_df = _flatten_content_pairs(
            _read_jsonl(Path(str(manifest["outputs"]["content_anchor_set"]))),
            seed=int(args.seed),
            max_pairs=int(args.content_max_pairs),
        )

    lora_adapter_dir = run_dir / "lora_adapter"
    value_head_path = run_dir / "value_head.pt"
    if not value_head_path.is_file():
        raise FileNotFoundError(f"Missing value head: {value_head_path}")
    scorer, tokenizer = load_reward_scorer(
        model_id=str(args.model_id),
        cache_dir=Path(args.cache_dir),
        lora_adapter_dir=lora_adapter_dir if lora_adapter_dir.is_dir() else None,
        value_head_path=value_head_path,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        use_4bit=bool(args.use_4bit),
    )

    selected_layers = _parse_int_list(args.selected_layers)
    num_layers = int(getattr(scorer.backbone.config, "num_hidden_layers", 0))
    selected_layers = [layer for layer in selected_layers if 1 <= int(layer) <= int(num_layers)]
    if not selected_layers:
        raise ValueError("No selected layers remain after model layer filtering.")

    device = next(p for p in scorer.parameters() if p.device.type != "meta").device
    atom_feature_rows: list[pd.DataFrame] = []
    decision_rows: list[pd.DataFrame] = []
    content_rows: list[pd.DataFrame] = []
    examples: dict[str, Any] = {}
    skipped_layers: list[dict[str, Any]] = []

    for hidden_layer in selected_layers:
        sae_id = format_sae_id(str(args.sae_id_template), hidden_layer=int(hidden_layer))
        try:
            sae = _load_sae(release=str(args.sae_release), sae_id=sae_id, device=device)
        except Exception as exc:
            if bool(args.skip_missing_sae):
                skipped_layers.append({"hidden_layer": int(hidden_layer), "sae_id": sae_id, "error": str(exc)})
                continue
            raise

        probe_features = _encode_texts_sae(
            scorer=scorer,
            tokenizer=tokenizer,
            sae=sae,
            texts=atom_probe_df["text"].astype(str).tolist(),
            hidden_layer=int(hidden_layer),
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            aggregation=str(args.aggregation),
            token_chunk_size=int(args.sae_token_chunk_size),
        )
        laurito_features = _encode_texts_sae(
            scorer=scorer,
            tokenizer=tokenizer,
            sae=sae,
            texts=laurito_df["text"].astype(str).tolist(),
            hidden_layer=int(hidden_layer),
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            aggregation=str(args.aggregation),
            token_chunk_size=int(args.sae_token_chunk_size),
        )
        content_features = None
        if content_anchor_df is not None and not content_anchor_df.empty:
            content_features = _encode_texts_sae(
                scorer=scorer,
                tokenizer=tokenizer,
                sae=sae,
                texts=content_anchor_df["text"].astype(str).tolist(),
                hidden_layer=int(hidden_layer),
                batch_size=int(args.batch_size),
                max_length=int(args.max_length),
                aggregation=str(args.aggregation),
                token_chunk_size=int(args.sae_token_chunk_size),
            )

        layer_atom_df = _rank_atom_features_for_layer(
            features=probe_features,
            probe_df=atom_probe_df,
            atoms=atoms,
            hidden_layer=int(hidden_layer),
            sae_release=str(args.sae_release),
            sae_id=sae_id,
            aggregation=str(args.aggregation),
            min_train_examples=int(args.min_train_examples),
            min_eval_examples=int(args.min_eval_examples),
            feature_candidate_k=int(args.feature_candidate_k),
            top_k_features_per_atom=int(args.top_k_features_per_atom),
        )
        layer_atom_df = _attach_laurito_transfer(
            feature_rows=layer_atom_df,
            laurito_features=laurito_features,
            laurito_df=laurito_df,
        )
        layer_decision_df = _laurito_decision_alignment(
            feature_rows=layer_atom_df,
            laurito_features=laurito_features,
            laurito_df=laurito_df,
            pair_df=pair_df,
            run_id=canonical_run_id,
        )
        layer_content_df = _content_utility_overlap(
            feature_rows=layer_atom_df,
            content_features=content_features,
            content_df=content_anchor_df,
        )
        examples.update(
            _feature_examples_for_layer(
                layer_rows=layer_atom_df,
                probe_features=probe_features,
                probe_df=atom_probe_df,
                laurito_features=laurito_features,
                laurito_df=laurito_df,
                max_features_per_atom_layer=int(args.example_features_per_atom_layer),
                top_examples=int(args.top_examples_per_feature),
            )
        )

        atom_feature_rows.append(layer_atom_df)
        if not layer_decision_df.empty:
            decision_rows.append(layer_decision_df)
        if not layer_content_df.empty:
            content_rows.append(layer_content_df)

        atom_df, bundle_df = _write_tables(
            out_dir=out_dir,
            atom_feature_rows=atom_feature_rows,
            decision_rows=decision_rows,
            content_rows=content_rows,
            examples=examples,
            trace_bundles=manifest.get("trace_bundles", {}),
        )
        feature_manifest = _feature_set_manifest(
            args=args,
            manifest_path=manifest_path,
            run_dir=run_dir,
            selected_layers=selected_layers,
            atom_feature_df=atom_df,
            bundle_feature_df=bundle_df,
        )
        feature_manifest["completed_hidden_layers"] = [
            int(df["hidden_layer"].iloc[0]) for df in atom_feature_rows if not df.empty
        ]
        feature_manifest["skipped_layers"] = skipped_layers
        feature_manifest["atom_probe_rows"] = int(len(atom_probe_df))
        feature_manifest["laurito_rows"] = int(len(laurito_df))
        feature_manifest["content_anchor_rows"] = 0 if content_anchor_df is None else int(len(content_anchor_df))
        _write_json(out_dir / "sae_feature_set_manifest.json", feature_manifest)
        print(f"Completed hidden layer {hidden_layer} with SAE {sae_id}")

        del sae, probe_features, laurito_features, content_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Wrote D4 SAE feature analysis outputs to {out_dir}")
    print(f"Wrote feature manifest to {out_dir / 'sae_feature_set_manifest.json'}")


if __name__ == "__main__":
    main()
