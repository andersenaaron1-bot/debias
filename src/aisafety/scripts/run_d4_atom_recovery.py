"""Run the first D4 atom-recovery layer sweep for a reward judge.

This is the first mechanistic implementation step after the D4 dataset pack is
frozen. It is intentionally J0-first:

- freeze a broad atom panel from the D4 pack
- sweep a sparse probe across selected residual layers
- identify which atoms are recoverable where
- measure ecological transfer to Laurito texts
- measure overlap with a content-anchor utility signal

This stage does not make causal claims yet. It is the layer/site selection
stage for later SAE feature analysis and mechanistic permutability checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.data.cue_corpus import assign_group_split
from aisafety.features.token_positions import take_last_token
from aisafety.reward.model import load_reward_scorer


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


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
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--layer-stride", type=int, default=4)
    p.add_argument("--tail-layers", type=int, default=4)
    p.add_argument("--max-train-per-item-type", type=int, default=1500)
    p.add_argument("--max-val-per-item-type", type=int, default=300)
    p.add_argument("--max-test-per-item-type", type=int, default=300)
    p.add_argument("--label-quantile", type=float, default=0.8)
    p.add_argument("--min-train-examples", type=int, default=60)
    p.add_argument("--min-eval-examples", type=int, default=20)
    p.add_argument("--content-max-pairs", type=int, default=4000)
    p.add_argument("--probe-c", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument(
        "--content-anchor-only",
        action="store_true",
        help="Only recompute the content-anchor utility control outputs.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("artifacts") / "mechanistic" / "d4_j0_atom_recovery_v1")
    return p.parse_args()


def select_hidden_layers(num_layers: int, *, stride: int, tail_layers: int) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    selected = {1, num_layers}
    step = max(1, int(stride))
    selected.update(range(step, num_layers + 1, step))
    tail = max(0, int(tail_layers))
    if tail:
        selected.update(range(max(1, num_layers - tail + 1), num_layers + 1))
    return sorted(int(x) for x in selected if 1 <= int(x) <= num_layers)


def _sample_atom_probe_rows(
    rows: list[dict[str, Any]],
    *,
    max_train_per_item_type: int,
    max_val_per_item_type: int,
    max_test_per_item_type: int,
) -> pd.DataFrame:
    limits = {
        "train": int(max_train_per_item_type),
        "val": int(max_val_per_item_type),
        "test": int(max_test_per_item_type),
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("split") or ""), str(row.get("item_type") or ""))
        grouped.setdefault(key, []).append(row)

    kept: list[dict[str, Any]] = []
    for (split, item_type), grp in grouped.items():
        limit = limits.get(split, 0)
        ordered = sorted(grp, key=lambda r: _sha1_hex(str(r.get("example_id") or r.get("text") or "")))
        if limit > 0:
            ordered = ordered[:limit]
        kept.extend(ordered)
    return pd.DataFrame(kept)


def _build_atom_label_frame(
    df: pd.DataFrame,
    *,
    atoms: list[str],
    q: float,
) -> pd.DataFrame:
    out = df.copy()
    for atom in atoms:
        label_col = f"{atom}__label"
        out[label_col] = -1
        for item_type, grp in out.groupby("item_type"):
            train_scores = grp.loc[grp["split"] == "train", "atom_scores"].map(lambda d: float((d or {}).get(atom, 0.0)))
            if train_scores.empty:
                continue
            lo = float(train_scores.quantile(1.0 - float(q)))
            hi = float(train_scores.quantile(float(q)))
            grp_scores = grp["atom_scores"].map(lambda d: float((d or {}).get(atom, 0.0)))
            pos_idx = grp.index[grp_scores >= hi]
            neg_idx = grp.index[grp_scores <= lo]
            out.loc[pos_idx, label_col] = 1
            out.loc[neg_idx, label_col] = 0
        out[f"{atom}__score"] = out["atom_scores"].map(lambda d: float((d or {}).get(atom, 0.0)))
    return out


def _raw_content_pair_id(row: dict[str, Any]) -> str:
    pair_id = str(row.get("pair_id") or "").strip()
    if pair_id and pair_id.lower() not in {"nan", "none", "null"}:
        return pair_id
    return ""


def _content_pair_id(row: dict[str, Any], *, index: int, id_counts: Counter[str]) -> str:
    pair_id = _raw_content_pair_id(row)
    if pair_id and id_counts[pair_id] == 1:
        return pair_id
    parts = [
        pair_id,
        str(row.get("source_dataset") or ""),
        str(row.get("domain") or ""),
        str(row.get("prompt") or ""),
        str(row.get("chosen_text") or row.get("chosen") or ""),
        str(row.get("rejected_text") or row.get("rejected") or ""),
        str(index),
    ]
    return f"synthetic:{_sha1_hex(chr(31).join(parts))}"


def _flatten_content_pairs(rows: list[dict[str, Any]], *, seed: int, max_pairs: int) -> pd.DataFrame:
    id_counts = Counter(_raw_content_pair_id(row) for row in rows)
    indexed_rows = [(_content_pair_id(row, index=i, id_counts=id_counts), row) for i, row in enumerate(rows)]
    ordered = sorted(indexed_rows, key=lambda item: _sha1_hex(item[0]))
    if max_pairs > 0:
        ordered = ordered[: int(max_pairs)]
    text_rows: list[dict[str, Any]] = []
    for pair_id, row in ordered:
        split = assign_group_split(pair_id, seed=int(seed), train_frac=0.8, val_frac=0.1)
        for label, key in ((1, "chosen_text"), (0, "rejected_text")):
            text_rows.append(
                {
                    "pair_id": pair_id,
                    "split": split,
                    "label": int(label),
                    "domain": row.get("domain"),
                    "source_dataset": row.get("source_dataset"),
                    "text": str(row.get(key) or ""),
                }
            )
    return pd.DataFrame(text_rows)


@torch.inference_mode()
def _encode_texts_by_layer(
    *,
    scorer,
    tokenizer,
    texts: list[str],
    selected_layers: list[int],
    batch_size: int,
    max_length: int,
) -> dict[int, np.ndarray]:
    device = next(p for p in scorer.parameters() if p.device.type != "meta").device
    padding_side = getattr(tokenizer, "padding_side", "right")
    outputs: dict[int, list[np.ndarray]] = {int(layer): [] for layer in selected_layers}
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
        for layer in selected_layers:
            pooled = take_last_token(hs[int(layer)], enc["attention_mask"], padding_side=padding_side)
            outputs[int(layer)].append(pooled.detach().to(dtype=torch.float32, device="cpu").numpy())
        del out, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {layer: np.concatenate(chunks, axis=0) if chunks else np.zeros((0, scorer.hidden_size), dtype=np.float32) for layer, chunks in outputs.items()}


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def _fit_sparse_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    seed: int,
    c_value: float,
) -> tuple[Pipeline, dict[str, float | int | None]]:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=int(seed),
                    max_iter=2000,
                    C=float(c_value),
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    train_score = pipe.decision_function(x_train)
    eval_score = pipe.decision_function(x_eval)
    eval_pred = pipe.predict(x_eval)
    clf: LogisticRegression = pipe.named_steps["clf"]
    nnz = int(np.count_nonzero(np.abs(clf.coef_[0]) > 1e-8))
    metrics: dict[str, float | int | None] = {
        "train_auc": _safe_auc(y_train, train_score),
        "eval_auc": _safe_auc(y_eval, eval_score),
        "eval_accuracy": float(accuracy_score(y_eval, eval_pred)),
        "nnz": nnz,
    }
    return pipe, metrics


def _run_atom_layer_sweep(
    *,
    probe_df: pd.DataFrame,
    layer_features: dict[int, np.ndarray],
    atoms: list[str],
    selected_layers: list[int],
    min_train_examples: int,
    min_eval_examples: int,
    seed: int,
    c_value: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_models: dict[str, dict[str, Any]] = {}

    train_mask = probe_df["split"] == "train"
    val_mask = probe_df["split"] == "val"
    test_mask = probe_df["split"] == "test"

    for atom in atoms:
        label_col = f"{atom}__label"
        labels = probe_df[label_col].to_numpy(dtype=int)
        for layer in selected_layers:
            x = layer_features[int(layer)]
            train_idx = np.where(train_mask.to_numpy() & (labels >= 0))[0]
            val_idx = np.where(val_mask.to_numpy() & (labels >= 0))[0]
            test_idx = np.where(test_mask.to_numpy() & (labels >= 0))[0]
            if (
                len(train_idx) < int(min_train_examples)
                or len(val_idx) < int(min_eval_examples)
                or len(np.unique(labels[train_idx])) < 2
                or len(np.unique(labels[val_idx])) < 2
            ):
                rows.append(
                    {
                        "atom": atom,
                        "layer": int(layer),
                        "train_n": int(len(train_idx)),
                        "val_n": int(len(val_idx)),
                        "test_n": int(len(test_idx)),
                        "val_auc": None,
                        "test_auc": None,
                        "val_accuracy": None,
                        "test_accuracy": None,
                        "nnz": None,
                        "status": "insufficient",
                    }
                )
                continue

            pipe, val_metrics = _fit_sparse_probe(
                x[train_idx],
                labels[train_idx],
                x[val_idx],
                labels[val_idx],
                seed=int(seed),
                c_value=float(c_value),
            )
            test_auc = None
            test_accuracy = None
            if len(test_idx) >= int(min_eval_examples) and len(np.unique(labels[test_idx])) >= 2:
                test_score = pipe.decision_function(x[test_idx])
                test_pred = pipe.predict(x[test_idx])
                test_auc = _safe_auc(labels[test_idx], test_score)
                test_accuracy = float(accuracy_score(labels[test_idx], test_pred))

            row = {
                "atom": atom,
                "layer": int(layer),
                "train_n": int(len(train_idx)),
                "val_n": int(len(val_idx)),
                "test_n": int(len(test_idx)),
                "val_auc": val_metrics["eval_auc"],
                "test_auc": test_auc,
                "val_accuracy": val_metrics["eval_accuracy"],
                "test_accuracy": test_accuracy,
                "nnz": int(val_metrics["nnz"]) if val_metrics["nnz"] is not None else None,
                "status": "ok",
            }
            rows.append(row)

            current = best_models.get(atom)
            current_key = (
                -1.0 if current is None or current["row"]["val_auc"] is None else float(current["row"]["val_auc"]),
                -1.0 if current is None or current["row"]["test_auc"] is None else float(current["row"]["test_auc"]),
            )
            new_key = (
                -1.0 if row["val_auc"] is None else float(row["val_auc"]),
                -1.0 if row["test_auc"] is None else float(row["test_auc"]),
            )
            if current is None or new_key > current_key:
                best_models[atom] = {"layer": int(layer), "pipe": pipe, "row": row}

    return pd.DataFrame(rows), best_models


def _run_content_anchor_utility_sweep(
    *,
    content_df: pd.DataFrame,
    layer_features: dict[int, np.ndarray],
    selected_layers: list[int],
    min_train_examples: int,
    min_eval_examples: int,
    seed: int,
    c_value: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train_mask = content_df["split"] == "train"
    val_mask = content_df["split"] == "val"
    test_mask = content_df["split"] == "test"
    labels = content_df["label"].to_numpy(dtype=int)

    for layer in selected_layers:
        x = layer_features[int(layer)]
        train_idx = np.where(train_mask.to_numpy())[0]
        val_idx = np.where(val_mask.to_numpy())[0]
        test_idx = np.where(test_mask.to_numpy())[0]
        if (
            len(train_idx) < int(min_train_examples)
            or len(val_idx) < int(min_eval_examples)
            or len(np.unique(labels[train_idx])) < 2
            or len(np.unique(labels[val_idx])) < 2
        ):
            rows.append(
                {
                    "layer": int(layer),
                    "train_n": int(len(train_idx)),
                    "val_n": int(len(val_idx)),
                    "test_n": int(len(test_idx)),
                    "val_auc": None,
                    "test_auc": None,
                    "val_accuracy": None,
                    "test_accuracy": None,
                    "nnz": None,
                    "status": "insufficient",
                }
            )
            continue

        pipe, val_metrics = _fit_sparse_probe(
            x[train_idx],
            labels[train_idx],
            x[val_idx],
            labels[val_idx],
            seed=int(seed),
            c_value=float(c_value),
        )
        test_auc = None
        test_accuracy = None
        if len(test_idx) >= int(min_eval_examples) and len(np.unique(labels[test_idx])) >= 2:
            test_score = pipe.decision_function(x[test_idx])
            test_pred = pipe.predict(x[test_idx])
            test_auc = _safe_auc(labels[test_idx], test_score)
            test_accuracy = float(accuracy_score(labels[test_idx], test_pred))
        rows.append(
            {
                "layer": int(layer),
                "train_n": int(len(train_idx)),
                "val_n": int(len(val_idx)),
                "test_n": int(len(test_idx)),
                "val_auc": val_metrics["eval_auc"],
                "test_auc": test_auc,
                "val_accuracy": val_metrics["eval_accuracy"],
                "test_accuracy": test_accuracy,
                "nnz": int(val_metrics["nnz"]) if val_metrics["nnz"] is not None else None,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _best_content_anchor_layer(utility_df: pd.DataFrame) -> dict[str, Any] | None:
    if utility_df.empty or utility_df["val_auc"].dropna().empty:
        return None
    return utility_df.sort_values(["val_auc", "test_auc"], ascending=False).iloc[0].to_dict()


def _evaluate_laurito_transfer(
    *,
    laurito_df: pd.DataFrame,
    layer_features: dict[int, np.ndarray],
    best_models: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    top_examples: dict[str, list[dict[str, Any]]] = {}
    for atom, payload in best_models.items():
        layer = int(payload["layer"])
        pipe: Pipeline = payload["pipe"]
        scores = pipe.decision_function(layer_features[layer])
        gold = laurito_df[atom].to_numpy(dtype=float)
        rho = pd.Series(scores).corr(pd.Series(gold), method="spearman")
        rows.append(
            {
                "atom": atom,
                "best_layer": int(layer),
                "spearman_with_atom_score": None if pd.isna(rho) else float(rho),
                "mean_probe_score": float(np.mean(scores)),
            }
        )
        order = np.argsort(scores)[::-1][:10]
        top_examples[atom] = [
            {
                "rank": int(rank + 1),
                "title": str(laurito_df.iloc[idx]["title"]),
                "item_type": str(laurito_df.iloc[idx]["item_type"]),
                "source": str(laurito_df.iloc[idx]["source"]),
                "atom_score": float(laurito_df.iloc[idx][atom]),
                "probe_score": float(scores[idx]),
                "text_preview": str(laurito_df.iloc[idx]["text"])[:300],
            }
            for rank, idx in enumerate(order.tolist())
        ]
    return pd.DataFrame(rows), top_examples


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
    atom_probe_path = Path(str(manifest["outputs"]["atom_probe_set"]))
    content_anchor_path = Path(str(manifest["outputs"]["content_anchor_set"]))
    laurito_text_scores_path = Path(str(manifest["outputs"]["laurito_text_atom_scores"]))

    atom_probe_rows = _read_jsonl(atom_probe_path)
    atom_probe_df = _sample_atom_probe_rows(
        atom_probe_rows,
        max_train_per_item_type=int(args.max_train_per_item_type),
        max_val_per_item_type=int(args.max_val_per_item_type),
        max_test_per_item_type=int(args.max_test_per_item_type),
    )
    d4_atoms = [str(x) for x in manifest.get("d4_atoms", [])]
    atom_probe_df = _build_atom_label_frame(atom_probe_df, atoms=d4_atoms, q=float(args.label_quantile))

    content_anchor_rows = _read_jsonl(content_anchor_path)
    content_anchor_df = _flatten_content_pairs(
        content_anchor_rows,
        seed=int(args.seed),
        max_pairs=int(args.content_max_pairs),
    )

    laurito_df = pd.read_csv(laurito_text_scores_path)

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

    num_layers = int(getattr(scorer.backbone.config, "num_hidden_layers", 0))
    if num_layers <= 0:
        raise RuntimeError("Could not infer num_hidden_layers from backbone config.")
    selected_layers = select_hidden_layers(num_layers, stride=int(args.layer_stride), tail_layers=int(args.tail_layers))

    if bool(args.content_anchor_only):
        content_anchor_feats = _encode_texts_by_layer(
            scorer=scorer,
            tokenizer=tokenizer,
            texts=content_anchor_df["text"].astype(str).tolist(),
            selected_layers=selected_layers,
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
        )
        utility_df = _run_content_anchor_utility_sweep(
            content_df=content_anchor_df,
            layer_features=content_anchor_feats,
            selected_layers=selected_layers,
            min_train_examples=int(args.min_train_examples),
            min_eval_examples=int(args.min_eval_examples),
            seed=int(args.seed),
            c_value=float(args.probe_c),
        )
        utility_df.to_csv(out_dir / "content_anchor_utility_metrics.csv", index=False)
        utility_summary = {
            "manifest_json": str(manifest_path),
            "reward_run_dir": str(run_dir),
            "model_id": str(args.model_id),
            "selected_layers": selected_layers,
            "num_layers": int(num_layers),
            "content_anchor_rows": int(len(content_anchor_df)),
            "content_anchor_split_counts": {
                str(k): int(v) for k, v in content_anchor_df["split"].value_counts().sort_index().items()
            },
            "best_content_anchor_layer": _best_content_anchor_layer(utility_df),
        }
        _write_json(out_dir / "content_anchor_utility_summary.json", utility_summary)
        print(f"Wrote content-anchor utility outputs to {out_dir}")
        print(f"Wrote summary to {out_dir / 'content_anchor_utility_summary.json'}")
        return

    atom_probe_feats = _encode_texts_by_layer(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=atom_probe_df["text"].astype(str).tolist(),
        selected_layers=selected_layers,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    content_anchor_feats = _encode_texts_by_layer(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=content_anchor_df["text"].astype(str).tolist(),
        selected_layers=selected_layers,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    laurito_feats = _encode_texts_by_layer(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=laurito_df["text"].astype(str).tolist(),
        selected_layers=selected_layers,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )

    atom_metrics_df, best_models = _run_atom_layer_sweep(
        probe_df=atom_probe_df,
        layer_features=atom_probe_feats,
        atoms=d4_atoms,
        selected_layers=selected_layers,
        min_train_examples=int(args.min_train_examples),
        min_eval_examples=int(args.min_eval_examples),
        seed=int(args.seed),
        c_value=float(args.probe_c),
    )
    utility_df = _run_content_anchor_utility_sweep(
        content_df=content_anchor_df,
        layer_features=content_anchor_feats,
        selected_layers=selected_layers,
        min_train_examples=int(args.min_train_examples),
        min_eval_examples=int(args.min_eval_examples),
        seed=int(args.seed),
        c_value=float(args.probe_c),
    )
    laurito_transfer_df, top_examples = _evaluate_laurito_transfer(
        laurito_df=laurito_df,
        layer_features=laurito_feats,
        best_models=best_models,
    )

    best_rows = [payload["row"] for payload in best_models.values()]
    best_layers_df = pd.DataFrame(best_rows).sort_values(["val_auc", "test_auc"], ascending=False).reset_index(drop=True)

    atom_metrics_df.to_csv(out_dir / "atom_probe_metrics.csv", index=False)
    utility_df.to_csv(out_dir / "content_anchor_utility_metrics.csv", index=False)
    laurito_transfer_df.to_csv(out_dir / "laurito_transfer_metrics.csv", index=False)
    best_layers_df.to_csv(out_dir / "best_layers_by_atom.csv", index=False)
    _write_json(out_dir / "top_examples_by_atom.json", top_examples)

    summary = {
        "manifest_json": str(manifest_path),
        "reward_run_dir": str(run_dir),
        "model_id": str(args.model_id),
        "selected_layers": selected_layers,
        "num_layers": int(num_layers),
        "atom_probe_rows": int(len(atom_probe_df)),
        "content_anchor_rows": int(len(content_anchor_df)),
        "laurito_rows": int(len(laurito_df)),
        "d4_atoms": d4_atoms,
        "best_atoms_by_val_auc": best_layers_df.head(15).to_dict(orient="records"),
        "best_content_anchor_layer": _best_content_anchor_layer(utility_df),
    }
    _write_json(out_dir / "summary.json", summary)

    print(f"Wrote D4 atom recovery outputs to {out_dir}")
    print(f"Wrote summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
