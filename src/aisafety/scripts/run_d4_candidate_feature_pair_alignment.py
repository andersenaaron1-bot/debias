"""Run broad human-vs-LLM judge alignment for frozen D4 SAE candidates.

This pass answers a different question than SAE discovery. Discovery asks
whether atom labels have sparse feature detectors. This pass asks whether those
candidate features separate human from LLM text and whether their paired
activation deltas align with a judge's LLM-vs-human reward margin.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.mech.labels import parse_int_list
from aisafety.mech.metrics import safe_auc, safe_spearman
from aisafety.mech.sae import (
    aggregate_sae_features,
    format_sae_id,
    hidden_layer_to_sae_layer,
    load_sae,
    sae_d_sae,
)
from aisafety.reward.model import load_reward_scorer


DEFAULT_SOURCE_DIR = Path("artifacts") / "mechanistic" / "d4_j0_sae_merged_ontology_discovery_v1"
DEFAULT_PAIR_JSONL = Path("data") / "derived" / "d4_human_llm_alignment_pairs_v1" / "pairs.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--candidate-source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--candidate-registry-csv",
        type=Path,
        default=None,
        help="Optional frozen registry. If omitted, one is built from merged SAE discovery outputs.",
    )
    parser.add_argument("--reward-run-dir", type=Path, default=Path("artifacts") / "reward" / "j0_anchor_v1_h100compact")
    parser.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--rescore-pairs", action="store_true")

    parser.add_argument("--sae-release", type=str, default="gemma-scope-9b-pt-res-canonical")
    parser.add_argument("--sae-id-template", type=str, default="layer_{sae_layer}/width_16k/canonical")
    parser.add_argument("--aggregation", choices=["last", "max"], default="max")
    parser.add_argument(
        "--selected-layers",
        type=str,
        default="",
        help="Optional comma-separated hidden-state layers. Defaults to all candidate layers.",
    )
    parser.add_argument("--skip-missing-sae", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--sae-batch-size", type=int, default=4)
    parser.add_argument("--sae-token-chunk-size", type=int, default=1024)

    parser.add_argument("--max-candidates", type=int, default=900)
    parser.add_argument("--max-features-per-layer", type=int, default=90)
    parser.add_argument("--min-atom-val-auc", type=float, default=0.75)
    parser.add_argument("--min-atom-test-auc", type=float, default=0.70)
    parser.add_argument("--min-abs-cohen-d", type=float, default=0.80)
    parser.add_argument("--min-laurito-abs-spearman", type=float, default=0.08)
    parser.add_argument("--min-laurito-auc-delta", type=float, default=0.03)
    parser.add_argument("--min-bundle-member-atoms", type=int, default=2)
    parser.add_argument(
        "--atoms",
        type=str,
        default="",
        help="Optional comma-separated atom subset for candidate registry construction.",
    )
    parser.add_argument("--random-controls-per-layer", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source-min-pairs", type=int, default=25)
    parser.add_argument("--top-examples-per-feature", type=int, default=3)
    parser.add_argument("--example-top-features", type=int, default=120)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts") / "mechanistic" / "d4_j0_human_llm_candidate_alignment_v1",
    )
    return parser.parse_args()


def _norm(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split()).strip()


def _csv_set(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def _optional_int_list(value: str) -> list[int]:
    if not str(value or "").strip():
        return []
    return parse_int_list(str(value))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not np.isfinite(out):
        return None
    return out


def _int_or_none(value: Any) -> int | None:
    val = _float_or_none(value)
    if val is None:
        return None
    return int(val)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).is_file():
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_key(row: dict[str, Any]) -> tuple[int, int, str, str, str]:
    hidden_layer = _int_or_none(row.get("hidden_layer"))
    feature_idx = _int_or_none(row.get("feature_idx"))
    if hidden_layer is None or feature_idx is None:
        raise ValueError(f"Candidate row lacks hidden_layer or feature_idx: {row}")
    aggregation = str(row.get("aggregation") or "max")
    sae_release = str(row.get("sae_release") or "")
    sae_id = str(row.get("sae_id") or "")
    return hidden_layer, feature_idx, aggregation, sae_release, sae_id


def _new_candidate_entry(
    *,
    hidden_layer: int,
    feature_idx: int,
    aggregation: str,
    sae_release: str,
    sae_id: str,
) -> dict[str, Any]:
    return {
        "hidden_layer": int(hidden_layer),
        "sae_layer": hidden_layer_to_sae_layer(int(hidden_layer)),
        "sae_release": str(sae_release),
        "sae_id": str(sae_id),
        "aggregation": str(aggregation),
        "feature_idx": int(feature_idx),
        "atoms": set(),
        "bundles": set(),
        "member_atoms": set(),
        "source_runs": set(),
        "candidate_reasons": set(),
        "max_atom_val_auc": None,
        "max_atom_test_auc": None,
        "max_abs_cohen_d": None,
        "max_laurito_abs_spearman": None,
        "max_laurito_auc_delta": None,
        "max_laurito_abs_margin_spearman": None,
        "max_content_auc_delta": None,
        "max_train_pos_activation_rate": None,
        "max_train_neg_activation_rate": None,
        "max_bundle_member_atoms": 0,
        "candidate_score": 0.0,
        "candidate_kind": "discovered",
    }


def _update_max(entry: dict[str, Any], key: str, value: float | None, *, abs_value: bool = False) -> None:
    if value is None:
        return
    value = abs(float(value)) if abs_value else float(value)
    old = entry.get(key)
    if old is None or float(value) > float(old):
        entry[key] = float(value)


def _entry_score(entry: dict[str, Any]) -> float:
    val = float(entry.get("max_atom_val_auc") or 0.5)
    test = float(entry.get("max_atom_test_auc") or 0.5)
    d_val = min(float(entry.get("max_abs_cohen_d") or 0.0), 4.0) / 4.0
    transfer = min(float(entry.get("max_laurito_abs_spearman") or 0.0), 1.0)
    decision_auc = min(float(entry.get("max_laurito_auc_delta") or 0.0) * 2.0, 1.0)
    decision_rho = min(float(entry.get("max_laurito_abs_margin_spearman") or 0.0), 1.0)
    content_auc = min(float(entry.get("max_content_auc_delta") or 0.0) * 2.0, 1.0)
    bundle = min(int(entry.get("max_bundle_member_atoms") or 0), 5) / 5.0
    score = (
        max(val - 0.5, 0.0) * 2.0
        + max(test - 0.5, 0.0)
        + d_val
        + transfer
        + decision_auc
        + decision_rho
        + 0.5 * bundle
        - 0.25 * content_auc
    )
    return float(score)


def _finalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    out["atoms"] = ";".join(sorted(str(x) for x in entry.get("atoms", set()) if str(x)))
    out["bundles"] = ";".join(sorted(str(x) for x in entry.get("bundles", set()) if str(x)))
    out["member_atoms"] = ";".join(sorted(str(x) for x in entry.get("member_atoms", set()) if str(x)))
    out["source_runs"] = ";".join(sorted(str(x) for x in entry.get("source_runs", set()) if str(x)))
    out["candidate_reasons"] = ";".join(
        sorted(str(x) for x in entry.get("candidate_reasons", set()) if str(x))
    )
    pos_rate = _float_or_none(out.get("max_train_pos_activation_rate"))
    neg_rate = _float_or_none(out.get("max_train_neg_activation_rate"))
    out["always_on_atom_probe_risk"] = bool(
        pos_rate is not None and neg_rate is not None and pos_rate >= 0.995 and neg_rate >= 0.995
    )
    out["candidate_score"] = _entry_score(out)
    out["candidate_id"] = (
        f"hidden_{int(out['hidden_layer'])}|feature_{int(out['feature_idx'])}|"
        f"aggregation_{out['aggregation']}"
    )
    return out


def build_candidate_registry(
    *,
    source_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Build a broad frozen registry from merged SAE discovery outputs."""

    atom_subset = _csv_set(str(args.atoms))
    atom_rows = _read_csv(source_dir / "merged_sae_atom_feature_scores.csv")
    bundle_rows = _read_csv(source_dir / "merged_sae_bundle_feature_scores.csv")
    candidates: dict[tuple[int, int, str, str, str], dict[str, Any]] = {}

    for row in atom_rows:
        if str(row.get("status") or "ok") != "ok" or not str(row.get("feature_idx") or "").strip():
            continue
        atom = str(row.get("atom") or "").strip()
        if atom_subset and atom not in atom_subset:
            continue
        try:
            key = _candidate_key(row)
        except ValueError:
            continue
        entry = candidates.setdefault(
            key,
            _new_candidate_entry(
                hidden_layer=key[0],
                feature_idx=key[1],
                aggregation=key[2],
                sae_release=key[3] or str(args.sae_release),
                sae_id=key[4],
            ),
        )
        if atom:
            entry["atoms"].add(atom)
        if row.get("source_run"):
            entry["source_runs"].add(str(row["source_run"]))

        val_auc = _float_or_none(row.get("val_auc"))
        test_auc = _float_or_none(row.get("test_auc"))
        abs_d = _float_or_none(row.get("abs_cohen_d"))
        laurito_rho = _float_or_none(row.get("laurito_spearman_with_atom_score"))
        choice_auc = _float_or_none(row.get("auc_llm_choice"))
        margin_rho = _float_or_none(row.get("spearman_with_llm_margin"))
        content_auc = _float_or_none(
            row.get("content_auc_chosen_vs_rejected") or row.get("auc_chosen_utility")
        )
        _update_max(entry, "max_atom_val_auc", val_auc)
        _update_max(entry, "max_atom_test_auc", test_auc)
        _update_max(entry, "max_abs_cohen_d", abs_d)
        _update_max(entry, "max_laurito_abs_spearman", laurito_rho, abs_value=True)
        _update_max(entry, "max_laurito_auc_delta", None if choice_auc is None else abs(choice_auc - 0.5))
        _update_max(entry, "max_laurito_abs_margin_spearman", margin_rho, abs_value=True)
        _update_max(entry, "max_content_auc_delta", None if content_auc is None else abs(content_auc - 0.5))
        _update_max(entry, "max_train_pos_activation_rate", _float_or_none(row.get("train_pos_activation_rate")))
        _update_max(entry, "max_train_neg_activation_rate", _float_or_none(row.get("train_neg_activation_rate")))

        if val_auc is not None and val_auc >= float(args.min_atom_val_auc):
            entry["candidate_reasons"].add("atom_val_auc")
        if test_auc is not None and test_auc >= float(args.min_atom_test_auc):
            entry["candidate_reasons"].add("atom_test_auc")
        if abs_d is not None and abs_d >= float(args.min_abs_cohen_d):
            entry["candidate_reasons"].add("atom_effect_size")
        if laurito_rho is not None and abs(laurito_rho) >= float(args.min_laurito_abs_spearman):
            entry["candidate_reasons"].add("laurito_atom_transfer")
        if choice_auc is not None and abs(choice_auc - 0.5) >= float(args.min_laurito_auc_delta):
            entry["candidate_reasons"].add("laurito_decision_auc")
        if margin_rho is not None and abs(margin_rho) >= float(args.min_laurito_abs_spearman):
            entry["candidate_reasons"].add("laurito_margin_spearman")

    for row in bundle_rows:
        if not str(row.get("feature_idx") or "").strip():
            continue
        try:
            key = _candidate_key(row)
        except ValueError:
            continue
        entry = candidates.setdefault(
            key,
            _new_candidate_entry(
                hidden_layer=key[0],
                feature_idx=key[1],
                aggregation=key[2],
                sae_release=key[3] or str(args.sae_release),
                sae_id=key[4],
            ),
        )
        bundle_id = str(row.get("bundle_id") or "").strip()
        if bundle_id:
            entry["bundles"].add(bundle_id)
        if row.get("source_run"):
            entry["source_runs"].add(str(row["source_run"]))
        for atom in str(row.get("member_atoms_hit") or "").split(";"):
            atom = atom.strip()
            if atom:
                entry["member_atoms"].add(atom)
        n_member = _int_or_none(row.get("n_member_atoms_hit")) or 0
        if n_member > int(entry.get("max_bundle_member_atoms") or 0):
            entry["max_bundle_member_atoms"] = int(n_member)
        _update_max(entry, "max_abs_cohen_d", _float_or_none(row.get("mean_abs_cohen_d")))
        _update_max(entry, "max_atom_val_auc", _float_or_none(row.get("max_val_auc")))
        _update_max(
            entry,
            "max_laurito_abs_spearman",
            _float_or_none(row.get("mean_laurito_abs_spearman")),
            abs_value=True,
        )
        if n_member >= int(args.min_bundle_member_atoms):
            entry["candidate_reasons"].add("multi_atom_bundle")
        if bundle_id and n_member > 0:
            entry["candidate_reasons"].add("bundle_member")

    finalized = [
        _finalize_entry(entry)
        for entry in candidates.values()
        if entry.get("candidate_reasons") or entry.get("bundles") or entry.get("atoms")
    ]
    selected_layers = set(_optional_int_list(str(args.selected_layers)))
    if selected_layers:
        finalized = [row for row in finalized if int(row["hidden_layer"]) in selected_layers]

    finalized.sort(
        key=lambda row: (
            int(row["hidden_layer"]),
            -float(row.get("candidate_score") or 0.0),
            int(row["feature_idx"]),
        )
    )
    by_layer: Counter[int] = Counter()
    capped: list[dict[str, Any]] = []
    for row in sorted(finalized, key=lambda r: (-float(r.get("candidate_score") or 0.0), int(r["hidden_layer"]))):
        layer = int(row["hidden_layer"])
        if int(args.max_features_per_layer) > 0 and by_layer[layer] >= int(args.max_features_per_layer):
            continue
        capped.append(row)
        by_layer[layer] += 1
        if int(args.max_candidates) > 0 and len(capped) >= int(args.max_candidates):
            break
    capped.sort(key=lambda row: (int(row["hidden_layer"]), int(row["feature_idx"]), str(row["aggregation"])))
    return capped


def load_candidate_registry(args: argparse.Namespace, workspace_root: Path) -> list[dict[str, Any]]:
    if args.candidate_registry_csv is not None:
        path = resolve_path(workspace_root, args.candidate_registry_csv)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Candidate registry not found: {args.candidate_registry_csv}")
        rows = _read_csv(path)
        selected_layers = set(_optional_int_list(str(args.selected_layers)))
        out: list[dict[str, Any]] = []
        for row in rows:
            hidden_layer = _int_or_none(row.get("hidden_layer"))
            feature_idx = _int_or_none(row.get("feature_idx"))
            if hidden_layer is None or feature_idx is None:
                continue
            if selected_layers and hidden_layer not in selected_layers:
                continue
            row = dict(row)
            row["hidden_layer"] = hidden_layer
            row["feature_idx"] = feature_idx
            row["sae_layer"] = _int_or_none(row.get("sae_layer")) or hidden_layer_to_sae_layer(hidden_layer)
            row["aggregation"] = str(row.get("aggregation") or args.aggregation)
            row["sae_release"] = str(row.get("sae_release") or args.sae_release)
            row["sae_id"] = str(row.get("sae_id") or format_sae_id(str(args.sae_id_template), hidden_layer=hidden_layer))
            row["candidate_id"] = str(
                row.get("candidate_id")
                or f"hidden_{hidden_layer}|feature_{feature_idx}|aggregation_{row['aggregation']}"
            )
            out.append(row)
        return out

    source_dir = resolve_path(workspace_root, args.candidate_source_dir)
    if source_dir is None or not source_dir.is_dir():
        raise FileNotFoundError(f"Candidate source directory not found: {args.candidate_source_dir}")
    return build_candidate_registry(source_dir=source_dir, args=args)


def _read_pair_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(read_jsonl(path))
    if df.empty:
        raise ValueError(f"No pair rows found in {path}")
    if {"A_text", "B_text", "A_source", "B_source"}.issubset(df.columns):
        human_texts: list[str] = []
        llm_texts: list[str] = []
        for row in df.itertuples(index=False):
            a_source = str(getattr(row, "A_source"))
            b_source = str(getattr(row, "B_source"))
            if a_source == "human" and b_source == "llm":
                human_texts.append(str(getattr(row, "A_text")))
                llm_texts.append(str(getattr(row, "B_text")))
            elif a_source == "llm" and b_source == "human":
                human_texts.append(str(getattr(row, "B_text")))
                llm_texts.append(str(getattr(row, "A_text")))
            else:
                human_texts.append("")
                llm_texts.append("")
        df["human_text"] = human_texts
        df["llm_text"] = llm_texts
    required = {"human_text", "llm_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pair file missing required columns: {sorted(missing)}")

    out = df.copy()
    out["human_text"] = out["human_text"].map(_norm)
    out["llm_text"] = out["llm_text"].map(_norm)
    out = out[(out["human_text"] != "") & (out["llm_text"] != "")].copy()
    if out.empty:
        raise ValueError("No nonempty human/LLM pairs remain after normalization.")
    if "pair_id" not in out.columns:
        out["pair_id"] = [
            sha1_hex(f"{idx}|{human}|{llm}")
            for idx, human, llm in zip(out.index, out["human_text"], out["llm_text"], strict=False)
        ]
    for col, default in (
        ("source_dataset", "unknown"),
        ("item_type", "unknown"),
        ("subset", ""),
        ("split", ""),
        ("title", ""),
        ("question", ""),
        ("llm_generator", ""),
    ):
        if col not in out.columns:
            out[col] = default
    out["human_token_count"] = out["human_text"].map(lambda text: len(str(text).split()))
    out["llm_token_count"] = out["llm_text"].map(lambda text: len(str(text).split()))
    out["token_delta_llm_minus_human"] = out["llm_token_count"] - out["human_token_count"]
    out["char_delta_llm_minus_human"] = out["llm_text"].map(len) - out["human_text"].map(len)
    return out.reset_index(drop=True)


def _cap_pairs(df: pd.DataFrame, *, max_pairs: int, seed: int) -> pd.DataFrame:
    if int(max_pairs) <= 0 or len(df) <= int(max_pairs):
        return df.reset_index(drop=True)
    order = sorted(
        range(len(df)),
        key=lambda idx: sha1_hex(f"{seed}:pair-sample:{df.iloc[idx]['pair_id']}"),
    )
    chosen = set(order[: int(max_pairs)])
    return df.iloc[[idx for idx in range(len(df)) if idx in chosen]].reset_index(drop=True)


def _scores_present(df: pd.DataFrame) -> bool:
    if "llm_margin_pair" not in df.columns or "y_llm_chosen" not in df.columns:
        return False
    margins = pd.to_numeric(df["llm_margin_pair"], errors="coerce")
    labels = pd.to_numeric(df["y_llm_chosen"], errors="coerce")
    return bool(margins.notna().all() and labels.notna().all())


def _scorer_device(scorer: Any):
    import torch

    try:
        return next(p for p in scorer.parameters() if p.device.type != "meta").device
    except StopIteration:
        return torch.device("cpu")


def _load_scorer_and_tokenizer(args: argparse.Namespace, workspace_root: Path):
    import torch

    run_dir = resolve_path(workspace_root, args.reward_run_dir)
    if run_dir is None:
        raise ValueError("Could not resolve reward-run-dir.")
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
    scorer.eval()
    return scorer, tokenizer


def _score_texts(
    *,
    scorer: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch

    device = _scorer_device(scorer)
    outputs: list[np.ndarray] = []
    scorer.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), int(batch_size)):
            batch = texts[start : start + int(batch_size)]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            scores = scorer(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            outputs.append(scores.detach().float().cpu().numpy())
            del enc, scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0,), dtype=np.float32)


def _attach_pair_scores(
    *,
    pair_df: pd.DataFrame,
    scorer: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if _scores_present(pair_df) and not bool(args.rescore_pairs):
        out = pair_df.copy()
        out["llm_margin_pair"] = pd.to_numeric(out["llm_margin_pair"], errors="coerce").astype(float)
        out["y_llm_chosen"] = pd.to_numeric(out["y_llm_chosen"], errors="coerce").astype(int)
        return out

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
    out = pair_df.copy()
    out["human_reward"] = human_scores.astype(float)
    out["llm_reward"] = llm_scores.astype(float)
    out["llm_margin_pair"] = out["llm_reward"] - out["human_reward"]
    out["y_llm_chosen"] = (out["llm_margin_pair"] > 0.0).astype(int)
    return out


def _control_matrix(pair_df: pd.DataFrame) -> np.ndarray:
    numeric_cols = [
        "human_token_count",
        "llm_token_count",
        "token_delta_llm_minus_human",
        "char_delta_llm_minus_human",
    ]
    cols = [np.ones(len(pair_df), dtype=float)]
    for col in numeric_cols:
        vals = pd.to_numeric(pair_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        std = float(np.std(vals))
        cols.append((vals - float(np.mean(vals))) / std if std > 1e-12 else vals * 0.0)
    source_counts = pair_df["source_dataset"].astype(str).value_counts()
    common_sources = [src for src, count in source_counts.items() if int(count) >= 25][:24]
    for source in common_sources:
        cols.append((pair_df["source_dataset"].astype(str) == source).to_numpy(dtype=float))
    return np.stack(cols, axis=1)


def _residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if len(y) == 0:
        return y
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return y - design @ beta
    except np.linalg.LinAlgError:
        return y - float(np.mean(y))


def _spearman_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    rho = safe_spearman(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if rho is None:
        return None, None
    try:
        from scipy.stats import spearmanr

        res = spearmanr(x, y)
        p_value = None if res.pvalue is None or not np.isfinite(res.pvalue) else float(res.pvalue)
    except Exception:
        p_value = None
    return rho, p_value


def _auc_with_p(y_true: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None]:
    auc = safe_auc(np.asarray(y_true), np.asarray(scores, dtype=float))
    if auc is None:
        return None, None
    try:
        from scipy.stats import mannwhitneyu

        y = np.asarray(y_true)
        s = np.asarray(scores, dtype=float)
        pos = s[y == 1]
        neg = s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return auc, None
        res = mannwhitneyu(pos, neg, alternative="two-sided")
        p_value = None if not np.isfinite(res.pvalue) else float(res.pvalue)
    except Exception:
        p_value = None
    return auc, p_value


def _paired_effect_size(delta: np.ndarray) -> float | None:
    if len(delta) == 0:
        return None
    std = float(np.std(delta))
    if std <= 1e-12:
        return None
    return float(np.mean(delta) / std)


def _bh_q_values(p_values: list[float | None]) -> list[float | None]:
    indexed = [(idx, float(p)) for idx, p in enumerate(p_values) if p is not None and np.isfinite(float(p))]
    if not indexed:
        return [None for _ in p_values]
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    q_by_idx: dict[int, float] = {}
    running = 1.0
    for rank, (idx, p_value) in enumerate(reversed(indexed), start=1):
        original_rank = m - rank + 1
        running = min(running, p_value * m / float(original_rank))
        q_by_idx[idx] = min(1.0, running)
    return [q_by_idx.get(idx) for idx in range(len(p_values))]


def _add_q_values(rows: list[dict[str, Any]], *, p_col: str, q_col: str) -> None:
    q_values = _bh_q_values([_float_or_none(row.get(p_col)) for row in rows])
    for row, q_value in zip(rows, q_values, strict=True):
        row[q_col] = q_value


def _registry_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "candidate_id",
        "candidate_kind",
        "hidden_layer",
        "sae_layer",
        "sae_release",
        "sae_id",
        "aggregation",
        "feature_idx",
        "atoms",
        "bundles",
        "member_atoms",
        "candidate_reasons",
        "source_runs",
        "candidate_score",
        "max_atom_val_auc",
        "max_atom_test_auc",
        "max_abs_cohen_d",
        "max_laurito_abs_spearman",
        "max_laurito_auc_delta",
        "max_laurito_abs_margin_spearman",
        "max_content_auc_delta",
        "max_bundle_member_atoms",
        "max_train_pos_activation_rate",
        "max_train_neg_activation_rate",
        "always_on_atom_probe_risk",
    ]
    extra: list[str] = []
    for row in rows:
        for key in row:
            if key not in preferred and key not in extra:
                extra.append(key)
    return preferred + extra


def _alignment_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "candidate_id",
        "candidate_kind",
        "hidden_layer",
        "sae_layer",
        "sae_release",
        "sae_id",
        "aggregation",
        "feature_idx",
        "atoms",
        "bundles",
        "member_atoms",
        "n_pairs",
        "n_llm_chosen",
        "mean_llm_minus_human_activation",
        "paired_delta_d",
        "activation_auc_llm_vs_human",
        "activation_auc_llm_vs_human_p",
        "auc_j0_llm_choice_from_activation_delta",
        "auc_j0_llm_choice_p",
        "spearman_delta_with_j0_margin",
        "spearman_delta_with_j0_margin_p",
        "spearman_delta_with_j0_margin_q",
        "length_controlled_spearman_delta_with_j0_margin",
        "length_controlled_spearman_p",
        "length_controlled_spearman_q",
        "spearman_delta_with_length_delta",
        "mean_human_activation",
        "mean_llm_activation",
        "human_activation_rate",
        "llm_activation_rate",
        "source_sign_consistency",
        "n_sources_with_min_pairs",
        "candidate_score",
        "candidate_reasons",
        "always_on_atom_probe_risk",
    ]
    extra: list[str] = []
    for row in rows:
        for key in row:
            if key not in preferred and key not in extra:
                extra.append(key)
    return preferred + extra


def _random_control_candidates(
    *,
    layer: int,
    sae_release: str,
    sae_id: str,
    aggregation: str,
    width: int,
    existing: set[int],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if int(count) <= 0:
        return []
    candidates = [
        idx
        for idx in range(int(width))
        if idx not in existing
    ]
    candidates.sort(key=lambda idx: sha1_hex(f"{seed}:random-control:{layer}:{idx}"))
    rows: list[dict[str, Any]] = []
    for feature_idx in candidates[: int(count)]:
        rows.append(
            {
                "candidate_id": f"control|hidden_{layer}|feature_{feature_idx}|aggregation_{aggregation}",
                "candidate_kind": "random_control",
                "hidden_layer": int(layer),
                "sae_layer": hidden_layer_to_sae_layer(int(layer)),
                "sae_release": str(sae_release),
                "sae_id": str(sae_id),
                "aggregation": str(aggregation),
                "feature_idx": int(feature_idx),
                "atoms": "",
                "bundles": "",
                "member_atoms": "",
                "source_runs": "",
                "candidate_reasons": "matched_random_control",
                "candidate_score": 0.0,
                "always_on_atom_probe_risk": False,
            }
        )
    return rows


def _encode_texts_candidate_sae(
    *,
    scorer: Any,
    tokenizer: Any,
    sae: Any,
    texts: list[str],
    hidden_layer: int,
    feature_indices: list[int],
    batch_size: int,
    max_length: int,
    aggregation: str,
    token_chunk_size: int,
) -> np.ndarray:
    import torch

    if not texts:
        return np.zeros((0, len(feature_indices)), dtype=np.float32)
    device = _scorer_device(scorer)
    padding_side = getattr(tokenizer, "padding_side", "right")
    idx_tensor = torch.tensor(feature_indices, device=device, dtype=torch.long)
    outputs: list[np.ndarray] = []
    scorer.eval()

    with torch.inference_mode():
        for start in range(0, len(texts), int(batch_size)):
            batch = texts[start : start + int(batch_size)]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            out = scorer.backbone(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = out.hidden_states
            if hidden_states is None:
                raise RuntimeError("Backbone did not return hidden_states.")
            if int(hidden_layer) >= len(hidden_states):
                raise ValueError(f"hidden_layer {hidden_layer} out of range for {len(hidden_states)} hidden states.")
            full_features = aggregate_sae_features(
                sae=sae,
                hidden=hidden_states[int(hidden_layer)],
                attention_mask=enc["attention_mask"],
                padding_side=padding_side,
                aggregation=str(aggregation),
                token_chunk_size=int(token_chunk_size),
            )
            selected = full_features.index_select(dim=1, index=idx_tensor)
            outputs.append(selected.detach().cpu().numpy().astype(np.float32, copy=False))
            del out, enc, full_features, selected
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, len(feature_indices)), dtype=np.float32)


def _source_consistency(source_rows: list[dict[str, Any]], *, layer: int, feature_idx: int, min_pairs: int) -> tuple[float | None, int]:
    signs: list[int] = []
    for row in source_rows:
        if int(row.get("hidden_layer") or -1) != int(layer) or int(row.get("feature_idx") or -1) != int(feature_idx):
            continue
        if int(row.get("n_pairs") or 0) < int(min_pairs):
            continue
        value = _float_or_none(row.get("mean_llm_minus_human_activation"))
        if value is None or abs(value) <= 1e-12:
            continue
        signs.append(1 if value > 0 else -1)
    if not signs:
        return None, 0
    counts = Counter(signs)
    return float(max(counts.values()) / len(signs)), len(signs)


def _metric_rows_for_layer(
    *,
    layer_candidates: list[dict[str, Any]],
    pair_df: pd.DataFrame,
    human_mat: np.ndarray,
    llm_mat: np.ndarray,
    feature_to_col: dict[int, int],
    design: np.ndarray,
    source_min_pairs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y = pair_df["y_llm_chosen"].to_numpy(dtype=int)
    margin = pair_df["llm_margin_pair"].to_numpy(dtype=float)
    length_delta = pair_df["token_delta_llm_minus_human"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for cand in layer_candidates:
        feature_idx = int(cand["feature_idx"])
        if feature_idx not in feature_to_col:
            continue
        col = feature_to_col[feature_idx]
        human_vals = human_mat[:, col]
        llm_vals = llm_mat[:, col]
        delta = llm_vals - human_vals
        activation_labels = np.concatenate([np.zeros(len(human_vals), dtype=int), np.ones(len(llm_vals), dtype=int)])
        activation_values = np.concatenate([human_vals, llm_vals])
        text_auc, text_auc_p = _auc_with_p(activation_labels, activation_values)
        choice_auc, choice_auc_p = _auc_with_p(y, delta)
        rho, rho_p = _spearman_with_p(delta, margin)
        len_rho = safe_spearman(delta, length_delta)
        delta_resid = _residualize(delta, design)
        margin_resid = _residualize(margin, design)
        controlled_rho, controlled_p = _spearman_with_p(delta_resid, margin_resid)

        base = {
            "candidate_id": str(cand.get("candidate_id") or ""),
            "candidate_kind": str(cand.get("candidate_kind") or "discovered"),
            "hidden_layer": int(cand["hidden_layer"]),
            "sae_layer": int(cand.get("sae_layer") or hidden_layer_to_sae_layer(int(cand["hidden_layer"]))),
            "sae_release": str(cand.get("sae_release") or ""),
            "sae_id": str(cand.get("sae_id") or ""),
            "aggregation": str(cand.get("aggregation") or ""),
            "feature_idx": feature_idx,
            "atoms": str(cand.get("atoms") or ""),
            "bundles": str(cand.get("bundles") or ""),
            "member_atoms": str(cand.get("member_atoms") or ""),
            "candidate_score": _float_or_none(cand.get("candidate_score")),
            "candidate_reasons": str(cand.get("candidate_reasons") or ""),
            "always_on_atom_probe_risk": cand.get("always_on_atom_probe_risk", False),
        }
        rows.append(
            {
                **base,
                "n_pairs": int(len(delta)),
                "n_llm_chosen": int(np.sum(y)),
                "mean_llm_minus_human_activation": float(np.mean(delta)),
                "paired_delta_d": _paired_effect_size(delta),
                "activation_auc_llm_vs_human": text_auc,
                "activation_auc_llm_vs_human_p": text_auc_p,
                "auc_j0_llm_choice_from_activation_delta": choice_auc,
                "auc_j0_llm_choice_p": choice_auc_p,
                "spearman_delta_with_j0_margin": rho,
                "spearman_delta_with_j0_margin_p": rho_p,
                "length_controlled_spearman_delta_with_j0_margin": controlled_rho,
                "length_controlled_spearman_p": controlled_p,
                "spearman_delta_with_length_delta": len_rho,
                "mean_human_activation": float(np.mean(human_vals)),
                "mean_llm_activation": float(np.mean(llm_vals)),
                "human_activation_rate": float(np.mean(human_vals > 0.0)),
                "llm_activation_rate": float(np.mean(llm_vals > 0.0)),
            }
        )

        domain_groups = pair_df.groupby(
            [pair_df["source_dataset"].astype(str), pair_df["item_type"].astype(str)]
        ).groups
        for (source, item_type), idx in domain_groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            if len(idx_arr) < int(source_min_pairs):
                continue
            sub_delta = delta[idx_arr]
            sub_y = y[idx_arr]
            sub_margin = margin[idx_arr]
            sub_text_labels = np.concatenate(
                [np.zeros(len(idx_arr), dtype=int), np.ones(len(idx_arr), dtype=int)]
            )
            sub_text_vals = np.concatenate([human_vals[idx_arr], llm_vals[idx_arr]])
            sub_text_auc, _ = _auc_with_p(sub_text_labels, sub_text_vals)
            sub_choice_auc, _ = _auc_with_p(sub_y, sub_delta)
            sub_rho = safe_spearman(sub_delta, sub_margin)
            source_rows.append(
                {
                    **base,
                    "source_dataset": str(source),
                    "item_type": str(item_type),
                    "n_pairs": int(len(idx_arr)),
                    "n_llm_chosen": int(np.sum(sub_y)),
                    "mean_llm_minus_human_activation": float(np.mean(sub_delta)),
                    "activation_auc_llm_vs_human": sub_text_auc,
                    "auc_j0_llm_choice_from_activation_delta": sub_choice_auc,
                    "spearman_delta_with_j0_margin": sub_rho,
                }
            )

    for row in rows:
        consistency, n_sources = _source_consistency(
            source_rows,
            layer=int(row["hidden_layer"]),
            feature_idx=int(row["feature_idx"]),
            min_pairs=int(source_min_pairs),
        )
        row["source_sign_consistency"] = consistency
        row["n_sources_with_min_pairs"] = int(n_sources)
    return rows, source_rows


def _examples_for_layer(
    *,
    layer_rows: list[dict[str, Any]],
    pair_df: pd.DataFrame,
    human_mat: np.ndarray,
    llm_mat: np.ndarray,
    feature_to_col: dict[int, int],
    top_features: int,
    top_examples: int,
) -> dict[str, Any]:
    if int(top_examples) <= 0 or not layer_rows:
        return {}

    scored = sorted(
        layer_rows,
        key=lambda row: abs(
            float(
                row.get("length_controlled_spearman_delta_with_j0_margin")
                or row.get("spearman_delta_with_j0_margin")
                or 0.0
            )
        ),
        reverse=True,
    )[: int(top_features)]
    examples: dict[str, Any] = {}
    for row in scored:
        feature_idx = int(row["feature_idx"])
        if feature_idx not in feature_to_col:
            continue
        col = feature_to_col[feature_idx]
        delta = llm_mat[:, col] - human_mat[:, col]
        pos_order = np.argsort(delta)[::-1][: int(top_examples)]
        neg_order = np.argsort(delta)[: int(top_examples)]
        key = str(row["candidate_id"])
        examples[key] = {
            "candidate": {k: row.get(k) for k in _alignment_fieldnames([row]) if k in row},
            "largest_llm_minus_human": [
                _pair_example(pair_df.iloc[int(idx)], delta=float(delta[int(idx)]))
                for idx in pos_order.tolist()
            ],
            "largest_human_minus_llm": [
                _pair_example(pair_df.iloc[int(idx)], delta=float(delta[int(idx)]))
                for idx in neg_order.tolist()
            ],
        }
    return examples


def _pair_example(row: pd.Series, *, delta: float) -> dict[str, Any]:
    return {
        "pair_id": str(row.get("pair_id") or ""),
        "source_dataset": str(row.get("source_dataset") or ""),
        "item_type": str(row.get("item_type") or ""),
        "subset": str(row.get("subset") or ""),
        "llm_generator": str(row.get("llm_generator") or ""),
        "llm_margin_pair": _float_or_none(row.get("llm_margin_pair")),
        "y_llm_chosen": _int_or_none(row.get("y_llm_chosen")),
        "feature_delta": float(delta),
        "human_text_preview": str(row.get("human_text") or "")[:300],
        "llm_text_preview": str(row.get("llm_text") or "")[:300],
    }


def _write_outputs(
    *,
    out_dir: Path,
    registry_rows: list[dict[str, Any]],
    pair_df: pd.DataFrame,
    alignment_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    examples: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    _add_q_values(alignment_rows, p_col="spearman_delta_with_j0_margin_p", q_col="spearman_delta_with_j0_margin_q")
    _add_q_values(alignment_rows, p_col="length_controlled_spearman_p", q_col="length_controlled_spearman_q")
    _write_csv(out_dir / "candidate_feature_registry.csv", registry_rows, _registry_fieldnames(registry_rows))
    pair_cols = [
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
        "human_reward",
        "llm_reward",
        "llm_margin_pair",
        "y_llm_chosen",
    ]
    pair_cols = [col for col in pair_cols if col in pair_df.columns]
    pair_df[pair_cols].to_csv(out_dir / "pair_scores.csv", index=False)
    _write_csv(
        out_dir / "candidate_feature_human_llm_alignment.csv",
        alignment_rows,
        _alignment_fieldnames(alignment_rows),
    )
    _write_csv(out_dir / "candidate_feature_source_alignment.csv", source_rows)
    with (out_dir / "candidate_feature_examples.json").open("w", encoding="utf-8") as handle:
        json.dump(examples, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_json(out_dir / "alignment_manifest.json", manifest)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    out_dir = resolve_path(workspace_root, args.out_dir)
    pair_path = resolve_path(workspace_root, args.pair_jsonl)
    if out_dir is None or pair_path is None:
        raise ValueError("Could not resolve output or pair path.")
    out_dir.mkdir(parents=True, exist_ok=True)

    registry_rows = load_candidate_registry(args, workspace_root)
    if not registry_rows:
        raise ValueError("No candidate features were selected.")
    pair_df = _cap_pairs(_read_pair_file(pair_path), max_pairs=int(args.max_pairs), seed=int(args.seed))

    scorer, tokenizer = _load_scorer_and_tokenizer(args, workspace_root)
    pair_df = _attach_pair_scores(pair_df=pair_df, scorer=scorer, tokenizer=tokenizer, args=args)
    design = _control_matrix(pair_df)

    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        by_layer[int(row["hidden_layer"])].append(row)
    selected_layers = sorted(by_layer)

    text_by_id: dict[str, str] = {}
    for text in pair_df["human_text"].astype(str).tolist() + pair_df["llm_text"].astype(str).tolist():
        text_by_id.setdefault(sha1_hex(text), text)
    unique_ids = sorted(text_by_id)
    unique_texts = [text_by_id[text_id] for text_id in unique_ids]
    text_id_to_row = {text_id: idx for idx, text_id in enumerate(unique_ids)}
    human_rows = np.asarray([text_id_to_row[sha1_hex(text)] for text in pair_df["human_text"].astype(str)], dtype=int)
    llm_rows = np.asarray([text_id_to_row[sha1_hex(text)] for text in pair_df["llm_text"].astype(str)], dtype=int)

    all_alignment_rows: list[dict[str, Any]] = []
    all_source_rows: list[dict[str, Any]] = []
    examples: dict[str, Any] = {}
    completed_layers: list[int] = []
    skipped_layers: list[dict[str, Any]] = []
    registry_with_controls = list(registry_rows)

    manifest = {
        "stage": "D4-human-LLM-candidate-feature-alignment",
        "pair_jsonl": str(pair_path),
        "candidate_source_dir": str(resolve_path(workspace_root, args.candidate_source_dir)),
        "candidate_registry_csv": None if args.candidate_registry_csv is None else str(args.candidate_registry_csv),
        "reward_run_dir": str(resolve_path(workspace_root, args.reward_run_dir)),
        "model_id": str(args.model_id),
        "sae_release": str(args.sae_release),
        "sae_id_template": str(args.sae_id_template),
        "aggregation": str(args.aggregation),
        "n_pairs": int(len(pair_df)),
        "n_unique_texts": int(len(unique_texts)),
        "n_initial_candidates": int(len(registry_rows)),
        "random_controls_per_layer": int(args.random_controls_per_layer),
        "completed_hidden_layers": completed_layers,
        "skipped_layers": skipped_layers,
        "pair_counts_by_source_dataset": {
            str(k): int(v) for k, v in pair_df["source_dataset"].astype(str).value_counts().sort_index().items()
        },
        "y_llm_chosen_rate": float(pair_df["y_llm_chosen"].mean()),
        "mean_llm_margin_pair": float(pair_df["llm_margin_pair"].mean()),
    }

    for hidden_layer in selected_layers:
        layer_candidates = list(by_layer[int(hidden_layer)])
        sae_id = str(layer_candidates[0].get("sae_id") or format_sae_id(str(args.sae_id_template), hidden_layer=int(hidden_layer)))
        sae_release = str(layer_candidates[0].get("sae_release") or args.sae_release)
        try:
            sae = load_sae(release=sae_release, sae_id=sae_id, device=_scorer_device(scorer))
        except Exception as exc:
            if bool(args.skip_missing_sae):
                skipped_layers.append({"hidden_layer": int(hidden_layer), "sae_id": sae_id, "error": str(exc)})
                _write_outputs(
                    out_dir=out_dir,
                    registry_rows=registry_with_controls,
                    pair_df=pair_df,
                    alignment_rows=all_alignment_rows,
                    source_rows=all_source_rows,
                    examples=examples,
                    manifest=manifest,
                )
                continue
            raise

        width = sae_d_sae(sae)
        existing = {int(row["feature_idx"]) for row in layer_candidates}
        controls = _random_control_candidates(
            layer=int(hidden_layer),
            sae_release=sae_release,
            sae_id=sae_id,
            aggregation=str(args.aggregation),
            width=int(width),
            existing=existing,
            count=int(args.random_controls_per_layer),
            seed=int(args.seed),
        )
        layer_candidates.extend(controls)
        registry_with_controls.extend(controls)
        feature_indices = sorted({int(row["feature_idx"]) for row in layer_candidates})
        feature_to_col = {feature_idx: col for col, feature_idx in enumerate(feature_indices)}

        unique_features = _encode_texts_candidate_sae(
            scorer=scorer,
            tokenizer=tokenizer,
            sae=sae,
            texts=unique_texts,
            hidden_layer=int(hidden_layer),
            feature_indices=feature_indices,
            batch_size=int(args.sae_batch_size),
            max_length=int(args.max_length),
            aggregation=str(args.aggregation),
            token_chunk_size=int(args.sae_token_chunk_size),
        )
        human_mat = unique_features[human_rows]
        llm_mat = unique_features[llm_rows]
        layer_rows, layer_source_rows = _metric_rows_for_layer(
            layer_candidates=layer_candidates,
            pair_df=pair_df,
            human_mat=human_mat,
            llm_mat=llm_mat,
            feature_to_col=feature_to_col,
            design=design,
            source_min_pairs=int(args.source_min_pairs),
        )
        all_alignment_rows.extend(layer_rows)
        all_source_rows.extend(layer_source_rows)
        examples.update(
            _examples_for_layer(
                layer_rows=layer_rows,
                pair_df=pair_df,
                human_mat=human_mat,
                llm_mat=llm_mat,
                feature_to_col=feature_to_col,
                top_features=int(args.example_top_features),
                top_examples=int(args.top_examples_per_feature),
            )
        )
        completed_layers.append(int(hidden_layer))
        manifest["completed_hidden_layers"] = completed_layers
        manifest["skipped_layers"] = skipped_layers
        _write_outputs(
            out_dir=out_dir,
            registry_rows=registry_with_controls,
            pair_df=pair_df,
            alignment_rows=all_alignment_rows,
            source_rows=all_source_rows,
            examples=examples,
            manifest=manifest,
        )
        print(
            f"Completed hidden layer {hidden_layer}: "
            f"{len(layer_candidates)} candidates, {len(pair_df)} pairs"
        )
        del sae, unique_features, human_mat, llm_mat
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"Wrote candidate alignment outputs to {out_dir}")
    print(f"Wrote manifest to {out_dir / 'alignment_manifest.json'}")


if __name__ == "__main__":
    main()
