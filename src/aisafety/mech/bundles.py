"""Bundle-level aggregation helpers for SAE feature outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_trace_bundles(trace_bundles: Any) -> dict[str, dict[str, Any]]:
    """Normalize trace bundle payloads from manifest dicts or ontology lists."""

    if isinstance(trace_bundles, dict):
        return {str(bundle_id): dict(payload) for bundle_id, payload in trace_bundles.items()}
    if isinstance(trace_bundles, list):
        out: dict[str, dict[str, Any]] = {}
        for payload in trace_bundles:
            if not isinstance(payload, dict):
                continue
            bundle_id = str(payload.get("bundle_id") or "").strip()
            if bundle_id:
                out[bundle_id] = dict(payload)
        return out
    return {}


def build_bundle_feature_scores(feature_rows: pd.DataFrame, trace_bundles: Any) -> pd.DataFrame:
    """Aggregate atom-feature rows into bundle candidate rows."""

    trace_bundle_map = normalize_trace_bundles(trace_bundles)
    if feature_rows.empty:
        return pd.DataFrame()
    ok = feature_rows[feature_rows["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for bundle_id, payload in sorted(trace_bundle_map.items()):
        member_atoms = [str(atom) for atom in (payload.get("member_atoms") or [])]
        if not member_atoms:
            continue
        sub = ok[ok["atom"].isin(member_atoms)]
        if sub.empty:
            continue
        group_cols = ["hidden_layer", "sae_layer", "sae_release", "sae_id", "aggregation", "feature_idx"]
        for key, group in sub.groupby(group_cols):
            hidden_layer, sae_layer, sae_release, sae_id, aggregation, feature_idx = key
            atoms = sorted(set(str(atom) for atom in group["atom"].tolist()))
            row = {
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
                "mean_abs_cohen_d": float(group["abs_cohen_d"].astype(float).mean()),
                "max_val_auc": float(group["val_auc"].astype(float).max()),
            }
            if "laurito_spearman_with_atom_score" in group.columns:
                row["mean_laurito_abs_spearman"] = float(
                    group["laurito_spearman_with_atom_score"].astype(float).abs().mean()
                )
            else:
                row["mean_laurito_abs_spearman"] = None
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["n_member_atoms_hit", "mean_abs_cohen_d", "max_val_auc"], ascending=False
    )

