"""Feature-level intervention helpers for D4 SAE perturbation runs."""

from __future__ import annotations

from collections.abc import Iterator
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from aisafety.mech.sae import hidden_layer_to_sae_layer, sae_encode


@dataclass(frozen=True)
class FeatureSpec:
    """A single SAE feature selected for perturbation or control comparisons."""

    bundle_id: str
    hidden_layer: int
    feature_idx: int
    sae_release: str
    sae_id: str
    aggregation: str = "max"
    feature_role: str = "target"
    freeze_status: str = ""
    atoms: str = ""
    signed_alignment: float = 0.0
    source_row: dict[str, Any] | None = None

    @property
    def sae_layer(self) -> int:
        return hidden_layer_to_sae_layer(self.hidden_layer)

    @property
    def feature_id(self) -> str:
        return f"L{int(self.hidden_layer)}F{int(self.feature_idx)}"

    @property
    def direction(self) -> int:
        if float(self.signed_alignment) < 0:
            return -1
        return 1


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).is_file():
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _feature_alignment(row: dict[str, Any]) -> float:
    value = _float_or_none(row.get("length_controlled_spearman_delta_with_j0_margin"))
    if value is not None:
        return float(value)
    value = _float_or_none(row.get("spearman_delta_with_j0_margin"))
    if value is not None:
        return float(value)
    auc = _float_or_none(row.get("auc_j0_llm_choice_from_activation_delta"))
    if auc is not None:
        return float(auc) - 0.5
    return 0.0


def load_bundle_feature_specs(
    registry_dir: Path,
    *,
    bundle_id: str,
    include_source_sensitive: bool = False,
    max_features: int = 0,
) -> list[FeatureSpec]:
    """Load frozen target features for one bundle from the registry artifact."""

    rows = _read_csv(Path(registry_dir) / "bundle_candidate_features.csv")
    allowed_statuses = {"intervention_eligible"}
    if include_source_sensitive:
        allowed_statuses.add("source_sensitive")

    specs: list[FeatureSpec] = []
    for row in rows:
        if str(row.get("bundle_id") or "") != str(bundle_id):
            continue
        if str(row.get("freeze_status") or "") not in allowed_statuses:
            continue
        specs.append(
            FeatureSpec(
                bundle_id=str(row.get("bundle_id") or bundle_id),
                hidden_layer=int(row["hidden_layer"]),
                feature_idx=int(row["feature_idx"]),
                sae_release=str(row.get("sae_release") or ""),
                sae_id=str(row.get("sae_id") or ""),
                aggregation=str(row.get("aggregation") or "max"),
                feature_role="target",
                freeze_status=str(row.get("freeze_status") or ""),
                atoms=str(row.get("atoms") or row.get("member_atoms_hit") or ""),
                signed_alignment=_feature_alignment(row),
                source_row=dict(row),
            )
        )

    specs.sort(
        key=lambda spec: (
            -abs(float(spec.signed_alignment)),
            int(spec.hidden_layer),
            int(spec.feature_idx),
        )
    )
    if int(max_features) > 0:
        specs = specs[: int(max_features)]
    return specs


def load_matched_random_control_specs(
    registry_dir: Path,
    *,
    bundle_id: str,
    allowed_target_keys: set[tuple[int, int]] | None = None,
    control_rank: int = 1,
    max_features: int = 0,
) -> list[FeatureSpec]:
    """Load layer-matched random controls from the frozen registry artifact."""

    rows = _read_csv(Path(registry_dir) / "matched_random_feature_controls.csv")
    specs: list[FeatureSpec] = []
    for row in rows:
        if str(row.get("bundle_id") or "") != str(bundle_id):
            continue
        if allowed_target_keys is not None:
            target_key = (
                int(float(row.get("target_hidden_layer") or -1)),
                int(float(row.get("target_feature_idx") or -1)),
            )
            if target_key not in allowed_target_keys:
                continue
        rank_text = str(row.get("control_rank") or "").strip()
        if rank_text and int(float(rank_text)) > int(control_rank):
            continue
        hidden_layer = int(float(row["control_hidden_layer"]))
        feature_idx = int(float(row["control_feature_idx"]))
        specs.append(
            FeatureSpec(
                bundle_id=str(row.get("bundle_id") or bundle_id),
                hidden_layer=hidden_layer,
                feature_idx=feature_idx,
                sae_release=str(row.get("sae_release") or ""),
                sae_id=str(row.get("sae_id") or ""),
                aggregation="max",
                feature_role="matched_random_control",
                freeze_status="matched_random_control",
                atoms="",
                signed_alignment=_feature_alignment(row),
                source_row=dict(row),
            )
        )

    specs.sort(key=lambda spec: (int(spec.hidden_layer), int(spec.feature_idx)))
    if int(max_features) > 0:
        specs = specs[: int(max_features)]
    return specs


def group_features_by_layer(features: list[FeatureSpec]) -> dict[int, list[FeatureSpec]]:
    """Group feature specs by Hugging Face hidden-state index."""

    grouped: dict[int, list[FeatureSpec]] = {}
    for spec in features:
        grouped.setdefault(int(spec.hidden_layer), []).append(spec)
    return {layer: sorted(rows, key=lambda spec: int(spec.feature_idx)) for layer, rows in grouped.items()}


def iter_feature_rows(features: list[FeatureSpec]) -> Iterator[dict[str, Any]]:
    """Yield stable CSV-friendly feature rows."""

    for spec in features:
        yield {
            "bundle_id": spec.bundle_id,
            "feature_role": spec.feature_role,
            "hidden_layer": int(spec.hidden_layer),
            "sae_layer": int(spec.sae_layer),
            "sae_release": spec.sae_release,
            "sae_id": spec.sae_id,
            "aggregation": spec.aggregation,
            "feature_idx": int(spec.feature_idx),
            "feature_id": spec.feature_id,
            "freeze_status": spec.freeze_status,
            "atoms": spec.atoms,
            "signed_alignment": float(spec.signed_alignment),
            "direction": int(spec.direction),
        }


def assign_quantile_bins(values: np.ndarray, *, high_low_frac: float) -> list[str]:
    """Assign low/middle/high bins from a continuous pair-level occurrence score."""

    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return []
    frac = min(max(float(high_low_frac), 0.01), 0.49)
    low_cut = float(np.quantile(arr, frac))
    high_cut = float(np.quantile(arr, 1.0 - frac))
    bins: list[str] = []
    for value in arr:
        if float(value) <= low_cut:
            bins.append("low")
        elif float(value) >= high_cut:
            bins.append("high")
        else:
            bins.append("middle")
    return bins


def find_decoder_layer_module(model: nn.Module, *, hidden_layer: int) -> nn.Module:
    """Find a decoder block module for a HF hidden-state index.

    Gemma hidden_states[0] is embeddings; hidden_states[N] is the output after
    decoder block N - 1. This function therefore maps hidden_layer to
    ``layers[hidden_layer - 1]`` and handles both bare HF models and PEFT
    wrappers by first checking common attributes and then falling back to
    named-module pattern matching.
    """

    layer_idx = hidden_layer_to_sae_layer(int(hidden_layer))
    candidates = [
        "model.layers",
        "base_model.model.model.layers",
        "base_model.model.layers",
        "module.model.layers",
        "module.base_model.model.model.layers",
    ]
    for path in candidates:
        obj: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, (nn.ModuleList, list, tuple)) and layer_idx < len(obj):
            return obj[layer_idx]

    pattern = re.compile(rf"(?:^|\.)layers\.{layer_idx}$")
    for name, module in model.named_modules():
        if pattern.search(name):
            return module
    raise ValueError(f"Could not locate decoder layer for hidden_layer={hidden_layer}.")


def _decoder_vectors(sae: Any, feature_indices: list[int], *, hidden_dim: int) -> torch.Tensor:
    if not hasattr(sae, "W_dec"):
        raise RuntimeError("SAE object does not expose W_dec for feature damping.")
    w_dec = getattr(sae, "W_dec")
    if not isinstance(w_dec, torch.Tensor):
        w_dec = torch.as_tensor(w_dec)
    idx = torch.tensor(feature_indices, device=w_dec.device, dtype=torch.long)
    if w_dec.ndim != 2:
        raise RuntimeError(f"Expected SAE W_dec to be 2D, got shape {tuple(w_dec.shape)}.")
    if w_dec.shape[0] > max(feature_indices) and w_dec.shape[1] == int(hidden_dim):
        return w_dec.index_select(0, idx)
    if w_dec.shape[1] > max(feature_indices) and w_dec.shape[0] == int(hidden_dim):
        return w_dec.index_select(1, idx).T
    raise RuntimeError(
        "Could not align SAE W_dec with feature indices and hidden dimension: "
        f"W_dec={tuple(w_dec.shape)} hidden_dim={hidden_dim}"
    )


class SaeFeatureDampingHook:
    """Forward hook that subtracts selected SAE decoder contributions."""

    def __init__(self, *, sae: Any, feature_indices: list[int], strength: float):
        if not feature_indices:
            raise ValueError("feature_indices must be nonempty.")
        self.sae = sae
        self.feature_indices = [int(idx) for idx in feature_indices]
        self.strength = float(strength)

    def _damp_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or self.strength == 0.0:
            return hidden
        original_dtype = hidden.dtype
        batch, seq, hidden_dim = hidden.shape
        flat = hidden.reshape(batch * seq, hidden_dim).to(dtype=torch.float32)
        acts = sae_encode(self.sae, flat)
        idx = torch.tensor(self.feature_indices, device=acts.device, dtype=torch.long)
        selected = acts.index_select(dim=1, index=idx)
        vectors = _decoder_vectors(self.sae, self.feature_indices, hidden_dim=hidden_dim).to(
            device=flat.device,
            dtype=torch.float32,
        )
        delta = selected @ vectors
        perturbed = flat - float(self.strength) * delta
        return perturbed.reshape_as(hidden).to(dtype=original_dtype)

    def __call__(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return self._damp_hidden(output)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (self._damp_hidden(output[0]), *output[1:])
        return output


def register_feature_damping_hooks(
    model: nn.Module,
    *,
    features_by_layer: dict[int, list[FeatureSpec]],
    sae_by_layer: dict[int, Any],
    strength: float,
) -> list[Any]:
    """Register SAE feature-damping hooks and return removable handles."""

    handles: list[Any] = []
    for hidden_layer, specs in sorted(features_by_layer.items()):
        if hidden_layer not in sae_by_layer:
            raise KeyError(f"Missing SAE for hidden_layer={hidden_layer}.")
        layer_module = find_decoder_layer_module(model, hidden_layer=int(hidden_layer))
        feature_indices = sorted({int(spec.feature_idx) for spec in specs})
        hook = SaeFeatureDampingHook(
            sae=sae_by_layer[int(hidden_layer)],
            feature_indices=feature_indices,
            strength=float(strength),
        )
        handles.append(layer_module.register_forward_hook(hook))
    return handles


def remove_hooks(handles: list[Any]) -> None:
    """Remove forward hooks, ignoring already-removed handles."""

    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass
