"""Freeze D4 bundle candidates for workshop-scale feature interventions.

This CPU-only step joins the broad human-vs-LLM candidate-alignment outputs
with an explicit bundle-candidate config. It produces the fixed registry that
later text perturbation and SAE feature-damping runs must consume.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_json, sha1_hex, write_json


DEFAULT_ALIGNMENT_RUN_DIR = (
    Path("artifacts") / "mechanistic" / "d4_j0_human_llm_candidate_alignment_strat10k_v3"
)
DEFAULT_CANDIDATE_SOURCE_DIR = (
    Path("artifacts") / "mechanistic" / "d4_j0_sae_merged_ontology_discovery_v1"
)
DEFAULT_CONFIG = Path("configs") / "ontology" / "d4_bundle_intervention_candidates_v1.json"
DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_j0_bundle_candidate_registry_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--alignment-run-dir", type=Path, default=DEFAULT_ALIGNMENT_RUN_DIR)
    parser.add_argument("--candidate-source-dir", type=Path, default=DEFAULT_CANDIDATE_SOURCE_DIR)
    parser.add_argument("--bundle-config-json", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--controls-per-feature", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _resolve(base: Path, value: Path | str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(base) / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).is_file():
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    val = _float_or_none(value)
    if val is None:
        return None
    return int(val)


def _split_set(value: Any) -> set[str]:
    return {part.strip() for part in str(value or "").split(";") if part.strip()}


def _feature_key(layer: Any, feature_idx: Any) -> tuple[int, int] | None:
    hidden_layer = _int_or_none(layer)
    feature = _int_or_none(feature_idx)
    if hidden_layer is None or feature is None:
        return None
    return int(hidden_layer), int(feature)


def _feature_id(layer: int, feature_idx: int) -> str:
    return f"L{int(layer)}F{int(feature_idx)}"


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[int(0.9 * (len(ordered) - 1))])


def _abs_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = _float_or_none(row.get(key))
    return default if val is None else abs(float(val))


def _choice_auc_delta(row: dict[str, Any]) -> float:
    val = _float_or_none(row.get("auc_j0_llm_choice_from_activation_delta"))
    return 0.0 if val is None else abs(float(val) - 0.5)


def _alignment_by_key(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, str]]:
    out: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        key = _feature_key(row.get("hidden_layer"), row.get("feature_idx"))
        if key is None:
            continue
        out[key] = row
    return out


def _discovery_metadata(source_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    meta: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {"atoms": set(), "bundles": set(), "member_atoms": set(), "source_runs": set()}
    )

    for row in _read_csv(source_dir / "merged_sae_atom_feature_scores.csv"):
        key = _feature_key(row.get("hidden_layer"), row.get("feature_idx"))
        if key is None:
            continue
        atom = str(row.get("atom") or "").strip()
        if atom:
            meta[key]["atoms"].add(atom)
        if row.get("source_run"):
            meta[key]["source_runs"].add(str(row["source_run"]))

    for row in _read_csv(source_dir / "merged_sae_bundle_feature_scores.csv"):
        key = _feature_key(row.get("hidden_layer"), row.get("feature_idx"))
        if key is None:
            continue
        bundle = str(row.get("bundle_id") or "").strip()
        if bundle:
            meta[key]["bundles"].add(bundle)
        if row.get("source_run"):
            meta[key]["source_runs"].add(str(row["source_run"]))
        meta[key]["member_atoms"].update(_split_set(row.get("member_atoms_hit")))

    return meta


def _source_support_by_key(source_rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, Any]]:
    by_key: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        key = _feature_key(row.get("hidden_layer"), row.get("feature_idx"))
        if key is not None:
            by_key[key].append(row)

    out: dict[tuple[int, int], dict[str, Any]] = {}
    for key, rows in by_key.items():
        datasets = sorted({str(row.get("source_dataset") or "") for row in rows if row.get("source_dataset")})
        source_rhos = [
            _float_or_none(row.get("spearman_delta_with_j0_margin"))
            for row in rows
            if _float_or_none(row.get("spearman_delta_with_j0_margin")) is not None
        ]
        out[key] = {
            "source_datasets_observed": ";".join(datasets),
            "n_source_rows": int(len(rows)),
            "max_source_abs_spearman": max([abs(float(x)) for x in source_rhos], default=0.0),
        }
    return out


def _control_baseline(alignment_rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    controls = [row for row in alignment_rows if row.get("candidate_kind") == "random_control"]
    control_rhos = [
        _abs_float(row, "length_controlled_spearman_delta_with_j0_margin")
        for row in controls
        if _float_or_none(row.get("length_controlled_spearman_delta_with_j0_margin")) is not None
    ]
    control_auc = [
        _choice_auc_delta(row)
        for row in controls
        if _float_or_none(row.get("auc_j0_llm_choice_from_activation_delta")) is not None
    ]
    return {
        "n_random_control_rows": len(controls),
        "p90_abs_controlled_rho": _p90(control_rhos),
        "p90_abs_choice_auc_delta": _p90(control_auc),
    }


def _row_atoms(
    *,
    align_row: dict[str, Any] | None,
    discovery: dict[str, Any] | None,
) -> set[str]:
    atoms: set[str] = set()
    if align_row is not None:
        atoms.update(_split_set(align_row.get("atoms")))
    if discovery is not None:
        atoms.update(discovery.get("atoms", set()))
        atoms.update(discovery.get("member_atoms", set()))
    return atoms


def _row_bundles(
    *,
    align_row: dict[str, Any] | None,
    discovery: dict[str, Any] | None,
) -> set[str]:
    bundles: set[str] = set()
    if align_row is not None:
        bundles.update(_split_set(align_row.get("bundles")))
    if discovery is not None:
        bundles.update(discovery.get("bundles", set()))
    return bundles


def _eligibility_status(
    *,
    row: dict[str, Any],
    bundle_atom_count: int,
    cfg: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[str, str]:
    if row.get("missing_alignment") == "1":
        return "missing_alignment", "feature absent from broad alignment output"

    elig = cfg.get("eligibility") if isinstance(cfg.get("eligibility"), dict) else {}
    min_atoms = int(elig.get("min_bundle_atoms_for_paper_claim", 3))
    min_sources = int(elig.get("min_sources_with_pairs", 4))
    preferred_cons = float(elig.get("preferred_source_sign_consistency", 0.8))
    allowed_cons = float(elig.get("allow_source_sensitive_consistency", 0.6))
    p90_rho = baseline.get("p90_abs_controlled_rho")
    p90_auc = baseline.get("p90_abs_choice_auc_delta")

    n_sources = _int_or_none(row.get("n_sources_with_min_pairs")) or 0
    sign_cons = _float_or_none(row.get("source_sign_consistency")) or 0.0
    abs_rho = _abs_float(row, "length_controlled_spearman_delta_with_j0_margin")
    auc_delta = _choice_auc_delta(row)
    beats_control = (
        (p90_rho is not None and abs_rho > float(p90_rho))
        or (p90_auc is not None and auc_delta > float(p90_auc))
    )

    reasons: list[str] = []
    if bundle_atom_count < min_atoms:
        reasons.append(f"bundle has fewer than {min_atoms} configured atoms")
    if n_sources < min_sources:
        reasons.append(f"n_sources_with_min_pairs={n_sources} < {min_sources}")
    if not beats_control:
        reasons.append("effect does not exceed matched-random p90")
    if sign_cons < allowed_cons:
        reasons.append(f"source_sign_consistency={sign_cons:.3f} < {allowed_cons}")

    if reasons:
        return "diagnostic", "; ".join(reasons)
    if sign_cons < preferred_cons:
        return "source_sensitive", f"source_sign_consistency={sign_cons:.3f} below preferred {preferred_cons}"
    return "intervention_eligible", "passes workshop freeze gates"


def _bundle_feature_rows(
    *,
    cfg: dict[str, Any],
    alignment_by_key: dict[tuple[int, int], dict[str, str]],
    discovery_by_key: dict[tuple[int, int], dict[str, Any]],
    source_support: dict[tuple[int, int], dict[str, Any]],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    default_release = str(cfg.get("sae_release") or "")
    default_aggregation = str(cfg.get("aggregation") or "max")
    for bundle in cfg.get("bundles", []):
        bundle_id = str(bundle.get("bundle_id") or "")
        atoms = [str(atom) for atom in bundle.get("atoms", [])]
        support_atoms = [str(atom) for atom in bundle.get("support_atoms", [])]
        configured_atoms = set(atoms) | set(support_atoms)
        for rank, spec in enumerate(bundle.get("primary_features", []), start=1):
            key = _feature_key(spec.get("hidden_layer"), spec.get("feature_idx"))
            if key is None:
                continue
            align = alignment_by_key.get(key)
            disc = discovery_by_key.get(key)
            row_atoms = _row_atoms(align_row=align, discovery=disc)
            row_bundles = _row_bundles(align_row=align, discovery=disc)
            overlap = sorted(row_atoms & configured_atoms)
            out: dict[str, Any] = {
                "bundle_id": bundle_id,
                "paper_label": str(bundle.get("paper_label") or bundle_id),
                "bundle_role": str(bundle.get("role") or ""),
                "bundle_feature_rank": int(rank),
                "hidden_layer": key[0],
                "feature_idx": key[1],
                "feature_id": _feature_id(*key),
                "sae_release": default_release,
                "aggregation": default_aggregation,
                "configured_bundle_atoms": ";".join(atoms),
                "configured_support_atoms": ";".join(support_atoms),
                "n_configured_bundle_atoms": len(atoms),
                "feature_atoms": ";".join(sorted(row_atoms)),
                "feature_bundles": ";".join(sorted(row_bundles)),
                "n_feature_atoms_in_bundle": len(overlap),
                "feature_atoms_in_bundle": ";".join(overlap),
                "source_runs": ";".join(sorted(disc.get("source_runs", set()))) if disc else "",
                "missing_alignment": "0" if align is not None else "1",
            }
            if align:
                for col in (
                    "candidate_kind",
                    "n_pairs",
                    "mean_llm_minus_human_activation",
                    "activation_auc_llm_vs_human",
                    "auc_j0_llm_choice_from_activation_delta",
                    "length_controlled_spearman_delta_with_j0_margin",
                    "length_controlled_spearman_q",
                    "source_sign_consistency",
                    "n_sources_with_min_pairs",
                    "candidate_reasons",
                ):
                    out[col] = align.get(col, "")
            out.update(source_support.get(key, {}))
            status, reason = _eligibility_status(
                row=out,
                bundle_atom_count=len(atoms),
                cfg=cfg,
                baseline=baseline,
            )
            out["freeze_status"] = status
            out["freeze_reason"] = reason
            rows.append(out)
    rows.sort(key=lambda row: (row["bundle_id"], int(row["bundle_feature_rank"])))
    return rows


def _atom_feature_rows(bundle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in bundle_rows:
        atoms = _split_set(row.get("feature_atoms_in_bundle")) or _split_set(row.get("feature_atoms"))
        for atom in sorted(atoms):
            rows.append(
                {
                    "atom": atom,
                    "bundle_id": row["bundle_id"],
                    "hidden_layer": row["hidden_layer"],
                    "feature_idx": row["feature_idx"],
                    "feature_id": row["feature_id"],
                    "freeze_status": row["freeze_status"],
                    "length_controlled_spearman_delta_with_j0_margin": row.get(
                        "length_controlled_spearman_delta_with_j0_margin", ""
                    ),
                    "auc_j0_llm_choice_from_activation_delta": row.get(
                        "auc_j0_llm_choice_from_activation_delta", ""
                    ),
                    "source_sign_consistency": row.get("source_sign_consistency", ""),
                    "n_sources_with_min_pairs": row.get("n_sources_with_min_pairs", ""),
                }
            )
    rows.sort(key=lambda row: (row["bundle_id"], row["atom"], int(row["hidden_layer"]), int(row["feature_idx"])))
    return rows


def _matched_control_rows(
    *,
    bundle_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, str]],
    controls_per_feature: int,
    seed: int,
) -> list[dict[str, Any]]:
    controls_by_layer: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in alignment_rows:
        if row.get("candidate_kind") != "random_control":
            continue
        key = _feature_key(row.get("hidden_layer"), row.get("feature_idx"))
        if key is not None:
            controls_by_layer[key[0]].append(row)

    target_keys = {(int(row["hidden_layer"]), int(row["feature_idx"])) for row in bundle_rows}
    all_controls = [row for rows in controls_by_layer.values() for row in rows]
    out: list[dict[str, Any]] = []
    for target in bundle_rows:
        layer = int(target["hidden_layer"])
        candidates = controls_by_layer.get(layer, []) or all_controls
        candidates = [
            row
            for row in candidates
            if _feature_key(row.get("hidden_layer"), row.get("feature_idx")) not in target_keys
        ]
        ordered = sorted(
            candidates,
            key=lambda row: sha1_hex(
                f"{seed}:control:{target['bundle_id']}:{target['hidden_layer']}:{target['feature_idx']}:"
                f"{row.get('hidden_layer')}:{row.get('feature_idx')}"
            ),
        )
        for rank, control in enumerate(ordered[: max(0, int(controls_per_feature))], start=1):
            control_key = _feature_key(control.get("hidden_layer"), control.get("feature_idx"))
            if control_key is None:
                continue
            out.append(
                {
                    "bundle_id": target["bundle_id"],
                    "target_hidden_layer": target["hidden_layer"],
                    "target_feature_idx": target["feature_idx"],
                    "target_feature_id": target["feature_id"],
                    "control_rank": rank,
                    "control_hidden_layer": control_key[0],
                    "control_feature_idx": control_key[1],
                    "control_feature_id": _feature_id(*control_key),
                    "layer_matched": str(control_key[0] == layer).lower(),
                    "length_controlled_spearman_delta_with_j0_margin": control.get(
                        "length_controlled_spearman_delta_with_j0_margin", ""
                    ),
                    "auc_j0_llm_choice_from_activation_delta": control.get(
                        "auc_j0_llm_choice_from_activation_delta", ""
                    ),
                    "source_sign_consistency": control.get("source_sign_consistency", ""),
                    "n_sources_with_min_pairs": control.get("n_sources_with_min_pairs", ""),
                }
            )
    return out


def build_registry(
    *,
    alignment_run_dir: Path,
    candidate_source_dir: Path,
    bundle_config_json: Path,
    out_dir: Path,
    controls_per_feature: int,
    seed: int,
) -> dict[str, Any]:
    cfg = read_json(bundle_config_json)
    alignment_rows = _read_csv(alignment_run_dir / "candidate_feature_human_llm_alignment.csv")
    source_rows = _read_csv(alignment_run_dir / "candidate_feature_source_alignment.csv")
    manifest = read_json(alignment_run_dir / "alignment_manifest.json") if (alignment_run_dir / "alignment_manifest.json").is_file() else {}
    if not alignment_rows:
        raise FileNotFoundError(f"No alignment rows found in {alignment_run_dir}")

    alignment_lookup = _alignment_by_key([row for row in alignment_rows if row.get("candidate_kind") != "random_control"])
    discovery_lookup = _discovery_metadata(candidate_source_dir)
    source_support = _source_support_by_key(source_rows)
    baseline = _control_baseline(alignment_rows)
    bundle_rows = _bundle_feature_rows(
        cfg=cfg,
        alignment_by_key=alignment_lookup,
        discovery_by_key=discovery_lookup,
        source_support=source_support,
        baseline=baseline,
    )
    atom_rows = _atom_feature_rows(bundle_rows)
    control_rows = _matched_control_rows(
        bundle_rows=bundle_rows,
        alignment_rows=alignment_rows,
        controls_per_feature=controls_per_feature,
        seed=seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "bundle_candidate_features.csv", bundle_rows)
    _write_csv(out_dir / "atom_candidate_features.csv", atom_rows)
    _write_csv(out_dir / "matched_random_feature_controls.csv", control_rows)

    by_bundle_status = Counter((row["bundle_id"], row["freeze_status"]) for row in bundle_rows)
    summary = {
        "alignment_run_dir": str(alignment_run_dir),
        "candidate_source_dir": str(candidate_source_dir),
        "bundle_config_json": str(bundle_config_json),
        "out_dir": str(out_dir),
        "seed": int(seed),
        "controls_per_feature": int(controls_per_feature),
        "alignment_manifest": manifest,
        "random_control_baseline": baseline,
        "n_bundle_feature_rows": len(bundle_rows),
        "n_atom_feature_rows": len(atom_rows),
        "n_matched_random_control_rows": len(control_rows),
        "by_freeze_status": dict(Counter(row["freeze_status"] for row in bundle_rows)),
        "by_bundle_status": {f"{bundle}::{status}": int(count) for (bundle, status), count in by_bundle_status.items()},
        "bundles": [
            {
                "bundle_id": str(bundle.get("bundle_id") or ""),
                "paper_label": str(bundle.get("paper_label") or ""),
                "n_configured_atoms": len(bundle.get("atoms", []) or []),
                "n_configured_features": len(bundle.get("primary_features", []) or []),
                "n_intervention_eligible_features": sum(
                    1
                    for row in bundle_rows
                    if row["bundle_id"] == str(bundle.get("bundle_id") or "")
                    and row["freeze_status"] == "intervention_eligible"
                ),
                "n_source_sensitive_features": sum(
                    1
                    for row in bundle_rows
                    if row["bundle_id"] == str(bundle.get("bundle_id") or "")
                    and row["freeze_status"] == "source_sensitive"
                ),
            }
            for bundle in cfg.get("bundles", [])
        ],
        "outputs": {
            "bundle_candidate_features_csv": str(out_dir / "bundle_candidate_features.csv"),
            "atom_candidate_features_csv": str(out_dir / "atom_candidate_features.csv"),
            "matched_random_feature_controls_csv": str(out_dir / "matched_random_feature_controls.csv"),
            "feature_freeze_manifest_json": str(out_dir / "feature_freeze_manifest.json"),
        },
    }
    write_json(out_dir / "feature_freeze_manifest.json", summary)
    return summary


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    summary = build_registry(
        alignment_run_dir=_resolve(workspace_root, args.alignment_run_dir),
        candidate_source_dir=_resolve(workspace_root, args.candidate_source_dir),
        bundle_config_json=_resolve(workspace_root, args.bundle_config_json),
        out_dir=_resolve(workspace_root, args.out_dir),
        controls_per_feature=int(args.controls_per_feature),
        seed=int(args.seed),
    )
    print(f"out_dir={summary['out_dir']}")
    print(f"bundle_features={summary['outputs']['bundle_candidate_features_csv']}")
    print(f"atom_features={summary['outputs']['atom_candidate_features_csv']}")
    print(f"random_controls={summary['outputs']['matched_random_feature_controls_csv']}")
    print(f"manifest={summary['outputs']['feature_freeze_manifest_json']}")
    print(f"by_freeze_status={summary['by_freeze_status']}")


if __name__ == "__main__":
    main()
