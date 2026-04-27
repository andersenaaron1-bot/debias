"""Manifest and ontology audit helpers for D4 tracing runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aisafety.mech.bundles import normalize_trace_bundles
from aisafety.mech.d4_io import file_status, read_json


def ontology_atoms_and_bundles(ontology: dict[str, Any]) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Return current ontology atoms and normalized bundle payloads."""

    bundles = normalize_trace_bundles(ontology.get("trace_bundles", []))
    atoms: set[str] = set(str(atom) for atom in ontology.get("priority_atoms", []))
    for payload in bundles.values():
        atoms.update(str(atom) for atom in payload.get("member_atoms", []))
    return atoms, bundles


def manifest_atoms_and_bundles(manifest: dict[str, Any]) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Return traced D4 atoms and bundles from a materialized manifest."""

    return (
        set(str(atom) for atom in manifest.get("d4_atoms", [])),
        normalize_trace_bundles(manifest.get("trace_bundles", {})),
    )


def build_manifest_audit(
    *,
    ontology_json: Path,
    manifest_json: Path,
    workspace_root: Path,
) -> dict[str, Any]:
    """Compare the current ontology config against a materialized D4 manifest."""

    ontology = read_json(ontology_json)
    manifest = read_json(manifest_json)
    ontology_atoms, ontology_bundles = ontology_atoms_and_bundles(ontology)
    manifest_atoms, manifest_bundles = manifest_atoms_and_bundles(manifest)

    bundle_support: dict[str, Any] = {}
    for bundle_id, payload in sorted(ontology_bundles.items()):
        member_atoms = [str(atom) for atom in payload.get("member_atoms", [])]
        present_atoms = [atom for atom in member_atoms if atom in manifest_atoms]
        missing_atoms = [atom for atom in member_atoms if atom not in manifest_atoms]
        if not present_atoms:
            support_status = "missing"
        elif missing_atoms:
            support_status = "partial"
        else:
            support_status = "complete"
        bundle_support[bundle_id] = {
            "status": str(payload.get("status") or ""),
            "member_atoms": member_atoms,
            "present_atoms": present_atoms,
            "missing_atoms": missing_atoms,
            "support_status": support_status,
        }

    output_status = {
        str(name): file_status(workspace_root, value)
        for name, value in sorted((manifest.get("outputs") or {}).items())
    }
    missing_outputs = [name for name, status in output_status.items() if status["path"] and not status["exists"]]
    missing_current_atoms = sorted(ontology_atoms - manifest_atoms)
    extra_manifest_atoms = sorted(manifest_atoms - ontology_atoms)
    partial_or_missing_bundles = [
        bundle_id
        for bundle_id, payload in bundle_support.items()
        if payload["support_status"] in {"partial", "missing"}
    ]

    return {
        "ontology_json": str(Path(ontology_json).resolve()),
        "manifest_json": str(Path(manifest_json).resolve()),
        "workspace_root": str(Path(workspace_root).resolve()),
        "ontology_atom_count": int(len(ontology_atoms)),
        "manifest_atom_count": int(len(manifest_atoms)),
        "manifest_bundle_count": int(len(manifest_bundles)),
        "traced_atoms": sorted(manifest_atoms),
        "missing_current_ontology_atoms": missing_current_atoms,
        "extra_manifest_atoms": extra_manifest_atoms,
        "bundle_support": bundle_support,
        "partial_or_missing_bundles": partial_or_missing_bundles,
        "output_status": output_status,
        "missing_outputs": missing_outputs,
        "rebuild_recommended": bool(missing_current_atoms or missing_outputs),
        "rebuild_reasons": {
            "missing_current_ontology_atoms": missing_current_atoms,
            "missing_outputs": missing_outputs,
            "partial_or_missing_bundles": partial_or_missing_bundles,
        },
    }

