"""Audit a D4 dataset-pack manifest against the current D4 ontology config.

This is a CPU-safe preflight step. It checks which ontology atoms are present
in the materialized D4 pack, which bundle families are complete or partial, and
whether manifest output paths resolve before LRZ model/SAE execution starts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aisafety.config import PROJECT_ROOT
from aisafety.mech.audit import build_manifest_audit
from aisafety.mech.d4_io import resolve_path, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ontology-json",
        type=Path,
        default=Path("configs") / "ontology" / "d4_reduced_ontology_v1.json",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=Path("data") / "derived" / "d4_dataset_pack_v1" / "manifest.json",
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts") / "mechanistic" / "d4_manifest_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    ontology_json = resolve_path(workspace_root, args.ontology_json)
    manifest_json = resolve_path(workspace_root, args.manifest_json)
    out_json = resolve_path(workspace_root, args.out_json)
    if ontology_json is None or manifest_json is None or out_json is None:
        raise ValueError("Could not resolve required paths.")

    audit = build_manifest_audit(
        ontology_json=ontology_json,
        manifest_json=manifest_json,
        workspace_root=workspace_root,
    )
    write_json(out_json, audit)

    print(f"Wrote D4 manifest audit to {out_json}")
    print(f"Manifest atoms: {audit['manifest_atom_count']}")
    print(f"Current ontology atoms: {audit['ontology_atom_count']}")
    print(f"Missing current ontology atoms: {len(audit['missing_current_ontology_atoms'])}")
    print(f"Missing outputs: {len(audit['missing_outputs'])}")
    print(f"Rebuild recommended: {audit['rebuild_recommended']}")


if __name__ == "__main__":
    main()

