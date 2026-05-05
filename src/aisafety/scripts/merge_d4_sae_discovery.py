"""Merge D4 SAE discovery outputs into one portable artifact bundle."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_RUNS = (
    ("coarse_late", "d4_j0_sae_coarse_late_fp_v1"),
    ("dense_midlate", "d4_j0_sae_dense_midlate_fp_v1"),
    ("dense_bridge", "d4_j0_sae_dense_bridge_fp_v1"),
)

DEFAULT_CSV_FILES = (
    "sae_atom_feature_scores.csv",
    "sae_bundle_feature_scores.csv",
    "sae_laurito_decision_alignment.csv",
    "sae_content_utility_overlap.csv",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Root containing artifacts/mechanistic, typically the DSS debias root.",
    )
    parser.add_argument(
        "--out-name",
        default="d4_j0_sae_merged_ontology_discovery_v1",
        help="Output directory name under artifacts/mechanistic.",
    )
    parser.add_argument(
        "--source-run",
        action="append",
        default=[],
        metavar="LABEL=DIRNAME",
        help="Optional source run mapping. May be repeated. Defaults to the J0 discovery runs.",
    )
    parser.add_argument("--archive", action="store_true", help="Also write a .tar.gz archive.")
    return parser.parse_args()


def _source_runs(args: argparse.Namespace) -> dict[str, str]:
    if not args.source_run:
        return dict(DEFAULT_SOURCE_RUNS)

    parsed: dict[str, str] = {}
    for item in args.source_run:
        if "=" not in item:
            raise ValueError(f"--source-run must be LABEL=DIRNAME, got {item!r}")
        label, dirname = item.split("=", 1)
        label = label.strip()
        dirname = dirname.strip()
        if not label or not dirname:
            raise ValueError(f"--source-run must be LABEL=DIRNAME, got {item!r}")
        parsed[label] = dirname
    return parsed


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _merge_csvs(
    *,
    source_dirs: dict[str, Path],
    out_dir: Path,
    manifest: dict[str, Any],
) -> None:
    for csv_name in DEFAULT_CSV_FILES:
        merged_rows: list[dict[str, str]] = []
        fieldnames: list[str] = []
        for source_label, source_dir in source_dirs.items():
            path = source_dir / csv_name
            if not path.exists():
                continue
            rows, source_fields = _read_csv_rows(path)
            for field in source_fields + ["source_run"]:
                if field not in fieldnames:
                    fieldnames.append(field)
            for row in rows:
                row["source_run"] = source_label
                merged_rows.append(row)

        if not merged_rows:
            continue

        out_path = out_dir / f"merged_{csv_name}"
        _write_csv(out_path, merged_rows, fieldnames)
        manifest["merged_csvs"].append(str(out_path))
        manifest["row_counts"][out_path.name] = len(merged_rows)


def _merge_examples(
    *,
    source_dirs: dict[str, Path],
    out_dir: Path,
    manifest: dict[str, Any],
) -> None:
    examples: dict[str, Any] = {}
    for source_label, source_dir in source_dirs.items():
        path = source_dir / "sae_feature_examples.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            examples[f"{source_label}|{key}"] = value

    if not examples:
        return

    out_path = out_dir / "merged_sae_feature_examples.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(examples, handle, indent=2, ensure_ascii=False)
    manifest["merged_jsons"].append(str(out_path))
    manifest["row_counts"][out_path.name] = len(examples)


def _copy_source_manifests(
    *,
    source_dirs: dict[str, Path],
    out_dir: Path,
    manifest: dict[str, Any],
) -> None:
    for source_label, source_dir in source_dirs.items():
        path = source_dir / "sae_feature_set_manifest.json"
        if not path.exists():
            continue
        out_path = out_dir / f"{source_label}_sae_feature_set_manifest.json"
        out_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest["copied_manifests"].append(str(out_path))


def _write_archive(out_dir: Path) -> Path:
    archive_path = out_dir.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return archive_path


def main() -> None:
    args = _parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    mech_root = artifact_root / "artifacts" / "mechanistic"
    out_dir = mech_root / str(args.out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = {
        label: mech_root / dirname for label, dirname in _source_runs(args).items()
    }
    missing = {label: str(path) for label, path in source_dirs.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing source runs: {missing}")

    manifest: dict[str, Any] = {
        "artifact_root": str(artifact_root),
        "out_dir": str(out_dir),
        "sources": {label: str(path) for label, path in source_dirs.items()},
        "merged_csvs": [],
        "merged_jsons": [],
        "copied_manifests": [],
        "row_counts": {},
    }

    _merge_csvs(source_dirs=source_dirs, out_dir=out_dir, manifest=manifest)
    _merge_examples(source_dirs=source_dirs, out_dir=out_dir, manifest=manifest)
    _copy_source_manifests(source_dirs=source_dirs, out_dir=out_dir, manifest=manifest)

    archive_path: Path | None = None
    if bool(args.archive):
        archive_path = _write_archive(out_dir)
        manifest["archive"] = str(archive_path)

    manifest_path = out_dir / "merge_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"merged_dir={out_dir}")
    if archive_path is not None:
        print(f"archive={archive_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
