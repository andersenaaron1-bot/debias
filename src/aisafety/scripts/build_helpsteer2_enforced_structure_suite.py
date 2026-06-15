"""Build the locked HelpSteer2 enforced-structure follow-up suite."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_json, read_jsonl, resolve_path, sha1_hex
from aisafety.mech.d4_io import write_json
from aisafety.scripts.build_helpsteer2_criterion_confirmation import (
    _displayed_options,
)
from aisafety.scripts.build_helpsteer2_criterion_switch_suite import (
    _semantic_swap,
)
from aisafety.scripts.build_helpsteer2_matched_criterion_suite import CRITERIA


CONDITIONS = (
    "free_long",
    "prompted_long",
    "enforced_generic",
    "enforced_criterion",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-suite-dir", type=Path, required=True)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "data/derived/helpsteer2_enforced_structure_suite_v1"
        ),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in values:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    return len(values)


def build_episodes(
    pairs: Iterable[dict[str, Any]],
    *,
    branches: int,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for pair in pairs:
        criterion_id = str(pair["updated_criterion_id"])
        target_semantic = str(pair["criterion_targets"][criterion_id])
        for order in ("original", "swapped"):
            option_a, option_b = _displayed_options(pair, order)
            target_option = (
                _semantic_swap(target_semantic)
                if order == "swapped"
                else target_semantic
            )
            for condition in CONDITIONS:
                episode_id = sha1_hex(
                    f"{pair['pair_id']}|enforced-structure|"
                    f"{condition}|{order}"
                )
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "comparison_id": episode_id,
                        "pair_id": str(pair["pair_id"]),
                        "origin_pair_id": str(pair["pair_id"]),
                        "source_dataset": (
                            "helpsteer2_enforced_structure"
                        ),
                        "task_type": (
                            "criterion_operationalization_enforced"
                        ),
                        "comparison_dimension": "criterion_use",
                        "condition_id": condition,
                        "condition_label": condition,
                        "transition_type": str(
                            pair["transition_type"]
                        ),
                        "presentation_order": order,
                        "prompt": str(pair["prompt"]),
                        "option_a_text": option_a,
                        "option_b_text": option_b,
                        "criterion_id": criterion_id,
                        "criterion_text": CRITERIA[criterion_id],
                        "target_semantic": target_semantic,
                        "target_option": target_option,
                        "branches_per_episode": int(branches),
                        "metadata": {
                            "criterion_targets": pair[
                                "criterion_targets"
                            ],
                            "criterion_gaps_a_minus_b": pair[
                                "criterion_gaps_a_minus_b"
                            ],
                            "option_a_attributes_canonical": pair[
                                "option_a_attributes"
                            ],
                            "option_b_attributes_canonical": pair[
                                "option_b_attributes"
                            ],
                            "pair_signature": str(
                                pair["pair_signature"]
                            ),
                            "confirmation_locked": True,
                        },
                    }
                )
    return episodes


def materialize(
    *,
    source_suite_dir: Path,
    out_dir: Path,
    branches: int,
    seed: int,
) -> dict[str, Any]:
    source_manifest = read_json(source_suite_dir / "manifest.json")
    pairs = read_jsonl(source_suite_dir / "pairs.jsonl")
    if not pairs:
        raise ValueError(f"No source pairs found in {source_suite_dir}")
    episodes = build_episodes(pairs, branches=int(branches))
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "pairs.jsonl", pairs)
    _write_jsonl(out_dir / "episodes.jsonl", episodes)
    manifest = {
        "stage": "helpsteer2-enforced-structure-suite",
        "source_suite_dir": str(source_suite_dir),
        "source_confirmation_freeze_hash": source_manifest.get(
            "confirmation_freeze_hash"
        ),
        "out_dir": str(out_dir),
        "seed": int(seed),
        "conditions": list(CONDITIONS),
        "branches": int(branches),
        "n_pairs": int(len(pairs)),
        "n_episodes": int(len(episodes)),
        "n_planned_traces": int(len(episodes) * int(branches)),
        "counts_by_condition": dict(
            sorted(Counter(row["condition_id"] for row in episodes).items())
        ),
        "counts_by_transition": dict(
            sorted(Counter(row["transition_type"] for row in pairs).items())
        ),
        "pairs_jsonl": str(out_dir / "pairs.jsonl"),
        "episodes_jsonl": str(out_dir / "episodes.jsonl"),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    source_suite_dir = _resolve(workspace_root, args.source_suite_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    manifest = materialize(
        source_suite_dir=source_suite_dir,
        out_dir=out_dir,
        branches=int(args.branches),
        seed=int(args.seed),
    )
    print(f"out_dir={out_dir}")
    print(f"n_pairs={manifest['n_pairs']}")
    print(f"n_episodes={manifest['n_episodes']}")
    print(f"n_planned_traces={manifest['n_planned_traces']}")


if __name__ == "__main__":
    main()
