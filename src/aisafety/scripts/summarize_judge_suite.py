"""Print a compact readout for the current judge matrix.

This is intended for login-node friendly inspection after:

- full reward eval for `J0` and `Jrepair-all`
- `D3` ecological validation for the first-wave suite
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ANCHOR_RUN = "j0_anchor_v1_h100compact"
DEFAULT_REPAIR_RUN = "jrepair_all_v1"
DEFAULT_ABLATION_RUNS = [
    "jrepair_loo_joint_academic_formality_v1",
    "jrepair_loo_cue_template_boilerplate_v1",
    "jrepair_loo_cue_hedging_certainty_v1",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace-root", type=Path, default=Path.cwd())
    p.add_argument("--anchor-run", type=str, default=DEFAULT_ANCHOR_RUN)
    p.add_argument("--repair-run", type=str, default=DEFAULT_REPAIR_RUN)
    p.add_argument(
        "--ablation-runs",
        type=str,
        default=",".join(DEFAULT_ABLATION_RUNS),
        help="Comma-separated ablation run ids to include in the D3 view.",
    )
    p.add_argument("--top-bundles", type=int, default=6)
    p.add_argument("--top-contrasts", type=int, default=8)
    return p.parse_args()


def _csv_list(val: str) -> list[str]:
    return [part.strip() for part in str(val or "").split(",") if part.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def _load_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float_or_nan(val: str | None) -> float:
    if val is None or str(val).strip() == "":
        return float("nan")
    return float(val)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _reward_eval_dir(workspace_root: Path, run_id: str) -> Path:
    return workspace_root / "artifacts" / "reward" / run_id / "eval"


def _d3_dir(workspace_root: Path, run_id: str) -> Path:
    return workspace_root / "artifacts" / "style_groups" / f"d3_{run_id}"


def load_core_eval_summary(workspace_root: Path, run_id: str) -> dict[str, Any]:
    eval_dir = _reward_eval_dir(workspace_root, run_id)
    pref = _load_json(_require(eval_dir / "pref_retention.json"))
    style = _load_json(_require(eval_dir / "style_sensitivity.json"))
    laur = _load_json(_require(eval_dir / "laurito_bias.json"))
    bench = _load_csv_rows(_require(eval_dir / "reward_benchmarks" / "summary.csv"))
    return {"pref": pref, "style": style, "laurito": laur, "benchmarks": bench}


def load_bundle_effects(workspace_root: Path, run_id: str) -> dict[str, dict[str, str]]:
    rows = _load_tsv_rows(_require(_d3_dir(workspace_root, run_id) / "bundle_effects.tsv"))
    return {row["name"]: row for row in rows if row.get("name")}


def format_core_eval_summary(run_id: str, payload: dict[str, Any]) -> list[str]:
    pref = payload["pref"]
    style = payload["style"]
    laur = payload["laurito"]
    bench = payload["benchmarks"]
    mean_acc = (
        sum(_float_or_nan(row.get("accuracy")) for row in bench) / float(len(bench))
        if bench
        else float("nan")
    )
    style_summary = {
        row["style_axis"]: float(row["mean_d"])
        for row in style.get("summary", [])
        if isinstance(row, dict) and row.get("style_axis") is not None and row.get("mean_d") is not None
    }
    lines = [
        f"=== {run_id} ===",
        (
            f"pref_acc={float(pref['pairwise_acc']):.4f} "
            f"auc={float(pref['separation_auc']):.4f} "
            f"margin={float(pref['mean_margin']):.4f}"
        ),
        (
            f"laurito overall={float(laur['raw']['overall']['prop_llm_chosen']):.4f} "
            f"paper={float(laur['raw']['paper']['prop_llm_chosen']):.4f} "
            f"product={float(laur['raw']['product']['prop_llm_chosen']):.4f} "
            f"movie={float(laur['raw']['movie']['prop_llm_chosen']):.4f}"
        ),
        "style " + " ".join(f"{k}={v:.4f}" for k, v in sorted(style_summary.items())),
        f"bench_mean_acc={mean_acc:.4f}",
    ]
    for row in bench:
        lines.append(f"  {row['benchmark']}: acc={float(row['accuracy']):.4f}")
    return lines


def format_top_bundles(
    run_id: str,
    bundle_rows: dict[str, dict[str, str]],
    *,
    top_n: int,
) -> list[str]:
    rows = [row for row in bundle_rows.values() if row.get("signed_effect_z")]
    rows.sort(key=lambda row: abs(float(row["signed_effect_z"])), reverse=True)
    lines = [f"=== {run_id} ==="]
    for row in rows[: int(top_n)]:
        lines.append(
            f"{row['name']:<34} "
            f"z={float(row['signed_effect_z']): .3f}  "
            f"ci=[{float(row['signed_effect_ci_95_low']): .3f},{float(row['signed_effect_ci_95_high']): .3f}]  "
            f"auc={float((row.get('auc_llm_choice') or 'nan')):.3f}  "
            f"d2={row.get('d2_status', '')}"
        )
    return lines


def format_bundle_contrast(
    reference_run: str,
    other_run: str,
    *,
    reference_rows: dict[str, dict[str, str]],
    current_rows: dict[str, dict[str, str]],
    top_n: int,
) -> list[str]:
    diffs: list[tuple[float, str, float, float]] = []
    for name in sorted(set(reference_rows) | set(current_rows)):
        z_ref = _float_or_nan(reference_rows.get(name, {}).get("signed_effect_z"))
        z_cur = _float_or_nan(current_rows.get(name, {}).get("signed_effect_z"))
        if z_ref != z_ref:  # nan
            z_ref = 0.0
        if z_cur != z_cur:
            z_cur = 0.0
        diffs.append((abs(z_cur - z_ref), name, z_cur, z_ref))
    diffs.sort(reverse=True)
    lines = [f"=== {other_run} vs {reference_run} ==="]
    for _, name, z_cur, z_ref in diffs[: int(top_n)]:
        lines.append(f"{name:<34} {other_run}={z_cur: .3f}  all={z_ref: .3f}  d={z_cur - z_ref: .3f}")
    return lines


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    ablation_runs = _csv_list(args.ablation_runs)
    d3_runs = [str(args.anchor_run), str(args.repair_run), *ablation_runs]

    print("## Core Eval")
    for run_id in [str(args.anchor_run), str(args.repair_run)]:
        for line in format_core_eval_summary(run_id, load_core_eval_summary(workspace_root, run_id)):
            print(line)
        print()

    print("## D3 Top Bundles")
    bundle_tables = {run_id: load_bundle_effects(workspace_root, run_id) for run_id in d3_runs}
    for run_id in d3_runs:
        for line in format_top_bundles(run_id, bundle_tables[run_id], top_n=int(args.top_bundles)):
            print(line)
        print()

    print("## D3 Contrast Vs Jrepair-all")
    ref_run = str(args.repair_run)
    for run_id in [str(args.anchor_run), *ablation_runs]:
        for line in format_bundle_contrast(
            ref_run,
            run_id,
            reference_rows=bundle_tables[ref_run],
            current_rows=bundle_tables[run_id],
            top_n=int(args.top_contrasts),
        ):
            print(line)
        print()


if __name__ == "__main__":
    main()
