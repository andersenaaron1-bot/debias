"""Run the full post-train reward evaluation suite for one adapter run.

This wrapper centralizes the evaluation battery for reward-model runs:
  - preference retention on SHP validation pairs
  - style sensitivity on the matching held-out invariance set
  - Laurito-style authorship bias on locally reconstructed A/B trials
  - reward-model benchmark retention on multiple-choice tasks
  - optional H/G/R triads if rewrite data are available

It resolves the relevant eval inputs from the matching experiment config when
possible, so a finished run directory is usually enough.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

from aisafety.config import DATA_DIR, DEFAULT_CACHE_DIR, DEFAULT_MODEL_ID, DEFAULT_SEED
from aisafety.data import DOMAINS, build_all_trials


DEFAULT_BENCHMARKS = "arc_challenge,hellaswag,winogrande,piqa,social_iqa,boolq,mmlu"


def _default_cache_dir() -> Path:
    return Path(os.environ.get("HF_HOME") or DEFAULT_CACHE_DIR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workspace-root", type=Path, default=Path.cwd())
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--experiment-config", type=Path, default=None)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--cache-dir", type=Path, default=_default_cache_dir())
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--pref-jsonl", type=Path, default=None)
    p.add_argument("--style-jsonl", type=Path, default=None)
    p.add_argument("--laurito-trials-csv", type=Path, default=None)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARKS)
    p.add_argument("--benchmark-max-examples", type=int, default=250)
    p.add_argument("--triad-rewrite-jsonl", type=Path, default=None)
    p.add_argument("--triad-rewrite-dimension", type=str, default="ai_tone")
    p.add_argument("--triad-rewrite-label", type=str, default="rlhf_ai_tone")
    p.add_argument("--skip-triads", action="store_true")
    p.add_argument("--skip-benchmarks", action="store_true")
    return p.parse_args()


def _load_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def infer_experiment_config_path(run_dir: Path, workspace_root: Path) -> Path | None:
    candidate = workspace_root / "configs" / "experiments" / f"{run_dir.name}.json"
    return candidate if candidate.exists() else None


def resolve_run_context(
    *,
    run_dir: Path,
    workspace_root: Path,
    experiment_config: Path | None,
    model_id_override: str | None,
    pref_jsonl_override: Path | None,
    style_jsonl_override: Path | None,
) -> dict[str, Any]:
    exp_path = experiment_config if experiment_config is not None else infer_experiment_config_path(run_dir, workspace_root)
    exp_cfg = _load_json_dict(exp_path) if exp_path is not None and exp_path.exists() else {}
    run_cfg_path = run_dir / "run_config.json"
    run_cfg = _load_json_dict(run_cfg_path) if run_cfg_path.exists() else {}

    model_id = str(
        model_id_override
        or exp_cfg.get("model_id")
        or run_cfg.get("model_id")
        or DEFAULT_MODEL_ID
    )
    pref_jsonl = pref_jsonl_override
    if pref_jsonl is None:
        pref_raw = exp_cfg.get("pref_val_jsonl")
        pref_jsonl = None if pref_raw is None else (workspace_root / str(pref_raw)).resolve()

    style_jsonl = style_jsonl_override
    if style_jsonl is None:
        style_raw = exp_cfg.get("style_val_jsonl")
        style_jsonl = None if style_raw is None else (workspace_root / str(style_raw)).resolve()

    return {
        "experiment_config": None if exp_path is None else exp_path.resolve(),
        "model_id": model_id,
        "pref_jsonl": None if pref_jsonl is None else pref_jsonl.resolve(),
        "style_jsonl": None if style_jsonl is None else style_jsonl.resolve(),
    }


def build_laurito_trials_csv(*, out_csv: Path, seed: int) -> Path:
    df = build_all_trials(DOMAINS, seed=int(seed), balance_order=True)
    if df.empty:
        raise RuntimeError("Could not build Laurito trials from local domain data.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def _run_stage(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    summary: dict[str, Any],
    optional: bool = False,
) -> bool:
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        summary["stages"][name] = {"status": "ok", "command": cmd, "optional": optional}
        return True
    except subprocess.CalledProcessError as exc:
        summary["stages"][name] = {
            "status": "failed",
            "command": cmd,
            "optional": optional,
            "returncode": int(exc.returncode),
        }
        return False


def main() -> None:
    args = parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    out_dir = (run_dir / "eval") if args.out_dir is None else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = run_dir / "lora_adapter"
    value_head = run_dir / "value_head.pt"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")
    if not value_head.exists():
        raise FileNotFoundError(f"Missing value head: {value_head}")

    ctx = resolve_run_context(
        run_dir=run_dir,
        workspace_root=workspace_root,
        experiment_config=args.experiment_config,
        model_id_override=args.model_id,
        pref_jsonl_override=args.pref_jsonl,
        style_jsonl_override=args.style_jsonl,
    )

    if ctx["pref_jsonl"] is None:
        raise ValueError("Could not resolve preference validation JSONL; pass --pref-jsonl or provide a matching experiment config.")
    if ctx["style_jsonl"] is None:
        raise ValueError("Could not resolve style validation JSONL; pass --style-jsonl or provide a matching experiment config.")

    laurito_trials_csv = (
        Path(args.laurito_trials_csv).resolve()
        if args.laurito_trials_csv is not None
        else build_laurito_trials_csv(out_csv=out_dir / "inputs" / "laurito_trials.csv", seed=int(args.seed))
    )

    triad_rewrite_jsonl = (
        Path(args.triad_rewrite_jsonl).resolve()
        if args.triad_rewrite_jsonl is not None
        else (workspace_root / "data" / "derived" / "openrouter_style_pairs_test" / "ai_tone.jsonl").resolve()
    )

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "model_id": str(ctx["model_id"]),
        "adapter_dir": str(adapter_dir),
        "value_head": str(value_head),
        "experiment_config": None if ctx["experiment_config"] is None else str(ctx["experiment_config"]),
        "pref_jsonl": str(ctx["pref_jsonl"]),
        "style_jsonl": str(ctx["style_jsonl"]),
        "laurito_trials_csv": str(laurito_trials_csv),
        "triad_rewrite_jsonl": str(triad_rewrite_jsonl),
        "stages": {},
    }

    py = sys.executable
    common = [
        "--model-id",
        str(ctx["model_id"]),
        "--cache-dir",
        str(Path(args.cache_dir).resolve()),
        "--lora-adapter-dir",
        str(adapter_dir),
        "--value-head",
        str(value_head),
        "--max-length",
        str(int(args.max_length)),
        "--batch-size",
        str(int(args.batch_size)),
    ]

    ok_pref = _run_stage(
        "pref_retention",
        [
            py,
            "-m",
            "aisafety.scripts.eval_pref_retention",
            "--pref-jsonl",
            str(ctx["pref_jsonl"]),
            *common,
            "--out-json",
            str(out_dir / "pref_retention.json"),
        ],
        cwd=workspace_root,
        summary=summary,
        optional=False,
    )

    ok_style = _run_stage(
        "style_sensitivity",
        [
            py,
            "-m",
            "aisafety.scripts.eval_style_sensitivity",
            "--style-jsonl",
            str(ctx["style_jsonl"]),
            *common,
            "--out-json",
            str(out_dir / "style_sensitivity.json"),
            "--out-csv",
            str(out_dir / "style_sensitivity.csv"),
        ],
        cwd=workspace_root,
        summary=summary,
        optional=False,
    )

    ok_laurito = _run_stage(
        "laurito_bias",
        [
            py,
            "-m",
            "aisafety.scripts.eval_laurito_bias_reward",
            "--trials-csv",
            str(laurito_trials_csv),
            *common,
            "--seed",
            str(int(args.seed)),
            "--bootstrap",
            str(int(args.bootstrap)),
            "--out-json",
            str(out_dir / "laurito_bias.json"),
        ],
        cwd=workspace_root,
        summary=summary,
        optional=False,
    )

    ok_bench = True
    if not bool(args.skip_benchmarks):
        ok_bench = _run_stage(
            "reward_benchmarks",
            [
                py,
                "-m",
                "aisafety.scripts.eval_reward_benchmarks",
                "--benchmark",
                str(args.benchmark),
                "--model-id",
                str(ctx["model_id"]),
                "--cache-dir",
                str(Path(args.cache_dir).resolve()),
                "--max-length",
                str(int(args.max_length)),
                "--batch-size",
                str(int(args.batch_size)),
                "--max-examples",
                str(int(args.benchmark_max_examples)),
                "--no-base-run",
                "--run",
                f"{run_dir.name}={adapter_dir}::{value_head}",
                "--out-dir",
                str(out_dir / "reward_benchmarks"),
            ],
            cwd=workspace_root,
            summary=summary,
            optional=False,
        )
    else:
        summary["stages"]["reward_benchmarks"] = {"status": "skipped", "optional": False, "reason": "skip flag"}

    if bool(args.skip_triads):
        summary["stages"]["triads"] = {"status": "skipped", "optional": True, "reason": "skip flag"}
    elif triad_rewrite_jsonl.exists():
        _run_stage(
            "triads",
            [
                py,
                "-m",
                "aisafety.scripts.eval_triads_reward",
                "--trials-csv",
                str(laurito_trials_csv),
                *common,
                "--seed",
                str(int(args.seed)),
                "--rewrite-jsonl",
                str(triad_rewrite_jsonl),
                "--rewrite-dimension",
                str(args.triad_rewrite_dimension),
                "--rewrite-label",
                str(args.triad_rewrite_label),
                "--out-json",
                str(out_dir / "triads.json"),
                "--out-csv",
                str(out_dir / "triads.csv"),
            ],
            cwd=workspace_root,
            summary=summary,
            optional=True,
        )
    else:
        summary["stages"]["triads"] = {
            "status": "skipped",
            "optional": True,
            "reason": f"rewrite jsonl not found: {triad_rewrite_jsonl}",
        }

    if (out_dir / "reward_benchmarks" / "summary.csv").exists():
        try:
            df = pd.read_csv(out_dir / "reward_benchmarks" / "summary.csv")
            summary["reward_benchmarks_rows"] = int(len(df))
        except Exception:
            pass

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote {summary_path}")

    failed_required = not all([ok_pref, ok_style, ok_laurito, ok_bench])
    if failed_required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
