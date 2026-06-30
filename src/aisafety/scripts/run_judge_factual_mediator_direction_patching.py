"""Steer factual forced readouts along mediator/decision probe directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_json, read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.mech.judge_reasoning import render_model_prompt
from aisafety.scripts.run_d4_bt_stage_contrast import _load_lm
from aisafety.scripts.run_judge_criterion_confirmation_patching import (
    _probabilities,
    _score_readout,
)
from aisafety.scripts.run_judge_prose_decision_direction_patching import (
    _direction_for_probe,
)
from aisafety.scripts.run_judge_reasoning_budget_sweep import (
    _forced_prompt,
    _single_token_label_ids,
)


DEFAULT_DATASETS = (
    "arc_challenge",
    "bbh_logical_deduction",
    "gsm8k_verification",
    "math500_verification",
    "truthfulqa",
)
PATCH_METRICS = (
    "target_selected",
    "target_probability",
    "target_logit_margin",
    "choice_confidence",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--budget-run-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--labels", default="A,B")
    parser.add_argument(
        "--probe-targets",
        default="criterion_target,current_choice,final_choice",
    )
    parser.add_argument("--include-datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--budget-tokens", type=int, default=2048)
    parser.add_argument("--branch-index", type=int, default=0)
    parser.add_argument("--alphas", default="-2.0,-1.0,0.0,1.0,2.0")
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--max-score-length", type=int, default=8192)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--raw-directions",
        action="store_true",
        help="Do not L2-normalize probe coefficients before steering.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/"
            "judge_factual_mediator_direction_patching_v1"
        ),
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    resolved = resolve_path(root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _csv(raw: str) -> list[str]:
    return [value.strip() for value in str(raw).split(",") if value.strip()]


def _floats(raw: str) -> list[float]:
    return [float(value) for value in _csv(raw)]


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in values:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _bootstrap_mean(
    values: pd.Series,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    point = float(np.mean(array)) if len(array) else np.nan
    if int(samples) <= 0 or not len(array):
        return point, np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    draws = [
        float(np.mean(rng.choice(array, size=len(array), replace=True)))
        for _ in range(int(samples))
    ]
    return (
        point,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _load_probe_payload(analysis_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = read_json(analysis_dir / "manifest.json")
    candidates: list[Path] = []
    raw_arrays = manifest.get("probe_arrays")
    if raw_arrays:
        raw_path = Path(str(raw_arrays))
        candidates.append(raw_path if raw_path.is_absolute() else analysis_dir / raw_path)
    candidates.append(analysis_dir / "probe_arrays.npz")
    arrays_path = next((path for path in candidates if path.exists()), None)
    if arrays_path is None:
        raise FileNotFoundError(
            "Could not find probe arrays; checked "
            + ", ".join(str(path) for path in candidates)
        )
    arrays_file = np.load(arrays_path, allow_pickle=False)
    arrays = {name: arrays_file[name] for name in arrays_file.files}
    specs = manifest.get("probe_specs") or {}
    if not isinstance(specs, dict) or not specs:
        raise ValueError(f"No probe_specs found in {analysis_dir / 'manifest.json'}")
    return manifest, arrays


def _score_index(
    rows: list[dict[str, Any]],
    *,
    budget_tokens: int,
    datasets: set[str],
) -> dict[str, dict[str, Any]]:
    return {
        str(row["trace_id"]): row
        for row in rows
        if str(row.get("reasoning_mode") or "") == "thinking"
        and int(row.get("budget_tokens") or 0) == int(budget_tokens)
        and str(row.get("source_dataset") or "") in datasets
        and str(row.get("target_option") or "").strip().upper() in {"A", "B"}
    }


def _selected_traces(
    traces: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    *,
    datasets: set[str],
    branch_index: int,
    max_pairs: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = []
    for row in traces:
        if str(row.get("reasoning_mode") or "") != "thinking":
            continue
        if str(row.get("source_dataset") or "") not in datasets:
            continue
        if int(row.get("branch_index") or 0) != int(branch_index):
            continue
        if str(row.get("trace_id") or "") not in scores:
            continue
        selected.append(row)
    if int(max_pairs) <= 0:
        return selected
    pair_ids = sorted(
        {str(row["pair_id"]) for row in selected},
        key=lambda pair_id: sha1_hex(f"{seed}:factual-mediator-patch-cap:{pair_id}"),
    )[: int(max_pairs)]
    allowed = set(pair_ids)
    return [row for row in selected if str(row["pair_id"]) in allowed]


def _forced_readout(
    trace: dict[str, Any],
    *,
    tokenizer: Any,
    prompt_style: str,
    budget_tokens: int,
    max_score_length: int,
) -> str:
    prompt = render_model_prompt(
        trace,
        tokenizer,
        prompt_style=str(prompt_style),
        reasoning_mode="thinking",
    )
    generated = [int(value) for value in trace.get("generated_token_ids") or []]
    prefix = tokenizer.decode(
        generated[: min(int(budget_tokens), len(generated))],
        skip_special_tokens=False,
    )
    forced = _forced_prompt(prompt, prefix, thinking=True)
    encoded = tokenizer(
        forced,
        add_special_tokens=True,
        truncation=True,
        max_length=int(max_score_length),
    )["input_ids"]
    return tokenizer.decode(encoded, skip_special_tokens=False)


def _desired_label(probe_target: str, target_option: str) -> str | None:
    if probe_target in {"criterion_target", "current_choice", "final_choice"}:
        return str(target_option)
    if probe_target == "target_reached":
        return "1"
    return None


def _summarize_patch_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["probe_target", "source_dataset", "alpha"],
        sort=True,
    ):
        rows.append(
            {
                "probe_target": keys[0],
                "source_dataset": keys[1],
                "alpha": float(keys[2]),
                "n_rows": int(len(group)),
                "n_pairs": int(group["pair_id"].nunique()),
                "target_adoption_rate": float(group["target_selected"].mean()),
                "mean_target_probability": float(group["target_probability"].mean()),
                "mean_target_logit_margin": float(group["target_logit_margin"].mean()),
                "mean_choice_confidence": float(group["choice_confidence"].mean()),
            }
        )
    all_rows = []
    for keys, group in frame.groupby(["probe_target", "alpha"], sort=True):
        all_rows.append(
            {
                "probe_target": keys[0],
                "source_dataset": "all",
                "alpha": float(keys[1]),
                "n_rows": int(len(group)),
                "n_pairs": int(group["pair_id"].nunique()),
                "target_adoption_rate": float(group["target_selected"].mean()),
                "mean_target_probability": float(group["target_probability"].mean()),
                "mean_target_logit_margin": float(group["target_logit_margin"].mean()),
                "mean_choice_confidence": float(group["choice_confidence"].mean()),
            }
        )
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(all_rows)], ignore_index=True)


def _patch_effects(
    frame: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped_frames = []
    grouped_frames.append(("source_dataset", frame.copy()))
    all_frame = frame.copy()
    all_frame["source_dataset"] = "all"
    grouped_frames.append(("all", all_frame))
    key_cols = ["probe_target", "source_dataset", "pair_id", "trace_id"]
    for _, source_frame in grouped_frames:
        for keys, group in source_frame.groupby(["probe_target", "source_dataset"], sort=True):
            baseline = group[group["alpha"].eq(0.0)][key_cols + list(PATCH_METRICS)]
            if baseline.empty:
                continue
            for alpha in sorted(value for value in group["alpha"].unique() if float(value) != 0.0):
                selected = group[group["alpha"].eq(float(alpha))]
                merged = selected[key_cols + list(PATCH_METRICS)].merge(
                    baseline,
                    on=key_cols,
                    suffixes=("_left", "_baseline"),
                )
                if merged.empty:
                    continue
                for metric in PATCH_METRICS:
                    merged["difference"] = (
                        merged[f"{metric}_left"].astype(float)
                        - merged[f"{metric}_baseline"].astype(float)
                    )
                    pair_values = merged.groupby("pair_id", sort=True)["difference"].mean()
                    point, low, high = _bootstrap_mean(
                        pair_values,
                        samples=int(bootstrap),
                        seed=int(
                            sha1_hex(
                                f"{seed}:factual-patch:{keys[0]}:{keys[1]}:{alpha}:{metric}"
                            )[:8],
                            16,
                        ),
                    )
                    rows.append(
                        {
                            "probe_target": keys[0],
                            "source_dataset": keys[1],
                            "alpha": float(alpha),
                            "reference_alpha": 0.0,
                            "metric": metric,
                            "n_pairs": int(len(pair_values)),
                            "mean": point,
                            "ci95_low": low,
                            "ci95_high": high,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    budget_run_dir = _resolve(workspace_root, args.budget_run_dir)
    analysis_dir = _resolve(workspace_root, args.analysis_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    run_manifest = read_json(budget_run_dir / "manifest.json")
    analysis_manifest, arrays = _load_probe_payload(analysis_dir)
    specs = analysis_manifest.get("probe_specs") or {}
    probe_targets = [target for target in _csv(args.probe_targets) if target in specs]
    if not probe_targets:
        raise ValueError("None of --probe-targets are present in analysis manifest.")
    datasets = set(_csv(args.include_datasets))
    scores = _score_index(
        read_jsonl(budget_run_dir / "budget_scores.jsonl"),
        budget_tokens=int(args.budget_tokens),
        datasets=datasets,
    )
    traces = _selected_traces(
        read_jsonl(budget_run_dir / "reasoning_traces.jsonl"),
        scores,
        datasets=datasets,
        branch_index=int(args.branch_index),
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
    )
    if not traces:
        raise ValueError("No factual traces match the patching filters.")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "patch_rows.jsonl"
    if rows_path.exists() and not bool(args.resume):
        raise FileExistsError(
            f"{rows_path} exists; use --resume or choose a new output directory."
        )
    existing = read_jsonl(rows_path) if rows_path.exists() else []
    completed = {
        (
            str(row["probe_target"]),
            str(row["trace_id"]),
            str(row["direction_label"]),
            float(row["alpha"]),
        )
        for row in existing
    }

    model, tokenizer = _load_lm(args)
    labels = _csv(args.labels)
    label_ids = _single_token_label_ids(tokenizer, labels)
    label_index = {label: index for index, label in enumerate(labels)}
    alphas = _floats(args.alphas)
    if 0.0 not in alphas:
        alphas = [0.0] + alphas
    prompt_style = str(run_manifest.get("prompt_style") or "chat_template")
    new_rows: list[dict[str, Any]] = []
    for trace in traces:
        score = scores[str(trace["trace_id"])]
        target_option = str(score.get("target_option") or "").strip().upper()
        if target_option not in label_index:
            continue
        prompt = _forced_readout(
            trace,
            tokenizer=tokenizer,
            prompt_style=prompt_style,
            budget_tokens=int(args.budget_tokens),
            max_score_length=int(args.max_score_length),
        )
        target_position = label_index[target_option]
        for probe_target in probe_targets:
            spec = specs[probe_target]
            direction, direction_label = _direction_for_probe(
                arrays,
                probe_target,
                desired_label=_desired_label(probe_target, target_option),
                normalize=not bool(args.raw_directions),
            )
            for alpha in alphas:
                completion_key = (
                    probe_target,
                    str(trace["trace_id"]),
                    direction_label,
                    float(alpha),
                )
                if completion_key in completed:
                    continue
                vector = None if float(alpha) == 0.0 else direction
                logits = _score_readout(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    label_ids=label_ids,
                    hidden_layer=int(spec["hidden_layer"]),
                    vector=vector,
                    alpha=float(alpha),
                    max_length=int(args.max_score_length),
                )
                probabilities = _probabilities(logits)
                predicted = labels[int(np.argmax(logits))]
                alternatives = np.delete(logits, target_position)
                sorted_probabilities = np.sort(probabilities)[::-1]
                output = {
                    "patch_id": sha1_hex(
                        f"{probe_target}:{trace['trace_id']}:{direction_label}:{alpha}"
                    ),
                    "probe_target": probe_target,
                    "direction_label": str(direction_label),
                    "trace_id": str(trace["trace_id"]),
                    "pair_id": str(trace["pair_id"]),
                    "source_dataset": str(trace.get("source_dataset") or ""),
                    "branch_index": int(trace.get("branch_index") or 0),
                    "hidden_layer": int(spec["hidden_layer"]),
                    "point_index": int(spec["point_index"]),
                    "budget_tokens": int(args.budget_tokens),
                    "alpha": float(alpha),
                    "direction_norm": float(np.linalg.norm(direction)),
                    "target_option": target_option,
                    "predicted_option": predicted,
                    "target_selected": bool(predicted == target_option),
                    "target_probability": float(probabilities[target_position]),
                    "target_logit_margin": float(
                        logits[target_position] - float(np.max(alternatives))
                    ),
                    "choice_confidence": float(
                        sorted_probabilities[0] - sorted_probabilities[1]
                    ),
                }
                for index, label in enumerate(labels):
                    output[f"logit_{label}"] = float(logits[index])
                    output[f"prob_{label}"] = float(probabilities[index])
                _append_jsonl(rows_path, [output])
                new_rows.append(output)
                completed.add(completion_key)

    frame = pd.DataFrame(read_jsonl(rows_path))
    summary = _summarize_patch_rows(frame)
    effects = _patch_effects(
        frame,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    frame.to_csv(out_dir / "patch_rows.csv", index=False)
    summary.to_csv(out_dir / "patch_summary.csv", index=False)
    effects.to_csv(out_dir / "patch_effects.csv", index=False)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-factual-mediator-direction-patching",
            "budget_run_dir": str(budget_run_dir),
            "analysis_dir": str(analysis_dir),
            "out_dir": str(out_dir),
            "model_id": str(args.model_id),
            "probe_targets": probe_targets,
            "include_datasets": sorted(datasets),
            "budget_tokens": int(args.budget_tokens),
            "branch_index": int(args.branch_index),
            "alphas": alphas,
            "directions_normalized": not bool(args.raw_directions),
            "max_pairs": int(args.max_pairs),
            "n_patch_rows": int(len(frame)),
            "n_new_patch_rows": int(len(new_rows)),
            "outputs": {
                "patch_rows_jsonl": str(rows_path),
                "patch_rows_csv": str(out_dir / "patch_rows.csv"),
                "patch_summary_csv": str(out_dir / "patch_summary.csv"),
                "patch_effects_csv": str(out_dir / "patch_effects.csv"),
            },
            "seed": int(args.seed),
        },
    )
    print(f"out_dir={out_dir}")
    print(f"n_patch_rows={len(frame)}")
    if not summary.empty:
        print("\n=== FACTUAL DIRECTION PATCH SUMMARY ===")
        print(summary.round(3).to_string(index=False))
    if not effects.empty:
        print("\n=== FACTUAL DIRECTION PATCH EFFECTS ===")
        print(effects.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
