"""Evaluate reward-model runs on candidate-scoring NLP benchmarks.

This suite is designed for scalar reward scorers trained via `train_reward_lora`.
It evaluates runs by scoring candidate answers independently and selecting the
highest-scoring option per example.

Example:
  python -m aisafety.scripts.eval_reward_benchmarks ^
    --model-id google/gemma-2-9b-it ^
    --base-value-head artifacts\\reward\\baseline_pref_only_v2\\value_head.pt ^
    --run pref_only=artifacts\\reward\\baseline_pref_only_v2\\lora_adapter::artifacts\\reward\\baseline_pref_only_v2\\value_head.pt ^
    --run invariance=artifacts\\reward\\baseline_pref_only\\lora_adapter::artifacts\\reward\\baseline_pref_only\\value_head.pt ^
    --max-examples 500 ^
    --out-dir artifacts\\reward_benchmarks\\pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED
from aisafety.eval.benchmark_tasks import (
    BENCHMARKS,
    available_benchmarks,
    benchmark_descriptions,
    compute_mcq_metrics,
    load_benchmark_examples,
    make_mcq_record,
    parse_run_spec,
)
from aisafety.reward.model import load_reward_scorer
from aisafety.reward.text_format import format_prompt_response


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--benchmark",
        type=str,
        default="arc_challenge,hellaswag,winogrande,piqa,social_iqa,boolq,mmlu",
        help=f"Comma-separated benchmarks. Available: {', '.join(available_benchmarks())}",
    )
    p.add_argument(
        "--split-override",
        action="append",
        default=[],
        help="Optional benchmark split override as name=split (repeatable).",
    )
    p.add_argument("--list-benchmarks", action="store_true", help="Print benchmark names/descriptions and exit.")

    p.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-examples", type=int, default=None, help="Optional per-benchmark cap after shuffling.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--base-run-name", type=str, default="base")
    p.add_argument("--base-value-head", type=Path, default=None, help="Value head for the no-adapter base run.")
    p.add_argument("--include-base-run", action="store_true")
    p.add_argument("--no-base-run", dest="include_base_run", action="store_false")
    p.set_defaults(include_base_run=True)
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="Adapter run as name=ADAPTER_DIR::VALUE_HEAD (repeatable).",
    )

    p.add_argument("--out-dir", type=Path, default=Path("artifacts/reward_benchmarks"))
    return p.parse_args()


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _parse_overrides(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        raw = str(item or "").strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Split override must be formatted as name=split: {item!r}")
        name, split = raw.split("=", 1)
        name = name.strip()
        split = split.strip()
        if not name or not split:
            raise ValueError(f"Split override must be formatted as name=split: {item!r}")
        out[name] = split
    return out


@torch.no_grad()
def _score_texts(model, tok, texts: list[str], *, max_length: int, batch_size: int, device) -> list[float]:
    scores: list[float] = []
    model.eval()
    for i in range(0, len(texts), int(batch_size)):
        batch = texts[i : i + int(batch_size)]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        s = model(enc["input_ids"], enc["attention_mask"]).detach().float().cpu().numpy()
        scores.extend([float(x) for x in s.tolist()])
    return scores


def main() -> None:
    args = parse_args()

    if bool(args.list_benchmarks):
        for name, desc in benchmark_descriptions():
            print(f"{name}: {desc}")
        return

    benchmarks = _parse_csv_list(args.benchmark)
    if not benchmarks:
        raise ValueError("Provide at least one --benchmark.")
    unknown = [x for x in benchmarks if x not in BENCHMARKS]
    if unknown:
        raise ValueError(f"Unknown benchmarks: {unknown}. Available: {available_benchmarks()}")

    split_overrides = _parse_overrides(list(args.split_override or []))

    run_specs = [parse_run_spec(x) for x in list(args.run or [])]
    if bool(args.include_base_run):
        if args.base_value_head is None:
            raise ValueError("--base-value-head is required when --include-base-run is enabled.")
        run_specs = [
            parse_run_spec(f"{args.base_run_name}=::{args.base_value_head}")
        ] + run_specs
    if not run_specs:
        raise ValueError("No runs configured. Provide --base-value-head and/or one or more --run specs.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples_by_benchmark = {}
    bench_meta = {}
    for bench in benchmarks:
        split = split_overrides.get(bench)
        examples = load_benchmark_examples(
            bench,
            split=split,
            cache_dir=Path(args.cache_dir),
            max_examples=None if args.max_examples is None else int(args.max_examples),
            seed=int(args.seed),
        )
        if not examples:
            raise ValueError(f"No normalized examples loaded for benchmark {bench!r}")
        examples_by_benchmark[bench] = examples
        bench_meta[bench] = {
            "dataset_id": BENCHMARKS[bench].dataset_id,
            "config_name": BENCHMARKS[bench].config_name,
            "split": split or BENCHMARKS[bench].default_split,
            "n_examples": int(len(examples)),
            "description": BENCHMARKS[bench].description,
        }

    all_records: list[dict] = []
    summary_rows: list[dict] = []
    group_rows: list[dict] = []

    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    for run in run_specs:
        print(f"Scoring run: {run.name}")
        scorer, tok = load_reward_scorer(
            model_id=str(args.model_id),
            cache_dir=Path(args.cache_dir),
            lora_adapter_dir=run.adapter_dir,
            value_head_path=run.value_head,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        device = next(p for p in scorer.parameters() if p.device.type != "meta").device

        for bench in benchmarks:
            examples = examples_by_benchmark[bench]
            texts: list[str] = []
            spans: list[tuple[int, int]] = []
            for ex in examples:
                start = len(texts)
                texts.extend([format_prompt_response(ex.prompt, r) for r in ex.responses])
                spans.append((start, len(texts)))

            scores = _score_texts(
                scorer,
                tok,
                texts,
                max_length=int(args.max_length),
                batch_size=int(args.batch_size),
                device=device,
            )
            if len(scores) != len(texts):
                raise RuntimeError(f"Expected {len(texts)} scores for {bench}/{run.name}, got {len(scores)}")

            bench_records = []
            for ex, (start, end) in zip(examples, spans, strict=True):
                rec = make_mcq_record(ex, scores=scores[start:end], run_name=str(run.name))
                bench_records.append(rec)
                all_records.append(rec)

            summary = compute_mcq_metrics(bench_records)
            summary_rows.append({"run": str(run.name), "benchmark": str(bench), **summary})

            df_bench = pd.DataFrame(bench_records)
            if "group" in df_bench.columns:
                valid_groups = df_bench[df_bench["group"].notna() & (df_bench["group"].astype(str) != "")]
                for group_name, grp in valid_groups.groupby("group", dropna=False):
                    group_summary = compute_mcq_metrics(grp.to_dict(orient="records"))
                    group_rows.append(
                        {"run": str(run.name), "benchmark": str(bench), "group": str(group_name), **group_summary}
                    )

        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_records = pd.DataFrame(all_records)
    df_summary = pd.DataFrame(summary_rows).sort_values(["benchmark", "run"]).reset_index(drop=True)
    df_summary_groups = pd.DataFrame(group_rows).sort_values(["benchmark", "group", "run"]).reset_index(drop=True)

    df_records.to_json(out_dir / "per_question.jsonl", orient="records", lines=True, force_ascii=False)
    df_summary.to_csv(out_dir / "summary.csv", index=False)
    if not df_summary_groups.empty:
        df_summary_groups.to_csv(out_dir / "summary_by_group.csv", index=False)

    if bool(args.include_base_run):
        base_name = str(args.base_run_name)
        df_base = df_summary[df_summary["run"] == base_name].copy()
        if not df_base.empty:
            merged = df_summary.merge(
                df_base,
                on="benchmark",
                suffixes=("", "_base"),
                how="left",
            )
            metric_cols = [
                "accuracy",
                "mean_correct_margin",
                "mean_score_spread",
                "mean_gold_rank",
                "mrr",
            ]
            for col in metric_cols:
                merged[f"delta_{col}_vs_base"] = merged[col] - merged[f"{col}_base"]
            merged.to_csv(out_dir / "summary_vs_base.csv", index=False)

    meta = {
        "model_id": str(args.model_id),
        "cache_dir": str(args.cache_dir),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "max_examples": None if args.max_examples is None else int(args.max_examples),
        "seed": int(args.seed),
        "benchmarks": bench_meta,
        "runs": [
            {
                "name": str(run.name),
                "adapter_dir": None if run.adapter_dir is None else str(run.adapter_dir),
                "value_head": str(run.value_head),
            }
            for run in run_specs
        ],
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote {out_dir / 'summary.csv'}")
    if not df_summary_groups.empty:
        print(f"Wrote {out_dir / 'summary_by_group.csv'}")
    if bool(args.include_base_run):
        print(f"Wrote {out_dir / 'summary_vs_base.csv'}")
    print(f"Wrote {out_dir / 'per_question.jsonl'}")
    print(f"Wrote {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
