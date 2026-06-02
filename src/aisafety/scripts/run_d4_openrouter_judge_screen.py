"""Run a small resumable OpenRouter judge screen on order-swapped BT probes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import httpx
import pandas as pd

from aisafety.config import DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.scripts.run_d4_bt_stage_contrast import _comparison_user_content


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
SYSTEM_PROMPT = "Return exactly one uppercase letter: A or B. Do not explain your answer."
CHOICE_RE = re.compile(r"^(?:answer\s*:\s*)?([AB])(?:[.)])?\s*$", re.I)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset", action="append", default=[], help="Probe as NAME=BT_PAIRS_JSONL.")
    parser.add_argument("--model", action="append", default=[], help="Hosted judge as LABEL=OPENROUTER_MODEL_ID.")
    parser.add_argument("--contrast", action="append", default=[], help="Model contrast as NAME=LEFT-RIGHT.")
    parser.add_argument("--comparison-template", default="standard")
    parser.add_argument("--max-source-comparisons", type=int, default=100)
    parser.add_argument("--max-completion-tokens", type=int, default=4)
    parser.add_argument(
        "--reasoning-effort",
        choices=["", "none", "minimal", "low", "medium", "high", "xhigh"],
        default="",
        help="Optional OpenRouter reasoning effort. Use none for short forced-choice verdicts.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--estimate-chars-per-token", type=float, default=4.0)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _named(value: str) -> tuple[str, str]:
    if "=" not in str(value):
        raise ValueError(f"Expected NAME=VALUE, got: {value}")
    name, raw = str(value).split("=", 1)
    if not name.strip() or not raw.strip():
        raise ValueError(f"Expected nonempty NAME=VALUE, got: {value}")
    return name.strip(), raw.strip()


def _parse_contrast(value: str) -> tuple[str, str, str]:
    name, expr = _named(value) if "=" in str(value) else ("", str(value))
    if "-" not in expr:
        raise ValueError(f"Expected NAME=LEFT-RIGHT or LEFT-RIGHT, got: {value}")
    left, right = expr.split("-", 1)
    if not left.strip() or not right.strip():
        raise ValueError(f"Expected nonempty LEFT-RIGHT, got: {value}")
    return name.strip() or f"{left.strip()}_minus_{right.strip()}", left.strip(), right.strip()


def _source_id(row: dict[str, Any]) -> str:
    return str(row.get("counterfactual_id") or row.get("pair_id") or row.get("bt_pair_id") or "")


def _target_option(row: dict[str, Any]) -> str:
    value = str(row.get("cue_plus_option") or row.get("llm_option") or "").strip().upper()
    if value not in {"A", "B"}:
        raise ValueError("BT rows must provide cue_plus_option or llm_option as A/B.")
    return value


def _cap_rows(rows: list[dict[str, Any]], *, max_sources: int, seed: int, dataset: str) -> list[dict[str, Any]]:
    source_ids = sorted(
        {_source_id(row) for row in rows if _source_id(row)},
        key=lambda value: sha1_hex(f"{seed}:openrouter-screen:{dataset}:{value}"),
    )
    if int(max_sources) > 0:
        source_ids = source_ids[: int(max_sources)]
    keep = set(source_ids)
    return [row for row in rows if _source_id(row) in keep]


def _prompt(row: dict[str, Any], *, comparison_template: str) -> str:
    return _comparison_user_content(row, comparison_template=str(comparison_template))


def _request_key(*, dataset: str, model_label: str, model_id: str, row: dict[str, Any], prompt: str, args: argparse.Namespace) -> str:
    return sha1_hex(
        json.dumps(
            {
                "dataset": dataset,
                "model_label": model_label,
                "model_id": model_id,
                "bt_pair_id": str(row.get("bt_pair_id") or ""),
                "prompt_sha1": sha1_hex(prompt),
                "comparison_template": str(args.comparison_template),
                "temperature": float(args.temperature),
                "max_completion_tokens": int(args.max_completion_tokens),
                "reasoning_effort": str(args.reasoning_effort),
                "system_prompt": SYSTEM_PROMPT,
            },
            sort_keys=True,
        )
    )


def parse_choice(text: str) -> str:
    match = CHOICE_RE.search(str(text or "").strip())
    return "" if match is None else match.group(1).upper()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _cached_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {str(row.get("request_key") or ""): row for row in read_jsonl(path) if str(row.get("request_key") or "")}


def _catalog(client: httpx.Client) -> dict[str, dict[str, Any]]:
    response = client.get(OPENROUTER_MODELS_URL)
    response.raise_for_status()
    payload = response.json()
    return {str(row.get("id") or ""): row for row in payload.get("data", []) if isinstance(row, dict)}


def _estimated_tokens(prompt: str, *, chars_per_token: float) -> int:
    return max(1, int(math.ceil(len(str(prompt)) / max(float(chars_per_token), 0.1))))


def _estimate_rows(
    jobs: list[dict[str, Any]],
    *,
    models: list[tuple[str, str]],
    catalog: dict[str, dict[str, Any]],
    chars_per_token: float,
    completion_tokens: int,
) -> pd.DataFrame:
    prompt_tokens = sum(_estimated_tokens(str(job["prompt"]), chars_per_token=chars_per_token) for job in jobs)
    rows: list[dict[str, Any]] = []
    for model_label, model_id in models:
        entry = catalog.get(model_id, {})
        pricing = entry.get("pricing") if isinstance(entry.get("pricing"), dict) else {}
        prompt_price = None if not entry else float(pricing.get("prompt") or 0.0)
        completion_price = None if not entry else float(pricing.get("completion") or 0.0)
        output_tokens = int(len(jobs) * int(completion_tokens))
        rows.append(
            {
                "model_label": model_label,
                "model_id": model_id,
                "catalog_found": bool(entry),
                "n_requests": int(len(jobs)),
                "estimated_prompt_tokens": int(prompt_tokens),
                "estimated_completion_tokens": int(output_tokens),
                "prompt_price_per_token": prompt_price,
                "completion_price_per_token": completion_price,
                "estimated_cost_usd": (
                    None
                    if prompt_price is None or completion_price is None
                    else float(prompt_tokens * prompt_price + output_tokens * completion_price)
                ),
            }
        )
    return pd.DataFrame(rows)


def _call_openrouter(
    client: httpx.Client,
    *,
    api_key: str,
    model_id: str,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = {
        "model": str(model_id),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_completion_tokens),
        "seed": int(args.seed),
    }
    if str(args.reasoning_effort):
        payload["reasoning"] = {"effort": str(args.reasoning_effort)}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "AISafety D4 judge screen",
    }
    last_error = ""
    for attempt in range(1, int(args.max_retries) + 1):
        try:
            response = client.post(OPENROUTER_CHAT_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - API errors are persisted after bounded retries.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < int(args.max_retries):
                time.sleep(float(args.retry_sleep_seconds) * attempt)
    raise RuntimeError(last_error)


def _usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else {}


def _response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message")
    return str(message.get("content") or "") if isinstance(message, dict) else ""


def _score_row(
    *,
    dataset: str,
    model_label: str,
    model_id: str,
    row: dict[str, Any],
    prompt: str,
    request_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    text = _response_text(payload)
    choice = parse_choice(text)
    target = _target_option(row)
    usage = _usage(payload)
    return {
        "request_key": request_key,
        "dataset": dataset,
        "model_label": model_label,
        "model_id": model_id,
        "response_model": str(payload.get("model") or ""),
        "response_id": str(payload.get("id") or ""),
        "bt_pair_id": str(row.get("bt_pair_id") or ""),
        "source_id": _source_id(row),
        "pair_id": str(row.get("pair_id") or ""),
        "counterfactual_id": str(row.get("counterfactual_id") or ""),
        "source_dataset": str(row.get("source_dataset") or ""),
        "subset": str(row.get("subset") or ""),
        "item_type": str(row.get("item_type") or ""),
        "role": str(row.get("role") or ""),
        "presentation_order": str(row.get("presentation_order") or ""),
        "target_option": target,
        "judge_choice": choice,
        "valid_choice": bool(choice in {"A", "B"}),
        "target_preferred": bool(choice == target) if choice in {"A", "B"} else None,
        "response_text": text,
        "prompt_sha1": sha1_hex(prompt),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cost_usd": float(usage.get("cost") or 0.0),
    }


def summarize_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = scores[scores["valid_choice"].astype(bool)].copy()
    pair_rows: list[dict[str, Any]] = []
    for (dataset, model_label, model_id, source_id), group in valid.groupby(
        ["dataset", "model_label", "model_id", "source_id"], sort=True
    ):
        shares = pd.to_numeric(group["target_preferred"], errors="coerce")
        pair_rows.append(
            {
                "dataset": dataset,
                "model_label": model_label,
                "model_id": model_id,
                "source_id": source_id,
                "n_valid_order_rows": int(len(group)),
                "target_preference_share": float(shares.mean()),
                "order_consistent": bool(shares.nunique() <= 1),
                "source_dataset": str(group.iloc[0].get("source_dataset") or ""),
                "item_type": str(group.iloc[0].get("item_type") or ""),
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    summary_rows: list[dict[str, Any]] = []
    if not pair_df.empty:
        for (dataset, model_label, model_id), group in pair_df.groupby(["dataset", "model_label", "model_id"], sort=True):
            share = pd.to_numeric(group["target_preference_share"], errors="coerce")
            score_group = scores[(scores["dataset"] == dataset) & (scores["model_label"] == model_label)]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model_label": model_label,
                    "model_id": model_id,
                    "n_source_comparisons": int(len(group)),
                    "mean_target_preference_share": float(share.mean()),
                    "strict_target_preference_rate": float((share == 1.0).mean()),
                    "strict_opposite_preference_rate": float((share == 0.0).mean()),
                    "order_inconsistent_rate": float((~group["order_consistent"].astype(bool)).mean()),
                    "invalid_response_rate": float((~score_group["valid_choice"].astype(bool)).mean()),
                    "prompt_tokens": int(pd.to_numeric(score_group["prompt_tokens"], errors="coerce").sum()),
                    "completion_tokens": int(pd.to_numeric(score_group["completion_tokens"], errors="coerce").sum()),
                    "cost_usd": float(pd.to_numeric(score_group["cost_usd"], errors="coerce").sum()),
                }
            )
    return pair_df, pd.DataFrame(summary_rows)


def contrast_summaries(pair_df: pd.DataFrame, contrasts: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = set(pair_df["model_label"].astype(str)) if not pair_df.empty else set()
    for raw in contrasts:
        name, left, right = _parse_contrast(raw)
        if left not in labels or right not in labels:
            raise ValueError(f"Contrast {raw!r} references unavailable labels: {sorted(labels)}")
        for dataset in sorted(set(pair_df["dataset"].astype(str))):
            left_df = pair_df[(pair_df["dataset"] == dataset) & (pair_df["model_label"] == left)]
            right_df = pair_df[(pair_df["dataset"] == dataset) & (pair_df["model_label"] == right)]
            merged = left_df[["source_id", "target_preference_share"]].merge(
                right_df[["source_id", "target_preference_share"]],
                on="source_id",
                suffixes=("_left", "_right"),
            )
            delta = merged["target_preference_share_left"] - merged["target_preference_share_right"]
            rows.append(
                {
                    "contrast": name,
                    "left_model": left,
                    "right_model": right,
                    "dataset": dataset,
                    "n_source_comparisons": int(len(merged)),
                    "mean_delta_target_preference_share": float(delta.mean()) if len(delta) else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    out_dir = resolve_path(workspace_root, args.out_dir)
    if out_dir is None:
        raise ValueError("Could not resolve --out-dir.")
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [_named(value) for value in args.dataset]
    models = [_named(value) for value in args.model]
    if not datasets or not models:
        raise ValueError("Pass at least one --dataset and one --model.")
    jobs: list[dict[str, Any]] = []
    for dataset, raw_path in datasets:
        path = resolve_path(workspace_root, raw_path)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing dataset {dataset}: {path}")
        rows = _cap_rows(read_jsonl(path), max_sources=int(args.max_source_comparisons), seed=int(args.seed), dataset=dataset)
        for row in rows:
            jobs.append({"dataset": dataset, "row": row, "prompt": _prompt(row, comparison_template=str(args.comparison_template))})

    with httpx.Client(timeout=float(args.timeout_seconds)) as client:
        catalog = _catalog(client)
        estimate_df = _estimate_rows(
            jobs,
            models=models,
            catalog=catalog,
            chars_per_token=float(args.estimate_chars_per_token),
            completion_tokens=int(args.max_completion_tokens),
        )
        estimate_df.to_csv(out_dir / "estimated_costs.csv", index=False)
        print("\n=== OpenRouter estimated cost ===")
        print(estimate_df.to_string(index=False))
        if bool(args.estimate_only):
            return
        api_key = str(os.environ.get("OPENROUTER_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be set unless --estimate-only is used.")
        cache_path = out_dir / "request_cache.jsonl"
        cache = _cached_rows(cache_path)
        for model_label, model_id in models:
            for job in jobs:
                key = _request_key(
                    dataset=str(job["dataset"]),
                    model_label=model_label,
                    model_id=model_id,
                    row=job["row"],
                    prompt=str(job["prompt"]),
                    args=args,
                )
                cached = cache.get(key)
                if cached is not None and not str(cached.get("error") or "").strip():
                    continue
                try:
                    payload = _call_openrouter(client, api_key=api_key, model_id=model_id, prompt=str(job["prompt"]), args=args)
                    scored = _score_row(
                        dataset=str(job["dataset"]),
                        model_label=model_label,
                        model_id=model_id,
                        row=job["row"],
                        prompt=str(job["prompt"]),
                        request_key=key,
                        payload=payload,
                    )
                except Exception as exc:  # noqa: BLE001 - persist failed requests for auditability.
                    scored = {
                        "request_key": key,
                        "dataset": str(job["dataset"]),
                        "model_label": model_label,
                        "model_id": model_id,
                        "bt_pair_id": str(job["row"].get("bt_pair_id") or ""),
                        "source_id": _source_id(job["row"]),
                        "valid_choice": False,
                        "target_preferred": None,
                        "response_text": "",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                _append_jsonl(cache_path, scored)
                cache[key] = scored

    scores = pd.DataFrame(list(cache.values()))
    requested_models = {model_id for _label, model_id in models}
    requested_datasets = {dataset for dataset, _path in datasets}
    scores = scores[scores["model_id"].astype(str).isin(requested_models) & scores["dataset"].astype(str).isin(requested_datasets)]
    scores.to_csv(out_dir / "scores.csv", index=False)
    pair_df, summary_df = summarize_scores(scores)
    if pair_df.empty:
        error_counts = (
            scores.fillna({"error": "", "response_text": ""})
            .groupby(["model_label", "error", "response_text"], dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values(["model_label", "n_rows"], ascending=[True, False])
        )
        error_counts.to_csv(out_dir / "invalid_response_summary.csv", index=False)
        raise ValueError(
            "No valid A/B responses were emitted. "
            f"Inspect {out_dir / 'invalid_response_summary.csv'}."
        )
    contrast_df = contrast_summaries(pair_df, [str(value) for value in args.contrast])
    pair_df.to_csv(out_dir / "pair_summary.csv", index=False)
    summary_df.to_csv(out_dir / "model_dataset_summary.csv", index=False)
    contrast_df.to_csv(out_dir / "model_contrasts.csv", index=False)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "D4-OpenRouter-judge-screen",
            "datasets": dict(datasets),
            "models": dict(models),
            "contrasts": [str(value) for value in args.contrast],
            "n_request_jobs": int(len(jobs) * len(models)),
            "n_cached_rows": int(len(scores)),
            "comparison_template": str(args.comparison_template),
            "max_source_comparisons": int(args.max_source_comparisons),
        },
    )
    print("\n=== OpenRouter judge screen ===")
    print(summary_df.to_string(index=False))
    if not contrast_df.empty:
        print("\n=== Model contrasts ===")
        print(contrast_df.to_string(index=False))


if __name__ == "__main__":
    main()
