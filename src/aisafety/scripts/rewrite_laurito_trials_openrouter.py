"""Rewrite Laurito A/B trials into a tone-normalized variant via OpenRouter.

This script reads an existing trials/results CSV (with columns like A_text/B_text)
and rewrites both sides into a single target style (e.g., "human_plain") using the
same OpenRouter prompting mechanism as `build_openrouter_style_pairs`.

Typical usage (tone-normalize to human style):
  python -m aisafety.scripts.rewrite_laurito_trials_openrouter ^
    --in-csv artifacts\\trials.csv ^
    --out-csv artifacts\\trials_human_plain.csv ^
    --dimension ai_tone ^
    --target-label human_plain ^
    --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd
from tqdm import tqdm

from aisafety.scripts.build_openrouter_style_pairs import (
    OPENROUTER_URL,
    STYLE_SPECS,
    build_messages,
    call_openrouter,
    truncate_text,
)


REQUIRED_TRIAL_COLS = ("item_type", "title", "A_text", "B_text", "A_source", "B_source")


@dataclass(frozen=True)
class RewriteRequest:
    key: str
    item_type: str
    title: str
    source: str
    dimension: str
    target_label: str
    style_desc: str
    original_text: str
    seed_text: str


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_rewrite_key(
    *,
    dimension: str,
    target_label: str,
    style_desc: str,
    source: str,
    item_type: str,
    title: str,
    original_text: str,
    model: str,
    temperature: float,
    top_p: float | None,
    max_tokens: int,
    max_chars: int,
) -> str:
    payload = {
        "dimension": str(dimension),
        "target_label": str(target_label),
        "style_desc": str(style_desc),
        "source": str(source),
        "item_type": str(item_type),
        "title": str(title),
        "text_sha256": _sha256_hex(str(original_text)),
        "model": str(model),
        "temperature": float(temperature),
        "top_p": None if top_p is None else float(top_p),
        "max_tokens": int(max_tokens),
        "max_chars": int(max_chars),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_hex(raw)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_trials_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_TRIAL_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV missing required columns: {missing}")


def _style_for_label(dimension: str, label: str) -> str:
    spec = STYLE_SPECS.get(dimension)
    if spec is None:
        raise ValueError(f"Unknown dimension: {dimension}")
    if getattr(spec, "mode", None) != "rewrite":
        raise ValueError(f"Dimension {dimension} uses mode={spec.mode!r}; only rewrite dimensions are supported.")
    if label == spec.pos_label:
        return str(spec.pos_style)
    if label == spec.neg_label:
        return str(spec.neg_style)
    raise ValueError(f"Label {label!r} is not valid for dimension={dimension} ({spec.pos_label!r}/{spec.neg_label!r}).")


def build_rewrite_requests(
    df_trials: pd.DataFrame,
    *,
    dimension: str,
    target_label: str,
    model: str,
    temperature: float,
    top_p: float | None,
    max_tokens: int,
    max_chars: int,
    human_target_label: str | None = None,
    llm_target_label: str | None = None,
) -> list[RewriteRequest]:
    validate_trials_df(df_trials)

    dimension = str(dimension)
    style_default = _style_for_label(dimension, str(target_label))

    if (human_target_label is None) ^ (llm_target_label is None):
        raise ValueError("Set both --human-target-label and --llm-target-label, or neither.")

    per_source = None
    if human_target_label is not None and llm_target_label is not None:
        per_source = {
            "human": (str(human_target_label), _style_for_label(dimension, str(human_target_label))),
            "llm": (str(llm_target_label), _style_for_label(dimension, str(llm_target_label))),
        }

    recs = []
    for r in df_trials.itertuples(index=False):
        d = r._asdict()
        for side in ("A", "B"):
            source = str(d[f"{side}_source"])
            text = str(d[f"{side}_text"])
            if source not in {"human", "llm"}:
                raise ValueError(f"Unexpected source {source!r}; expected 'human' or 'llm'.")
            if per_source is None:
                label = str(target_label)
                style_desc = style_default
            else:
                label, style_desc = per_source[source]

            recs.append(
                {
                    "item_type": str(d["item_type"]),
                    "title": str(d["title"]),
                    "source": source,
                    "dimension": dimension,
                    "target_label": label,
                    "style_desc": style_desc,
                    "original_text": text,
                    "seed_text": truncate_text(text, int(max_chars)),
                }
            )

    uniq = pd.DataFrame(recs).drop_duplicates(
        subset=["item_type", "title", "source", "dimension", "target_label", "seed_text", "original_text"]
    )
    out: list[RewriteRequest] = []
    for row in uniq.itertuples(index=False):
        key = make_rewrite_key(
            dimension=row.dimension,
            target_label=row.target_label,
            style_desc=row.style_desc,
            source=row.source,
            item_type=row.item_type,
            title=row.title,
            original_text=row.original_text,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_chars=max_chars,
        )
        out.append(
            RewriteRequest(
                key=key,
                item_type=row.item_type,
                title=row.title,
                source=row.source,
                dimension=row.dimension,
                target_label=row.target_label,
                style_desc=row.style_desc,
                original_text=row.original_text,
                seed_text=row.seed_text,
            )
        )
    return out


def apply_rewrites_to_trials(
    df_trials: pd.DataFrame,
    *,
    rewrite_map: dict[str, str],
    dimension: str,
    target_label: str,
    keep_original: bool,
    model: str,
    temperature: float,
    top_p: float | None,
    max_tokens: int,
    max_chars: int,
    human_target_label: str | None = None,
    llm_target_label: str | None = None,
) -> pd.DataFrame:
    validate_trials_df(df_trials)
    out = df_trials.copy()
    if keep_original:
        if "A_text_original" not in out.columns:
            out["A_text_original"] = out["A_text"]
        if "B_text_original" not in out.columns:
            out["B_text_original"] = out["B_text"]

    if (human_target_label is None) ^ (llm_target_label is None):
        raise ValueError("Set both human_target_label and llm_target_label, or neither.")

    per_source = None
    if human_target_label is not None and llm_target_label is not None:
        per_source = {"human": str(human_target_label), "llm": str(llm_target_label)}

    def pick_label(source: str) -> str:
        if per_source is None:
            return str(target_label)
        return per_source[source]

    def rewrite_row_texts(r: pd.Series) -> tuple[str, str]:
        # NOTE: style_desc is included in the key, derived from the label and STYLE_SPECS.
        A_key = make_rewrite_key(
            dimension=str(dimension),
            target_label=pick_label(str(r["A_source"])),
            style_desc=_style_for_label(str(dimension), pick_label(str(r["A_source"]))),
            source=str(r["A_source"]),
            item_type=str(r["item_type"]),
            title=str(r["title"]),
            original_text=str(r["A_text"]),
            model=str(model),
            temperature=float(temperature),
            top_p=None if top_p is None else float(top_p),
            max_tokens=int(max_tokens),
            max_chars=int(max_chars),
        )
        B_key = make_rewrite_key(
            dimension=str(dimension),
            target_label=pick_label(str(r["B_source"])),
            style_desc=_style_for_label(str(dimension), pick_label(str(r["B_source"]))),
            source=str(r["B_source"]),
            item_type=str(r["item_type"]),
            title=str(r["title"]),
            original_text=str(r["B_text"]),
            model=str(model),
            temperature=float(temperature),
            top_p=None if top_p is None else float(top_p),
            max_tokens=int(max_tokens),
            max_chars=int(max_chars),
        )
        return rewrite_map.get(A_key, str(r["A_text"])), rewrite_map.get(B_key, str(r["B_text"]))

    rewritten = out.apply(rewrite_row_texts, axis=1, result_type="expand")
    out["A_text"] = rewritten[0]
    out["B_text"] = rewritten[1]
    out["rewrite_dimension"] = str(dimension)
    out["rewrite_model"] = str(model)
    out["rewrite_label_default"] = str(target_label)
    out["rewrite_label_human"] = None if human_target_label is None else str(human_target_label)
    out["rewrite_label_llm"] = None if llm_target_label is None else str(llm_target_label)
    out["rewrite_temperature"] = float(temperature)
    out["rewrite_top_p"] = None if top_p is None else float(top_p)
    out["rewrite_max_tokens"] = int(max_tokens)
    out["rewrite_max_chars"] = int(max_chars)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-csv", type=Path, required=True, help="Trials/results CSV with A_text/B_text columns.")
    p.add_argument("--out-csv", type=Path, required=True, help="Output CSV with rewritten A_text/B_text.")
    p.add_argument(
        "--dimension",
        type=str,
        default="ai_tone",
        help="Style dimension (controls which prompt style to apply).",
    )
    p.add_argument(
        "--target-label",
        type=str,
        default=None,
        help="Label whose style to rewrite into (default: dimension's neg label).",
    )
    p.add_argument("--human-target-label", type=str, default=None, help="Optional target label for human texts.")
    p.add_argument("--llm-target-label", type=str, default=None, help="Optional target label for llm texts.")
    p.add_argument("--model", type=str, required=True, help="OpenRouter model id.")
    p.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (or env OPENROUTER_API_KEY).")
    p.add_argument("--base-url", type=str, default=OPENROUTER_URL)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-chars", type=int, default=2000, help="Max chars from each text to include in prompt.")
    p.add_argument("--sleep-seconds", type=float, default=0.2)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-sleep", type=float, default=2.0)
    p.add_argument(
        "--cache-jsonl",
        type=Path,
        default=None,
        help="JSONL cache for individual text rewrites (enables resume).",
    )
    p.add_argument("--include-prompts", action="store_true", help="Store OpenRouter messages in the cache JSONL.")
    p.add_argument("--keep-original", action="store_true", help="Keep A_text_original/B_text_original in output.")
    p.add_argument("--no-keep-original", dest="keep_original", action="store_false")
    p.set_defaults(keep_original=True)
    p.add_argument(
        "--on-failure",
        type=str,
        choices=["raise", "keep_original"],
        default="raise",
        help="What to do if a text rewrite fails.",
    )
    p.add_argument("--error-log", type=Path, default=None, help="Optional file to append errors to.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite --out-csv if it exists.")
    return p.parse_args()


def _log_error(path: Path | None, msg: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def main() -> None:
    args = parse_args()
    if args.out_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out_csv} (pass --overwrite to replace).")

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set and --api-key was not provided.")

    spec = STYLE_SPECS.get(str(args.dimension))
    if spec is None:
        raise ValueError(f"Unknown dimension: {args.dimension!r}")
    if spec.mode != "rewrite":
        raise ValueError(f"Dimension {args.dimension!r} uses mode={spec.mode!r}; only rewrite dimensions are supported.")

    target_label = str(args.target_label) if args.target_label else str(spec.neg_label)

    cache_path: Path = args.cache_jsonl or (
        args.out_csv.parent / f"{args.out_csv.stem}_rewrites.jsonl"
    )
    cached_rows = read_jsonl(cache_path)
    rewrite_map: dict[str, str] = {}
    for row in cached_rows:
        key = row.get("key")
        text = row.get("rewritten_text")
        if isinstance(key, str) and isinstance(text, str):
            rewrite_map[key] = text

    df = pd.read_csv(args.in_csv)

    requests = build_rewrite_requests(
        df,
        dimension=str(args.dimension),
        target_label=target_label,
        model=str(args.model),
        temperature=float(args.temperature),
        top_p=float(args.top_p) if args.top_p is not None else None,
        max_tokens=int(args.max_tokens),
        max_chars=int(args.max_chars),
        human_target_label=str(args.human_target_label) if args.human_target_label else None,
        llm_target_label=str(args.llm_target_label) if args.llm_target_label else None,
    )

    pending = [r for r in requests if r.key not in rewrite_map]
    if pending:
        with httpx.Client(timeout=60) as client:
            for req in tqdm(pending, desc="OpenRouter rewrites"):
                messages = build_messages("rewrite", req.style_desc, req.seed_text)
                attempt = 0
                rewritten = None
                while True:
                    try:
                        rewritten = call_openrouter(
                            client,
                            api_key=api_key,
                            model=str(args.model),
                            messages=messages,
                            temperature=float(args.temperature),
                            max_tokens=int(args.max_tokens),
                            top_p=float(args.top_p) if args.top_p is not None else None,
                            base_url=str(args.base_url),
                        )
                        break
                    except httpx.HTTPError:
                        attempt += 1
                        if attempt > int(args.max_retries):
                            raise
                        time.sleep(float(args.retry_sleep))
                    except RuntimeError as exc:
                        msg = (
                            f"{req.dimension} label={req.target_label} "
                            f"{req.item_type} title={req.title!r} source={req.source} key={req.key[:8]} "
                            f"{exc}"
                        )
                        _log_error(args.error_log, msg)
                        if str(args.on_failure) == "keep_original":
                            rewritten = req.original_text
                            break
                        raise

                assert rewritten is not None
                rewrite_map[req.key] = rewritten
                cache_row = {
                    "key": req.key,
                    "dimension": req.dimension,
                    "target_label": req.target_label,
                    "item_type": req.item_type,
                    "title": req.title,
                    "source": req.source,
                    "seed_text": req.seed_text,
                    "original_text": req.original_text,
                    "rewritten_text": rewritten,
                    "model": str(args.model),
                    "temperature": float(args.temperature),
                    "top_p": None if args.top_p is None else float(args.top_p),
                    "max_tokens": int(args.max_tokens),
                    "max_chars": int(args.max_chars),
                }
                if args.include_prompts:
                    cache_row["messages"] = messages
                append_jsonl(cache_path, [cache_row])
                if float(args.sleep_seconds) > 0:
                    time.sleep(float(args.sleep_seconds))

    out_df = apply_rewrites_to_trials(
        df,
        rewrite_map=rewrite_map,
        dimension=str(args.dimension),
        target_label=target_label,
        keep_original=bool(args.keep_original),
        model=str(args.model),
        temperature=float(args.temperature),
        top_p=float(args.top_p) if args.top_p is not None else None,
        max_tokens=int(args.max_tokens),
        max_chars=int(args.max_chars),
        human_target_label=str(args.human_target_label) if args.human_target_label else None,
        llm_target_label=str(args.llm_target_label) if args.llm_target_label else None,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} rewritten trials to {args.out_csv}")
    print(f"Rewrite cache: {cache_path}")


if __name__ == "__main__":
    main()
