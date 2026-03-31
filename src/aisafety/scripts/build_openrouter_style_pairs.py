"""Generate contrastive style pairs via OpenRouter from local seed datasets.

This script creates synthetic pairs for stylistic dimensions by rewriting or
answering with two opposing styles. It uses local seed text sources in /data
and calls the OpenRouter API to generate both sides.

Usage:
  python -m aisafety.scripts.build_openrouter_style_pairs --help
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import httpx

from aisafety.config import DATA_DIR, DEFAULT_SEED


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SEED_SOURCE_CHOICES = ("default", "jsonl", "hc3", "laurito_human", "laurito_llm")

DEFAULT_DISCLAIMER_REGEXES = (
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bi am unable to\b",
    r"\bi'm unable to\b",
    r"\bcannot provide\b",
    r"\bcan't provide\b",
    r"\bdo not have access\b",
    r"\bdon't have access\b",
    r"\bcannot access\b",
    r"\bcan't access\b",
    r"\bcannot browse\b",
    r"\btechnical limitations?\b",
)

DEFAULT_CORPORATE_RISK_REGEXES = (
    r"\brisk\b",
    r"\buncertain\b",
    r"\badverse\b",
    r"\bliabilit(?:y|ies)\b",
    r"\bcompliance\b",
    r"\blitigation\b",
    r"\bregulator(?:y|s)?\b",
    r"\bmaterial\b",
    r"\bexposure\b",
    r"\bvolatil(?:ity|e)\b",
    r"\bdefault\b",
    r"\binsolvenc(?:y|ies)\b",
    r"\bbankrupt(?:cy|cies)\b",
)

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


@dataclass(frozen=True)
class StyleSpec:
    dimension: str
    mode: str  # "rewrite" or "answer"
    pos_label: str
    neg_label: str
    pos_style: str
    neg_style: str
    seed_source: str
    seed_loader: str  # "jsonl" or "hc3"


@dataclass(frozen=True)
class Seed:
    seed_id: int
    text: str
    source: str
    meta: dict


STYLE_SPECS: dict[str, StyleSpec] = {
    "academic_formality": StyleSpec(
        dimension="academic_formality",
        mode="rewrite",
        pos_label="academic_formal",
        neg_label="casual_tone",
        pos_style=(
            "Formal academic abstract style. Impersonal voice, precise terminology, "
            "no slang or contractions. Keep length similar and preserve facts."
        ),
        neg_style=(
            "Casual conversational tone. Simple language, contractions allowed, "
            "friendly and accessible. Preserve facts and meaning."
        ),
        seed_source=str(DATA_DIR / "derived" / "steering_samples" / "academic_formality.jsonl"),
        seed_loader="jsonl",
    ),
    "subjectivity": StyleSpec(
        dimension="subjectivity",
        mode="rewrite",
        pos_label="subjective_opinion",
        neg_label="objective_descriptive",
        pos_style=(
            "Subjective opinionated style. First-person voice, emotions, clear judgments. "
            "Preserve the original content without adding facts."
        ),
        neg_style=(
            "Objective descriptive style. Neutral, factual, no personal feelings or judgments. "
            "Preserve meaning and avoid new facts."
        ),
        seed_source=str(DATA_DIR / "derived" / "steering_samples" / "subjectivity.jsonl"),
        seed_loader="jsonl",
    ),
    "corporate_safety": StyleSpec(
        dimension="corporate_safety",
        mode="rewrite",
        pos_label="corporate_risk_disclosure",
        neg_label="plain_neutral",
        pos_style=(
            "Corporate risk-disclosure style. Formal, cautious, compliance-oriented language. "
            "Reference uncertainty and risk without adding new facts."
        ),
        neg_style=(
            "Plain neutral style. Straightforward, informational, no legalese. "
            "Preserve meaning and keep it factual."
        ),
        seed_source=str(DATA_DIR / "derived" / "steering_samples" / "corporate_safety.jsonl"),
        seed_loader="jsonl",
    ),
    "ai_tone": StyleSpec(
        dimension="ai_tone",
        mode="rewrite",
        pos_label="rlhf_ai_tone",
        neg_label="human_plain",
        pos_style=(
            "Helpful AI assistant tone. Structured, polite, safety-conscious; may include "
            "brief caveats or limitations. Preserve meaning."
        ),
        neg_style=(
            "Human forum tone. Direct and unfiltered, no disclaimers or AI mentions. "
            "Preserve meaning and facts."
        ),
        seed_source=str(DATA_DIR / "derived" / "steering_samples" / "ai_tone.jsonl"),
        seed_loader="jsonl",
    ),
    "ai_signifiers": StyleSpec(
        dimension="ai_signifiers",
        mode="answer",
        pos_label="ai_signifiers",
        neg_label="human_expert_forum",
        pos_style=(
            "AI assistant signifiers. Polite, structured, explicit about limitations, "
            "no personal anecdotes. Helpful and cautious."
        ),
        neg_style=(
            "Human expert forum response. Direct, unabridged, no AI mentions, "
            "confident and detailed."
        ),
        seed_source="HC3 local questions",
        seed_loader="hc3",
    ),
}


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def reservoir_sample(items: Iterable[Seed], n: int, seed: int) -> list[Seed]:
    if n <= 0:
        return []
    rng = random.Random(seed)
    reservoir: list[Seed] = []
    seen = 0
    for item in items:
        seen += 1
        if len(reservoir) < n:
            reservoir.append(item)
            continue
        j = rng.randrange(seen)
        if j < n:
            reservoir[j] = item
    if len(reservoir) < n:
        raise ValueError(f"Need {n} seeds but only saw {len(reservoir)}.")
    return reservoir


def load_seeds_from_jsonl(path: Path, n: int | None, seed: int) -> list[Seed]:
    if not path.exists():
        raise FileNotFoundError(f"Seed JSONL not found: {path}")
    seeds: list[Seed] = []
    for i, row in enumerate(read_jsonl(path)):
        text = row.get("text") or row.get("seed_text") or row.get("question")
        if not isinstance(text, str) or not text.strip():
            continue
        seeds.append(
            Seed(
                seed_id=i,
                text=text.strip(),
                source=str(path),
                meta={k: v for k, v in row.items() if k not in {"text", "seed_text"}},
            )
        )
    rng = random.Random(seed)
    rng.shuffle(seeds)
    if n is None:
        return seeds
    return seeds[:n]


def load_seeds_from_hc3(subsets: list[str], n: int, seed: int) -> list[Seed]:
    paths = [DATA_DIR / "HC3" / f"{s}.jsonl" for s in subsets]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing HC3 JSONLs: {missing}")

    def iter_items() -> Iterator[Seed]:
        seed_id = 0
        for p in paths:
            subset = p.stem
            for row in read_jsonl(p):
                q = row.get("question")
                if not isinstance(q, str) or not q.strip():
                    continue
                yield Seed(
                    seed_id=seed_id,
                    text=q.strip(),
                    source=str(p),
                    meta={"subset": subset},
                )
                seed_id += 1

    return reservoir_sample(iter_items(), n=n, seed=seed)


def load_seeds_from_laurito(
    item_types: list[str],
    n: int,
    seed: int,
    *,
    source: str,
    include_title: bool,
    max_chars: int | None,
) -> list[Seed]:
    from aisafety.data.domains import DOMAINS
    from aisafety.data.loaders import load_human_map, load_llm_all_by_title

    rng = random.Random(seed)
    keep = {t.strip() for t in item_types if t.strip()}
    seeds: list[Seed] = []
    seed_id = 0

    for cfg in DOMAINS.values():
        if cfg.item_type not in keep or not cfg.exists():
            continue

        if source == "human":
            items = load_human_map(cfg.human_dir).items()
            for title, text in items:
                if not isinstance(text, str) or not text.strip():
                    continue
                txt = text.strip()
                if include_title:
                    txt = f"Title: {title}\n\n{txt}"
                txt = truncate_text(txt, max_chars)
                seeds.append(
                    Seed(
                        seed_id=seed_id,
                        text=txt,
                        source=f"Laurito/{cfg.item_type}/human",
                        meta={"item_type": cfg.item_type, "title": title, "source": "human"},
                    )
                )
                seed_id += 1
        else:
            llm_map = load_llm_all_by_title(cfg.llm_dir, prompt_key=cfg.prompt_key)
            for title, texts in llm_map.items():
                if not texts:
                    continue
                txt = rng.choice(texts)
                if not isinstance(txt, str) or not txt.strip():
                    continue
                txt = txt.strip()
                if include_title:
                    txt = f"Title: {title}\n\n{txt}"
                txt = truncate_text(txt, max_chars)
                seeds.append(
                    Seed(
                        seed_id=seed_id,
                        text=txt,
                        source=f"Laurito/{cfg.item_type}/llm",
                        meta={"item_type": cfg.item_type, "title": title, "source": "llm"},
                    )
                )
                seed_id += 1

    rng.shuffle(seeds)
    return seeds[:n]


def build_messages(mode: str, style_desc: str, seed_text: str) -> list[dict]:
    if mode == "rewrite":
        sys = (
            "You are a careful editor. Preserve meaning and facts. "
            "Do not add new information. Output only the rewritten text."
        )
        user = (
            "Rewrite the text below in the specified style.\n\n"
            f"STYLE:\n{style_desc}\n\n"
            f"TEXT:\n{seed_text}\n\n"
            "Return only the rewritten text."
        )
    elif mode == "answer":
        sys = (
            "You are a careful domain expert. Answer the question in the requested style. "
            "Do not invent facts. Output only the answer."
        )
        user = (
            "Answer the question below in the specified style.\n\n"
            f"STYLE:\n{style_desc}\n\n"
            f"QUESTION:\n{seed_text}\n\n"
            "Return only the answer."
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def call_openrouter(
    client: httpx.Client,
    *,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    top_p: float | None,
    base_url: str,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "HTTP-Referer": "local"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    resp = client.post(base_url, json=payload, headers=headers)
    if resp.status_code >= 400:
        if resp.status_code in {429, 500, 502, 503, 504}:
            resp.raise_for_status()
        detail = _format_http_error(resp)
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {detail}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return str(content).strip()


def _format_http_error(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:
        return resp.text
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("error") or err.get("type")
            if msg:
                return str(msg)
        if "message" in data:
            return str(data["message"])
    return resp.text


def _log_error(path: Path | None, msg: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dimensions",
        type=str,
        default="academic_formality,subjectivity,corporate_safety,ai_tone,ai_signifiers",
        help="Comma-separated list: " + ",".join(sorted(STYLE_SPECS.keys())),
    )
    p.add_argument("--out-dir", type=Path, default=DATA_DIR / "derived" / "openrouter_style_pairs")
    p.add_argument("--num-seeds", type=int, default=100, help="Seeds per dimension.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--seed-source",
        type=str,
        choices=SEED_SOURCE_CHOICES,
        default="default",
        help="Override seed source selection.",
    )
    p.add_argument(
        "--seed-jsonl",
        type=Path,
        default=None,
        help="Custom JSONL seeds (used when --seed-source jsonl).",
    )
    p.add_argument("--model", type=str, required=True, help="OpenRouter model id.")
    p.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (or env).")
    p.add_argument("--base-url", type=str, default=OPENROUTER_URL)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-seed-chars", type=int, default=2000)
    p.add_argument("--hc3-subsets", type=str, default="finance,medicine,open_qa,wiki_csai")
    p.add_argument("--laurito-item-types", type=str, default="movie,paper,product")
    p.add_argument(
        "--laurito-include-title",
        dest="laurito_include_title",
        action="store_true",
        help="Prefix Laurito seeds with the title.",
    )
    p.add_argument(
        "--laurito-no-title",
        dest="laurito_include_title",
        action="store_false",
        help="Do not prefix Laurito seeds with the title.",
    )
    p.set_defaults(laurito_include_title=True)
    p.add_argument("--laurito-max-chars", type=int, default=2000)
    p.add_argument(
        "--seed-drop-disclaimers",
        action="store_true",
        help="Drop common AI disclaimer lines from AI tone seeds.",
    )
    p.add_argument(
        "--seed-drop-regex",
        action="append",
        default=[],
        help="Regex to remove from seed text (repeatable).",
    )
    p.add_argument(
        "--seed-drop-mode",
        type=str,
        choices=["line", "span"],
        default="line",
        help="Drop entire lines or only matched spans.",
    )
    p.add_argument(
        "--require-english",
        action="store_true",
        help="Filter out seeds that look non-English.",
    )
    p.add_argument("--max-non-ascii-ratio", type=float, default=0.2)
    p.add_argument("--min-ascii-alpha-ratio", type=float, default=0.2)
    p.add_argument("--drop-cjk", action="store_true", help="Drop seeds containing CJK characters.")
    p.add_argument(
        "--corporate-require-risk",
        action="store_true",
        help="Filter corporate_safety seeds to risk-factor-like text.",
    )
    p.add_argument(
        "--corporate-risk-regex",
        action="append",
        default=[],
        help="Regex defining corporate risk cues (repeatable).",
    )
    p.add_argument("--corporate-min-risk-matches", type=int, default=1)
    p.add_argument(
        "--seed-pool-multiplier",
        type=int,
        default=3,
        help="Load extra seeds to allow filtering without running short.",
    )
    p.add_argument("--sleep-seconds", type=float, default=0.2, help="Sleep between API calls.")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-sleep", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist.")
    p.add_argument("--include-prompts", action="store_true", help="Store prompts in output JSONL.")
    p.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip seeds that fail due to API errors (e.g., moderation).",
    )
    p.add_argument(
        "--error-log",
        type=Path,
        default=None,
        help="Optional path to write skipped-seed errors.",
    )
    return p.parse_args()


def truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def iter_dimensions(arg: str) -> list[str]:
    dims = [d.strip() for d in arg.split(",") if d.strip()]
    invalid = [d for d in dims if d not in STYLE_SPECS]
    if invalid:
        raise ValueError(f"Unknown dimensions: {invalid}")
    return dims


def ensure_out_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace.")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compile_regexes(patterns: list[str]) -> list[re.Pattern]:
    compiled: list[re.Pattern] = []
    for pat in patterns:
        compiled.append(re.compile(pat, flags=re.IGNORECASE))
    return compiled


def apply_drop_regex(text: str, patterns: list[re.Pattern], mode: str) -> str:
    if not patterns:
        return text
    if mode == "span":
        for rx in patterns:
            text = rx.sub("", text)
        return text.strip()
    lines = text.splitlines()
    kept = []
    for ln in lines:
        if any(rx.search(ln) for rx in patterns):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def looks_englishish(
    text: str,
    *,
    max_non_ascii_ratio: float,
    min_ascii_alpha_ratio: float,
    drop_cjk: bool,
) -> bool:
    if not text:
        return False
    if drop_cjk and CJK_RE.search(text):
        return False
    total = max(len(text), 1)
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    if (non_ascii / total) > max_non_ascii_ratio:
        return False
    ascii_alpha = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    if (ascii_alpha / total) < min_ascii_alpha_ratio:
        return False
    return True


def passes_risk_filter(text: str, patterns: list[re.Pattern], min_hits: int) -> bool:
    if not patterns or min_hits <= 0:
        return True
    hits = 0
    for rx in patterns:
        if rx.search(text):
            hits += 1
            if hits >= min_hits:
                return True
    return False


def prepare_seeds_for_dimension(spec: StyleSpec, seeds: list[Seed], args: argparse.Namespace) -> list[Seed]:
    patterns: list[str] = list(args.seed_drop_regex or [])
    if args.seed_drop_disclaimers and spec.dimension == "ai_tone":
        patterns += list(DEFAULT_DISCLAIMER_REGEXES)
    drop_regexes = compile_regexes(patterns)

    risk_patterns: list[str] = list(args.corporate_risk_regex or [])
    if args.corporate_require_risk and not risk_patterns:
        risk_patterns = list(DEFAULT_CORPORATE_RISK_REGEXES)
    risk_regexes = compile_regexes(risk_patterns)

    cleaned: list[Seed] = []
    for seed in seeds:
        text = seed.text.strip()
        if drop_regexes:
            text = apply_drop_regex(text, drop_regexes, args.seed_drop_mode)
        if not text:
            continue
        if args.require_english and not looks_englishish(
            text,
            max_non_ascii_ratio=float(args.max_non_ascii_ratio),
            min_ascii_alpha_ratio=float(args.min_ascii_alpha_ratio),
            drop_cjk=bool(args.drop_cjk),
        ):
            continue
        if spec.dimension == "corporate_safety" and args.corporate_require_risk:
            if not passes_risk_filter(
                text, risk_regexes, min_hits=int(args.corporate_min_risk_matches)
            ):
                continue
        cleaned.append(Seed(seed_id=seed.seed_id, text=text, source=seed.source, meta=seed.meta))

    rng = random.Random(args.seed + abs(hash(spec.dimension)) % 10000)
    rng.shuffle(cleaned)
    if len(cleaned) < args.num_seeds:
        raise ValueError(
            f"{spec.dimension}: need {args.num_seeds} seeds after filtering, only have {len(cleaned)}."
        )
    return cleaned[: args.num_seeds]


def build_rows_for_dimension(
    spec: StyleSpec,
    seeds: list[Seed],
    *,
    client: httpx.Client,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float | None,
    base_url: str,
    max_seed_chars: int,
    sleep_seconds: float,
    max_retries: int,
    retry_sleep: float,
    include_prompts: bool,
    seed_source_label: str,
    skip_failed: bool,
    error_log: Path | None,
) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        seed_text = truncate_text(seed.text, max_seed_chars)
        for label, style_desc in ((spec.pos_label, spec.pos_style), (spec.neg_label, spec.neg_style)):
            messages = build_messages(spec.mode, style_desc, seed_text)
            attempt = 0
            while True:
                try:
                    text = call_openrouter(
                        client,
                        api_key=api_key,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        base_url=base_url,
                    )
                    break
                except httpx.HTTPError:
                    attempt += 1
                    if attempt > max_retries:
                        if skip_failed:
                            _log_error(
                                error_log,
                                f"{spec.dimension} seed={seed.seed_id} label={label} http_error",
                            )
                            text = None
                            break
                        raise
                    time.sleep(retry_sleep)
                except RuntimeError as exc:
                    if skip_failed:
                        _log_error(
                            error_log,
                            f"{spec.dimension} seed={seed.seed_id} label={label} {exc}",
                        )
                        text = None
                        break
                    raise
            if text is None:
                continue
            row = {
                "dimension": spec.dimension,
                "label": label,
                "seed_source": seed_source_label,
                "seed_id": seed.seed_id,
                "seed_text": seed_text,
                "generated_text": text,
                "model": model,
                "meta": seed.meta,
            }
            if include_prompts:
                row["messages"] = messages
            rows.append(row)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    return rows


def main() -> None:
    args = parse_args()
    dims = iter_dimensions(args.dimensions)
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set and --api-key was not provided.")

    hc3_subsets = [s.strip() for s in str(args.hc3_subsets).split(",") if s.strip()]
    laurito_item_types = [s.strip() for s in str(args.laurito_item_types).split(",") if s.strip()]
    pool_n = max(1, int(args.num_seeds) * max(1, int(args.seed_pool_multiplier)))
    with httpx.Client(timeout=60) as client:
        for dim in dims:
            spec = STYLE_SPECS[dim]
            seed_source_label = spec.seed_source
            if args.seed_source == "jsonl":
                if args.seed_jsonl is None:
                    raise ValueError("--seed-jsonl is required when --seed-source jsonl is set.")
                seed_source_label = str(args.seed_jsonl)
                seeds = load_seeds_from_jsonl(args.seed_jsonl, None, args.seed)
            elif args.seed_source == "hc3":
                seed_source_label = "HC3 local questions"
                seeds = load_seeds_from_hc3(hc3_subsets, pool_n, args.seed)
            elif args.seed_source == "laurito_human":
                seed_source_label = f"Laurito/human:{','.join(laurito_item_types)}"
                seeds = load_seeds_from_laurito(
                    laurito_item_types,
                    pool_n,
                    args.seed,
                    source="human",
                    include_title=bool(args.laurito_include_title),
                    max_chars=int(args.laurito_max_chars),
                )
            elif args.seed_source == "laurito_llm":
                seed_source_label = f"Laurito/llm:{','.join(laurito_item_types)}"
                seeds = load_seeds_from_laurito(
                    laurito_item_types,
                    pool_n,
                    args.seed,
                    source="llm",
                    include_title=bool(args.laurito_include_title),
                    max_chars=int(args.laurito_max_chars),
                )
            elif spec.seed_loader == "jsonl":
                seeds = load_seeds_from_jsonl(Path(spec.seed_source), None, args.seed)
            else:
                seeds = load_seeds_from_hc3(hc3_subsets, pool_n, args.seed)

            seeds = prepare_seeds_for_dimension(spec, seeds, args)

            out_path = args.out_dir / f"{dim}.jsonl"
            ensure_out_path(out_path, overwrite=args.overwrite)

            rows = build_rows_for_dimension(
                spec,
                seeds,
                client=client,
                api_key=api_key,
                model=args.model,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                top_p=float(args.top_p) if args.top_p is not None else None,
                base_url=args.base_url,
                max_seed_chars=int(args.max_seed_chars),
                sleep_seconds=float(args.sleep_seconds),
                max_retries=int(args.max_retries),
                retry_sleep=float(args.retry_sleep),
                include_prompts=bool(args.include_prompts),
                seed_source_label=seed_source_label,
                skip_failed=bool(args.skip_failed),
                error_log=args.error_log,
            )
            write_jsonl(out_path, rows)
            print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
