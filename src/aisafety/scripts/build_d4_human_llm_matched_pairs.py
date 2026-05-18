"""Build a deterministic matched subset of D4 human-vs-LLM pairs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any

from aisafety.config import DATA_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.counterfactuals import flat_text, token_count
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json


DEFAULT_PAIR_JSONL = DATA_DIR / "derived" / "d4_human_llm_alignment_pairs_v1" / "pairs.jsonl"
DEFAULT_OUT_DIR = DATA_DIR / "derived" / "d4_human_llm_alignment_pairs_matched_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-total-pairs", type=int, default=1000)
    parser.add_argument("--max-pairs-per-stratum", type=int, default=0)
    parser.add_argument("--strata", choices=["dataset", "dataset_subset"], default="dataset_subset")
    parser.add_argument("--min-response-tokens", type=int, default=20)
    parser.add_argument("--max-response-tokens", type=int, default=900)
    parser.add_argument("--max-abs-token-delta", type=int, default=80)
    parser.add_argument("--max-length-ratio", type=float, default=1.35)
    parser.add_argument("--max-prompt-overlap-delta", type=float, default=0.20)
    parser.add_argument("--max-type-token-ratio-delta", type=float, default=0.25)
    parser.add_argument("--max-punct-rate-delta", type=float, default=0.08)
    parser.add_argument(
        "--require-all-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false, only token length controls are enforced; other metrics are recorded.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in _WORD_RE.findall(flat_text(text))]


def _safe_ratio(a: float, b: float) -> float:
    return float(max(a, b) / max(min(a, b), 1.0))


def _text_metrics(text: str, prompt: str) -> dict[str, float]:
    toks = _tokens(text)
    prompt_toks = set(_tokens(prompt))
    n = len(toks)
    unique = len(set(toks))
    overlap = len([tok for tok in toks if tok in prompt_toks]) / max(float(n), 1.0)
    chars = len(flat_text(text))
    punct = len(_PUNCT_RE.findall(text)) / max(float(chars), 1.0)
    avg_word_len = sum(len(tok) for tok in toks) / max(float(n), 1.0)
    return {
        "tokens": float(n),
        "type_token_ratio": float(unique / max(float(n), 1.0)),
        "prompt_overlap": float(overlap),
        "punct_rate": float(punct),
        "avg_word_len": float(avg_word_len),
    }


def _pair_prompt(row: dict[str, Any]) -> str:
    for key in ("prompt", "question", "title"):
        value = flat_text(str(row.get(key) or ""))
        if value:
            return value
    return ""


def _stratum_key(row: dict[str, Any], strata: str) -> tuple[str, ...]:
    source = str(row.get("source_dataset") or "")
    if strata == "dataset_subset":
        return (source, str(row.get("subset") or row.get("item_type") or ""))
    return (source,)


def _match_score(row: dict[str, Any]) -> float:
    return float(
        abs(float(row["token_delta_llm_minus_human"]))
        + 30.0 * abs(float(row["prompt_overlap_delta_llm_minus_human"]))
        + 20.0 * abs(float(row["type_token_ratio_delta_llm_minus_human"]))
        + 100.0 * abs(float(row["punct_rate_delta_llm_minus_human"]))
    )


def _stable_sort(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _match_score(row),
            sha1_hex(f"{seed}:matched:{row.get('source_dataset')}:{row.get('subset')}:{row.get('pair_id')}"),
        ),
    )


def _balanced_cap(
    rows: list[dict[str, Any]],
    *,
    strata: str,
    max_total_pairs: int,
    max_pairs_per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_stratum: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[_stratum_key(row, strata)].append(row)

    queues: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for key, values in by_stratum.items():
        ordered = _stable_sort(values, seed=seed)
        if int(max_pairs_per_stratum) > 0:
            ordered = ordered[: int(max_pairs_per_stratum)]
        queues[key] = ordered

    if int(max_total_pairs) <= 0:
        capped = [row for key in sorted(queues) for row in queues[key]]
    else:
        keys = sorted(queues, key=lambda key: sha1_hex(f"{seed}:matched-stratum:{'::'.join(key)}"))
        capped = []
        cursor = 0
        while len(capped) < int(max_total_pairs) and any(queues.values()):
            key = keys[cursor % len(keys)]
            cursor += 1
            if queues[key]:
                capped.append(queues[key].pop(0))
    return sorted(
        capped,
        key=lambda row: (
            str(row.get("source_dataset") or ""),
            str(row.get("subset") or ""),
            str(row.get("pair_id") or ""),
        ),
    )


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def build_matched_pairs(
    rows: list[dict[str, Any]],
    *,
    min_response_tokens: int,
    max_response_tokens: int,
    max_abs_token_delta: int,
    max_length_ratio: float,
    max_prompt_overlap_delta: float,
    max_type_token_ratio_delta: float,
    max_punct_rate_delta: float,
    require_all_controls: bool,
    strata: str,
    max_total_pairs: int,
    max_pairs_per_stratum: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    skipped = Counter()
    metric_rows: list[dict[str, Any]] = []

    for row in rows:
        human_text = flat_text(str(row.get("human_text") or ""))
        llm_text = flat_text(str(row.get("llm_text") or ""))
        if not human_text or not llm_text:
            skipped["missing_text"] += 1
            continue
        prompt = _pair_prompt(row)
        human = _text_metrics(human_text, prompt)
        llm = _text_metrics(llm_text, prompt)
        human_tokens = int(human["tokens"])
        llm_tokens = int(llm["tokens"])
        token_delta = llm_tokens - human_tokens
        length_ratio = _safe_ratio(float(llm_tokens), float(human_tokens))
        prompt_overlap_delta = llm["prompt_overlap"] - human["prompt_overlap"]
        type_token_delta = llm["type_token_ratio"] - human["type_token_ratio"]
        punct_delta = llm["punct_rate"] - human["punct_rate"]

        annotated = {
            **row,
            "human_token_count": human_tokens,
            "llm_token_count": llm_tokens,
            "token_delta_llm_minus_human": int(token_delta),
            "abs_token_delta": int(abs(token_delta)),
            "length_ratio_max_over_min": float(length_ratio),
            "prompt_overlap_human": human["prompt_overlap"],
            "prompt_overlap_llm": llm["prompt_overlap"],
            "prompt_overlap_delta_llm_minus_human": float(prompt_overlap_delta),
            "type_token_ratio_human": human["type_token_ratio"],
            "type_token_ratio_llm": llm["type_token_ratio"],
            "type_token_ratio_delta_llm_minus_human": float(type_token_delta),
            "punct_rate_human": human["punct_rate"],
            "punct_rate_llm": llm["punct_rate"],
            "punct_rate_delta_llm_minus_human": float(punct_delta),
            "avg_word_len_human": human["avg_word_len"],
            "avg_word_len_llm": llm["avg_word_len"],
        }
        metric_rows.append(annotated)

        if human_tokens < int(min_response_tokens) or llm_tokens < int(min_response_tokens):
            skipped["too_short"] += 1
            continue
        if human_tokens > int(max_response_tokens) or llm_tokens > int(max_response_tokens):
            skipped["too_long"] += 1
            continue
        if abs(token_delta) > int(max_abs_token_delta):
            skipped["token_delta"] += 1
            continue
        if length_ratio > float(max_length_ratio):
            skipped["length_ratio"] += 1
            continue
        if bool(require_all_controls):
            if abs(prompt_overlap_delta) > float(max_prompt_overlap_delta):
                skipped["prompt_overlap_delta"] += 1
                continue
            if abs(type_token_delta) > float(max_type_token_ratio_delta):
                skipped["type_token_ratio_delta"] += 1
                continue
            if abs(punct_delta) > float(max_punct_rate_delta):
                skipped["punct_rate_delta"] += 1
                continue
        kept.append(annotated)

    capped = _balanced_cap(
        kept,
        strata=str(strata),
        max_total_pairs=int(max_total_pairs),
        max_pairs_per_stratum=int(max_pairs_per_stratum),
        seed=int(seed),
    )
    summary = {
        "n_input_pairs": int(len(rows)),
        "n_metric_pairs": int(len(metric_rows)),
        "n_matched_uncapped": int(len(kept)),
        "n_matched_pairs": int(len(capped)),
        "skipped": dict(skipped),
        "strata": str(strata),
        "max_total_pairs": int(max_total_pairs),
        "max_pairs_per_stratum": int(max_pairs_per_stratum),
        "constraints": {
            "min_response_tokens": int(min_response_tokens),
            "max_response_tokens": int(max_response_tokens),
            "max_abs_token_delta": int(max_abs_token_delta),
            "max_length_ratio": float(max_length_ratio),
            "max_prompt_overlap_delta": float(max_prompt_overlap_delta),
            "max_type_token_ratio_delta": float(max_type_token_ratio_delta),
            "max_punct_rate_delta": float(max_punct_rate_delta),
            "require_all_controls": bool(require_all_controls),
        },
        "by_dataset": dict(Counter(str(row.get("source_dataset") or "") for row in capped)),
        "by_dataset_subset": {
            f"{dataset}::{subset}": int(count)
            for (dataset, subset), count in Counter(
                (str(row.get("source_dataset") or ""), str(row.get("subset") or row.get("item_type") or ""))
                for row in capped
            ).items()
        },
        "mean_metrics": _mean_metrics(capped),
    }
    return capped, summary


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = [
        "human_token_count",
        "llm_token_count",
        "token_delta_llm_minus_human",
        "abs_token_delta",
        "length_ratio_max_over_min",
        "prompt_overlap_delta_llm_minus_human",
        "type_token_ratio_delta_llm_minus_human",
        "punct_rate_delta_llm_minus_human",
    ]
    out: dict[str, float | None] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and _is_finite(row[key])]
        out[key] = None if not vals else float(sum(vals) / len(vals))
    return out


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    pair_path = _resolve(workspace_root, args.pair_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    rows = read_jsonl(pair_path)
    matched, summary = build_matched_pairs(
        rows,
        min_response_tokens=int(args.min_response_tokens),
        max_response_tokens=int(args.max_response_tokens),
        max_abs_token_delta=int(args.max_abs_token_delta),
        max_length_ratio=float(args.max_length_ratio),
        max_prompt_overlap_delta=float(args.max_prompt_overlap_delta),
        max_type_token_ratio_delta=float(args.max_type_token_ratio_delta),
        max_punct_rate_delta=float(args.max_punct_rate_delta),
        require_all_controls=bool(args.require_all_controls),
        strata=str(args.strata),
        max_total_pairs=int(args.max_total_pairs),
        max_pairs_per_stratum=int(args.max_pairs_per_stratum),
        seed=int(args.seed),
    )
    if not matched:
        raise ValueError("No pairs survived the matching constraints.")

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_out = out_dir / "pairs.jsonl"
    summary_out = out_dir / "summary.json"
    _write_jsonl(pair_out, matched)
    write_json(
        summary_out,
        {
            "stage": "D4-human-LLM-matched-pair-subset",
            "pair_jsonl": str(pair_path),
            "out_dir": str(out_dir),
            "matched_pair_jsonl": str(pair_out),
            "summary_json": str(summary_out),
            "seed": int(args.seed),
            **summary,
        },
    )
    print(f"pairs={pair_out}")
    print(f"summary={summary_out}")
    print(f"n_matched_pairs={len(matched)}")
    print(f"by_dataset={summary['by_dataset']}")
    print(f"mean_metrics={summary['mean_metrics']}")


if __name__ == "__main__":
    main()
