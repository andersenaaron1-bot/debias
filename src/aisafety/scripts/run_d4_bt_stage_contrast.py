"""Score D4 Bradley-Terry cue contrasts across base, instruction, and reward stages."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.scripts.run_d4_candidate_feature_pair_alignment import (
    _load_scorer_and_tokenizer,
    _score_texts,
    _write_csv,
)
from aisafety.scripts.run_d4_surface_counterfactual_audit import _cap_rows


DEFAULT_BT_JSONL = Path("data") / "derived" / "d4_bt_stage_contrast_pairs_v1" / "bt_pairs.jsonl"
DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_bt_stage_contrast_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--bt-pairs-jsonl", type=Path, default=DEFAULT_BT_JSONL)
    parser.add_argument(
        "--stage",
        choices=["base_lm", "it_lm", "reward_j0"],
        required=True,
        help="Scoring stage. base_lm and it_lm use forced-choice logprobs; reward_j0 uses scalar reward margins.",
    )
    parser.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--reward-run-dir", type=Path, default=Path("artifacts") / "reward" / "j0_anchor_v1_h100compact")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--prompt-style", choices=["plain", "chat_template"], default="plain")
    parser.add_argument("--labels", type=str, default="A,B")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


def _comparison_user_content(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "Compare the two responses.").strip()
    a = str(row.get("option_a_text") or "").strip()
    b = str(row.get("option_b_text") or "").strip()
    return (
        "Compare two responses to the same prompt.\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Response A:\n{a}\n\n"
        f"Response B:\n{b}\n\n"
        "Which response is better? Reply with only A or B."
    )


def _comparison_prompt(row: dict[str, Any], tokenizer: Any, *, prompt_style: str) -> str:
    content = _comparison_user_content(row)
    if prompt_style == "chat_template":
        if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
            raise ValueError("Tokenizer has no chat template; use --prompt-style plain.")
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return content + "\nAnswer: "


def _load_lm(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cache_dir = str(args.cache_dir) if args.cache_dir else None
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), cache_dir=cache_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if bool(args.use_4bit):
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_id),
            cache_dir=cache_dir,
            quantization_config=qconf,
            device_map="auto",
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_id),
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
    model.eval()
    return model, tokenizer


def _label_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    labels: list[str],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch

    if len(labels) != 2:
        raise ValueError("--labels must contain exactly two labels, usually A,B.")
    device = next(p for p in model.parameters() if p.device.type != "meta").device
    requests: list[dict[str, Any]] = []
    max_label_len = max(len(tokenizer(label, add_special_tokens=False)["input_ids"]) for label in labels)
    prompt_max = max(8, int(max_length) - int(max_label_len) - 1)
    for prompt_index, prompt in enumerate(prompts):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=prompt_max,
        )["input_ids"]
        for label_index, label in enumerate(labels):
            label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
            if not label_ids:
                raise ValueError(f"Label {label!r} tokenized to no ids.")
            input_ids = list(prompt_ids) + list(label_ids)
            requests.append(
                {
                    "prompt_index": prompt_index,
                    "label_index": label_index,
                    "input_ids": input_ids,
                    "label_start": len(prompt_ids),
                    "label_len": len(label_ids),
                }
            )

    out = np.full((len(prompts), len(labels)), np.nan, dtype=float)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for start in range(0, len(requests), int(batch_size)):
        batch = requests[start : start + int(batch_size)]
        max_len = max(len(req["input_ids"]) for req in batch)
        input_ids = []
        attention_mask = []
        for req in batch:
            ids = list(req["input_ids"])
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
        enc_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        enc_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=enc_ids, attention_mask=enc_mask, return_dict=True).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        for row_idx, req in enumerate(batch):
            total = 0.0
            label_start = int(req["label_start"])
            label_len = int(req["label_len"])
            ids = req["input_ids"]
            for pos in range(label_start, label_start + label_len):
                if pos <= 0:
                    continue
                token_id = int(ids[pos])
                total += float(log_probs[row_idx, pos - 1, token_id].detach().cpu())
            out[int(req["prompt_index"]), int(req["label_index"])] = total
        del enc_ids, enc_mask, logits, log_probs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _score_lm_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    model, tokenizer = _load_lm(args)
    labels = _csv_list(str(args.labels))
    if len(labels) != 2:
        raise ValueError("--labels must contain exactly two labels, usually A,B.")
    prompts = [
        _comparison_prompt(row._asdict(), tokenizer, prompt_style=str(args.prompt_style))
        for row in df.itertuples(index=False)
    ]
    logprobs = _label_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        labels=labels,
        batch_size=int(args.score_batch_size),
        max_length=int(args.max_length),
    )
    out = df.copy()
    out["stage"] = str(args.stage)
    out["model_id"] = str(args.model_id)
    out["prompt_style"] = str(args.prompt_style)
    out["logprob_A"] = logprobs[:, 0]
    out["logprob_B"] = logprobs[:, 1]
    out["bt_margin_A_minus_B"] = out["logprob_A"] - out["logprob_B"]
    cue_plus_is_a = out["cue_plus_option"].astype(str) == "A"
    out["cue_plus_margin"] = np.where(cue_plus_is_a, out["bt_margin_A_minus_B"], -out["bt_margin_A_minus_B"])
    out["cue_plus_prob"] = out["cue_plus_margin"].map(lambda value: _sigmoid(float(value)))
    out["cue_plus_preferred"] = out["cue_plus_margin"].astype(float) > 0.0
    return out


def _score_reward_rows(df: pd.DataFrame, args: argparse.Namespace, workspace_root: Path) -> pd.DataFrame:
    scorer, tokenizer = _load_scorer_and_tokenizer(args, workspace_root)
    text_by_id: dict[str, str] = {}
    for text in df["cue_plus_text"].astype(str).tolist() + df["cue_minus_text"].astype(str).tolist():
        text_by_id.setdefault(sha1_hex(text), text)
    text_ids = sorted(text_by_id)
    scores = _score_texts(
        scorer=scorer,
        tokenizer=tokenizer,
        texts=[text_by_id[text_id] for text_id in text_ids],
        batch_size=int(args.score_batch_size),
        max_length=int(args.max_length),
    )
    score_by_id = {text_id: float(score) for text_id, score in zip(text_ids, scores, strict=True)}
    out = df.copy()
    out["stage"] = str(args.stage)
    out["model_id"] = str(args.model_id)
    out["prompt_style"] = "reward_scalar"
    out["cue_plus_reward"] = [score_by_id[sha1_hex(text)] for text in out["cue_plus_text"].astype(str)]
    out["cue_minus_reward"] = [score_by_id[sha1_hex(text)] for text in out["cue_minus_text"].astype(str)]
    out["cue_plus_margin"] = out["cue_plus_reward"] - out["cue_minus_reward"]
    out["cue_plus_prob"] = out["cue_plus_margin"].map(lambda value: _sigmoid(float(value)))
    out["cue_plus_preferred"] = out["cue_plus_margin"].astype(float) > 0.0
    return out


def _mean_bool(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.astype(bool).mean())


def _summary_row(df: pd.DataFrame, *, group_type: str, group_value: str) -> dict[str, Any]:
    margin = pd.to_numeric(df["cue_plus_margin"], errors="coerce")
    prob = pd.to_numeric(df["cue_plus_prob"], errors="coerce")
    return {
        "group_type": group_type,
        "group_value": group_value,
        "n_pairs": int(len(df)),
        "mean_cue_plus_margin": float(margin.mean()),
        "median_cue_plus_margin": float(margin.median()),
        "mean_abs_cue_plus_margin": float(margin.abs().mean()),
        "mean_cue_plus_prob": float(prob.mean()),
        "cue_plus_preference_rate": _mean_bool(df["cue_plus_preferred"]),
        "mean_token_delta_plus_minus": float(pd.to_numeric(df["cue_plus_minus_cue_minus_tokens"], errors="coerce").mean()),
    }


def _build_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = [_summary_row(df, group_type="all", group_value="all")]
    for key in ("axis", "axis_direction", "axis_role", "source_axis", "presentation_order"):
        if key == "axis":
            groups = [(axis, group) for axis, group in df.groupby("axis", sort=True)]
        elif key == "axis_direction":
            groups = [
                (f"{axis}::{direction}", group)
                for (axis, direction), group in df.groupby(["axis", "direction"], sort=True)
            ]
        elif key == "axis_role":
            groups = [(f"{axis}::{role}", group) for (axis, role), group in df.groupby(["axis", "role"], sort=True)]
        elif key == "source_axis":
            groups = [
                (f"{source}::{axis}", group)
                for (source, axis), group in df.groupby(["source_dataset", "axis"], sort=True)
            ]
        else:
            groups = [(order, group) for order, group in df.groupby("presentation_order", sort=True)]
        for group_value, group in groups:
            rows.append(_summary_row(group, group_type=key, group_value=str(group_value)))
    return rows


def _score_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "bt_pair_id",
        "counterfactual_id",
        "pair_id",
        "source_dataset",
        "subset",
        "split",
        "item_type",
        "role",
        "axis",
        "direction",
        "presentation_order",
        "cue_plus_option",
        "stage",
        "model_id",
        "prompt_style",
        "cue_plus_margin",
        "cue_plus_prob",
        "cue_plus_preferred",
        "logprob_A",
        "logprob_B",
        "bt_margin_A_minus_B",
        "cue_plus_reward",
        "cue_minus_reward",
        "cue_plus_tokens",
        "cue_minus_tokens",
        "cue_plus_minus_cue_minus_tokens",
        "length_ratio_plus_over_minus",
        "transform_id",
        "content_preservation_flags",
    ]
    return [col for col in preferred if col in df.columns]


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    bt_path = _resolve(workspace_root, args.bt_pairs_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    rows = _cap_rows(read_jsonl(bt_path), max_rows=int(args.max_pairs), seed=int(args.seed))
    if not rows:
        raise ValueError(f"No BT rows found in {bt_path}")
    df = pd.DataFrame(rows)

    if str(args.stage) == "reward_j0":
        scored = _score_reward_rows(df, args, workspace_root)
    else:
        scored = _score_lm_rows(df, args)

    summary_rows = _build_summary(scored)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "bt_stage_scores.csv"
    summary_path = out_dir / "stage_summary.csv"
    scored[_score_columns(scored)].to_csv(scores_path, index=False)
    _write_csv(summary_path, summary_rows)
    manifest = {
        "stage": "D4-BT-stage-contrast-score",
        "bt_pairs_jsonl": str(bt_path),
        "out_dir": str(out_dir),
        "scoring_stage": str(args.stage),
        "model_id": str(args.model_id),
        "prompt_style": str(args.prompt_style),
        "reward_run_dir": str(args.reward_run_dir),
        "max_pairs": int(args.max_pairs),
        "n_pairs": int(len(scored)),
        "counts_by_axis_direction": {
            f"{axis}::{direction}": int(count)
            for (axis, direction), count in Counter(zip(scored["axis"].astype(str), scored["direction"].astype(str))).items()
        },
        "outputs": {
            "bt_stage_scores_csv": str(scores_path),
            "stage_summary_csv": str(summary_path),
            "manifest_json": str(out_dir / "manifest.json"),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"out_dir={out_dir}")
    print(f"scores={scores_path}")
    print(f"summary={summary_path}")
    print(f"manifest={out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
