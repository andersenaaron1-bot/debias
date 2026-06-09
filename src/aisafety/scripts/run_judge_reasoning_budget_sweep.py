"""Run nested token-budget interventions on pairwise judge reasoning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.mech.judge_reasoning import (
    normalize_choice,
    parse_final_choice,
    render_model_prompt,
)
from aisafety.scripts.run_d4_bt_stage_contrast import _csv_list, _load_lm
from aisafety.scripts.run_judge_reasoning_trajectories import _cap_comparisons


DEFAULT_COMPARISONS = (
    Path("data") / "derived" / "judge_deliberation_suite_v1" / "comparisons.jsonl"
)
DEFAULT_OUT_DIR = (
    Path("artifacts") / "mechanistic" / "judge_deliberation_budget_sweep_v1"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--comparisons-jsonl", type=Path, default=DEFAULT_COMPARISONS)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument(
        "--prompt-style",
        choices=["plain", "chat_template"],
        default="chat_template",
    )
    parser.add_argument("--labels", default="A,B")
    parser.add_argument("--budget-tokens", default="0,128,256,512,1024,2048")
    parser.add_argument("--branches-per-comparison", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--cap-strategy",
        choices=["global", "source_round_robin"],
        default="source_round_robin",
    )
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-score-length", type=int, default=8192)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens-direct", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _parse_budgets(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in str(raw).split(",") if value.strip()})
    if not values or values[0] < 0:
        raise ValueError("--budget-tokens requires nonnegative integers.")
    return values


def _single_token_label_ids(tokenizer: Any, labels: list[str]) -> tuple[int, int]:
    if len(labels) != 2:
        raise ValueError("--labels requires exactly two labels.")
    encoded = [
        tokenizer(label, add_special_tokens=False)["input_ids"]
        for label in labels
    ]
    if any(len(ids) != 1 for ids in encoded):
        raise ValueError(f"Budget verdict labels must be single tokens, got {encoded}")
    return int(encoded[0][0]), int(encoded[1][0])


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _generate(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> tuple[str, list[int], bool]:
    import torch

    device = next(
        parameter for parameter in model.parameters()
        if parameter.device.type != "meta"
    ).device
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_prompt_length),
    )
    prompt_tokens = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(device) for key, value in encoded.items()}
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    token_ids = output.sequences[0, prompt_tokens:].detach().cpu().tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    saturated = len(token_ids) >= int(max_new_tokens)
    return text, token_ids, saturated


def _forced_prompt(prompt: str, prefix_text: str, *, thinking: bool) -> str:
    text = prompt + prefix_text
    if thinking and "</think>" not in prefix_text:
        text += "\n</think>"
    return text + "\nFINAL:"


def _score_label_prompts(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    label_ids: tuple[int, int],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    import torch

    device = next(
        parameter for parameter in model.parameters()
        if parameter.device.type != "meta"
    ).device
    outputs: list[np.ndarray] = []
    for start in range(0, len(prompts), max(int(batch_size), 1)):
        batch = prompts[start : start + max(int(batch_size), 1)]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length),
        )
        lengths = encoded["attention_mask"].sum(dim=1).long()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded, return_dict=True).logits.float()
        row_index = torch.arange(len(batch), device=logits.device)
        final_logits = logits[row_index, lengths.to(logits.device) - 1]
        selected = final_logits[:, [int(label_ids[0]), int(label_ids[1])]]
        outputs.append(selected.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 2))


def _verdict_row(
    *,
    comparison: dict[str, Any],
    run_label: str,
    model_id: str,
    prompt_style: str,
    mode: str,
    branch_index: int,
    branch_seed: int,
    budget_tokens: int,
    available_prefix_tokens: int,
    natural_choice: str,
    full_natural_choice: str,
    full_generated_tokens: int,
    max_budget_saturated: bool,
    logits: np.ndarray,
    labels: list[str],
    trace_id: str,
) -> dict[str, Any]:
    scores = np.asarray(logits, dtype=float)
    margin = float(scores[0] - scores[1])
    shifted = scores - float(np.max(scores))
    probabilities = np.exp(shifted) / float(np.exp(shifted).sum())
    forced_choice = labels[0] if margin >= 0 else labels[1]
    target = normalize_choice(comparison.get("target_option"))
    selected_text = (
        str(comparison.get("option_a_text") or "")
        if forced_choice == "A"
        else str(comparison.get("option_b_text") or "")
    )
    budget_eval_id = sha1_hex(
        f"{trace_id}|{mode}|{budget_tokens}|{available_prefix_tokens}"
    )
    return {
        **comparison,
        "budget_eval_id": budget_eval_id,
        "trace_id": trace_id,
        "run_label": run_label,
        "model_id": model_id,
        "prompt_style": prompt_style,
        "reasoning_mode": mode,
        "branch_index": int(branch_index),
        "branch_seed": int(branch_seed),
        "budget_tokens": int(budget_tokens),
        "available_prefix_tokens": int(available_prefix_tokens),
        "natural_choice_at_budget": normalize_choice(natural_choice),
        "natural_valid_at_budget": bool(normalize_choice(natural_choice)),
        "full_natural_choice": normalize_choice(full_natural_choice),
        "full_natural_valid": bool(normalize_choice(full_natural_choice)),
        "full_generated_tokens": int(full_generated_tokens),
        "generation_finished_before_budget": bool(
            int(full_generated_tokens) < int(budget_tokens)
        ),
        "max_budget_saturated": bool(max_budget_saturated),
        "forced_choice": forced_choice,
        "forced_margin_a_minus_b": margin,
        "forced_prob_a": float(probabilities[0]),
        "forced_choice_confidence": float(abs(probabilities[0] - probabilities[1])),
        "forced_target_selected": (
            None if not target else bool(forced_choice == target)
        ),
        "natural_target_selected": (
            None
            if not target or not normalize_choice(natural_choice)
            else bool(normalize_choice(natural_choice) == target)
        ),
        "forced_selected_text_hash": sha1_hex(selected_text),
    }


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    comparisons_path = _resolve(workspace_root, args.comparisons_jsonl)
    out_dir = _resolve(workspace_root, args.out_dir)
    comparisons = _cap_comparisons(
        read_jsonl(comparisons_path),
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
        strategy=str(args.cap_strategy),
    )
    if not comparisons:
        raise ValueError(f"No comparisons selected from {comparisons_path}")
    budgets = _parse_budgets(str(args.budget_tokens))
    max_budget = max(budgets)
    if max_budget <= 0:
        raise ValueError("At least one positive thinking budget is required.")

    model, tokenizer = _load_lm(args)
    labels = _csv_list(str(args.labels))
    label_ids = _single_token_label_ids(tokenizer, labels)
    run_label = str(args.run_label or args.model_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "budget_scores.jsonl"
    traces_path = out_dir / "reasoning_traces.jsonl"
    existing_scores = (
        read_jsonl(scores_path)
        if bool(args.resume) and scores_path.is_file()
        else []
    )
    existing_traces = (
        read_jsonl(traces_path)
        if bool(args.resume) and traces_path.is_file()
        else []
    )
    if not bool(args.resume) and (scores_path.exists() or traces_path.exists()):
        raise FileExistsError(
            f"Budget output already exists in {out_dir}; use --resume or a new directory."
        )
    if not bool(args.resume):
        scores_path.write_text("", encoding="utf-8")
        traces_path.write_text("", encoding="utf-8")
    completed_eval_ids = {
        str(row.get("budget_eval_id") or "") for row in existing_scores
    }
    completed_trace_budgets = {
        (
            str(row.get("trace_id") or ""),
            str(row.get("reasoning_mode") or ""),
            int(row.get("budget_tokens") or 0),
        )
        for row in existing_scores
    }
    completed_trace_ids = {
        str(row.get("trace_id") or "") for row in existing_traces
    }

    n_new_scores = 0
    n_new_traces = 0
    for comparison in comparisons:
        direct_seed = int(
            sha1_hex(f"{args.seed}:{comparison['comparison_id']}:direct")[:8],
            16,
        )
        direct_trace_id = sha1_hex(
            f"{comparison['comparison_id']}|direct|{direct_seed}"
        )
        direct_eval_id = sha1_hex(f"{direct_trace_id}|direct|0|0")
        if direct_eval_id not in completed_eval_ids:
            direct_prompt = render_model_prompt(
                comparison,
                tokenizer,
                prompt_style=str(args.prompt_style),
                reasoning_mode="direct",
            )
            direct_text, direct_tokens, direct_saturated = _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=direct_prompt,
                max_prompt_length=int(args.max_prompt_length),
                max_new_tokens=int(args.max_new_tokens_direct),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                seed=direct_seed,
            )
            direct_logits = _score_label_prompts(
                model=model,
                tokenizer=tokenizer,
                prompts=[_forced_prompt(direct_prompt, "", thinking=False)],
                label_ids=label_ids,
                max_length=int(args.max_score_length),
                batch_size=1,
            )[0]
            row = _verdict_row(
                comparison=comparison,
                run_label=run_label,
                model_id=str(args.model_id),
                prompt_style=str(args.prompt_style),
                mode="direct",
                branch_index=-1,
                branch_seed=direct_seed,
                budget_tokens=0,
                available_prefix_tokens=0,
                natural_choice=parse_final_choice(direct_text),
                full_natural_choice=parse_final_choice(direct_text),
                full_generated_tokens=len(direct_tokens),
                max_budget_saturated=direct_saturated,
                logits=direct_logits,
                labels=labels,
                trace_id=direct_trace_id,
            )
            _append_jsonl(scores_path, [row])
            completed_eval_ids.add(direct_eval_id)
            n_new_scores += 1

        thinking_prompt = render_model_prompt(
            comparison,
            tokenizer,
            prompt_style=str(args.prompt_style),
            reasoning_mode="thinking",
        )
        for branch_index in range(max(int(args.branches_per_comparison), 1)):
            branch_seed = int(
                sha1_hex(
                    f"{args.seed}:{comparison['comparison_id']}:thinking:{branch_index}"
                )[:8],
                16,
            )
            trace_id = sha1_hex(
                f"{comparison['comparison_id']}|thinking|{branch_index}|{branch_seed}"
            )
            expected_budgets = {
                (trace_id, "thinking", int(budget))
                for budget in budgets
            }
            if expected_budgets.issubset(completed_trace_budgets):
                continue
            response_text, token_ids, saturated = _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=thinking_prompt,
                max_prompt_length=int(args.max_prompt_length),
                max_new_tokens=int(max_budget),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                seed=branch_seed,
            )
            prefix_texts = [
                tokenizer.decode(
                    token_ids[: min(int(budget), len(token_ids))],
                    skip_special_tokens=False,
                )
                for budget in budgets
            ]
            forced_prompts = [
                _forced_prompt(thinking_prompt, prefix, thinking=True)
                for prefix in prefix_texts
            ]
            logits = _score_label_prompts(
                model=model,
                tokenizer=tokenizer,
                prompts=forced_prompts,
                label_ids=label_ids,
                max_length=int(args.max_score_length),
                batch_size=int(args.score_batch_size),
            )
            full_choice = parse_final_choice(response_text)
            score_rows: list[dict[str, Any]] = []
            for budget, prefix, current_logits in zip(
                budgets,
                prefix_texts,
                logits,
                strict=True,
            ):
                available = min(int(budget), len(token_ids))
                row = _verdict_row(
                    comparison=comparison,
                    run_label=run_label,
                    model_id=str(args.model_id),
                    prompt_style=str(args.prompt_style),
                    mode="thinking",
                    branch_index=branch_index,
                    branch_seed=branch_seed,
                    budget_tokens=int(budget),
                    available_prefix_tokens=available,
                    natural_choice=parse_final_choice(prefix),
                    full_natural_choice=full_choice,
                    full_generated_tokens=len(token_ids),
                    max_budget_saturated=saturated,
                    logits=current_logits,
                    labels=labels,
                    trace_id=trace_id,
                )
                if row["budget_eval_id"] not in completed_eval_ids:
                    score_rows.append(row)
                    completed_eval_ids.add(str(row["budget_eval_id"]))
                    completed_trace_budgets.add(
                        (trace_id, "thinking", int(budget))
                    )
            _append_jsonl(scores_path, score_rows)
            n_new_scores += len(score_rows)
            if trace_id not in completed_trace_ids:
                _append_jsonl(
                    traces_path,
                    [
                        {
                            **comparison,
                            "trace_id": trace_id,
                            "run_label": run_label,
                            "model_id": str(args.model_id),
                            "prompt_style": str(args.prompt_style),
                            "reasoning_mode": "thinking",
                            "branch_index": int(branch_index),
                            "branch_seed": int(branch_seed),
                            "response_text": response_text,
                            "generated_token_ids": token_ids,
                            "generated_tokens": int(len(token_ids)),
                            "max_budget_saturated": bool(saturated),
                            "final_choice": full_choice,
                            "valid_choice": bool(full_choice),
                        }
                    ],
                )
                completed_trace_ids.add(trace_id)
                n_new_traces += 1

    all_scores = read_jsonl(scores_path)
    all_traces = read_jsonl(traces_path)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-reasoning-token-budget-sweep",
            "comparisons_jsonl": str(comparisons_path),
            "out_dir": str(out_dir),
            "budget_scores_jsonl": str(scores_path),
            "reasoning_traces_jsonl": str(traces_path),
            "run_label": run_label,
            "model_id": str(args.model_id),
            "prompt_style": str(args.prompt_style),
            "labels": labels,
            "budget_tokens": budgets,
            "branches_per_comparison": int(args.branches_per_comparison),
            "max_pairs": int(args.max_pairs),
            "cap_strategy": str(args.cap_strategy),
            "max_prompt_length": int(args.max_prompt_length),
            "max_score_length": int(args.max_score_length),
            "max_new_tokens_direct": int(args.max_new_tokens_direct),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "resume": bool(args.resume),
            "n_comparisons": int(len(comparisons)),
            "n_scores": int(len(all_scores)),
            "n_reasoning_traces": int(len(all_traces)),
            "n_new_scores": int(n_new_scores),
            "n_new_reasoning_traces": int(n_new_traces),
        },
    )
    print(f"out_dir={out_dir}")
    print(f"n_comparisons={len(comparisons)}")
    print(f"n_scores={len(all_scores)}")
    print(f"n_reasoning_traces={len(all_traces)}")


if __name__ == "__main__":
    main()
