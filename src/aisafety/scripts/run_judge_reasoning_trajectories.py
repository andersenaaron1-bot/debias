"""Sample judge reasoning branches and capture selected residual trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.mech.judge_reasoning import (
    LastTokenTrajectoryRecorder,
    normalize_choice,
    parse_final_choice,
    render_model_prompt,
    resample_trajectory,
)
from aisafety.mech.labels import parse_int_list, select_hidden_layers
from aisafety.scripts.run_d4_bt_stage_contrast import _csv_list, _load_lm


DEFAULT_COMPARISONS = Path("data") / "derived" / "judge_reasoning_pairs_v1" / "comparisons.jsonl"
DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "judge_reasoning_trajectories_v1"


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
    parser.add_argument("--prompt-style", choices=["plain", "chat_template"], default="chat_template")
    parser.add_argument("--reasoning-modes", default="thinking,direct")
    parser.add_argument("--labels", default="A,B")
    parser.add_argument("--branches-per-comparison", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument(
        "--cap-strategy",
        choices=["global", "source_round_robin"],
        default="source_round_robin",
    )
    parser.add_argument("--selected-layers", default="")
    parser.add_argument("--layer-stride", type=int, default=6)
    parser.add_argument("--tail-layers", type=int, default=2)
    parser.add_argument("--trajectory-points", type=int, default=17)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--compress-shards", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens-thinking", type=int, default=512)
    parser.add_argument("--max-new-tokens-direct", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _selected_layers(model: Any, *, raw: str, stride: int, tail_layers: int) -> list[int]:
    num_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if num_layers <= 0:
        raise ValueError("Model config does not expose num_hidden_layers.")
    layers = parse_int_list(raw) if str(raw).strip() else select_hidden_layers(
        num_layers,
        stride=int(stride),
        tail_layers=int(tail_layers),
    )
    layers = [int(layer) for layer in layers if 1 <= int(layer) <= num_layers]
    if not layers:
        raise ValueError("No selected hidden layers remain.")
    return layers


def _label_ids(tokenizer: Any, labels: list[str]) -> tuple[int, int]:
    if len(labels) != 2:
        raise ValueError("--labels requires exactly two comma-separated labels.")
    encoded = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in labels]
    if any(len(ids) != 1 for ids in encoded):
        raise ValueError(f"Trajectory label margins require single-token labels, got {encoded}")
    return int(encoded[0][0]), int(encoded[1][0])


def _cap_comparisons(
    rows: list[dict[str, Any]],
    *,
    max_pairs: int,
    seed: int,
    strategy: str,
) -> list[dict[str, Any]]:
    pair_sources: dict[str, str] = {}
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if pair_id:
            pair_sources.setdefault(pair_id, str(row.get("source_dataset") or "unknown"))
    pair_ids = sorted(
        pair_sources,
        key=lambda item: sha1_hex(f"{seed}:judge-trajectory:{item}"),
    )
    if int(max_pairs) > 0 and str(strategy) == "global":
        pair_ids = pair_ids[: int(max_pairs)]
    elif int(max_pairs) > 0:
        buckets: dict[str, list[str]] = {}
        for pair_id in pair_ids:
            buckets.setdefault(pair_sources[pair_id], []).append(pair_id)
        selected: list[str] = []
        offset = 0
        sources = sorted(buckets)
        while len(selected) < int(max_pairs):
            added = False
            for source in sources:
                if offset < len(buckets[source]):
                    selected.append(buckets[source][offset])
                    added = True
                    if len(selected) >= int(max_pairs):
                        break
            if not added:
                break
            offset += 1
        pair_ids = selected
    keep = set(pair_ids)
    return [row for row in rows if str(row.get("pair_id") or "") in keep]


class TraceShardWriter:
    def __init__(
        self,
        root: Path,
        *,
        n_points: int,
        hidden_layers: list[int],
        shard_size: int,
        compress: bool,
        start_shard_index: int = 0,
    ):
        self.root = Path(root)
        self.n_points = int(n_points)
        self.hidden_layers = [int(layer) for layer in hidden_layers]
        self.shard_size = max(int(shard_size), 1)
        self.compress = bool(compress)
        self.shard_index = int(start_shard_index)
        self.states: list[np.ndarray] = []
        self.point_mask: list[np.ndarray] = []
        self.step_indices: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []
        self.label_margins: list[np.ndarray] = []

    def add(
        self,
        *,
        states: np.ndarray,
        step_indices: np.ndarray,
        positions: np.ndarray,
        label_margins: np.ndarray,
    ) -> tuple[str, int]:
        if states.ndim != 3:
            raise ValueError("states must have shape [points, layers, hidden].")
        n_valid = min(len(states), self.n_points)
        padded = np.zeros(
            (self.n_points, len(self.hidden_layers), states.shape[-1]),
            dtype=np.float16,
        )
        padded[:n_valid] = states[:n_valid].astype(np.float16)
        mask = np.zeros((self.n_points,), dtype=bool)
        mask[:n_valid] = True
        indices = np.full((self.n_points,), -1, dtype=np.int32)
        indices[:n_valid] = step_indices[:n_valid]
        point_positions = np.full((self.n_points,), np.nan, dtype=np.float32)
        point_positions[:n_valid] = positions[:n_valid]
        margins = np.full((self.n_points,), np.nan, dtype=np.float32)
        n_margins = min(n_valid, len(label_margins))
        margins[:n_margins] = label_margins[:n_margins]

        shard_name = f"trajectory_states_{self.shard_index:05d}.npz"
        row_index = len(self.states)
        self.states.append(padded)
        self.point_mask.append(mask)
        self.step_indices.append(indices)
        self.positions.append(point_positions)
        self.label_margins.append(margins)
        if len(self.states) >= self.shard_size:
            self.flush()
        return shard_name, row_index

    def flush(self) -> None:
        if not self.states:
            return
        path = self.root / f"trajectory_states_{self.shard_index:05d}.npz"
        payload = {
            "states": np.stack(self.states, axis=0),
            "point_mask": np.stack(self.point_mask, axis=0),
            "step_indices": np.stack(self.step_indices, axis=0),
            "positions": np.stack(self.positions, axis=0),
            "label_margins": np.stack(self.label_margins, axis=0),
            "hidden_layers": np.asarray(self.hidden_layers, dtype=np.int32),
        }
        writer = np.savez_compressed if self.compress else np.savez
        writer(path, **payload)
        self.shard_index += 1
        self.states.clear()
        self.point_mask.clear()
        self.step_indices.clear()
        self.positions.clear()
        self.label_margins.clear()


def _trace_label_margins(scores: Any, *, label_ids: tuple[int, int]) -> np.ndarray:
    import torch

    values: list[float] = []
    for score in scores or ():
        logits = score[0].float()
        values.append(float((logits[int(label_ids[0])] - logits[int(label_ids[1])]).detach().cpu()))
    return np.asarray(values, dtype=np.float32)


def _generate_trace(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    reasoning_mode: str,
    hidden_layers: list[int],
    label_ids: tuple[int, int],
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> tuple[str, list[int], list[int], np.ndarray, np.ndarray]:
    import torch

    device = next(param for param in model.parameters() if param.device.type != "meta").device
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_prompt_length),
    )
    prompt_token_ids = encoded["input_ids"][0].detach().cpu().tolist()
    prompt_tokens = len(prompt_token_ids)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    with LastTokenTrajectoryRecorder(model, hidden_layers=hidden_layers) as recorder:
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    generated = output.sequences[0, prompt_tokens:].detach().cpu().tolist()
    text = tokenizer.decode(generated, skip_special_tokens=False)
    states = recorder.array()
    margins = _trace_label_margins(output.scores, label_ids=label_ids)
    n_steps = min(len(states), len(margins), len(generated))
    return text, prompt_token_ids, generated, states[:n_steps], margins[:n_steps]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _next_shard_index(out_dir: Path) -> int:
    indices: list[int] = []
    for path in Path(out_dir).glob("trajectory_states_*.npz"):
        try:
            indices.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=-1) + 1


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

    model, tokenizer = _load_lm(args)
    hidden_layers = _selected_layers(
        model,
        raw=str(args.selected_layers),
        stride=int(args.layer_stride),
        tail_layers=int(args.tail_layers),
    )
    labels = _csv_list(str(args.labels))
    label_ids = _label_ids(tokenizer, labels)
    modes = _csv_list(str(args.reasoning_modes))
    invalid_modes = sorted(set(modes) - {"thinking", "direct", "free_reasoning"})
    if invalid_modes:
        raise ValueError(f"Unsupported reasoning modes: {invalid_modes}")

    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.jsonl"
    existing_rows = read_jsonl(traces_path) if bool(args.resume) and traces_path.is_file() else []
    if not bool(args.resume) and (
        traces_path.exists() or any(out_dir.glob("trajectory_states_*.npz"))
    ):
        raise FileExistsError(
            f"Trace output already exists in {out_dir}; use --resume or a new --out-dir."
        )
    if not bool(args.resume):
        traces_path.write_text("", encoding="utf-8")
    existing_trace_ids = {str(row.get("trace_id") or "") for row in existing_rows}
    start_shard_index = _next_shard_index(out_dir)
    shard_writer = TraceShardWriter(
        out_dir,
        n_points=int(args.trajectory_points),
        hidden_layers=hidden_layers,
        shard_size=int(args.shard_size),
        compress=bool(args.compress_shards),
        start_shard_index=start_shard_index,
    )
    new_trace_rows: list[dict[str, Any]] = []
    pending_trace_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        for reasoning_mode in modes:
            prompt = render_model_prompt(
                comparison,
                tokenizer,
                prompt_style=str(args.prompt_style),
                reasoning_mode=reasoning_mode,
            )
            for branch_index in range(max(int(args.branches_per_comparison), 1)):
                branch_seed = int(
                    sha1_hex(
                        f"{args.seed}:{comparison['comparison_id']}:{reasoning_mode}:{branch_index}"
                    )[:8],
                    16,
                )
                max_new_tokens = (
                    int(args.max_new_tokens_direct)
                    if reasoning_mode == "direct"
                    else int(args.max_new_tokens_thinking)
                )
                trace_id = sha1_hex(
                    f"{comparison['comparison_id']}|{reasoning_mode}|{branch_index}|{branch_seed}"
                )
                if trace_id in existing_trace_ids:
                    continue
                text, prompt_token_ids, token_ids, states, margins = _generate_trace(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    reasoning_mode=reasoning_mode,
                    hidden_layers=hidden_layers,
                    label_ids=label_ids,
                    max_prompt_length=int(args.max_prompt_length),
                    max_new_tokens=max_new_tokens,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    seed=branch_seed,
                )
                if len(states) <= 0:
                    continue
                sampled, indices, positions = resample_trajectory(
                    states,
                    n_points=int(args.trajectory_points),
                )
                sampled_margins = margins[indices]
                shard_index_before = shard_writer.shard_index
                shard_name, shard_row = shard_writer.add(
                    states=sampled,
                    step_indices=indices,
                    positions=positions,
                    label_margins=sampled_margins,
                )
                final_choice = parse_final_choice(text)
                target_option = normalize_choice(comparison.get("target_option"))
                trace_row = {
                    **comparison,
                    "trace_id": trace_id,
                    "run_label": str(args.run_label or args.model_id),
                    "model_id": str(args.model_id),
                    "prompt_style": str(args.prompt_style),
                    "reasoning_mode": reasoning_mode,
                    "branch_index": int(branch_index),
                    "branch_seed": int(branch_seed),
                    "final_choice": final_choice,
                    "valid_choice": bool(final_choice),
                    "target_selected": None
                    if not final_choice or not target_option
                    else bool(final_choice == target_option),
                    "response_text": text,
                    "prompt_token_ids": prompt_token_ids,
                    "generated_token_ids": token_ids,
                    "prompt_text": prompt,
                    "prompt_tokens": int(len(prompt_token_ids)),
                    "generated_tokens": int(len(token_ids)),
                    "trajectory_steps": int(len(states)),
                    "trajectory_shard": shard_name,
                    "trajectory_shard_row": int(shard_row),
                }
                new_trace_rows.append(trace_row)
                pending_trace_rows.append(trace_row)
                if shard_writer.shard_index > shard_index_before:
                    _append_jsonl(traces_path, pending_trace_rows)
                    pending_trace_rows.clear()
    shard_writer.flush()
    _append_jsonl(traces_path, pending_trace_rows)
    trace_rows = existing_rows + new_trace_rows
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "judge-reasoning-trajectory-capture",
            "comparisons_jsonl": str(comparisons_path),
            "out_dir": str(out_dir),
            "traces_jsonl": str(traces_path),
            "run_label": str(args.run_label or args.model_id),
            "model_id": str(args.model_id),
            "model_config_name_or_path": str(
                getattr(model.config, "_name_or_path", "") or ""
            ),
            "model_config_commit_hash": str(
                getattr(model.config, "_commit_hash", "") or ""
            ),
            "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "") or ""),
            "prompt_style": str(args.prompt_style),
            "reasoning_modes": modes,
            "labels": labels,
            "hidden_layers": hidden_layers,
            "trajectory_points": int(args.trajectory_points),
            "branches_per_comparison": int(args.branches_per_comparison),
            "max_pairs": int(args.max_pairs),
            "cap_strategy": str(args.cap_strategy),
            "max_prompt_length": int(args.max_prompt_length),
            "max_new_tokens_thinking": int(args.max_new_tokens_thinking),
            "max_new_tokens_direct": int(args.max_new_tokens_direct),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "resume": bool(args.resume),
            "n_existing_traces": int(len(existing_rows)),
            "n_new_traces": int(len(new_trace_rows)),
            "n_comparisons": int(len({row["comparison_id"] for row in trace_rows})),
            "n_traces": int(len(trace_rows)),
            "n_valid_choices": int(sum(bool(row["valid_choice"]) for row in trace_rows)),
            "n_shards": int(len(list(out_dir.glob("trajectory_states_*.npz")))),
            "n_new_shards": int(shard_writer.shard_index - start_shard_index),
        },
    )
    print(f"traces={traces_path}")
    print(f"n_traces={len(trace_rows)}")
    print(f"n_new_traces={len(new_trace_rows)}")
    print(f"n_valid_choices={sum(bool(row['valid_choice']) for row in trace_rows)}")


if __name__ == "__main__":
    main()
