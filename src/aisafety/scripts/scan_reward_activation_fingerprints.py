"""Compare reward-model runs by layerwise value-lens and activation fingerprints.

This script is the reward-model analogue of `scan_lora_activation_fingerprints.py`.
It works with scalar reward scorers (backbone + value head), not CausalLM logits.

For each run, it computes on Laurito A/B trials:
- final scalar margin oriented as (LLM - human)
- layerwise "value lens": apply the run's value head to each layer's last-token hidden
- layerwise mean hidden delta vectors, oriented as (LLM - human)

Then it writes CSVs compatible with the existing pivot/render pipeline:
- `final_logit_margin_summary.csv`
- `logit_lens_by_layer.csv`
- `delta_hidden_mean_norm_by_layer.csv`
- `adapter_similarity_hidden.csv`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aisafety.config import DEFAULT_CACHE_DIR
from aisafety.features.token_positions import take_last_token
from aisafety.reward.model import load_reward_scorer


REQUIRED_TRIAL_COLS = ("item_type", "title", "A_text", "B_text", "A_source", "B_source")


@dataclass
class RunningStat:
    sum: torch.Tensor
    sum_sq: torch.Tensor
    n: int

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> "RunningStat":
        z = torch.zeros(shape, dtype=torch.float64)
        return cls(sum=z.clone(), sum_sq=z.clone(), n=0)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(dtype=torch.float32, device="cpu")
        self.sum += x.sum(dim=0, dtype=torch.float64)
        self.sum_sq += (x * x).sum(dim=0, dtype=torch.float64)
        self.n += int(x.shape[0])

    def mean(self) -> torch.Tensor:
        if self.n <= 0:
            raise ValueError("No samples accumulated.")
        return self.sum / float(self.n)

    def std(self) -> torch.Tensor:
        if self.n <= 0:
            raise ValueError("No samples accumulated.")
        m = self.mean()
        var = (self.sum_sq / float(self.n)) - (m * m)
        var = torch.clamp(var, min=0.0)
        return torch.sqrt(var)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False).ravel()
    b = b.astype(np.float64, copy=False).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


@torch.inference_mode()
def _fingerprint_reward(
    *,
    scorer,
    tokenizer,
    df_trials: pd.DataFrame,
    batch_size: int,
    max_length: int,
) -> dict[str, object]:
    padding_side = getattr(tokenizer, "padding_side", "right")
    device = next(p for p in scorer.parameters() if p.device.type != "meta").device
    value_dtype = scorer.value_head.weight.dtype
    hidden_size = int(getattr(scorer.backbone.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise RuntimeError("Could not infer hidden_size from backbone config.")

    hidden_stats: list[RunningStat] | None = None
    lens_stats: list[RunningStat] | None = None
    final_stats = RunningStat.zeros(())

    for i in tqdm(range(0, len(df_trials), int(batch_size)), desc="Fingerprint", leave=False):
        chunk = df_trials.iloc[i : i + int(batch_size)]
        if chunk.empty:
            continue

        a_src = chunk["A_source"].astype(str).str.lower().to_numpy()
        b_src = chunk["B_source"].astype(str).str.lower().to_numpy()
        valid = ((a_src == "llm") & (b_src == "human")) | ((a_src == "human") & (b_src == "llm"))
        if not bool(np.all(valid)):
            chunk = chunk.loc[valid].reset_index(drop=True)
            if chunk.empty:
                continue
            a_src = chunk["A_source"].astype(str).str.lower().to_numpy()

        sign = torch.as_tensor(np.where(a_src == "llm", 1.0, -1.0), device=device, dtype=torch.float32)

        a_texts = chunk["A_text"].astype(str).tolist()
        b_texts = chunk["B_text"].astype(str).tolist()
        texts = a_texts + b_texts
        batch_n = len(a_texts)

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length),
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = scorer.backbone(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("Backbone did not return hidden_states.")

        num_layers = len(hs) - 1
        if hidden_stats is None or lens_stats is None:
            hidden_stats = [RunningStat.zeros((hidden_size,)) for _ in range(num_layers)]
            lens_stats = [RunningStat.zeros(()) for _ in range(num_layers)]

        for l in range(1, num_layers + 1):
            h_last = take_last_token(hs[l], enc["attention_mask"], padding_side=padding_side)  # [2B, H]
            h_a = h_last[:batch_n]
            h_b = h_last[batch_n:]

            delta = (h_a - h_b).to(dtype=torch.float32) * sign.unsqueeze(1)
            hidden_stats[l - 1].update(delta)

            s_a = scorer.value_head(h_a.to(dtype=value_dtype)).squeeze(-1).to(dtype=torch.float32)
            s_b = scorer.value_head(h_b.to(dtype=value_dtype)).squeeze(-1).to(dtype=torch.float32)
            margin = sign * (s_a - s_b)
            lens_stats[l - 1].update(margin)
            if l == num_layers:
                final_stats.update(margin)

            del h_last, h_a, h_b, delta, s_a, s_b, margin

        del out, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if hidden_stats is None or lens_stats is None:
        raise RuntimeError("No valid Laurito batches processed; cannot compute fingerprints.")

    hidden_mean = np.stack([s.mean().numpy().astype(np.float32) for s in hidden_stats], axis=0)
    hidden_std = np.stack([s.std().numpy().astype(np.float32) for s in hidden_stats], axis=0)
    lens_mean = np.array([float(s.mean().item()) for s in lens_stats], dtype=np.float32)
    lens_std = np.array([float(s.std().item()) for s in lens_stats], dtype=np.float32)

    return {
        "hidden_mean": hidden_mean,
        "hidden_std": hidden_std,
        "lens_mean": lens_mean,
        "lens_std": lens_std,
        "final_logit_mean": float(final_stats.mean().item()),
        "final_logit_std": float(final_stats.std().item()),
        "num_layers": int(hidden_mean.shape[0]),
    }


def _validate_trials(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_TRIAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Trials CSV missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--trials-csv", type=Path, required=True)
    p.add_argument("--include-item-types", type=str, default="movie,paper,product")
    p.add_argument("--model-id", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)

    p.add_argument("--baseline-adapter-dir", type=Path, required=True)
    p.add_argument("--baseline-value-head", type=Path, required=True)
    p.add_argument("--invariance-adapter-dir", type=Path, required=True)
    p.add_argument("--invariance-value-head", type=Path, required=True)

    p.add_argument("--base-run-name", type=str, default="base")
    p.add_argument("--baseline-run-name", type=str, default="baseline")
    p.add_argument("--invariance-run-name", type=str, default="invariance")
    p.add_argument("--include-base-run", action="store_true")
    p.add_argument("--no-base-run", dest="include_base_run", action="store_false")
    p.set_defaults(include_base_run=True)
    p.add_argument(
        "--base-value-head",
        type=Path,
        default=None,
        help="Value head for no-LoRA base run. Defaults to --baseline-value-head.",
    )

    p.add_argument("--out-dir", type=Path, default=Path("artifacts/reward_activation_fingerprints"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_trials = pd.read_csv(args.trials_csv)
    _validate_trials(df_trials)
    include_item_types = [x.strip() for x in str(args.include_item_types).split(",") if x.strip()]
    if include_item_types:
        df_trials = df_trials[df_trials["item_type"].astype(str).isin(include_item_types)].reset_index(drop=True)
    if df_trials.empty:
        raise ValueError("No trials left after filtering --include-item-types.")

    base_head = args.base_value_head if args.base_value_head is not None else args.baseline_value_head
    runs: list[tuple[str, Path | None, Path]] = []
    if bool(args.include_base_run):
        runs.append((str(args.base_run_name), None, Path(base_head)))
    runs.append((str(args.baseline_run_name), Path(args.baseline_adapter_dir), Path(args.baseline_value_head)))
    runs.append((str(args.invariance_run_name), Path(args.invariance_adapter_dir), Path(args.invariance_value_head)))

    fps: dict[str, dict[str, object]] = {}
    run_meta: list[dict[str, str | None]] = []

    for run_name, adapter_dir, value_head in runs:
        print(f"Running reward fingerprint: {run_name}")
        scorer, tok = load_reward_scorer(
            model_id=str(args.model_id),
            cache_dir=Path(args.cache_dir),
            lora_adapter_dir=adapter_dir,
            value_head_path=value_head,
            device_map={"": 0} if torch.cuda.is_available() else "auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        scorer.eval()
        fps[run_name] = _fingerprint_reward(
            scorer=scorer,
            tokenizer=tok,
            df_trials=df_trials,
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
        )
        run_meta.append(
            {
                "run": run_name,
                "lora_adapter_dir": None if adapter_dir is None else str(adapter_dir),
                "value_head": str(value_head),
            }
        )
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_order = [r[0] for r in runs]
    num_layers = int(fps[run_order[0]]["num_layers"])

    final_rows = [
        {
            "run": run_name,
            "final_logit_mean": float(fps[run_name]["final_logit_mean"]),
            "final_logit_std": float(fps[run_name]["final_logit_std"]),
        }
        for run_name in run_order
    ]
    pd.DataFrame(final_rows).to_csv(out_dir / "final_logit_margin_summary.csv", index=False)

    lens_rows: list[dict[str, float | int | str]] = []
    for run_name in run_order:
        lens_mean = fps[run_name]["lens_mean"]
        lens_std = fps[run_name]["lens_std"]
        for layer_idx in range(num_layers):
            lens_rows.append(
                {
                    "run": run_name,
                    "layer": int(layer_idx),
                    "logit_lens_mean": float(lens_mean[layer_idx]),
                    "logit_lens_std": float(lens_std[layer_idx]),
                }
            )
    pd.DataFrame(lens_rows).to_csv(out_dir / "logit_lens_by_layer.csv", index=False)

    base_name = str(args.base_run_name) if bool(args.include_base_run) else str(args.baseline_run_name)
    if base_name not in fps:
        raise ValueError(f"Base run {base_name!r} not present in collected fingerprints.")
    base_h = fps[base_name]["hidden_mean"]

    delta_rows: list[dict[str, float | int | str]] = []
    for run_name in run_order:
        if run_name == base_name:
            continue
        d = fps[run_name]["hidden_mean"] - base_h
        norms = np.linalg.norm(d, axis=1)
        for layer_idx, n in enumerate(norms.tolist()):
            delta_rows.append({"run": run_name, "layer": int(layer_idx), "delta_hidden_mean_norm": float(n)})
    pd.DataFrame(delta_rows).to_csv(out_dir / "delta_hidden_mean_norm_by_layer.csv", index=False)

    runs_no_base = [n for n in run_order if n != base_name]
    deltas = {n: (fps[n]["hidden_mean"] - base_h) for n in runs_no_base}
    sim_rows: list[dict[str, float | int | str]] = []
    for i, a_name in enumerate(runs_no_base):
        for b_name in runs_no_base[i + 1 :]:
            a = deltas[a_name]
            b = deltas[b_name]
            for layer_idx in range(num_layers):
                sim_rows.append(
                    {
                        "run_a": a_name,
                        "run_b": b_name,
                        "layer": int(layer_idx),
                        "cosine": _cosine(a[layer_idx], b[layer_idx]),
                    }
                )
    pd.DataFrame(sim_rows).to_csv(out_dir / "adapter_similarity_hidden.csv", index=False)

    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": str(args.model_id),
                "trials_csv": str(args.trials_csv),
                "include_item_types": include_item_types,
                "max_length": int(args.max_length),
                "batch_size": int(args.batch_size),
                "base_run_name": base_name,
                "runs": run_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote reward activation fingerprints to {out_dir}")


if __name__ == "__main__":
    main()

