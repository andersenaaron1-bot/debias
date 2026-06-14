"""Patch matched criterion/evidence states into confirmation readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import (
    read_jsonl,
    resolve_path,
    sha1_hex,
    write_json,
)
from aisafety.mech.interventions import (
    find_decoder_layer_module,
    remove_hooks,
)
from aisafety.mech.judge_reasoning import (
    OneShotLastTokenSteeringHook,
    random_orthogonal_direction,
)
from aisafety.scripts.analyze_judge_reasoning_trajectories import TraceArtifact
from aisafety.scripts.run_d4_bt_stage_contrast import _load_lm
from aisafety.scripts.run_judge_criterion_switch_behavior import (
    _forced_prompt,
    _semantic_verdict,
)
from aisafety.scripts.run_judge_reasoning_budget_sweep import (
    _single_token_label_ids,
)


PATCH_SPECS = (
    {
        "patch_type": "criterion_update",
        "donor_condition": "early_criterion",
        "recipient_condition": "late_criterion",
        "point_name": "phase1_readout_128",
        "stage": "phase1",
        "budget_tokens": 128,
        "hidden_layer": 20,
    },
    {
        "patch_type": "evidence_operationalization",
        "donor_condition": "late_evidence",
        "recipient_condition": "late_criterion",
        "point_name": "phase2_readout_0",
        "stage": "phase2",
        "budget_tokens": 0,
        "hidden_layer": 32,
    },
    {
        "patch_type": "explicit_target",
        "donor_condition": "late_explicit_target",
        "recipient_condition": "late_criterion",
        "point_name": "phase2_readout_0",
        "stage": "phase2",
        "budget_tokens": 0,
        "hidden_layer": 32,
    },
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
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--labels", default="A,B,C")
    parser.add_argument("--alphas", default="1.0")
    parser.add_argument(
        "--include-patch-types",
        default="criterion_update,evidence_operationalization,explicit_target",
    )
    parser.add_argument("--include-orders", default="original,swapped")
    parser.add_argument("--branch-index", type=int, default=0)
    parser.add_argument("--max-pairs-per-patch", type=int, default=0)
    parser.add_argument("--max-score-length", type=int, default=8192)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/mechanistic/"
            "criterion_confirmation_patching_v1"
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


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["pair_id"]),
        str(row["presentation_order"]),
        int(row["branch_index"]),
    )


class StateIndex:
    """Lazy trace-id and layer lookup for one activation artifact."""

    def __init__(self, trace_dir: Path):
        self.artifact = TraceArtifact(trace_dir)
        self.locations = {
            str(row["trace_id"]): int(index)
            for index, row in self.artifact.frame.reset_index(
                drop=True
            ).iterrows()
        }
        self.cache: dict[int, np.ndarray] = {}

    def state(
        self,
        trace_id: str,
        *,
        hidden_layer: int,
        point_name: str,
    ) -> np.ndarray:
        index = self.locations[str(trace_id)]
        row = self.artifact.frame.iloc[index]
        point_names = list(row.get("point_names") or [])
        if point_name not in point_names:
            raise KeyError(f"{point_name!r} missing for trace {trace_id}")
        if int(hidden_layer) not in self.cache:
            self.cache[int(hidden_layer)] = self.artifact.layer_states(
                int(hidden_layer)
            )
        return self.cache[int(hidden_layer)][
            index, point_names.index(point_name)
        ].astype(np.float32)


def matched_condition_rows(
    rows: Iterable[dict[str, Any]],
    *,
    donor_condition: str,
    recipient_condition: str,
    branch_index: int,
    include_orders: set[str],
) -> list[dict[str, Any]]:
    by_key: dict[
        tuple[str, str, int], dict[str, dict[str, Any]]
    ] = {}
    for row in rows:
        if int(row.get("branch_index") or 0) != int(branch_index):
            continue
        if str(row.get("presentation_order") or "") not in include_orders:
            continue
        by_key.setdefault(_key(row), {})[
            str(row.get("condition_id") or "")
        ] = row
    matches: list[dict[str, Any]] = []
    for key, conditions in sorted(by_key.items()):
        if donor_condition not in conditions:
            continue
        if recipient_condition not in conditions:
            continue
        donor = conditions[donor_condition]
        recipient = conditions[recipient_condition]
        matches.append(
            {
                "key": key,
                "pair_id": key[0],
                "presentation_order": key[1],
                "branch_index": key[2],
                "transition_type": str(
                    recipient.get("transition_type") or ""
                ),
                "target_displayed": str(
                    recipient.get("phase2_target_option") or ""
                ),
                "target_semantic": str(
                    recipient.get("phase2_target_semantic") or ""
                ),
                "initial_target_displayed": str(
                    recipient.get("phase1_target_option") or ""
                ),
                "initial_target_semantic": str(
                    recipient.get("phase1_target_semantic") or ""
                ),
                "donor": donor,
                "recipient": recipient,
            }
        )
    return matches


def _cap_pairs(
    matches: list[dict[str, Any]],
    *,
    maximum: int,
    seed: int,
    patch_type: str,
) -> list[dict[str, Any]]:
    if int(maximum) <= 0:
        return matches
    pair_ids = sorted(
        {str(row["pair_id"]) for row in matches},
        key=lambda pair_id: sha1_hex(
            f"{seed}:confirmation-patch-cap:{patch_type}:{pair_id}"
        ),
    )[: int(maximum)]
    allowed = set(pair_ids)
    return [row for row in matches if str(row["pair_id"]) in allowed]


def _select_control(
    current: dict[str, Any],
    pool: list[dict[str, Any]],
    *,
    kind: str,
    seed: int,
    patch_type: str,
) -> tuple[dict[str, Any] | None, str]:
    different = [
        candidate
        for candidate in pool
        if str(candidate["pair_id"]) != str(current["pair_id"])
    ]
    same_order = [
        candidate
        for candidate in different
        if candidate["presentation_order"] == current["presentation_order"]
    ]
    if kind == "shuffled_same_target":
        candidates = [
            candidate
            for candidate in same_order
            if candidate["target_displayed"] == current["target_displayed"]
            and candidate["transition_type"] == current["transition_type"]
        ]
        quality = "same_order_target_transition"
        if not candidates:
            candidates = [
                candidate
                for candidate in different
                if candidate["target_displayed"]
                == current["target_displayed"]
            ]
            quality = "same_target"
    elif kind == "shuffled_opposite_target":
        candidates = [
            candidate
            for candidate in same_order
            if candidate["target_displayed"] in {"A", "B"}
            and candidate["target_displayed"]
            != current["target_displayed"]
            and candidate["transition_type"] == current["transition_type"]
        ]
        quality = "same_order_transition_opposite_target"
        if not candidates:
            candidates = [
                candidate
                for candidate in different
                if candidate["target_displayed"] in {"A", "B"}
                and candidate["target_displayed"]
                != current["target_displayed"]
            ]
            quality = "opposite_target"
    elif kind == "same_target_donor":
        candidates = [
            candidate
            for candidate in same_order
            if candidate["transition_type"] == "same_target"
            and candidate["target_displayed"] == current["target_displayed"]
        ]
        quality = "same_order_same_target_transition"
        if not candidates:
            candidates = [
                candidate
                for candidate in different
                if candidate["transition_type"] == "same_target"
            ]
            quality = "same_target_transition"
    else:
        raise ValueError(f"Unknown control kind: {kind}")
    if not candidates:
        return None, "unavailable"
    ordered = sorted(
        candidates,
        key=lambda candidate: sha1_hex(
            f"{seed}:{patch_type}:{kind}:{current['pair_id']}:"
            f"{current['presentation_order']}:{candidate['pair_id']}:"
            f"{candidate['presentation_order']}"
        ),
    )
    return ordered[0], quality


def _forced_readout(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    stage: str,
    budget_tokens: int,
) -> str:
    response_ids = [
        int(value) for value in row[f"{stage}_response_token_ids"]
    ]
    prefix = tokenizer.decode(
        response_ids[: min(int(budget_tokens), len(response_ids))],
        skip_special_tokens=False,
    )
    return _forced_prompt(
        str(row[f"{stage}_prompt_text"]),
        prefix,
        thinking=True,
    )


def _score_readout(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    label_ids: tuple[int, ...],
    hidden_layer: int,
    vector: np.ndarray | None,
    alpha: float,
    max_length: int,
) -> np.ndarray:
    import torch

    device = next(
        parameter
        for parameter in model.parameters()
        if parameter.device.type != "meta"
    ).device
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    handles: list[Any] = []
    if vector is not None and abs(float(alpha)) > 0:
        module = find_decoder_layer_module(
            model,
            hidden_layer=int(hidden_layer),
        )
        hook = OneShotLastTokenSteeringHook(
            vector,
            alpha=float(alpha),
        )
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            logits = model(**encoded, return_dict=True).logits.float()
    finally:
        remove_hooks(handles)
    length = int(encoded["attention_mask"][0].sum().item())
    selected = logits[0, length - 1, list(label_ids)]
    return selected.detach().cpu().numpy().astype(float)


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=float) - float(np.max(logits))
    values = np.exp(shifted)
    return values / float(values.sum())


def _settings(
    current: dict[str, Any],
    pool: list[dict[str, Any]],
    *,
    alphas: list[float],
    seed: int,
    patch_type: str,
) -> list[dict[str, Any]]:
    settings = [
        {
            "setting": "baseline",
            "alpha": 0.0,
            "source": current,
            "sign": 0.0,
            "control_match_quality": "exact_recipient",
        }
    ]
    controls = {}
    for kind in (
        "shuffled_same_target",
        "shuffled_opposite_target",
        "same_target_donor",
    ):
        controls[kind] = _select_control(
            current,
            pool,
            kind=kind,
            seed=int(seed),
            patch_type=patch_type,
        )
    for alpha in alphas:
        settings.extend(
            [
                {
                    "setting": "matched_delta",
                    "alpha": float(alpha),
                    "source": current,
                    "sign": 1.0,
                    "control_match_quality": "exact_pair",
                },
                {
                    "setting": "matched_negative",
                    "alpha": float(alpha),
                    "source": current,
                    "sign": -1.0,
                    "control_match_quality": "exact_pair",
                },
                {
                    "setting": "random_orthogonal",
                    "alpha": float(alpha),
                    "source": current,
                    "sign": 1.0,
                    "control_match_quality": "norm_matched",
                },
            ]
        )
        for kind, (source, quality) in controls.items():
            if source is None:
                continue
            settings.append(
                {
                    "setting": kind,
                    "alpha": float(alpha),
                    "source": source,
                    "sign": 1.0,
                    "control_match_quality": quality,
                }
            )
    return settings


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in values:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def summarize_patch_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    order_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["patch_type", "setting", "alpha", "pair_id"],
        sort=True,
    ):
        choices = [
            value
            for value in group["predicted_semantic"].astype(str)
            if value in {"A", "B", "C"}
        ]
        order_rows.append(
            {
                "patch_type": keys[0],
                "setting": keys[1],
                "alpha": float(keys[2]),
                "pair_id": keys[3],
                "order_complete": len(choices) >= 2,
                "order_consistent": (
                    len(choices) >= 2 and len(set(choices)) == 1
                ),
            }
        )
    order_frame = pd.DataFrame(order_rows)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["patch_type", "setting", "alpha"],
        sort=True,
    ):
        order = order_frame[
            order_frame["patch_type"].eq(keys[0])
            & order_frame["setting"].eq(keys[1])
            & order_frame["alpha"].eq(float(keys[2]))
        ]
        complete = order[order["order_complete"]]
        rows.append(
            {
                "patch_type": keys[0],
                "setting": keys[1],
                "alpha": float(keys[2]),
                "n_rows": int(len(group)),
                "n_pairs": int(group["pair_id"].nunique()),
                "target_adoption_rate": float(
                    group["target_selected"].mean()
                ),
                "mean_target_probability": float(
                    group["target_probability"].mean()
                ),
                "mean_target_logit_margin": float(
                    group["target_logit_margin"].mean()
                ),
                "mean_choice_confidence": float(
                    group["choice_confidence"].mean()
                ),
                "order_consistent_rate": (
                    float(complete["order_consistent"].mean())
                    if not complete.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_mean(
    values: pd.Series,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    point = float(np.mean(array))
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


def patch_effects(
    frame: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    exact_keys = [
        "patch_type",
        "pair_id",
        "presentation_order",
        "branch_index",
    ]
    for patch_type, patch_frame in frame.groupby(
        "patch_type",
        sort=True,
    ):
        baseline = patch_frame[
            patch_frame["setting"].eq("baseline")
        ][exact_keys + list(PATCH_METRICS)].copy()
        for alpha in sorted(
            patch_frame.loc[
                ~patch_frame["setting"].eq("baseline"), "alpha"
            ].unique()
        ):
            selected = patch_frame[patch_frame["alpha"].eq(alpha)]
            settings = sorted(
                set(selected["setting"]) - {"baseline"}
            )
            contrasts = [
                (setting, "baseline") for setting in settings
            ]
            contrasts.extend(
                ("matched_delta", setting)
                for setting in settings
                if setting != "matched_delta"
            )
            for left_setting, right_setting in contrasts:
                left = selected[
                    selected["setting"].eq(left_setting)
                ][exact_keys + list(PATCH_METRICS)]
                if left.empty:
                    continue
                if right_setting == "baseline":
                    right = baseline
                else:
                    right = selected[
                        selected["setting"].eq(right_setting)
                    ][exact_keys + list(PATCH_METRICS)]
                if right.empty:
                    continue
                merged = left.merge(
                    right,
                    on=exact_keys,
                    suffixes=("_left", "_right"),
                )
                if merged.empty:
                    continue
                for metric in PATCH_METRICS:
                    merged["difference"] = (
                        merged[f"{metric}_left"].astype(float)
                        - merged[f"{metric}_right"].astype(float)
                    )
                    pair_values = merged.groupby(
                        "pair_id", sort=True
                    )["difference"].mean()
                    point, low, high = _bootstrap_mean(
                        pair_values,
                        samples=int(bootstrap),
                        seed=int(
                            sha1_hex(
                                f"{seed}:{patch_type}:{alpha}:"
                                f"{left_setting}:{right_setting}:{metric}"
                            )[:8],
                            16,
                        ),
                    )
                    rows.append(
                        {
                            "patch_type": patch_type,
                            "left_setting": left_setting,
                            "right_setting": right_setting,
                            "alpha": float(alpha),
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
    behavior_dir = _resolve(workspace_root, args.behavior_dir)
    trace_dir = _resolve(workspace_root, args.trace_dir)
    out_dir = _resolve(workspace_root, args.out_dir)
    rows = read_jsonl(behavior_dir / "switch_traces.jsonl")
    if not rows:
        raise ValueError(f"No behavior traces found in {behavior_dir}")
    patch_types = set(_csv(args.include_patch_types))
    include_orders = set(_csv(args.include_orders))
    alphas = _floats(args.alphas)
    if not alphas:
        raise ValueError("At least one patch alpha is required.")
    state_index = StateIndex(trace_dir)
    matched: dict[str, list[dict[str, Any]]] = {}
    specs = [
        spec
        for spec in PATCH_SPECS
        if str(spec["patch_type"]) in patch_types
    ]
    for spec in specs:
        pool = matched_condition_rows(
            rows,
            donor_condition=str(spec["donor_condition"]),
            recipient_condition=str(spec["recipient_condition"]),
            branch_index=int(args.branch_index),
            include_orders=include_orders,
        )
        for match in pool:
            donor_state = state_index.state(
                str(match["donor"]["trace_id"]),
                hidden_layer=int(spec["hidden_layer"]),
                point_name=str(spec["point_name"]),
            )
            recipient_state = state_index.state(
                str(match["recipient"]["trace_id"]),
                hidden_layer=int(spec["hidden_layer"]),
                point_name=str(spec["point_name"]),
            )
            match["delta"] = donor_state - recipient_state
        matched[str(spec["patch_type"])] = pool

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "patch_rows.jsonl"
    if rows_path.exists() and not bool(args.resume):
        raise FileExistsError(
            f"{rows_path} exists; use --resume or a new output directory."
        )
    existing = read_jsonl(rows_path) if rows_path.exists() else []
    completed = {
        (
            str(row["patch_type"]),
            str(row["recipient_trace_id"]),
            str(row["setting"]),
            float(row["alpha"]),
        )
        for row in existing
    }

    model, tokenizer = _load_lm(args)
    labels = _csv(args.labels)
    label_ids = _single_token_label_ids(tokenizer, labels)
    label_index = {label: index for index, label in enumerate(labels)}
    new_rows: list[dict[str, Any]] = []
    for spec in specs:
        patch_type = str(spec["patch_type"])
        pool = matched[patch_type]
        recipients = [
            match
            for match in pool
            if str(match["transition_type"])
            in {"choice_to_choice", "tie_to_choice"}
        ]
        recipients = _cap_pairs(
            recipients,
            maximum=int(args.max_pairs_per_patch),
            seed=int(args.seed),
            patch_type=patch_type,
        )
        for match in recipients:
            recipient = match["recipient"]
            prompt = _forced_readout(
                recipient,
                tokenizer=tokenizer,
                stage=str(spec["stage"]),
                budget_tokens=int(spec["budget_tokens"]),
            )
            for setting in _settings(
                match,
                pool,
                alphas=alphas,
                seed=int(args.seed),
                patch_type=patch_type,
            ):
                completion_key = (
                    patch_type,
                    str(recipient["trace_id"]),
                    str(setting["setting"]),
                    float(setting["alpha"]),
                )
                if completion_key in completed:
                    continue
                source = setting["source"]
                vector: np.ndarray | None = None
                if setting["setting"] != "baseline":
                    vector = (
                        np.asarray(source["delta"], dtype=np.float32)
                        * float(setting["sign"])
                    )
                    if setting["setting"] == "random_orthogonal":
                        vector = random_orthogonal_direction(
                            vector,
                            seed=int(
                                sha1_hex(
                                    f"{args.seed}:{patch_type}:random:"
                                    f"{match['pair_id']}:"
                                    f"{match['presentation_order']}"
                                )[:8],
                                16,
                            ),
                        ) * float(np.linalg.norm(vector))
                logits = _score_readout(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    label_ids=label_ids,
                    hidden_layer=int(spec["hidden_layer"]),
                    vector=vector,
                    alpha=float(setting["alpha"]),
                    max_length=int(args.max_score_length),
                )
                probabilities = _probabilities(logits)
                predicted = labels[int(np.argmax(logits))]
                predicted_semantic = _semantic_verdict(
                    predicted,
                    str(match["presentation_order"]),
                )
                target = str(match["target_displayed"])
                target_position = label_index[target]
                alternatives = np.delete(logits, target_position)
                sorted_probabilities = np.sort(probabilities)[::-1]
                output = {
                    "patch_id": sha1_hex(
                        f"{patch_type}:{recipient['trace_id']}:"
                        f"{setting['setting']}:{setting['alpha']}"
                    ),
                    "patch_type": patch_type,
                    "pair_id": str(match["pair_id"]),
                    "transition_type": str(match["transition_type"]),
                    "presentation_order": str(
                        match["presentation_order"]
                    ),
                    "branch_index": int(match["branch_index"]),
                    "recipient_condition": str(
                        spec["recipient_condition"]
                    ),
                    "donor_condition": str(spec["donor_condition"]),
                    "recipient_trace_id": str(recipient["trace_id"]),
                    "matched_donor_trace_id": str(
                        match["donor"]["trace_id"]
                    ),
                    "control_donor_trace_id": (
                        str(source["donor"]["trace_id"])
                        if setting["setting"] != "baseline"
                        else ""
                    ),
                    "control_donor_pair_id": (
                        str(source["pair_id"])
                        if setting["setting"] != "baseline"
                        else ""
                    ),
                    "control_match_quality": str(
                        setting["control_match_quality"]
                    ),
                    "setting": str(setting["setting"]),
                    "alpha": float(setting["alpha"]),
                    "hidden_layer": int(spec["hidden_layer"]),
                    "point_name": str(spec["point_name"]),
                    "stage": str(spec["stage"]),
                    "budget_tokens": int(spec["budget_tokens"]),
                    "delta_norm": (
                        float(np.linalg.norm(vector))
                        if vector is not None
                        else 0.0
                    ),
                    "target_displayed": target,
                    "target_semantic": str(match["target_semantic"]),
                    "initial_target_displayed": str(
                        match["initial_target_displayed"]
                    ),
                    "initial_target_semantic": str(
                        match["initial_target_semantic"]
                    ),
                    "predicted_displayed": predicted,
                    "predicted_semantic": predicted_semantic,
                    "target_selected": bool(predicted == target),
                    "target_probability": float(
                        probabilities[target_position]
                    ),
                    "target_logit_margin": float(
                        logits[target_position] - float(np.max(alternatives))
                    ),
                    "choice_confidence": float(
                        sorted_probabilities[0]
                        - sorted_probabilities[1]
                    ),
                }
                for index, label in enumerate(labels):
                    output[f"logit_{label}"] = float(logits[index])
                    output[f"prob_{label}"] = float(probabilities[index])
                _append_jsonl(rows_path, [output])
                new_rows.append(output)
                completed.add(completion_key)

    all_rows = read_jsonl(rows_path)
    frame = pd.DataFrame(all_rows)
    summary = summarize_patch_rows(frame)
    effects = patch_effects(
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
            "stage": "judge-criterion-confirmation-patching",
            "behavior_dir": str(behavior_dir),
            "trace_dir": str(trace_dir),
            "out_dir": str(out_dir),
            "model_id": str(args.model_id),
            "patch_specs": list(specs),
            "alphas": alphas,
            "include_orders": sorted(include_orders),
            "branch_index": int(args.branch_index),
            "max_pairs_per_patch": int(args.max_pairs_per_patch),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "n_patch_rows": int(len(frame)),
            "n_new_patch_rows": int(len(new_rows)),
            "outputs": {
                "patch_rows_jsonl": str(rows_path),
                "patch_rows_csv": str(out_dir / "patch_rows.csv"),
                "patch_summary_csv": str(
                    out_dir / "patch_summary.csv"
                ),
                "patch_effects_csv": str(
                    out_dir / "patch_effects.csv"
                ),
            },
        },
    )
    print(f"out_dir={out_dir}")
    print(f"n_patch_rows={len(frame)}")
    print(summary.to_string(index=False))
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
