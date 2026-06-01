"""Localize and suppress LM judge surface-cue pathways with causal patching."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aisafety.config import DEFAULT_CACHE_DIR, DEFAULT_SEED, PROJECT_ROOT
from aisafety.mech.d4_io import read_jsonl, resolve_path, sha1_hex, write_json
from aisafety.mech.decision_patching import (
    AttentionHeadDecisionPatchPreHook,
    DecoderOutputPatchHook,
    DecoderOutputSuppressionHook,
    MlpDecisionPatchHook,
    deterministic_fit_mask,
    fit_low_rank_basis,
    normalized_recovery,
)
from aisafety.mech.interventions import find_decoder_layer_module, remove_hooks
from aisafety.mech.labels import parse_int_list, select_hidden_layers
from aisafety.scripts.run_d4_bt_stage_contrast import (
    _comparison_user_content,
    _csv_list,
    _load_lm,
)


DEFAULT_OUT_DIR = Path("artifacts") / "mechanistic" / "d4_lm_judge_decision_patching_v1"


@dataclass
class PromptRecord:
    dataset: str
    row: dict[str, Any]
    observed_prompt: str
    neutral_prompt: str
    observed_cue_span: tuple[int, int]
    neutral_cue_span: tuple[int, int]

    @property
    def counterfactual_id(self) -> str:
        return str(self.row.get("counterfactual_id") or "")

    @property
    def bt_pair_id(self) -> str:
        return str(self.row.get("bt_pair_id") or "")

    @property
    def cue_plus_option(self) -> str:
        return str(self.row.get("cue_plus_option") or "")


@dataclass
class StateCache:
    records: list[PromptRecord]
    observed_margin: np.ndarray
    neutral_margin: np.ndarray
    observed_decision: dict[int, np.ndarray]
    neutral_decision: dict[int, np.ndarray]
    observed_span: dict[int, np.ndarray]
    neutral_span: dict[int, np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--fit", required=True, help="Fit probe as NAME=BT_PAIRS_JSONL.")
    parser.add_argument("--eval", action="append", default=[], help="Evaluation probe as NAME=BT_PAIRS_JSONL.")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--prompt-style", choices=["plain", "chat_template"], default="plain")
    parser.add_argument("--comparison-template", default="standard")
    parser.add_argument("--labels", default="A,B")
    parser.add_argument("--max-fit-counterfactuals", type=int, default=0)
    parser.add_argument("--max-eval-counterfactuals", type=int, default=300)
    parser.add_argument("--selected-layers", default="")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--tail-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--fit-frac", type=float, default=0.5)
    parser.add_argument("--subspace-rank", type=int, default=3)
    parser.add_argument("--suppression-alpha", type=float, default=1.0)
    parser.add_argument("--component-max-counterfactuals", type=int, default=32)
    parser.add_argument("--component-verify-top-k", type=int, default=12)
    parser.add_argument("--skip-component-scout", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _resolve(workspace_root: Path, path: Path) -> Path:
    resolved = resolve_path(workspace_root, path)
    if resolved is None:
        raise ValueError(f"Could not resolve path: {path}")
    return resolved


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in str(value):
        raise ValueError(f"Expected NAME=PATH, got: {value}")
    name, path = str(value).split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"Expected NAME=PATH, got: {value}")
    return name.strip(), Path(path.strip())


def _reuse_fit_probe_for_eval(dataset: str, fit_name: str) -> bool:
    """Evaluate the fit probe on its own rows so fit/held-out labels are valid."""

    return str(dataset) == str(fit_name)


def _cap_bt_rows(rows: list[dict[str, Any]], *, max_counterfactuals: int, seed: int, salt: str) -> list[dict[str, Any]]:
    rows = [row for row in rows if str(row.get("counterfactual_id") or "") and str(row.get("bt_pair_id") or "")]
    counterfactual_ids = sorted(
        {str(row["counterfactual_id"]) for row in rows},
        key=lambda item: sha1_hex(f"{seed}:{salt}:{item}"),
    )
    if int(max_counterfactuals) > 0:
        counterfactual_ids = counterfactual_ids[: int(max_counterfactuals)]
    keep = set(counterfactual_ids)
    return [row for row in rows if str(row["counterfactual_id"]) in keep]


def _neutral_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    cue_plus = str(row.get("cue_plus_option") or "")
    cue_minus_text = str(row.get("cue_minus_text") or "")
    if cue_plus == "A":
        out["option_a_text"] = cue_minus_text
    elif cue_plus == "B":
        out["option_b_text"] = cue_minus_text
    else:
        raise ValueError(f"Expected cue_plus_option A or B, got {cue_plus!r}")
    return out


def _render_prompt(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    prompt_style: str,
    comparison_template: str,
) -> tuple[str, dict[str, tuple[int, int]]]:
    option_a = str(row.get("option_a_text") or "").strip()
    option_b = str(row.get("option_b_text") or "").strip()
    marker_a = f"__D4_OPTION_A_{sha1_hex(option_a)[:12]}__"
    marker_b = f"__D4_OPTION_B_{sha1_hex(option_b)[:12]}__"
    placeholder_row = dict(row)
    placeholder_row["option_a_text"] = marker_a
    placeholder_row["option_b_text"] = marker_b
    content = _comparison_user_content(placeholder_row, comparison_template=comparison_template)
    if str(prompt_style) == "chat_template":
        if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
            raise ValueError("Tokenizer has no chat template; use --prompt-style plain.")
        prompt = str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    else:
        prompt = content + "\nAnswer: "
    spans: dict[str, tuple[int, int]] = {}
    for option, marker, text in (("A", marker_a, option_a), ("B", marker_b, option_b)):
        start = prompt.find(marker)
        if start < 0:
            raise ValueError(f"Could not locate option {option} placeholder inside judge prompt.")
        stop = start + len(text)
        prompt = prompt.replace(marker, text, 1)
        spans[option] = (start, stop)
    return prompt, spans


def _prompt_record(
    dataset: str,
    row: dict[str, Any],
    tokenizer: Any,
    *,
    prompt_style: str,
    comparison_template: str,
) -> PromptRecord:
    observed_prompt, observed_spans = _render_prompt(
        row,
        tokenizer,
        prompt_style=prompt_style,
        comparison_template=comparison_template,
    )
    neutral_prompt, neutral_spans = _render_prompt(
        _neutral_row(row),
        tokenizer,
        prompt_style=prompt_style,
        comparison_template=comparison_template,
    )
    cue_plus = str(row.get("cue_plus_option") or "")
    return PromptRecord(
        dataset=str(dataset),
        row=row,
        observed_prompt=observed_prompt,
        neutral_prompt=neutral_prompt,
        observed_cue_span=observed_spans[cue_plus],
        neutral_cue_span=neutral_spans[cue_plus],
    )


def _label_token_ids(tokenizer: Any, labels: list[str]) -> tuple[int, int]:
    if len(labels) != 2:
        raise ValueError("--labels must contain exactly two labels.")
    encoded = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in labels]
    if any(len(ids) != 1 for ids in encoded):
        raise ValueError(f"Decision patching currently requires single-token labels, got: {encoded}")
    return int(encoded[0][0]), int(encoded[1][0])


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
        raise ValueError("No selected layers remain after model-layer filtering.")
    return layers


def _token_positions(offsets: Any, span: tuple[int, int]) -> list[int]:
    start, stop = span
    positions: list[int] = []
    for index, raw in enumerate(offsets):
        token_start, token_stop = map(int, raw)
        if token_stop > int(start) and token_start < int(stop):
            positions.append(int(index))
    return positions


def _encoded_batch(
    tokenizer: Any,
    records: list[PromptRecord],
    *,
    prompt_kind: str,
    max_length: int,
    device: Any,
) -> tuple[dict[str, Any], Any, list[list[int]]]:
    import torch

    prompts = [getattr(record, f"{prompt_kind}_prompt") for record in records]
    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in prompts
    ]
    if max(prompt_lengths, default=0) > int(max_length):
        raise ValueError(
            f"Judge prompt exceeds --max-length: longest={max(prompt_lengths)} max_length={int(max_length)}. "
            "Increase MAX_LENGTH so causal patches are not evaluated on truncated comparisons."
        )
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    tensor_inputs = {key: value.to(device) for key, value in encoded.items()}
    positions = tensor_inputs["attention_mask"].sum(dim=1).to(dtype=torch.long) - 1
    span_positions: list[list[int]] = []
    for row_index, record in enumerate(records):
        span = getattr(record, f"{prompt_kind}_cue_span")
        selected = _token_positions(offsets[row_index].tolist(), span)
        if not selected:
            raise ValueError(
                f"Cue span vanished after tokenization/truncation: dataset={record.dataset} "
                f"counterfactual_id={record.counterfactual_id}"
            )
        span_positions.append(selected)
    return tensor_inputs, positions, span_positions


def _margin_from_logits(logits: Any, positions: Any, records: list[PromptRecord], label_ids: tuple[int, int]) -> np.ndarray:
    import torch

    row_idx = torch.arange(len(records), device=logits.device)
    a = logits[row_idx, positions, int(label_ids[0])].float()
    b = logits[row_idx, positions, int(label_ids[1])].float()
    raw = a - b
    orient = torch.tensor(
        [1.0 if record.cue_plus_option == "A" else -1.0 for record in records],
        device=logits.device,
    )
    return (raw * orient).detach().cpu().numpy().astype(float)


def _collect_prompt_states(
    *,
    model: Any,
    tokenizer: Any,
    records: list[PromptRecord],
    prompt_kind: str,
    selected_layers: list[int],
    label_ids: tuple[int, int],
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    import torch

    device = next(param for param in model.parameters() if param.device.type != "meta").device
    margins: list[np.ndarray] = []
    decision: dict[int, list[np.ndarray]] = {layer: [] for layer in selected_layers}
    spans: dict[int, list[np.ndarray]] = {layer: [] for layer in selected_layers}
    for start in range(0, len(records), max(int(batch_size), 1)):
        batch = records[start : start + max(int(batch_size), 1)]
        encoded, positions, span_positions = _encoded_batch(
            tokenizer,
            batch,
            prompt_kind=prompt_kind,
            max_length=int(max_length),
            device=device,
        )
        with torch.inference_mode():
            result = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
        margins.append(_margin_from_logits(result.logits, positions, batch, label_ids))
        row_idx = torch.arange(len(batch), device=device)
        for layer in selected_layers:
            hidden = result.hidden_states[int(layer)].float()
            decision[int(layer)].append(hidden[row_idx, positions].detach().cpu().numpy().astype(np.float32))
            pooled = torch.stack(
                [hidden[index, token_positions].mean(dim=0) for index, token_positions in enumerate(span_positions)],
                dim=0,
            )
            spans[int(layer)].append(pooled.detach().cpu().numpy().astype(np.float32))
        del encoded, positions, result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (
        np.concatenate(margins, axis=0),
        {layer: np.concatenate(chunks, axis=0) for layer, chunks in decision.items()},
        {layer: np.concatenate(chunks, axis=0) for layer, chunks in spans.items()},
    )


def _collect_states(
    *,
    model: Any,
    tokenizer: Any,
    records: list[PromptRecord],
    selected_layers: list[int],
    label_ids: tuple[int, int],
    batch_size: int,
    max_length: int,
) -> StateCache:
    observed_margin, observed_decision, observed_span = _collect_prompt_states(
        model=model,
        tokenizer=tokenizer,
        records=records,
        prompt_kind="observed",
        selected_layers=selected_layers,
        label_ids=label_ids,
        batch_size=batch_size,
        max_length=max_length,
    )
    neutral_margin, neutral_decision, neutral_span = _collect_prompt_states(
        model=model,
        tokenizer=tokenizer,
        records=records,
        prompt_kind="neutral",
        selected_layers=selected_layers,
        label_ids=label_ids,
        batch_size=batch_size,
        max_length=max_length,
    )
    return StateCache(
        records=records,
        observed_margin=observed_margin,
        neutral_margin=neutral_margin,
        observed_decision=observed_decision,
        neutral_decision=neutral_decision,
        observed_span=observed_span,
        neutral_span=neutral_span,
    )


def _score_decoder_edit(
    *,
    model: Any,
    tokenizer: Any,
    records: list[PromptRecord],
    layer_module: Any,
    prompt_kind: str,
    label_ids: tuple[int, int],
    replacements: np.ndarray | None,
    span_patch: bool,
    basis_rows: np.ndarray | None,
    center: np.ndarray | None,
    alpha: float,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch

    device = next(param for param in model.parameters() if param.device.type != "meta").device
    margins: list[np.ndarray] = []
    for start in range(0, len(records), max(int(batch_size), 1)):
        batch = records[start : start + max(int(batch_size), 1)]
        encoded, positions, span_positions = _encoded_batch(
            tokenizer,
            batch,
            prompt_kind=prompt_kind,
            max_length=int(max_length),
            device=device,
        )
        if basis_rows is not None and center is not None:
            hook = DecoderOutputSuppressionHook(
                positions=positions,
                basis_rows=torch.as_tensor(basis_rows),
                center=torch.as_tensor(center),
                alpha=float(alpha),
            )
        elif replacements is not None:
            hook = DecoderOutputPatchHook(
                positions=positions,
                replacements=torch.as_tensor(replacements[start : start + len(batch)]),
                span_positions=span_positions if span_patch else None,
            )
        else:
            raise ValueError("Expected replacements or suppression basis.")
        handles = [layer_module.register_forward_hook(hook)]
        try:
            with torch.inference_mode():
                result = model(**encoded, return_dict=True, use_cache=False)
        finally:
            remove_hooks(handles)
        margins.append(_margin_from_logits(result.logits, positions, batch, label_ids))
        del encoded, positions, result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(margins, axis=0)


def _detail_metadata(records: list[PromptRecord]) -> pd.DataFrame:
    columns = (
        "bt_pair_id",
        "counterfactual_id",
        "pair_id",
        "source_dataset",
        "subset",
        "item_type",
        "role",
        "axis",
        "direction",
        "transform_id",
        "presentation_order",
        "cue_plus_option",
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {key: str(record.row.get(key) or "") for key in columns}
        row["dataset"] = record.dataset
        rows.append(row)
    return pd.DataFrame(rows)


def _counterfactual_summary(detail: pd.DataFrame, *, value_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    order_level = detail.groupby([*group_cols, "counterfactual_id"], sort=True)[value_cols].mean().reset_index()
    rows: list[dict[str, Any]] = []
    for key, group in order_level.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key, strict=True)}
        row["n_counterfactuals"] = int(len(group))
        for col in value_cols:
            row[f"mean_{col}"] = float(pd.to_numeric(group[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _heldout_component_recovery(
    patched: list[float],
    observed: list[float],
    neutral: list[float],
    verify_indices: list[int],
) -> np.ndarray:
    """Compute verified-component recovery on the held-out scout subset only."""

    return normalized_recovery(
        np.asarray(patched),
        np.asarray([observed[index] for index in verify_indices]),
        np.asarray([neutral[index] for index in verify_indices]),
    )


def _run_residual_and_suppression(
    *,
    model: Any,
    tokenizer: Any,
    label_ids: tuple[int, int],
    selected_layers: list[int],
    fit_cache: StateCache,
    eval_cache: dict[str, StateCache],
    fit_frac: float,
    rank: int,
    alpha: float,
    batch_size: int,
    max_length: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    fit_mask = deterministic_fit_mask(
        [record.counterfactual_id for record in fit_cache.records],
        fit_frac=float(fit_frac),
        seed=int(seed),
    )
    fit_ids = {
        record.counterfactual_id
        for record, is_fit in zip(fit_cache.records, fit_mask, strict=True)
        if bool(is_fit)
    }
    fit_dataset = str(fit_cache.records[0].dataset)
    patch_frames: list[pd.DataFrame] = []
    suppress_frames: list[pd.DataFrame] = []
    bases: dict[str, np.ndarray] = {}
    for layer in selected_layers:
        layer_module = find_decoder_layer_module(model, hidden_layer=int(layer))
        fit_deltas = fit_cache.observed_decision[int(layer)][fit_mask] - fit_cache.neutral_decision[int(layer)][fit_mask]
        basis = fit_low_rank_basis(fit_deltas, rank=int(rank))
        center = fit_cache.neutral_decision[int(layer)][fit_mask].mean(axis=0).astype(np.float32)
        bases[f"hidden_{int(layer)}_basis"] = basis
        bases[f"hidden_{int(layer)}_neutral_center"] = center
        for dataset, cache in eval_cache.items():
            metadata = _detail_metadata(cache.records)
            metadata["basis_eval_split"] = [
                ("fit" if record.counterfactual_id in fit_ids else "heldout")
                if str(dataset) == fit_dataset
                else "transfer"
                for record in cache.records
            ]
            for patch_type, replacements, span_patch in (
                ("residual_decision", cache.observed_decision[int(layer)], False),
                ("answer_span_pooled", cache.observed_span[int(layer)], True),
            ):
                patched = _score_decoder_edit(
                    model=model,
                    tokenizer=tokenizer,
                    records=cache.records,
                    layer_module=layer_module,
                    prompt_kind="neutral",
                    label_ids=label_ids,
                    replacements=replacements,
                    span_patch=span_patch,
                    basis_rows=None,
                    center=None,
                    alpha=float(alpha),
                    batch_size=int(batch_size),
                    max_length=int(max_length),
                )
                detail = metadata.copy()
                detail["hidden_layer"] = int(layer)
                detail["patch_type"] = patch_type
                detail["observed_margin"] = cache.observed_margin
                detail["neutral_margin"] = cache.neutral_margin
                detail["patched_margin"] = patched
                detail["normalized_recovery"] = normalized_recovery(patched, cache.observed_margin, cache.neutral_margin)
                patch_frames.append(detail)
            suppressed = _score_decoder_edit(
                model=model,
                tokenizer=tokenizer,
                records=cache.records,
                layer_module=layer_module,
                prompt_kind="observed",
                label_ids=label_ids,
                replacements=None,
                span_patch=False,
                basis_rows=basis,
                center=center,
                alpha=float(alpha),
                batch_size=int(batch_size),
                max_length=int(max_length),
            )
            detail = metadata.copy()
            detail["hidden_layer"] = int(layer)
            detail["subspace_rank"] = int(basis.shape[0])
            detail["suppression_alpha"] = float(alpha)
            detail["observed_margin"] = cache.observed_margin
            detail["neutral_margin"] = cache.neutral_margin
            detail["suppressed_margin"] = suppressed
            detail["attenuation_toward_neutral"] = normalized_recovery(
                suppressed,
                cache.neutral_margin,
                cache.observed_margin,
            )
            suppress_frames.append(detail)
    patch_detail = pd.concat(patch_frames, ignore_index=True) if patch_frames else pd.DataFrame()
    suppress_detail = pd.concat(suppress_frames, ignore_index=True) if suppress_frames else pd.DataFrame()
    patch_summary = _counterfactual_summary(
        patch_detail,
        value_cols=["observed_margin", "neutral_margin", "patched_margin", "normalized_recovery"],
        group_cols=["dataset", "basis_eval_split", "patch_type", "hidden_layer"],
    )
    suppress_summary = _counterfactual_summary(
        suppress_detail,
        value_cols=["observed_margin", "neutral_margin", "suppressed_margin", "attenuation_toward_neutral"],
        group_cols=["dataset", "basis_eval_split", "hidden_layer", "subspace_rank", "suppression_alpha"],
    )
    return patch_detail, patch_summary, suppress_detail, suppress_summary, bases


def _component_refs(model: Any, selected_layers: list[int]) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for layer in selected_layers:
        block = find_decoder_layer_module(model, hidden_layer=int(layer))
        if hasattr(block, "mlp"):
            refs[f"mlp:{int(layer)}"] = getattr(block, "mlp")
        attention = getattr(block, "self_attn", None)
        if attention is not None and hasattr(attention, "o_proj"):
            refs[f"attn:{int(layer)}"] = getattr(attention, "o_proj")
    return refs


def _single_encoded(tokenizer: Any, record: PromptRecord, *, prompt_kind: str, max_length: int, device: Any) -> tuple[dict[str, Any], int]:
    encoded, positions, _span_positions = _encoded_batch(
        tokenizer,
        [record],
        prompt_kind=prompt_kind,
        max_length=int(max_length),
        device=device,
    )
    return encoded, int(positions[0].item())


def _capture_component_states(
    *,
    model: Any,
    tokenizer: Any,
    record: PromptRecord,
    refs: dict[str, Any],
    label_ids: tuple[int, int],
    max_length: int,
) -> tuple[float, dict[str, np.ndarray]]:
    import torch

    device = next(param for param in model.parameters() if param.device.type != "meta").device
    encoded, position = _single_encoded(tokenizer, record, prompt_kind="observed", max_length=max_length, device=device)
    captured: dict[str, np.ndarray] = {}
    handles: list[Any] = []

    def mlp_hook(name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            captured[name] = hidden[0, position].detach().float().cpu().numpy()

        return hook

    def attn_hook(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            captured[name] = inputs[0][0, position].detach().float().cpu().numpy()

        return hook

    for name, module in refs.items():
        handles.append(module.register_forward_hook(mlp_hook(name)) if name.startswith("mlp:") else module.register_forward_pre_hook(attn_hook(name)))
    try:
        with torch.inference_mode():
            result = model(**encoded, return_dict=True, use_cache=False)
    finally:
        remove_hooks(handles)
    margin = float(_margin_from_logits(result.logits, torch.tensor([position], device=device), [record], label_ids)[0])
    return margin, captured


def _target_component_attribution(
    *,
    model: Any,
    tokenizer: Any,
    record: PromptRecord,
    refs: dict[str, Any],
    source_states: dict[str, np.ndarray],
    label_ids: tuple[int, int],
    max_length: int,
) -> tuple[float, list[dict[str, Any]]]:
    import torch

    device = next(param for param in model.parameters() if param.device.type != "meta").device
    encoded, position = _single_encoded(tokenizer, record, prompt_kind="neutral", max_length=max_length, device=device)
    input_ids = encoded.pop("input_ids")
    inputs_embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def mlp_hook(name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            hidden.retain_grad()
            captured[name] = hidden

        return hook

    def attn_hook(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            inputs[0].retain_grad()
            captured[name] = inputs[0]

        return hook

    for name, module in refs.items():
        handles.append(module.register_forward_hook(mlp_hook(name)) if name.startswith("mlp:") else module.register_forward_pre_hook(attn_hook(name)))
    try:
        result = model(inputs_embeds=inputs_embeds, **encoded, return_dict=True, use_cache=False)
        logits = result.logits[0, position]
        sign = 1.0 if record.cue_plus_option == "A" else -1.0
        score = float(sign) * (logits[int(label_ids[0])] - logits[int(label_ids[1])])
        score.backward()
    finally:
        remove_hooks(handles)
    rows: list[dict[str, Any]] = []
    num_heads = int(getattr(model.config, "num_attention_heads", 0))
    for name, tensor in captured.items():
        target = tensor[0, position].detach().float().cpu().numpy()
        gradient = tensor.grad[0, position].detach().float().cpu().numpy()
        source = source_states[name]
        if name.startswith("mlp:"):
            layer = int(name.split(":", 1)[1])
            rows.append(
                {
                    "component_type": "mlp",
                    "hidden_layer": layer,
                    "component_index": -1,
                    "attribution": float(np.dot(source - target, gradient)),
                }
            )
        else:
            layer = int(name.split(":", 1)[1])
            if num_heads <= 0 or len(target) % num_heads:
                raise ValueError("Could not split attention output-projection input into attention heads.")
            head_dim = len(target) // num_heads
            for head in range(num_heads):
                start = int(head * head_dim)
                stop = int(start + head_dim)
                rows.append(
                    {
                        "component_type": "attn_head",
                        "hidden_layer": layer,
                        "component_index": int(head),
                        "attribution": float(np.dot(source[start:stop] - target[start:stop], gradient[start:stop])),
                    }
                )
    return float(score.detach().cpu()), rows


def _score_component_patch(
    *,
    model: Any,
    tokenizer: Any,
    record: PromptRecord,
    refs: dict[str, Any],
    source_states: dict[str, np.ndarray],
    candidate: dict[str, Any],
    label_ids: tuple[int, int],
    max_length: int,
) -> float:
    import torch

    device = next(param for param in model.parameters() if param.device.type != "meta").device
    encoded, position = _single_encoded(tokenizer, record, prompt_kind="neutral", max_length=max_length, device=device)
    layer = int(candidate["hidden_layer"])
    component_type = str(candidate["component_type"])
    key = f"mlp:{layer}" if component_type == "mlp" else f"attn:{layer}"
    if component_type == "mlp":
        hook = MlpDecisionPatchHook(
            positions=torch.tensor([position]),
            replacements=torch.as_tensor(source_states[key]).reshape(1, -1),
        )
        handles = [refs[key].register_forward_hook(hook)]
    else:
        num_heads = int(getattr(model.config, "num_attention_heads", 0))
        hidden_dim = int(len(source_states[key]))
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("Could not split attention output-projection input into attention heads.")
        hook = AttentionHeadDecisionPatchPreHook(
            positions=torch.tensor([position]),
            replacements=torch.as_tensor(source_states[key]).reshape(1, -1),
            head_index=int(candidate["component_index"]),
            head_dim=int(hidden_dim // num_heads),
        )
        handles = [refs[key].register_forward_pre_hook(hook)]
    try:
        with torch.inference_mode():
            result = model(**encoded, return_dict=True, use_cache=False)
    finally:
        remove_hooks(handles)
    return float(_margin_from_logits(result.logits, torch.tensor([position], device=device), [record], label_ids)[0])


def _run_component_scout(
    *,
    model: Any,
    tokenizer: Any,
    records: list[PromptRecord],
    selected_layers: list[int],
    label_ids: tuple[int, int],
    max_counterfactuals: int,
    verify_top_k: int,
    fit_frac: float,
    max_length: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = _cap_bt_rows(
        [record.row for record in records],
        max_counterfactuals=int(max_counterfactuals),
        seed=int(seed),
        salt="component-scout",
    )
    keep = {str(row["bt_pair_id"]) for row in rows}
    selected = [record for record in records if record.bt_pair_id in keep]
    refs = _component_refs(model, selected_layers)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = _detail_metadata(selected)
    component_fit_mask = deterministic_fit_mask(
        [record.counterfactual_id for record in selected],
        fit_frac=float(fit_frac),
        seed=int(seed) + 17,
    )
    metadata["component_split"] = np.where(component_fit_mask, "fit", "heldout")
    source_by_row: list[dict[str, np.ndarray]] = []
    observed: list[float] = []
    neutral: list[float] = []
    attribution_rows: list[dict[str, Any]] = []
    for row_index, record in enumerate(selected):
        observed_margin, source_states = _capture_component_states(
            model=model,
            tokenizer=tokenizer,
            record=record,
            refs=refs,
            label_ids=label_ids,
            max_length=int(max_length),
        )
        neutral_margin, rows = _target_component_attribution(
            model=model,
            tokenizer=tokenizer,
            record=record,
            refs=refs,
            source_states=source_states,
            label_ids=label_ids,
            max_length=int(max_length),
        )
        source_by_row.append(source_states)
        observed.append(float(observed_margin))
        neutral.append(float(neutral_margin))
        for row in rows:
            attribution_rows.append(
                {
                    **metadata.iloc[row_index].to_dict(),
                    **row,
                    "observed_margin": float(observed_margin),
                    "neutral_margin": float(neutral_margin),
                }
            )
    attribution = pd.DataFrame(attribution_rows)
    shortlist = (
        attribution[attribution["component_split"] == "fit"]
        .groupby(["component_type", "hidden_layer", "component_index"], sort=True)["attribution"]
        .agg(["count", "mean", lambda series: float(series.abs().mean())])
        .reset_index()
        .rename(columns={"count": "n_bt_rows", "mean": "mean_attribution", "<lambda_0>": "mean_abs_attribution"})
        .sort_values("mean_abs_attribution", ascending=False)
        .head(max(int(verify_top_k), 1))
        .reset_index(drop=True)
    )
    verify_frames: list[pd.DataFrame] = []
    verify_indices = [index for index, is_fit in enumerate(component_fit_mask) if not bool(is_fit)]
    for candidate in shortlist.to_dict(orient="records"):
        patched = [
            _score_component_patch(
                model=model,
                tokenizer=tokenizer,
                record=record,
                refs=refs,
                source_states=source_by_row[row_index],
                candidate=candidate,
                label_ids=label_ids,
                max_length=int(max_length),
            )
            for row_index, record in enumerate(selected)
            if row_index in verify_indices
        ]
        detail = metadata.iloc[verify_indices].reset_index(drop=True).copy()
        detail["component_type"] = str(candidate["component_type"])
        detail["hidden_layer"] = int(candidate["hidden_layer"])
        detail["component_index"] = int(candidate["component_index"])
        detail["observed_margin"] = [observed[index] for index in verify_indices]
        detail["neutral_margin"] = [neutral[index] for index in verify_indices]
        detail["patched_margin"] = patched
        detail["normalized_recovery"] = _heldout_component_recovery(
            patched,
            observed,
            neutral,
            verify_indices,
        )
        verify_frames.append(detail)
    verification = pd.concat(verify_frames, ignore_index=True) if verify_frames else pd.DataFrame()
    verification_summary = _counterfactual_summary(
        verification,
        value_cols=["observed_margin", "neutral_margin", "patched_margin", "normalized_recovery"],
        group_cols=["component_type", "hidden_layer", "component_index"],
    )
    if not verification_summary.empty:
        verification_summary["aggregate_recovery"] = np.divide(
            verification_summary["mean_patched_margin"] - verification_summary["mean_neutral_margin"],
            verification_summary["mean_observed_margin"] - verification_summary["mean_neutral_margin"],
        )
        verification_summary = verification_summary.sort_values("aggregate_recovery", ascending=False)
    return attribution, shortlist, verification, verification_summary


def main() -> None:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    out_dir = _resolve(workspace_root, args.out_dir)
    fit_name, fit_path = _named_path(str(args.fit))
    fit_path = _resolve(workspace_root, fit_path)
    eval_specs = [_named_path(value) for value in args.eval] or [(fit_name, fit_path)]
    eval_specs = [(name, _resolve(workspace_root, path)) for name, path in eval_specs]

    model, tokenizer = _load_lm(args)
    labels = _csv_list(str(args.labels))
    label_ids = _label_token_ids(tokenizer, labels)
    layers = _selected_layers(
        model,
        raw=str(args.selected_layers),
        stride=int(args.layer_stride),
        tail_layers=int(args.tail_layers),
    )
    fit_rows = _cap_bt_rows(
        read_jsonl(fit_path),
        max_counterfactuals=int(args.max_fit_counterfactuals),
        seed=int(args.seed),
        salt=f"fit:{fit_name}",
    )
    fit_records = [
        _prompt_record(
            fit_name,
            row,
            tokenizer,
            prompt_style=str(args.prompt_style),
            comparison_template=str(args.comparison_template),
        )
        for row in fit_rows
    ]
    if not fit_records:
        raise ValueError(f"No fit rows selected from {fit_path}")
    eval_records: dict[str, list[PromptRecord]] = {}
    for dataset, path in eval_specs:
        rows = _cap_bt_rows(
            read_jsonl(path),
            max_counterfactuals=int(args.max_eval_counterfactuals),
            seed=int(args.seed),
            salt=f"eval:{dataset}",
        )
        eval_records[dataset] = [
            _prompt_record(
                dataset,
                row,
                tokenizer,
                prompt_style=str(args.prompt_style),
                comparison_template=str(args.comparison_template),
            )
            for row in rows
        ]
    print(f"Collecting fit states: dataset={fit_name} rows={len(fit_records)} layers={','.join(map(str, layers))}")
    fit_cache = _collect_states(
        model=model,
        tokenizer=tokenizer,
        records=fit_records,
        selected_layers=layers,
        label_ids=label_ids,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    eval_cache: dict[str, StateCache] = {}
    for dataset, records in eval_records.items():
        print(f"Collecting evaluation states: dataset={dataset} rows={len(records)}")
        if _reuse_fit_probe_for_eval(dataset, fit_name):
            eval_cache[dataset] = fit_cache
        else:
            eval_cache[dataset] = _collect_states(
                model=model,
                tokenizer=tokenizer,
                records=records,
                selected_layers=layers,
                label_ids=label_ids,
                batch_size=int(args.batch_size),
                max_length=int(args.max_length),
            )
    patch_detail, patch_summary, suppress_detail, suppress_summary, bases = _run_residual_and_suppression(
        model=model,
        tokenizer=tokenizer,
        label_ids=label_ids,
        selected_layers=layers,
        fit_cache=fit_cache,
        eval_cache=eval_cache,
        fit_frac=float(args.fit_frac),
        rank=int(args.subspace_rank),
        alpha=float(args.suppression_alpha),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        seed=int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_detail.to_csv(out_dir / "residual_patch_rows.csv", index=False)
    patch_summary.to_csv(out_dir / "residual_patch_summary.csv", index=False)
    suppress_detail.to_csv(out_dir / "subspace_suppression_rows.csv", index=False)
    suppress_summary.to_csv(out_dir / "subspace_suppression_summary.csv", index=False)
    np.savez_compressed(out_dir / "fitted_subspaces.npz", **bases)

    if not bool(args.skip_component_scout):
        print("Running decision-position component attribution and verified patching scout")
        attribution, shortlist, verification, verification_summary = _run_component_scout(
            model=model,
            tokenizer=tokenizer,
            records=fit_records,
            selected_layers=layers,
            label_ids=label_ids,
            max_counterfactuals=int(args.component_max_counterfactuals),
            verify_top_k=int(args.component_verify_top_k),
            fit_frac=float(args.fit_frac),
            max_length=int(args.max_length),
            seed=int(args.seed),
        )
        attribution.to_csv(out_dir / "component_attribution_rows.csv", index=False)
        shortlist.to_csv(out_dir / "component_shortlist.csv", index=False)
        verification.to_csv(out_dir / "component_verification_rows.csv", index=False)
        verification_summary.to_csv(out_dir / "component_verification_summary.csv", index=False)
    write_json(
        out_dir / "manifest.json",
        {
            "stage": "D4-LM-judge-decision-patching",
            "model_id": str(args.model_id),
            "run_label": str(args.run_label or args.model_id),
            "fit_probe": {"name": fit_name, "bt_pairs_jsonl": str(fit_path), "n_bt_rows": int(len(fit_records))},
            "eval_probes": {
                name: {"n_bt_rows": int(len(cache.records))}
                for name, cache in eval_cache.items()
            },
            "selected_hidden_layers": layers,
            "prompt_style": str(args.prompt_style),
            "comparison_template": str(args.comparison_template),
            "labels": labels,
            "subspace_rank": int(args.subspace_rank),
            "suppression_alpha": float(args.suppression_alpha),
            "fit_frac": float(args.fit_frac),
            "component_scout": not bool(args.skip_component_scout),
            "component_attribution_method": "first_order_activation_delta_dot_gradient_prefilter_then_verified_patch",
            "component_scope": "decision_position_mlp_outputs_and_attention_head_chunks_before_o_proj",
            "seed": int(args.seed),
        },
    )
    print(f"out_dir={out_dir}")
    print(f"layers={','.join(map(str, layers))}")


if __name__ == "__main__":
    main()
