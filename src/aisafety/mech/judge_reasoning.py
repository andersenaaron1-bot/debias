"""Shared schemas and numerical helpers for judge-reasoning trajectory studies."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from aisafety.mech.d4_io import sha1_hex
from aisafety.mech.interventions import find_decoder_layer_module, remove_hooks


FINAL_CHOICE_RE = re.compile(
    r"(?:FINAL(?:\s+ANSWER|\s+CHOICE)?|ANSWER|CHOICE|VERDICT)\s*[:\-]\s*([AB])\b",
    flags=re.IGNORECASE,
)
BARE_CHOICE_RE = re.compile(r"^\s*([AB])\s*[.!]?\s*$", flags=re.IGNORECASE | re.MULTILINE)
FINAL_VERDICT_RE = re.compile(
    r"(?:FINAL(?:\s+ANSWER|\s+CHOICE)?|ANSWER|CHOICE|VERDICT)\s*[:\-]\s*([A-Z]+)\b",
    flags=re.IGNORECASE,
)
BARE_VERDICT_RE = re.compile(
    r"^\s*([A-Z]+)\s*[.!]?\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class JudgeComparison:
    """Canonical pairwise decision record used by every reasoning suite stage."""

    comparison_id: str
    pair_id: str
    source_dataset: str
    prompt: str
    option_a_text: str
    option_b_text: str
    comparison_dimension: str = "overall_quality"
    task_type: str = "pairwise_judgment"
    target_option: str = ""
    target_kind: str = "none"
    subset: str = ""
    split: str = ""
    presentation_order: str = ""
    condition_id: str = ""
    condition_label: str = ""
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        row = {
            "comparison_id": self.comparison_id,
            "pair_id": self.pair_id,
            "source_dataset": self.source_dataset,
            "subset": self.subset,
            "split": self.split,
            "task_type": self.task_type,
            "comparison_dimension": self.comparison_dimension,
            "prompt": self.prompt,
            "option_a_text": self.option_a_text,
            "option_b_text": self.option_b_text,
            "target_option": self.target_option,
            "target_kind": self.target_kind,
            "presentation_order": self.presentation_order,
            "condition_id": self.condition_id,
            "condition_label": self.condition_label,
        }
        if self.metadata:
            row["metadata"] = self.metadata
        return row


def normalize_choice(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"A", "B"} else ""


def normalize_verdict(value: Any, *, labels: Iterable[str]) -> str:
    allowed = {str(label).strip().upper() for label in labels if str(label).strip()}
    text = str(value or "").strip().upper()
    return text if text in allowed else ""


def opposite_choice(value: str) -> str:
    choice = normalize_choice(value)
    if choice == "A":
        return "B"
    if choice == "B":
        return "A"
    return ""


def parse_final_choice(text: str) -> str:
    """Parse an explicit final A/B verdict without reading incidental mentions."""

    matches = FINAL_CHOICE_RE.findall(str(text or ""))
    if matches:
        return str(matches[-1]).upper()
    tail = str(text or "").strip()[-80:]
    matches = BARE_CHOICE_RE.findall(tail)
    return str(matches[-1]).upper() if matches else ""


def parse_final_verdict(text: str, *, labels: Iterable[str]) -> str:
    """Parse an explicit verdict from an arbitrary finite label set."""

    allowed = [str(label).strip().upper() for label in labels if str(label).strip()]
    matches = FINAL_VERDICT_RE.findall(str(text or ""))
    for match in reversed(matches):
        verdict = normalize_verdict(match, labels=allowed)
        if verdict:
            return verdict
    tail = str(text or "").strip()[-80:]
    matches = BARE_VERDICT_RE.findall(tail)
    for match in reversed(matches):
        verdict = normalize_verdict(match, labels=allowed)
        if verdict:
            return verdict
    return ""


def comparison_prompt_content(
    row: dict[str, Any],
    *,
    reasoning_mode: str,
) -> str:
    """Render a dimension-aware judge prompt independent of model serialization."""

    dimension = str(row.get("comparison_dimension") or "overall_quality").replace("_", " ")
    prompt = str(row.get("prompt") or "Compare the two options.").strip()
    option_a = str(row.get("option_a_text") or "").strip()
    option_b = str(row.get("option_b_text") or "").strip()
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    allow_tie = bool(row.get("allow_tie") or metadata.get("allow_tie"))
    criterion_text = str(
        row.get("criterion_text")
        or metadata.get("criterion_text")
        or ""
    ).strip()
    criterion = (
        f"Decision rule:\n{criterion_text}\n\n"
        if criterion_text
        else ""
    )
    base = (
        f"Evaluate the two options on the dimension: {dimension}.\n\n"
        f"{criterion}"
        f"Context or question:\n{prompt}\n\n"
        f"Option A:\n{option_a}\n\n"
        f"Option B:\n{option_b}\n\n"
    )
    verdict_instruction = (
        "Return exactly one line: FINAL: A, FINAL: B, or FINAL: C. "
        "Use C only when the decision rule leaves the options tied or "
        "underdetermined."
        if allow_tie
        else "Return exactly one line: FINAL: A or FINAL: B."
    )
    if str(reasoning_mode) == "direct":
        return base + verdict_instruction
    return (
        base
        + "Analyze the evidence for both options before deciding. Do not infer quality from option "
        f"position. {verdict_instruction}"
    )


def render_model_prompt(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    prompt_style: str,
    reasoning_mode: str,
) -> str:
    content = comparison_prompt_content(row, reasoning_mode=reasoning_mode)
    if str(prompt_style) != "chat_template":
        return content + "\n"
    if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("Tokenizer has no chat template; use prompt_style=plain.")
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if str(reasoning_mode) in {"thinking", "direct"}:
        kwargs["enable_thinking"] = str(reasoning_mode) == "thinking"
    try:
        return str(tokenizer.apply_chat_template([{"role": "user", "content": content}], **kwargs))
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return str(tokenizer.apply_chat_template([{"role": "user", "content": content}], **kwargs))


def deterministic_group_fold(group_id: str, *, n_folds: int, seed: int, salt: str) -> int:
    folds = max(int(n_folds), 2)
    return int(sha1_hex(f"{seed}:{salt}:{group_id}")[:12], 16) % folds


def normalized_sample_indices(n_steps: int, n_points: int) -> np.ndarray:
    """Return stable unique trajectory indices including both endpoints."""

    if int(n_steps) <= 0:
        return np.zeros((0,), dtype=int)
    if int(n_steps) == 1 or int(n_points) <= 1:
        return np.asarray([0], dtype=int)
    raw = np.linspace(0, int(n_steps) - 1, num=min(int(n_points), int(n_steps)))
    return np.unique(np.rint(raw).astype(int))


def resample_trajectory(
    states: np.ndarray,
    *,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample a [steps, layers, hidden] trajectory onto normalized points."""

    array = np.asarray(states)
    if array.ndim != 3:
        raise ValueError("states must have shape [steps, layers, hidden].")
    indices = normalized_sample_indices(len(array), int(n_points))
    if not len(indices):
        return (
            np.zeros((0, array.shape[1], array.shape[2]), dtype=array.dtype),
            indices,
            np.zeros((0,), dtype=np.float32),
        )
    denominator = max(len(array) - 1, 1)
    positions = indices.astype(np.float32) / float(denominator)
    return array[indices], indices, positions


def first_persistent_threshold(
    values: Iterable[float],
    *,
    threshold: float,
    persistence: int,
) -> int | None:
    arr = np.asarray(list(values), dtype=float)
    width = max(int(persistence), 1)
    for index in range(len(arr)):
        stop = min(index + width, len(arr))
        window = arr[index:stop]
        if len(window) == width and np.isfinite(window).all() and bool(np.all(window >= float(threshold))):
            return int(index)
    return None


def direction_angle_degrees(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    angle = direction_angle_degrees(left, right)
    if angle is None:
        return None
    return float(math.cos(math.radians(angle)))


def row_metadata(row: dict[str, Any], *, exclude: set[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in row.items():
        if key in exclude:
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        metadata[str(key)] = value
    return metadata


def random_orthogonal_direction(
    direction: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Create a deterministic unit random direction orthogonal to ``direction``."""

    target = np.asarray(direction, dtype=np.float32).reshape(-1)
    target_norm = float(np.linalg.norm(target))
    if target_norm <= 1e-12:
        raise ValueError("Cannot construct an orthogonal control for a zero direction.")
    unit_target = target / target_norm
    rng = np.random.default_rng(int(seed))
    for _ in range(16):
        control = rng.standard_normal(len(unit_target)).astype(np.float32)
        control = control - float(np.dot(control, unit_target)) * unit_target
        control_norm = float(np.linalg.norm(control))
        if control_norm > 1e-12:
            return control / control_norm
    raise RuntimeError("Could not construct a nonzero random orthogonal direction.")


def trace_shard_path(root: Path, shard_name: str) -> Path:
    path = Path(root) / str(shard_name)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class LastTokenTrajectoryRecorder:
    """Record selected decoder-block outputs at the last active token."""

    def __init__(self, model: Any, *, hidden_layers: list[int]):
        self.model = model
        self.hidden_layers = [int(layer) for layer in hidden_layers]
        self._steps: dict[int, list[np.ndarray]] = {layer: [] for layer in self.hidden_layers}
        self._handles: list[Any] = []

    @staticmethod
    def _hidden(output: Any) -> Any:
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, tuple):
            return output[0]
        return output

    def _hook(self, hidden_layer: int):
        def record(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = self._hidden(output)
            if getattr(hidden, "ndim", 0) != 3:
                raise ValueError("Decoder output must have shape [batch, sequence, hidden].")
            if int(hidden.shape[0]) != 1:
                raise ValueError("Trajectory capture currently requires generation batch size 1.")
            vector = hidden[0, -1].detach().float().cpu().numpy()
            self._steps[int(hidden_layer)].append(vector.astype(np.float32, copy=False))

        return record

    def __enter__(self) -> "LastTokenTrajectoryRecorder":
        for hidden_layer in self.hidden_layers:
            module = find_decoder_layer_module(self.model, hidden_layer=int(hidden_layer))
            self._handles.append(module.register_forward_hook(self._hook(int(hidden_layer))))
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        remove_hooks(self._handles)
        self._handles = []

    def array(self) -> np.ndarray:
        lengths = {len(rows) for rows in self._steps.values()}
        if not lengths:
            return np.zeros((0, 0, 0), dtype=np.float32)
        if len(lengths) != 1:
            raise RuntimeError(f"Selected layers produced inconsistent trajectory lengths: {lengths}")
        n_steps = next(iter(lengths))
        if n_steps <= 0:
            return np.zeros((0, len(self.hidden_layers), 0), dtype=np.float32)
        return np.stack(
            [
                np.stack([self._steps[layer][step] for layer in self.hidden_layers], axis=0)
                for step in range(n_steps)
            ],
            axis=0,
        )


class OneShotLastTokenSteeringHook:
    """Add a vector to the final token on the first decoder invocation only."""

    def __init__(self, vector: Any, *, alpha: float):
        self.vector = vector
        self.alpha = float(alpha)
        self.applied = False

    def __call__(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self.applied:
            return output
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            return output
        edited = hidden.clone()
        vector = torch.as_tensor(self.vector, device=hidden.device, dtype=hidden.dtype)
        edited[:, -1, :] = edited[:, -1, :] + self.alpha * vector
        self.applied = True
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited
