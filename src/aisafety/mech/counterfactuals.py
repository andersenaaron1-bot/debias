"""Deterministic text counterfactuals for D4 surface-cue audits."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable


DEFAULT_SURFACE_AXES = (
    "structured_assistant_packaging",
    "formal_institutional_packaging",
    "benefit_value_framing",
)


@dataclass(frozen=True)
class CounterfactualVariant:
    """One deterministic surface-form rewrite."""

    axis: str
    direction: str
    transform_id: str
    text: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class CounterfactualSkip:
    """Reason a transform was not emitted."""

    axis: str
    direction: str
    transform_id: str
    reason: str


TransformOutput = CounterfactualVariant | CounterfactualSkip
TransformFn = Callable[[str], tuple[str, tuple[str, ...], str | None]]


_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|[a-zA-Z][.)]\s+)")
_LIST_PREFIX_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+|[a-zA-Z][.)]\s+)")
_SPACE_RE = re.compile(r"[ \t]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""

    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    collapsed: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank and collapsed:
                collapsed.append("")
            previous_blank = True
            continue
        collapsed.append(line)
        previous_blank = False
    return "\n".join(collapsed).strip()


def flat_text(text: str) -> str:
    """Return a single-line normalized representation."""

    return " ".join(normalize_text(text).split())


def token_count(text: str) -> int:
    """Cheap whitespace token count used for deterministic filters."""

    return len(flat_text(text).split())


def split_sentences(text: str) -> list[str]:
    """Split text into coarse sentence-like units."""

    text = flat_text(text)
    if not text:
        return []
    parts = [part.strip(" ;") for part in _SENTENCE_SPLIT_RE.split(text) if part.strip(" ;")]
    if len(parts) <= 1 and ";" in text:
        parts = [part.strip(" ;") for part in text.split(";") if part.strip(" ;")]
    return parts


def _has_list_structure(text: str) -> bool:
    lines = [line for line in normalize_text(text).split("\n") if line.strip()]
    return sum(1 for line in lines if _BULLET_PREFIX_RE.search(line)) >= 2


def _strip_list_markers(text: str) -> str:
    lines = []
    for line in normalize_text(text).split("\n"):
        stripped = _BULLET_PREFIX_RE.sub("", line).strip()
        if stripped:
            lines.append(stripped)
    return " ".join(lines).strip()


def _ensure_terminal(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?":
        return text + "."
    return text


def structured_increase(text: str) -> tuple[str, tuple[str, ...], str | None]:
    """Increase assistant-style structure by turning sentences into bullets."""

    if _has_list_structure(text):
        return "", (), "already_structured"
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return "", (), "too_few_segments"
    bullets = [_ensure_terminal(sentence) for sentence in sentences[:6]]
    if len(sentences) > 6:
        bullets[-1] = bullets[-1].rstrip(".!?") + "..."
    return "Key points:\n" + "\n".join(f"- {sentence}" for sentence in bullets), (
        "added_wrapper",
        "added_list_structure",
    ), None


def structured_decrease(text: str) -> tuple[str, tuple[str, ...], str | None]:
    """Decrease visible assistant-style structure by paragraphizing lists."""

    original = normalize_text(text)
    if not _has_list_structure(original) and "\n" not in original:
        return "", (), "not_structured"
    text = re.sub(r"(?im)^\s*(?:key points|summary|answer|here(?:'s| is).{0,40}):\s*$", "", original)
    text = _strip_list_markers(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, ("removed_list_structure", "paragraphized"), None


_FORMAL_DECREASE_REPLACEMENTS = (
    (re.compile(r"\bit is important to note that\b", re.I), ""),
    (re.compile(r"\bit should be noted that\b", re.I), ""),
    (re.compile(r"\bin conclusion,?\b", re.I), ""),
    (re.compile(r"\boverall,?\b", re.I), ""),
    (re.compile(r"\bfurthermore,?\b", re.I), ""),
    (re.compile(r"\bmoreover,?\b", re.I), ""),
    (re.compile(r"\btherefore,?\b", re.I), "so"),
    (re.compile(r"\baccording to\b", re.I), "from"),
)


def formal_decrease(text: str) -> tuple[str, tuple[str, ...], str | None]:
    """Remove common formal/institutional discourse markers."""

    out = normalize_text(text)
    hits = 0
    for pattern, repl in _FORMAL_DECREASE_REPLACEMENTS:
        out, n = pattern.subn(repl, out)
        hits += n
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r"\s+", " ", out).strip()
    if hits <= 0:
        return "", (), "no_formal_markers"
    return out, ("removed_formal_markers",), None


def formal_increase(text: str) -> tuple[str, tuple[str, ...], str | None]:
    """Add light formal framing without changing the substantive content."""

    text = flat_text(text)
    if not text:
        return "", (), "empty"
    if re.match(r"(?i)^(overall|in summary|it is important to note)", text):
        return "", (), "already_formal_preface"
    return "Overall, " + text[0].lower() + text[1:], ("added_formal_preface",), None


_BENEFIT_DECREASE_REPLACEMENTS = (
    (re.compile(r"\bperfect for\b", re.I), "for"),
    (re.compile(r"\bideal for\b", re.I), "for"),
    (re.compile(r"\bdesigned to help\b", re.I), "can"),
    (re.compile(r"\bhelps you\b", re.I), "can"),
    (re.compile(r"\bso you can\b", re.I), "and you can"),
    (re.compile(r"\bwith ease\b", re.I), ""),
    (re.compile(r"\bvaluable\b", re.I), "useful"),
)


def benefit_decrease(text: str) -> tuple[str, tuple[str, ...], str | None]:
    """Remove sales/value framing markers where they already occur."""

    out = normalize_text(text)
    hits = 0
    for pattern, repl in _BENEFIT_DECREASE_REPLACEMENTS:
        out, n = pattern.subn(repl, out)
        hits += n
    out = re.sub(r"\s+", " ", out).strip()
    if hits <= 0:
        return "", (), "no_benefit_markers"
    return out, ("removed_benefit_markers",), None


def benefit_increase(text: str) -> tuple[str, tuple[str, ...], str | None]:
    """Add one short value-framing sentence for exploratory audits."""

    text = normalize_text(text)
    if not text:
        return "", (), "empty"
    marker = " This can make the information easier to apply in practice."
    if marker.strip().lower() in text.lower():
        return "", (), "already_benefit_framed"
    return text.rstrip() + marker, ("added_benefit_sentence", "exploratory"), None


TRANSFORMS: dict[tuple[str, str], tuple[str, TransformFn]] = {
    ("structured_assistant_packaging", "increase"): ("structured_listify_v1", structured_increase),
    ("structured_assistant_packaging", "decrease"): ("structured_paragraphize_v1", structured_decrease),
    ("formal_institutional_packaging", "increase"): ("formal_preface_v1", formal_increase),
    ("formal_institutional_packaging", "decrease"): ("formal_marker_strip_v1", formal_decrease),
    ("benefit_value_framing", "increase"): ("benefit_sentence_v1", benefit_increase),
    ("benefit_value_framing", "decrease"): ("benefit_marker_strip_v1", benefit_decrease),
}


def build_counterfactual_variants(
    text: str,
    *,
    axes: set[str],
    min_tokens: int = 20,
    min_length_ratio: float = 0.7,
    max_length_ratio: float = 1.3,
) -> list[TransformOutput]:
    """Return all deterministic variants or skip rows for selected axes."""

    base = normalize_text(text)
    base_tokens = token_count(base)
    outputs: list[TransformOutput] = []
    for axis, direction in sorted(TRANSFORMS):
        if axis not in axes:
            continue
        transform_id, fn = TRANSFORMS[(axis, direction)]
        if base_tokens < int(min_tokens):
            outputs.append(CounterfactualSkip(axis, direction, transform_id, "base_too_short"))
            continue
        variant, flags, reason = fn(base)
        variant = normalize_text(variant)
        if reason is not None:
            outputs.append(CounterfactualSkip(axis, direction, transform_id, reason))
            continue
        if not variant:
            outputs.append(CounterfactualSkip(axis, direction, transform_id, "empty_variant"))
            continue
        if flat_text(variant) == flat_text(base):
            outputs.append(CounterfactualSkip(axis, direction, transform_id, "unchanged"))
            continue
        ratio = token_count(variant) / max(float(base_tokens), 1.0)
        if ratio < float(min_length_ratio) or ratio > float(max_length_ratio):
            outputs.append(CounterfactualSkip(axis, direction, transform_id, "length_ratio_out_of_bounds"))
            continue
        outputs.append(
            CounterfactualVariant(
                axis=axis,
                direction=direction,
                transform_id=transform_id,
                text=variant,
                flags=tuple(flags) + ("length_ok", "changed", "nonempty"),
            )
        )
    return outputs
