"""Operational atom ontology for D2 bundle validation."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from aisafety.config import PROJECT_ROOT


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?")
ALPHA_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
PASSIVE_RE = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b", flags=re.IGNORECASE)
LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
TITLE_OPEN_RE = re.compile(r"^[A-Z][A-Za-z0-9'\"/:,-]*(?:\s+[A-Z][A-Za-z0-9'\"/:,-]*){1,7}$")
ENUM_WORD_RE = re.compile(
    r"\b(?:first|second|third|fourth|finally|lastly|next|overall|in conclusion|in summary)\b",
    flags=re.IGNORECASE,
)

NOMINALIZATION_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ism", "ance", "ence")
SUBORDINATORS = ("that", "which", "who", "because", "although", "while", "if", "when", "whereas", "since")
INSTRUCTION_VERBS = ("consider", "ensure", "check", "avoid", "remember", "review", "use", "try", "verify")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with",
}

FORMAL_CONNECTIVES = ("however", "therefore", "moreover", "furthermore", "consequently", "thus", "whereas", "nevertheless")
FRAME_MARKERS = ("overall", "in conclusion", "in summary", "to sum up", "first", "second", "finally", "in this paper", "this study")
CODE_GLOSS_MARKERS = ("for example", "for instance", "that is", "i.e.", "namely", "in other words", "such as")
EVIDENTIAL_MARKERS = ("according to", "prior work", "evidence suggests", "results show", "findings indicate", "literature suggests")
SELF_MENTION_MARKERS = ("i", "we", "our", "ours", "us", "this paper", "this study", "in our work")
ENGAGEMENT_MARKERS = ("note that", "consider", "you can", "you should", "remember", "let us", "keep in mind")
HEDGE_MARKERS = ("may", "might", "could", "likely", "appears", "appear", "seems", "seem", "possible", "possibly", "perhaps", "generally")
CERTAINTY_MARKERS = ("clearly", "definitely", "certainly", "undoubtedly", "always", "never", "must", "prove", "proves", "demonstrates", "establishes")
PROMOTIONAL_ADJECTIVES = ("innovative", "seamless", "powerful", "premium", "versatile", "compelling", "must-have", "ideal", "perfect", "high-quality", "exceptional", "stylish")
DISCLAIMER_LEXICON = ("consult a professional", "informational purposes", "not a substitute", "for safety", "ensure that", "make sure", "do not", "should not", "policy", "guidelines", "risk")
REPORTING_VERBS = ("demonstrate", "demonstrates", "suggest", "suggests", "indicate", "indicates", "show", "shows", "report", "reports", "argue", "argues", "examine", "examines", "evaluate", "evaluates", "propose", "proposes", "present", "presents")
TECHNICAL_ABSTRACT_NOUNS = ("framework", "methodology", "evaluation", "experiment", "dataset", "analysis", "model", "approach", "mechanism", "distribution", "evidence", "results", "objective")
PROBLEM_CUES = ("problem", "issue", "challenge", "risk", "concern", "difficulty")
SOLUTION_CUES = ("solution", "recommend", "recommended", "propose", "address", "resolve", "mitigate", "improve")
FEATURE_CUES = ("feature", "features", "includes", "offers", "equipped", "provides", "comes with")
BENEFIT_CUES = ("benefit", "value", "helps", "enables", "improves", "save", "convenient", "efficient", "ideal for", "perfect for")
NARRATIVE_CUES = ("follows", "story", "journey", "discovers", "confronts", "escape", "family", "love", "survival", "mission", "battle", "secret")
CONCLUSION_MARKERS = ("in conclusion", "overall", "to sum up", "in summary", "ultimately", "taken together")
ASSISTANT_WRAPPERS = ("here are", "here's", "key takeaways", "step-by-step", "in short", "to help")
BACKGROUND_CUES = ("recent work", "prior work", "background", "motivation", "challenge")
METHOD_CUES = ("method", "approach", "we propose", "we present", "our model", "our method")
RESULT_CUES = ("results show", "results indicate", "we find", "findings", "outperform")
CONFLICT_CUES = ("conflict", "threat", "danger", "obstacle", "mystery", "betrayal")
RESOLUTION_CUES = ("resolve", "resolution", "reconcile", "survive", "overcome", "save")
INSTITUTIONAL_CUES = ("policy", "guideline", "guidelines", "procedure", "protocol", "regulation", "requirement", "compliance", "organization", "institution")
PROFESSIONAL_CUES = ("experience", "experienced", "expertise", "skilled", "accomplished", "managed", "led", "delivered", "achievement", "results-driven", "collaborated")


ATOM_OPERATIONAL_CONFIGS: dict[str, dict[str, Any]] = {
    "enumeration_markers": {"normalization": "per_1k_tokens", "known_confounds": ("genuine instructional content", "outline formatting")},
    "bullet_or_list_structure": {"normalization": "ratio_of_nonempty_lines", "known_confounds": ("copied markdown", "reference lists")},
    "punctuation_density": {"normalization": "per_token_ratio", "known_confounds": ("quotes and citations", "dialogue-heavy text")},
    "quotation_style": {"normalization": "per_1k_tokens", "known_confounds": ("dialogue content", "titles or cited strings")},
    "title_case_or_heading_like_openers": {"normalization": "binary_or_opening_ratio", "known_confounds": ("copied headings", "metadata lines")},
    "formal_connectives": {"normalization": "per_1k_tokens", "known_confounds": ("genre-mandated exposition", "translation artifacts")},
    "frame_markers": {"normalization": "per_1k_tokens", "known_confounds": ("instructional answers", "academic abstracts")},
    "code_gloss_markers": {"normalization": "per_1k_tokens", "known_confounds": ("didactic explanations", "example-rich tasks")},
    "evidential_markers": {"normalization": "per_1k_tokens", "known_confounds": ("citation-heavy topical content", "news style")},
    "self_mention_markers": {"normalization": "per_1k_tokens", "known_confounds": ("first-person narrative", "job applications")},
    "engagement_markers": {"normalization": "per_1k_tokens", "known_confounds": ("direct advice tasks", "interactive instructions")},
    "hedge_markers": {"normalization": "per_1k_tokens", "known_confounds": ("topic uncertainty", "speculative content")},
    "booster_certainty_markers": {"normalization": "per_1k_tokens", "known_confounds": ("rhetorical emphasis", "marketing copy")},
    "promotional_adjectives": {"normalization": "per_1k_tokens", "known_confounds": ("product category terminology", "reviews")},
    "disclaimer_lexicon": {"normalization": "per_1k_tokens", "known_confounds": ("true safety content", "legal notices")},
    "reporting_verbs": {"normalization": "per_1k_tokens", "known_confounds": ("journalistic reporting", "summaries of studies")},
    "technical_abstract_nouns": {"normalization": "per_1k_tokens", "known_confounds": ("topic-heavy technical domains", "course material")},
    "passive_or_agentless_constructions": {"normalization": "per_sentence", "known_confounds": ("heuristic misses irregular passives", "domain-specific participles")},
    "nominalization_patterns": {"normalization": "per_token_ratio", "known_confounds": ("topic-specific terminology", "false positives from common nouns")},
    "complex_noun_phrase_chains": {"normalization": "per_sentence", "known_confounds": ("heuristic noun-chain approximation", "entity names")},
    "subordination_density": {"normalization": "per_sentence", "known_confounds": ("long narrative sentences", "quoted material")},
    "participial_modifier_usage": {"normalization": "per_sentence", "known_confounds": ("gerund-heavy content", "present progressive verbs")},
    "personal_pronoun_suppression": {"normalization": "inverse_pronoun_rate", "known_confounds": ("short texts", "topic-driven first-person discourse")},
    "imperative_or_instructional_constructions": {"normalization": "per_sentence", "known_confounds": ("recipes and manuals", "explicit user support tasks")},
    "sentence_length_balance": {"normalization": "inverse_sentence_length_cv", "known_confounds": ("very short texts", "quoted dialogue")},
    "compression_vs_elaboration": {"normalization": "hybrid_ratio", "known_confounds": ("document length", "topic complexity")},
    "definition_then_expansion_pattern": {"normalization": "per_document_or_sentence", "known_confounds": ("educational content", "FAQ-style text")},
    "benefit_first_packaging": {"normalization": "opening_window_ratio", "known_confounds": ("genuine persuasive writing", "job applications")},
    "balanced_multi_part_completion": {"normalization": "segment_balance_ratio", "known_confounds": ("structured lists", "template prompts")},
    "high_density_summary_style": {"normalization": "hybrid_ratio", "known_confounds": ("abstract content density", "length effects")},
    "background_method_result_script": {"normalization": "ordered_move_score", "known_confounds": ("true scientific abstracts", "paper summaries")},
    "problem_solution_script": {"normalization": "ordered_move_score", "known_confounds": ("helpdesk tasks", "risk advisories")},
    "feature_benefit_call_to_value_script": {"normalization": "ordered_move_score", "known_confounds": ("marketing copy", "job applications")},
    "setup_conflict_resolution_script": {"normalization": "ordered_move_score", "known_confounds": ("real narrative content", "spoilers")},
    "conclusion_or_takeaway_move": {"normalization": "closing_window_ratio", "known_confounds": ("explicit summaries", "essay endings")},
    "helpful_assistant_wrapper": {"normalization": "document_level_score", "known_confounds": ("prompted formatting", "FAQ or support content")},
    "entity_grid_coherence": {"normalization": "mean_sentence_overlap", "known_confounds": ("short texts", "named-entity repetition")},
    "epistemic_hedging": {"normalization": "per_1k_tokens", "known_confounds": ("speculative subject matter", "forecasting")},
    "authoritative_certainty": {"normalization": "per_1k_tokens", "known_confounds": ("true certainty language", "promotional copy")},
    "institutional_impersonality": {"normalization": "hybrid_ratio", "known_confounds": ("policy documents", "formal legal writing")},
    "compliance_or_safety_stance": {"normalization": "document_level_score", "known_confounds": ("actual safety guidance", "regulated domains")},
    "enthusiasm_or_salesmanship": {"normalization": "document_level_score", "known_confounds": ("genuine enthusiasm", "review text")},
    "narrative_engagement_stance": {"normalization": "document_level_score", "known_confounds": ("true storytelling", "fan-fiction-like tone")},
    "professional_self_presentation": {"normalization": "document_level_score", "known_confounds": ("resumes and cover letters", "professional bios")},
    "academic_abstract_register": {"normalization": "weak_classifier_score", "known_confounds": ("true academic abstracts", "paper summaries")},
    "product_pitch_register": {"normalization": "weak_classifier_score", "known_confounds": ("actual marketing content", "store listings")},
    "movie_synopsis_register": {"normalization": "weak_classifier_score", "known_confounds": ("real synopsis text", "plot summaries")},
    "job_application_professional_register": {"normalization": "weak_classifier_score", "known_confounds": ("authentic applications", "professional bios")},
    "helpdesk_assistant_register": {"normalization": "weak_classifier_score", "known_confounds": ("support documentation", "FAQ responses")},
}


@dataclass(frozen=True)
class AtomSpec:
    atom_id: str
    level: int
    level_name: str
    description: str
    theoretical_source: str
    extractor_type: str
    extractor_recipe: str
    content_leakage_risk: str
    likely_bundle_memberships: tuple[str, ...]
    priority: str
    normalization: str
    known_confounds: tuple[str, ...]
    validation_subset: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "level": self.level,
            "level_name": self.level_name,
            "description": self.description,
            "theoretical_source": self.theoretical_source,
            "extractor_type": self.extractor_type,
            "extractor_recipe": self.extractor_recipe,
            "content_leakage_risk": self.content_leakage_risk,
            "likely_bundle_memberships": list(self.likely_bundle_memberships),
            "priority": self.priority,
            "normalization": self.normalization,
            "known_confounds": list(self.known_confounds),
            "validation_subset": bool(self.validation_subset),
        }


@dataclass(frozen=True)
class TextContext:
    raw_text: str
    text_lower: str
    words: list[str]
    alpha_tokens: list[str]
    token_counts: Counter[str]
    sentences: list[str]
    sentence_tokens: list[list[str]]
    lines: list[str]

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def alpha_count(self) -> int:
        return len(self.alpha_tokens)

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


def _normalize_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _tokenize_words(text: str) -> list[str]:
    return [tok.group(0) for tok in WORD_RE.finditer(text)]


def _lower_alpha_tokens(text: str) -> list[str]:
    return [tok.group(0).lower() for tok in ALPHA_RE.finditer(text)]


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def _per_1k(count: float, denom: float) -> float:
    return 1000.0 * _safe_div(count, denom)


def _count_phrase_hits(text_lower: str, token_counts: Counter[str], phrases: tuple[str, ...]) -> int:
    hits = 0
    for phrase in phrases:
        if " " in phrase or "-" in phrase:
            pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
            hits += len(re.findall(pattern, text_lower))
        else:
            hits += int(token_counts.get(phrase.lower(), 0))
    return hits


def _window_text(sentences: list[str], *, first_n: int | None = None, last_n: int | None = None) -> str:
    if first_n is not None:
        return " ".join(sentences[:first_n]).lower()
    if last_n is not None:
        return " ".join(sentences[-last_n:]).lower()
    return ""


def _count_sentence_initial_verbs(sentences: list[str]) -> int:
    count = 0
    for sent in sentences:
        tokens = _lower_alpha_tokens(sent)
        if tokens and tokens[0] in INSTRUCTION_VERBS:
            count += 1
    return count


def _sentence_length_cv(sentence_tokens: list[list[str]]) -> float:
    lengths = [len(toks) for toks in sentence_tokens if toks]
    if not lengths:
        return 0.0
    mean = sum(lengths) / float(len(lengths))
    if mean <= 1e-12:
        return 0.0
    var = sum((length - mean) ** 2 for length in lengths) / float(len(lengths))
    return (var ** 0.5) / mean


def _segment_balance_score(text: str, sentences: list[str]) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3 and sum(1 for line in lines if LIST_LINE_RE.match(line)) >= 2:
        seg_lengths = [max(1, len(_tokenize_words(line))) for line in lines if LIST_LINE_RE.match(line)]
    else:
        seg_lengths = [max(1, len(_tokenize_words(sent))) for sent in sentences[:4]]
    if len(seg_lengths) < 2:
        return 0.0
    mean = sum(seg_lengths) / float(len(seg_lengths))
    var = sum((length - mean) ** 2 for length in seg_lengths) / float(len(seg_lengths))
    cv = (var ** 0.5) / mean if mean else 0.0
    return max(0.0, 1.0 - cv)


def _content_tokens(tokens: list[str]) -> set[str]:
    return {tok for tok in tokens if len(tok) >= 4 and tok not in STOPWORDS and not tok.isdigit()}


def _entity_grid_overlap(sentence_tokens: list[list[str]]) -> float:
    overlaps: list[float] = []
    content_sets = [_content_tokens(toks) for toks in sentence_tokens if toks]
    for left, right in zip(content_sets, content_sets[1:]):
        if left or right:
            overlaps.append(_safe_div(len(left & right), len(left | right)))
    return float(sum(overlaps) / len(overlaps)) if overlaps else 0.0


def _ordered_move_score(sentence_lowers: list[str], first: tuple[str, ...], second: tuple[str, ...], third: tuple[str, ...] | None = None) -> float:
    positions: list[int] = []
    for cue_group in (first, second, third):
        if cue_group is None:
            continue
        pos = None
        for idx, sent in enumerate(sentence_lowers):
            if any(cue in sent for cue in cue_group):
                pos = idx
                break
        if pos is None:
            return 0.0
        positions.append(pos)
    return 1.0 if positions == sorted(positions) else 0.25


def _make_context(text: str) -> TextContext:
    raw_text = _normalize_text(text)
    sentences = _split_sentences(raw_text)
    sentence_tokens = [_lower_alpha_tokens(sent) for sent in sentences]
    alpha_tokens = _lower_alpha_tokens(raw_text)
    return TextContext(
        raw_text=raw_text,
        text_lower=raw_text.lower(),
        words=_tokenize_words(raw_text),
        alpha_tokens=alpha_tokens,
        token_counts=Counter(alpha_tokens),
        sentences=sentences,
        sentence_tokens=sentence_tokens,
        lines=[line.strip() for line in raw_text.splitlines() if line.strip()],
    )


def _score_complex_noun_phrase_chains(ctx: TextContext) -> float:
    runs = 0
    for tokens in ctx.sentence_tokens:
        current = 0
        for tok in tokens:
            if len(tok) >= 4 and tok not in STOPWORDS and tok not in SUBORDINATORS and not tok.endswith("ing"):
                current += 1
            else:
                if current >= 3:
                    runs += 1
                current = 0
        if current >= 3:
            runs += 1
    return _safe_div(runs, max(ctx.sentence_count, 1))


def _score_participial_modifiers(ctx: TextContext) -> float:
    count = 0
    for sent in ctx.sentences:
        count += len(re.findall(r"\b\w+(?:ing|ed)\s+\w+\b", sent.lower()))
    return _safe_div(count, max(ctx.sentence_count, 1))


def _score_definition_then_expansion(ctx: TextContext) -> float:
    sent_lowers = [sent.lower() for sent in ctx.sentences]
    patterns = ("is defined as", "refers to", "is a", "are a", "means")
    for idx, sent in enumerate(sent_lowers):
        if any(pat in sent for pat in patterns):
            next_sent = sent_lowers[idx + 1] if idx + 1 < len(sent_lowers) else ""
            if any(marker in sent or marker in next_sent for marker in CODE_GLOSS_MARKERS):
                return 1.0
    return 0.0


def _score_benefit_first_packaging(ctx: TextContext) -> float:
    opening = _window_text(ctx.sentences, first_n=2)
    later = " ".join(ctx.sentences[2:]).lower()
    opening_hits = _count_phrase_hits(opening, Counter(_lower_alpha_tokens(opening)), BENEFIT_CUES + PROMOTIONAL_ADJECTIVES)
    later_hits = _count_phrase_hits(later, Counter(_lower_alpha_tokens(later)), FEATURE_CUES)
    return float(opening_hits > 0 and later_hits > 0)


def _score_high_density_summary(ctx: TextContext) -> float:
    long_words = sum(1 for tok in ctx.alpha_tokens if len(tok) >= 7)
    technical = _count_phrase_hits(ctx.text_lower, ctx.token_counts, TECHNICAL_ABSTRACT_NOUNS)
    pronouns = sum(ctx.token_counts.get(tok, 0) for tok in ("i", "we", "you", "they", "he", "she"))
    return _safe_div(long_words + technical, max(ctx.alpha_count, 1)) - _safe_div(pronouns, max(ctx.alpha_count, 1))


def _score_helpful_assistant_wrapper(ctx: TextContext) -> float:
    opener = _window_text(ctx.sentences, first_n=2)
    closer = _window_text(ctx.sentences, last_n=1)
    opener_hits = _count_phrase_hits(opener, Counter(_lower_alpha_tokens(opener)), ASSISTANT_WRAPPERS + FRAME_MARKERS)
    closer_hits = _count_phrase_hits(closer, Counter(_lower_alpha_tokens(closer)), CONCLUSION_MARKERS)
    list_bonus = 1.0 if any(LIST_LINE_RE.match(line) for line in ctx.lines) else 0.0
    return opener_hits + 0.5 * closer_hits + list_bonus


def _score_institutional_impersonality(ctx: TextContext) -> float:
    institutional = _count_phrase_hits(ctx.text_lower, ctx.token_counts, INSTITUTIONAL_CUES)
    pronouns = sum(ctx.token_counts.get(tok, 0) for tok in ("i", "we", "you", "he", "she", "they"))
    passive = len(PASSIVE_RE.findall(ctx.raw_text))
    return _safe_div(institutional + passive, max(ctx.sentence_count, 1)) + 0.5 * _safe_div(1.0, 1.0 + pronouns)


def _score_professional_self_presentation(ctx: TextContext) -> float:
    professional = _count_phrase_hits(ctx.text_lower, ctx.token_counts, PROFESSIONAL_CUES)
    self_mentions = _count_phrase_hits(ctx.text_lower, ctx.token_counts, ("i", "my", "we", "our"))
    return _per_1k(professional + 0.5 * self_mentions, max(ctx.alpha_count, 1))


def _score_register_classifier(atom_scores: dict[str, float], atoms: tuple[str, ...]) -> float:
    vals = [float(atom_scores.get(atom, 0.0)) for atom in atoms]
    return float(sum(vals) / len(vals)) if vals else 0.0


@lru_cache(maxsize=1)
def get_atom_specs() -> dict[str, AtomSpec]:
    path = PROJECT_ROOT / "data" / "derived" / "style_groups" / "candidate_atom_inventory_d1.tsv"
    specs: dict[str, AtomSpec] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            atom_id = str(row["atom_id"]).strip()
            cfg = ATOM_OPERATIONAL_CONFIGS.get(atom_id)
            if cfg is None:
                raise KeyError(f"Missing D2 operational config for atom {atom_id}")
            bundles = tuple(part.strip() for part in str(row["likely_bundle_memberships"]).split(";") if part.strip())
            specs[atom_id] = AtomSpec(
                atom_id=atom_id,
                level=int(row["level"]),
                level_name=str(row["level_name"]).strip(),
                description=str(row["description"]).strip(),
                theoretical_source=str(row["theoretical_source"]).strip(),
                extractor_type=str(row["extractor_type"]).strip(),
                extractor_recipe=str(row["extractor_recipe"]).strip(),
                content_leakage_risk=str(row["content_leakage_risk"]).strip(),
                likely_bundle_memberships=bundles,
                priority=str(row["priority"]).strip(),
                normalization=str(cfg["normalization"]),
                known_confounds=tuple(str(x).strip() for x in cfg["known_confounds"]),
                validation_subset=int(row["level"]) <= 6,
            )
    missing = set(ATOM_OPERATIONAL_CONFIGS) - set(specs)
    if missing:
        raise KeyError(f"Operational configs not present in D1 TSV: {sorted(missing)}")
    return specs


ATOM_SPEC_NAMES = tuple(sorted(get_atom_specs().keys()))


@lru_cache(maxsize=1)
def get_bundle_priors() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for atom_id, spec in get_atom_specs().items():
        for bundle in spec.likely_bundle_memberships:
            out.setdefault(bundle, []).append(atom_id)
    for bundle, atoms in out.items():
        out[bundle] = sorted(atoms)
    return dict(sorted(out.items()))


BUNDLE_PRIOR_NAMES = tuple(get_bundle_priors().keys())


def extract_atom_scores(text: str) -> dict[str, float]:
    """Extract grounded atom scores from text using the D2 operational ontology."""

    ctx = _make_context(text)
    sent_lowers = [sent.lower() for sent in ctx.sentences]
    alpha_count = max(ctx.alpha_count, 1)
    sentence_count = max(ctx.sentence_count, 1)
    punctuation_total = sum(ch in ",;:()[]-—\"'" for ch in ctx.raw_text)
    personal_pronouns = sum(
        ctx.token_counts.get(tok, 0)
        for tok in ("i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours")
    )

    atom_scores: dict[str, float] = {
        "enumeration_markers": _per_1k(
            len(ENUM_WORD_RE.findall(ctx.raw_text)) + len(re.findall(r"\b\d+[.)]\b", ctx.raw_text)),
            alpha_count,
        ),
        "bullet_or_list_structure": _safe_div(sum(1 for line in ctx.lines if LIST_LINE_RE.match(line)), max(len(ctx.lines), 1)),
        "punctuation_density": _safe_div(punctuation_total, max(ctx.word_count, 1)),
        "quotation_style": _per_1k(ctx.raw_text.count('"') + ctx.raw_text.count("'"), alpha_count),
        "title_case_or_heading_like_openers": float(bool(ctx.lines and TITLE_OPEN_RE.match(ctx.lines[0]) and len(ctx.lines[0].split()) <= 8)),
        "formal_connectives": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, FORMAL_CONNECTIVES), alpha_count),
        "frame_markers": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, FRAME_MARKERS), alpha_count),
        "code_gloss_markers": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, CODE_GLOSS_MARKERS), alpha_count),
        "evidential_markers": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, EVIDENTIAL_MARKERS), alpha_count),
        "self_mention_markers": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, SELF_MENTION_MARKERS), alpha_count),
        "engagement_markers": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, ENGAGEMENT_MARKERS), alpha_count),
        "hedge_markers": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, HEDGE_MARKERS), alpha_count),
        "booster_certainty_markers": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, CERTAINTY_MARKERS), alpha_count),
        "promotional_adjectives": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, PROMOTIONAL_ADJECTIVES), alpha_count),
        "disclaimer_lexicon": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, DISCLAIMER_LEXICON), alpha_count),
        "reporting_verbs": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, REPORTING_VERBS), alpha_count),
        "technical_abstract_nouns": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, TECHNICAL_ABSTRACT_NOUNS), alpha_count),
        "passive_or_agentless_constructions": _safe_div(len(PASSIVE_RE.findall(ctx.raw_text)), sentence_count),
        "nominalization_patterns": _safe_div(sum(1 for tok in ctx.alpha_tokens if tok.endswith(NOMINALIZATION_SUFFIXES)), alpha_count),
        "complex_noun_phrase_chains": _score_complex_noun_phrase_chains(ctx),
        "subordination_density": _safe_div(_count_phrase_hits(ctx.text_lower, ctx.token_counts, SUBORDINATORS), sentence_count),
        "participial_modifier_usage": _score_participial_modifiers(ctx),
        "personal_pronoun_suppression": 1.0 - _safe_div(personal_pronouns, alpha_count),
        "imperative_or_instructional_constructions": _safe_div(
            _count_sentence_initial_verbs(ctx.sentences) + _count_phrase_hits(ctx.text_lower, ctx.token_counts, ("please", "be sure", "make sure", "remember")),
            sentence_count,
        ),
        "sentence_length_balance": max(0.0, 1.0 - _sentence_length_cv(ctx.sentence_tokens)),
        "compression_vs_elaboration": _safe_div(
            sum(1 for tok in ctx.alpha_tokens if len(tok) >= 7)
            + sum(1 for tok in ctx.alpha_tokens if tok.endswith(NOMINALIZATION_SUFFIXES))
            + _count_phrase_hits(ctx.text_lower, ctx.token_counts, TECHNICAL_ABSTRACT_NOUNS),
            alpha_count,
        ) - 0.5 * _sentence_length_cv(ctx.sentence_tokens),
        "definition_then_expansion_pattern": _score_definition_then_expansion(ctx),
        "benefit_first_packaging": _score_benefit_first_packaging(ctx),
        "balanced_multi_part_completion": _segment_balance_score(ctx.raw_text, ctx.sentences),
        "high_density_summary_style": _score_high_density_summary(ctx),
        "background_method_result_script": _ordered_move_score(sent_lowers, BACKGROUND_CUES, METHOD_CUES, RESULT_CUES),
        "problem_solution_script": _ordered_move_score(sent_lowers, PROBLEM_CUES, SOLUTION_CUES, None),
        "feature_benefit_call_to_value_script": _ordered_move_score(sent_lowers, FEATURE_CUES, BENEFIT_CUES, ("ideal for", "perfect for", "value")),
        "setup_conflict_resolution_script": _ordered_move_score(sent_lowers, NARRATIVE_CUES, CONFLICT_CUES, RESOLUTION_CUES),
        "conclusion_or_takeaway_move": float(
            _count_phrase_hits(
                _window_text(ctx.sentences, last_n=2),
                Counter(_lower_alpha_tokens(_window_text(ctx.sentences, last_n=2))),
                CONCLUSION_MARKERS,
            ) > 0
        ),
        "helpful_assistant_wrapper": _score_helpful_assistant_wrapper(ctx),
        "entity_grid_coherence": _entity_grid_overlap(ctx.sentence_tokens),
        "epistemic_hedging": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, HEDGE_MARKERS + ("evidence suggests",)), alpha_count),
        "authoritative_certainty": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, CERTAINTY_MARKERS + ("clearly", "must")), alpha_count),
        "institutional_impersonality": _score_institutional_impersonality(ctx),
        "compliance_or_safety_stance": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, DISCLAIMER_LEXICON + INSTITUTIONAL_CUES), alpha_count),
        "enthusiasm_or_salesmanship": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, PROMOTIONAL_ADJECTIVES + BENEFIT_CUES + CERTAINTY_MARKERS), alpha_count) + _safe_div(ctx.raw_text.count("!"), max(ctx.word_count, 1)),
        "narrative_engagement_stance": _per_1k(_count_phrase_hits(ctx.text_lower, ctx.token_counts, NARRATIVE_CUES + CONFLICT_CUES), alpha_count),
        "professional_self_presentation": _score_professional_self_presentation(ctx),
    }

    atom_scores["academic_abstract_register"] = _score_register_classifier(atom_scores, ("formal_connectives", "technical_abstract_nouns", "nominalization_patterns", "background_method_result_script", "epistemic_hedging", "institutional_impersonality"))
    atom_scores["product_pitch_register"] = _score_register_classifier(atom_scores, ("promotional_adjectives", "benefit_first_packaging", "feature_benefit_call_to_value_script", "enthusiasm_or_salesmanship", "authoritative_certainty"))
    atom_scores["movie_synopsis_register"] = _score_register_classifier(atom_scores, ("setup_conflict_resolution_script", "narrative_engagement_stance", "entity_grid_coherence", "conclusion_or_takeaway_move"))
    atom_scores["job_application_professional_register"] = _score_register_classifier(atom_scores, ("professional_self_presentation", "benefit_first_packaging", "authoritative_certainty", "sentence_length_balance", "formal_connectives"))
    atom_scores["helpdesk_assistant_register"] = _score_register_classifier(atom_scores, ("helpful_assistant_wrapper", "problem_solution_script", "imperative_or_instructional_constructions", "compliance_or_safety_stance", "balanced_multi_part_completion"))

    return {atom_id: float(atom_scores.get(atom_id, 0.0)) for atom_id in ATOM_SPEC_NAMES}
