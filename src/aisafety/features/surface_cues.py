"""Lightweight surface-cue features for authorship and nuisance-style analysis."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
PASSIVE_RE = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
CONTRACTION_RE = re.compile(r"\b\w+'\w+\b")
PROPER_CASE_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

FIRST_PERSON_PLURAL = ("we", "our", "ours", "us")
SECOND_PERSON = ("you", "your", "yours")
THIRD_PERSON = ("he", "she", "they", "them", "their", "theirs", "his", "her", "hers")

HEDGE_LEXICON = (
    "may",
    "might",
    "could",
    "can",
    "appears",
    "appear",
    "seems",
    "seem",
    "suggests",
    "suggest",
    "likely",
    "possibly",
    "perhaps",
    "generally",
    "often",
    "approximately",
    "roughly",
    "typically",
    "indicates",
    "indicate",
)
CERTAINTY_LEXICON = (
    "clearly",
    "definitely",
    "certainly",
    "undoubtedly",
    "always",
    "never",
    "prove",
    "proves",
    "demonstrates",
    "show",
    "shows",
    "establishes",
    "must",
    "will",
)
ACADEMIC_LEXICON = (
    "this paper",
    "we propose",
    "we present",
    "our approach",
    "our method",
    "results show",
    "results demonstrate",
    "method",
    "approach",
    "analysis",
    "evaluation",
    "experiment",
    "experiments",
    "dataset",
    "model",
    "models",
    "study",
)
SAFETY_LEXICON = (
    "important to note",
    "please note",
    "ensure that",
    "make sure",
    "recommended",
    "for safety",
    "best practice",
    "avoid",
    "do not",
    "should not",
    "guidelines",
    "policy",
    "policies",
    "safe to",
    "risk",
    "responsibly",
)
PROMO_LEXICON = (
    "premium",
    "high-quality",
    "must-have",
    "perfect for",
    "designed to",
    "ideal for",
    "feature",
    "features",
    "enjoy",
    "elevate",
    "stylish",
    "durable",
    "versatile",
    "excellent choice",
    "best choice",
)
NARRATIVE_LEXICON = (
    "follows",
    "story of",
    "journey",
    "discovers",
    "confronts",
    "sets out",
    "unexpected",
    "secrets",
    "friendship",
    "family",
    "love",
    "survival",
    "must",
)
TEMPLATE_LEXICON = (
    "in conclusion",
    "overall",
    "to sum up",
    "in summary",
    "this paper presents",
    "this product",
    "this film",
    "the article",
    "it is important to note",
    "in this paper",
    "the authors",
)
DISCOURSE_MARKERS = (
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "additionally",
    "consequently",
    "meanwhile",
    "overall",
)
NOMINALIZATION_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ism", "ance", "ence")


SURFACE_FEATURE_NAMES = (
    "char_count",
    "word_count",
    "sentence_count",
    "avg_word_len",
    "avg_sentence_len_words",
    "long_word_ratio",
    "unique_token_ratio",
    "token_repetition_ratio",
    "contraction_rate",
    "digit_ratio",
    "uppercase_ratio",
    "comma_rate",
    "semicolon_rate",
    "colon_rate",
    "paren_rate",
    "dash_rate",
    "quote_rate",
    "exclamation_rate",
    "question_rate",
    "newline_rate",
    "bullet_line_ratio",
    "markdown_emphasis_rate",
    "proper_case_rate",
    "first_person_plural_rate",
    "second_person_rate",
    "third_person_pronoun_rate",
    "hedge_rate",
    "certainty_rate",
    "academic_phrase_rate",
    "safety_phrase_rate",
    "promo_phrase_rate",
    "narrative_phrase_rate",
    "template_phrase_rate",
    "discourse_marker_rate",
    "citation_year_rate",
    "passive_voice_rate",
    "nominalization_rate",
    "syllables_per_word",
    "flesch_reading_ease",
)


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def _normalize_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _tokenize_words(text: str) -> list[str]:
    return [tok.group(0) for tok in WORD_RE.finditer(text)]


def _lower_alpha_tokens(text: str) -> list[str]:
    return [tok.group(0).lower() for tok in TOKEN_RE.finditer(text)]


@lru_cache(maxsize=8192)
def _estimate_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", str(word or "").lower())
    if not w:
        return 0
    vowels = "aeiouy"
    syllables = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            syllables += 1
        prev_vowel = is_vowel
    if w.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def _count_phrase_hits(text_lower: str, token_counts: Counter[str], phrases: tuple[str, ...]) -> int:
    hits = 0
    for phrase in phrases:
        if " " in phrase or "-" in phrase:
            pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
            hits += len(re.findall(pattern, text_lower))
        else:
            hits += int(token_counts.get(phrase, 0))
    return hits


def _bullet_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    bulletish = 0
    for line in lines:
        if line.startswith(("-", "*", "\u2022")):
            bulletish += 1
            continue
        if re.match(r"^\d+[.)]\s", line):
            bulletish += 1
    return _safe_div(bulletish, len(lines))


def extract_surface_features(text: str) -> dict[str, float]:
    """Return lightweight lexical, formatting, and discourse cue features."""

    raw_text = _normalize_text(text)
    text_lower = raw_text.lower()
    words = _tokenize_words(raw_text)
    alpha_tokens = _lower_alpha_tokens(raw_text)
    token_counts = Counter(alpha_tokens)
    sentences = _split_sentences(raw_text)

    word_count = len(words)
    alpha_count = len(alpha_tokens)
    sentence_count = len(sentences)
    chars = len(raw_text)
    unique_tokens = len(set(alpha_tokens))
    long_words = sum(1 for tok in alpha_tokens if len(tok) >= 7)
    contractions = len(CONTRACTION_RE.findall(raw_text))
    digits = sum(ch.isdigit() for ch in raw_text)
    uppers = sum(ch.isupper() for ch in raw_text)
    commas = raw_text.count(",")
    semicolons = raw_text.count(";")
    colons = raw_text.count(":")
    parens = raw_text.count("(") + raw_text.count(")")
    dashes = raw_text.count("-") + raw_text.count("\u2014")
    quotes = raw_text.count('"') + raw_text.count("'")
    exclamations = raw_text.count("!")
    questions = raw_text.count("?")
    newlines = raw_text.count("\n")
    markdown_emphasis = raw_text.count("**") + raw_text.count("__")
    proper_case = len(PROPER_CASE_RE.findall(raw_text))
    years = len(YEAR_RE.findall(raw_text))
    passive = len(PASSIVE_RE.findall(raw_text))
    nominalizations = sum(1 for tok in alpha_tokens if tok.endswith(NOMINALIZATION_SUFFIXES))

    sentence_lengths = [len(_tokenize_words(sent)) for sent in sentences if sent.strip()]
    avg_sentence_len = _safe_div(sum(sentence_lengths), len(sentence_lengths))
    avg_word_len = _safe_div(sum(len(tok) for tok in alpha_tokens), alpha_count)
    repetition_ratio = _safe_div(alpha_count - unique_tokens, alpha_count)
    syllables_per_word = _safe_div(sum(_estimate_syllables(tok) for tok in alpha_tokens), alpha_count)
    flesch = 206.835 - 1.015 * avg_sentence_len - 84.6 * syllables_per_word if alpha_count else 0.0

    return {
        "char_count": float(chars),
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_word_len": float(avg_word_len),
        "avg_sentence_len_words": float(avg_sentence_len),
        "long_word_ratio": _safe_div(long_words, alpha_count),
        "unique_token_ratio": _safe_div(unique_tokens, alpha_count),
        "token_repetition_ratio": _safe_div(repetition_ratio, 1.0),
        "contraction_rate": _safe_div(contractions, alpha_count),
        "digit_ratio": _safe_div(digits, max(chars, 1)),
        "uppercase_ratio": _safe_div(uppers, max(chars, 1)),
        "comma_rate": _safe_div(commas, max(word_count, 1)),
        "semicolon_rate": _safe_div(semicolons, max(word_count, 1)),
        "colon_rate": _safe_div(colons, max(word_count, 1)),
        "paren_rate": _safe_div(parens, max(word_count, 1)),
        "dash_rate": _safe_div(dashes, max(word_count, 1)),
        "quote_rate": _safe_div(quotes, max(word_count, 1)),
        "exclamation_rate": _safe_div(exclamations, max(word_count, 1)),
        "question_rate": _safe_div(questions, max(word_count, 1)),
        "newline_rate": _safe_div(newlines, max(chars, 1)),
        "bullet_line_ratio": _bullet_line_ratio(raw_text),
        "markdown_emphasis_rate": _safe_div(markdown_emphasis, max(chars, 1)),
        "proper_case_rate": _safe_div(proper_case, max(word_count, 1)),
        "first_person_plural_rate": _safe_div(sum(token_counts[t] for t in FIRST_PERSON_PLURAL), alpha_count),
        "second_person_rate": _safe_div(sum(token_counts[t] for t in SECOND_PERSON), alpha_count),
        "third_person_pronoun_rate": _safe_div(sum(token_counts[t] for t in THIRD_PERSON), alpha_count),
        "hedge_rate": _safe_div(_count_phrase_hits(text_lower, token_counts, HEDGE_LEXICON), alpha_count),
        "certainty_rate": _safe_div(_count_phrase_hits(text_lower, token_counts, CERTAINTY_LEXICON), alpha_count),
        "academic_phrase_rate": _safe_div(_count_phrase_hits(text_lower, token_counts, ACADEMIC_LEXICON), alpha_count),
        "safety_phrase_rate": _safe_div(_count_phrase_hits(text_lower, token_counts, SAFETY_LEXICON), alpha_count),
        "promo_phrase_rate": _safe_div(_count_phrase_hits(text_lower, token_counts, PROMO_LEXICON), alpha_count),
        "narrative_phrase_rate": _safe_div(_count_phrase_hits(text_lower, token_counts, NARRATIVE_LEXICON), alpha_count),
        "template_phrase_rate": _safe_div(_count_phrase_hits(text_lower, token_counts, TEMPLATE_LEXICON), alpha_count),
        "discourse_marker_rate": _safe_div(_count_phrase_hits(text_lower, token_counts, DISCOURSE_MARKERS), alpha_count),
        "citation_year_rate": _safe_div(years, max(word_count, 1)),
        "passive_voice_rate": _safe_div(passive, max(sentence_count, 1)),
        "nominalization_rate": _safe_div(nominalizations, alpha_count),
        "syllables_per_word": float(syllables_per_word),
        "flesch_reading_ease": float(flesch),
    }
