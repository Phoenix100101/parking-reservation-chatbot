"""spaCy-backed PII detection and masking for chat guardrails.

Detection combines two sources:

* **spaCy NER** (``en_core_web_sm``) for free-text entities like ``PERSON``.
* **regex patterns** for structured identifiers (email, phone, credit card,
  SSN, IP) that NER models miss or tag inconsistently.

A central design constraint: values the reservation flow legitimately collects
(email, vehicle plate, dates/times) must **not** be masked. The input guardrail
runs before :mod:`core.nodes.reservation_node`, which extracts those fields from
``user_input`` via the LLM, and the booking summary echoes them back to the user.
Masking them would break slot-filling and redact confirmations.

Two mechanisms keep reservation data intact:

1. :data:`RESERVATION_LABELS` are never in the default mask set, so the *kinds*
   of data the booking collects pass through.
2. Callers pass an ``allow`` set of raw values (see :func:`reservation_allow_values`)
   so the specific email/plate currently being collected is exempt even if a
   label would otherwise be masked.

If the spaCy model isn't installed the detector degrades gracefully to
regex-only matching (download it with
``uv run python -m spacy download en_core_web_sm``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

try:  # spaCy is a hard dependency, but stay importable if the model is absent.
    import spacy
except ImportError:  # pragma: no cover - spacy declared in pyproject
    spacy = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------
# spaCy NER labels we surface (others are ignored as noise for this domain).
NER_LABELS = frozenset(
    {"PERSON", "GPE", "LOC", "ORG", "NORP", "FAC", "DATE", "TIME", "MONEY"}
)

# Labels tied to the reservation flow — never masked by default so slot-filling
# and confirmation summaries keep working.
RESERVATION_LABELS = frozenset({"EMAIL", "DATE", "TIME"})

# What the output guardrail redacts by default: clearly sensitive, structured
# data that the booking flow never legitimately needs. These are regex-matched
# and reliable across languages. NER labels like PERSON are intentionally NOT
# here — en_core_web_sm is English-only and noisy (it tags ordinary German/other
# nouns as PERSON), so masking on it would corrupt normal replies. Enable PERSON
# via GUARDRAIL_MASK_LABELS only if you switch to a suitable model.
DEFAULT_MASK_LABELS = frozenset({"PHONE", "CREDIT_CARD", "SSN", "IP_ADDRESS"})

# Structured identifiers matched by regex (label -> compiled pattern).
_REGEX_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "PHONE": re.compile(
        r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
    ),
}


@dataclass(frozen=True)
class PIIEntity:
    """A detected span of personally identifiable information."""

    text: str
    label: str
    start: int
    end: int


# ---------------------------------------------------------------------------
# spaCy loading (lazy + graceful fallback)
# ---------------------------------------------------------------------------
_NLP: "spacy.language.Language | None | bool" = None


def _model_name() -> str:
    """spaCy model name from guardrail settings, with a standalone fallback."""
    try:
        from config.configuration import get_settings

        return get_settings().guardrail.spacy_model
    except Exception:
        return "en_core_web_sm"


def _get_nlp():
    """Load spaCy once; fall back to a blank pipeline (regex-only) if needed."""
    global _NLP
    if _NLP is not None:
        return _NLP or None
    if spacy is None:
        _NLP = False
        return None
    try:
        _NLP = spacy.load(_model_name(), disable=["lemmatizer"])
    except OSError:
        # Model not downloaded — keep working with regex matching only.
        _NLP = spacy.blank("en")
    return _NLP


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _regex_entities(text: str) -> list[PIIEntity]:
    found: list[PIIEntity] = []
    for label, pattern in _REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            value = m.group().strip()
            if label == "CREDIT_CARD" and len(re.sub(r"\D", "", value)) < 13:
                continue
            found.append(PIIEntity(value, label, m.start(), m.start() + len(value)))
    return found


def _overlaps(a: PIIEntity, spans: list[PIIEntity]) -> bool:
    return any(a.start < s.end and s.start < a.end for s in spans)


def detect(text: str) -> list[PIIEntity]:
    """Return all PII spans in ``text``, sorted by position.

    Regex matches take precedence; spaCy entities overlapping a regex span are
    dropped to avoid double-tagging (e.g. an email mislabelled ``ORG``).
    """
    if not text:
        return []

    entities = _regex_entities(text)

    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        ner_spans = [
            PIIEntity(ent.text, ent.label_, ent.start_char, ent.end_char)
            for ent in getattr(doc, "ents", [])
            if ent.label_ in NER_LABELS
        ]
        entities.extend(e for e in ner_spans if not _overlaps(e, entities))

    return sorted(entities, key=lambda e: e.start)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def mask(
    text: str,
    *,
    mask_labels: frozenset[str] = DEFAULT_MASK_LABELS,
    allow: frozenset[str] | set[str] = frozenset(),
) -> tuple[str, list[PIIEntity]]:
    """Redact sensitive PII in ``text``.

    Args:
        text: Source text.
        mask_labels: Labels to redact. Defaults to :data:`DEFAULT_MASK_LABELS`,
            which deliberately excludes :data:`RESERVATION_LABELS`.
        allow: Raw values to leave untouched even if their label is masked —
            typically the reservation fields currently being collected (see
            :func:`reservation_allow_values`).

    Returns:
        ``(masked_text, redacted_entities)``.
    """
    if not text:
        return text, []

    allow_norm = {v.strip().lower() for v in allow if v}
    redacted: list[PIIEntity] = [
        e
        for e in detect(text)
        if e.label in mask_labels and e.text.strip().lower() not in allow_norm
    ]

    # Splice right-to-left so earlier offsets stay valid.
    masked = text
    for e in sorted(redacted, key=lambda e: e.start, reverse=True):
        masked = masked[: e.start] + f"[REDACTED_{e.label}]" + masked[e.end :]

    return masked, redacted


def reservation_allow_values(reservation_fields: dict | None) -> frozenset[str]:
    """Collect raw reservation values that must never be masked.

    Pulls string-valued slots (email, plate, …) plus formatted datetimes from
    ``reservation_fields`` so the booking summary and follow-up prompts keep
    showing the real data the user provided.
    """
    if not reservation_fields:
        return frozenset()

    values: set[str] = set()
    for value in reservation_fields.values():
        if isinstance(value, str) and value.strip():
            values.add(value)
        elif hasattr(value, "strftime"):  # datetime-like
            values.add(value.strftime("%Y-%m-%d %H:%M"))
            values.add(value.isoformat())
    return frozenset(values)