"""Input guardrail — pre-LLM screening of the user's message.

Runs first in the graph (before the router and reservation node). It blocks two
classes of input and otherwise passes the message through **unchanged** so the
reservation node can still extract booking fields (email, plate, dates) from it:

1. Prompt-injection / jailbreak attempts.
2. Highly sensitive data that the parking flow never needs (credit cards, SSNs)
   — we refuse to process it rather than risk storing/logging it.

Crucially, the guardrail does *not* mask reservation fields here: masking
``user_input`` would break slot-filling downstream. Redaction of leaked PII is
the output guardrail's job.
"""

import logging
import re

from config.configuration import get_settings
from core.state import ChatState
from guardrails import pii_detector

logger = logging.getLogger(__name__)

# Lightweight prompt-injection heuristics.
_INJECTION_PATTERNS = [
    re.compile(r"\bignore\b.{0,30}\b(previous|prior|above|all)\b.{0,30}\binstructions?\b", re.I),
    re.compile(r"\bdisregard\b.{0,30}\b(previous|prior|above|system)\b", re.I),
    re.compile(r"\b(you are now|act as|pretend to be|from now on you)\b", re.I),
    re.compile(r"\b(system|developer)\s*prompt\b", re.I),
    re.compile(r"\breveal\b.{0,30}\b(prompt|instructions?|system)\b", re.I),
]

_INJECTION_MSG = (
    "I can't follow that request. I'm here to help with parking facility "
    "questions, availability, and reservations."
)
_SENSITIVE_MSG = (
    "For your security, please don't share card numbers or government IDs. "
    "To book a spot I only need a start/end time, your vehicle plate, and an email."
)


def _has_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def input_filter_node(state: ChatState) -> dict:
    logger.debug("input_filter: start")
    cfg = get_settings().guardrail
    text = state.get("user_input", "") or ""

    if not cfg.enabled:
        return {"input_blocked": False}

    if cfg.injection_check and _has_injection(text):
        logger.warning("input blocked: prompt injection")
        return {"input_blocked": True, "response": _INJECTION_MSG}

    if cfg.block_input_labels:
        hits = {e.label for e in pii_detector.detect(text)} & cfg.block_input_labels
        if hits:
            # Log labels only — never the matched values.
            logger.warning("input blocked: sensitive PII labels=%s", sorted(hits))
            return {"input_blocked": True, "response": _SENSITIVE_MSG}

    # Clean input — pass through untouched (reservation fields stay intact).
    return {"input_blocked": False}