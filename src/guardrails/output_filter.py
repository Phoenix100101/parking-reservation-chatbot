"""Output guardrail — post-LLM screening of the assistant's reply.

Two responsibilities:

1. Redact sensitive PII (cards, SSNs, phone numbers, IPs, names) that may have
   leaked into the model's response — but **never** the reservation fields the
   user is actively providing. The booking summary echoes the email and plate
   back on purpose, so those values are added to an allowlist via
   :func:`guardrails.pii_detector.reservation_allow_values`.
2. Record this turn (user message + masked reply) into ``history``. It runs last
   on every branch, so it's the single accumulation point (via the
   ``append_history`` reducer on :class:`core.state.ChatState`).
"""

import logging

from config.configuration import get_settings
from core.state import ChatState
from guardrails import pii_detector

logger = logging.getLogger(__name__)


def output_filter_node(state: ChatState) -> dict:
    logger.debug("output_filter: start")
    cfg = get_settings().guardrail

    response = state.get("response") or ""
    redacted: list[pii_detector.PIIEntity] = []
    masked = response
    if cfg.enabled and cfg.mask_labels:
        allow = pii_detector.reservation_allow_values(state.get("reservation_fields"))
        masked, redacted = pii_detector.mask(
            response, mask_labels=cfg.mask_labels, allow=allow
        )
        if redacted:
            labels = sorted({e.label for e in redacted})
            # Log labels only — never the redacted values.
            logger.warning("redacted output PII labels=%s", labels)

    turns: list[dict] = []
    user_input = state.get("user_input")
    if user_input:
        turns.append({"role": "user", "content": user_input})
    if masked:
        turns.append({"role": "assistant", "content": masked})

    return {
        "output_blocked": bool(redacted),
        "response": masked,
        "history": turns,
    }