import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from config.configuration import build_chat_model
from core.state import ChatState

logger = logging.getLogger(__name__)

model = build_chat_model()

SYSTEM_PROMPT = """
You classify user messages for a parking facility chatbot.

Return JSON with a single field "intent" set to exactly one of these labels:

- "info_query": questions about the facility itself — locations, parking zones,
  amenities, how booking works, general policies. Static information that
  doesn't change minute-to-minute.
- "dynamic_query": questions about live state or the facility's schedule —
  current availability, what's open right now, today's operating hours, and the
  regular working days/hours (e.g. "what are your working days/hours?", "when
  are you open?"). The schedule lives in the live database, not the static docs.
- "reservation": the user wants to book, reserve, cancel, or modify a parking
  spot. Includes partial reservation requests like "I need a spot tomorrow".
- "out_of_scope": anything unrelated to this parking facility (weather,
  poems, general chitchat, other businesses, etc.).

When in doubt between info_query and dynamic_query, prefer dynamic_query if
the message contains words like "now", "today", "currently", "available".

IMPORTANT — multi-turn reservations:
A reservation is collected over several turns: the bot asks for one detail at a
time (start/end time, vehicle plate, contact email, yes/no confirmation). When
the conversation context below says a reservation is IN PROGRESS, the user's
message is almost certainly the answer to the bot's last question. Classify it
as "reservation" even when, on its own, it looks meaningless — e.g. a bare plate
like "XR-CAL21-R", an email, a date/time, or "yes"/"no". Only pick a different
intent if the user clearly changes the subject (asks an unrelated question) or
asks to stop/cancel the booking.

Respond with JSON only. Example: {"intent": "reservation"}
"""


def _reservation_in_progress(state: ChatState) -> bool:
    """True when a booking/cancel flow is mid-way (started but not finalised)."""
    rf = state.get("reservation_fields") or {}
    return bool(rf.get("operation")) and not rf.get("reservation_id")


def _build_context(state: ChatState) -> str:
    """Conversation context handed to the classifier so it routes follow-ups."""
    lines: list[str] = []
    if _reservation_in_progress(state):
        lines.append("A reservation is currently IN PROGRESS.")
    previous = state.get("response")
    if previous:
        lines.append(f'The bot last said: "{previous}"')
    return "\n".join(lines)


def _classify_intent(state: ChatState) -> str:
    user_input = state["user_input"]
    context = _build_context(state)
    human = (
        f"Conversation context:\n{context}\n\nUser message:\n{user_input}"
        if context
        else user_input
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)]
    response = model.invoke(messages)
    intent_dict = json.loads(response.content.strip())
    return intent_dict["intent"]


def router_node(state: ChatState) -> dict:
    logger.debug("router: start")
    intent = _classify_intent(state)
    logger.debug("router: intent=%s", intent)
    return {"intent": intent}