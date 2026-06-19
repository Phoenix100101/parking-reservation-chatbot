"""Reservation agent node — multi-turn slot-filling for parking bookings.

The graph runs this node once per user message, so the "conversation" is driven
across invocations via the accumulated ``reservation_fields`` (see
:class:`core.state.ReservationState`). Each call:

    1. extracts any booking fields present in the latest message (LLM),
    2. validates and merges them into the running state,
    3. asks for the next missing field, OR
    4. summarises and asks the user to confirm, OR
    5. on confirmation, writes the reservation to Postgres.

Requirements for multi-turn to work:
    * ``build_graph()`` must compile with a checkpointer and be invoked with a
      stable ``thread_id`` so ``reservation_fields`` survives between turns.
    * ``state`` should carry a ``session_id`` (used as ``user_session_id`` in the
      DB). Absent one, a placeholder is used — see :func:`_session_id`.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from config.configuration import build_chat_model
from core.state import ChatState
from data.sql_store import postgres_client as db

logger = logging.getLogger(__name__)

# Fields the user must supply before we can book (space is assigned separately).
REQUIRED_FIELDS = ("start_date_time", "end_date_time", "vehicle_plate", "contact_email")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_AFFIRM = {"yes", "y", "yeah", "yep", "confirm", "correct", "sure", "ok", "okay"}
_NEGATE = {"no", "n", "nope", "cancel", "stop", "wrong"}

_PROMPTS = {
    "start_date_time": "When would you like the reservation to start? (date and time)",
    "end_date_time": "And when should it end?",
    "vehicle_plate": "What's your vehicle plate number?",
    "contact_email": "What email should I send the confirmation to?",
}


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------
class ExtractedFields(BaseModel):
    """Booking fields the LLM could pull from the latest user message."""

    operation: Optional[Literal["book", "cancel", "modify"]] = Field(
        None, description="What the user is trying to do."
    )
    start_date_time: Optional[datetime] = Field(
        None, description="Reservation start as ISO 8601; resolve relative dates."
    )
    end_date_time: Optional[datetime] = Field(
        None, description="Reservation end as ISO 8601; resolve relative dates."
    )
    vehicle_plate: Optional[str] = Field(None, description="Vehicle licence plate.")
    contact_email: Optional[str] = Field(None, description="Contact email address.")
    space_id: Optional[int] = Field(None, description="Specific space id, if named.")
    floor_preference: Optional[int] = Field(None, description="Preferred floor, if any.")
    confirm: Optional[bool] = Field(
        None, description="True if the user affirms, False if they decline/cancel."
    )


_EXTRACT_PROMPT = """You extract parking-reservation details from a user message.
The current date and time is {now} (UTC). Resolve relative expressions like
"tomorrow 3pm" against it. Only fill a field if the user actually provided it;
leave everything else null. Output times in ISO 8601 with a UTC offset.
Set `confirm` to true only if the user is affirming a proposed booking, false if
they decline or want to change something."""

_model = build_chat_model()
_EXTRACTOR = _model.with_structured_output(ExtractedFields)


def _extract(state: ChatState) -> ExtractedFields:
    """Run the LLM extractor over the latest message; degrade to empty on error."""
    messages = [
        SystemMessage(content=_EXTRACT_PROMPT.format(now=_now().isoformat())),
    ]
    for turn in (state.get("history") or [])[-6:]:
        role, content = turn.get("role"), turn.get("content", "")
        if content:
            messages.append(HumanMessage(content=f"[{role}] {content}"))
    messages.append(HumanMessage(content=state.get("user_input", "")))
    try:
        return _EXTRACTOR.invoke(messages)  # type: ignore[return-value]
    except Exception:
        # Best-effort extraction: a transient LLM/parse failure must not crash
        # the turn — the node will simply re-prompt for whatever is still
        # missing. Log it (with traceback) so the degradation isn't silent.
        logger.exception("reservation field extraction failed; using empty fields")
        return ExtractedFields()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _session_id(state: ChatState) -> str:
    """Session id used as the reservation's ``user_session_id``.

    Falls back to a placeholder if the app didn't put a ``session_id`` on the
    state — wire one through (via the graph config / thread_id) for real use.
    """
    return state.get("session_id") or "anonymous-session"  # type: ignore[return-value]


def _ensure_tz(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _valid_email(value: str) -> bool:
    return bool(_EMAIL_RE.match(value.strip())) and len(value.strip()) <= 255


def _clean_plate(value: str) -> Optional[str]:
    plate = value.strip()
    return plate if 0 < len(plate) <= 20 else None


def _affirmative(text: str) -> bool:
    return text.strip().lower() in _AFFIRM


def _negative(text: str) -> bool:
    return text.strip().lower() in _NEGATE


def _respond(rf: dict, text: str) -> dict:
    return {"reservation_fields": rf, "response": text}


def _merge_extracted(rf: dict, ex: ExtractedFields) -> Optional[str]:
    """Validate and merge extracted fields into ``rf``.

    Returns an error message to show the user if a supplied value was invalid,
    otherwise ``None``.
    """
    if ex.start_date_time:
        rf["start_date_time"] = _ensure_tz(ex.start_date_time)
    if ex.end_date_time:
        rf["end_date_time"] = _ensure_tz(ex.end_date_time)
    if ex.space_id is not None:
        rf["space_id"] = ex.space_id
    if ex.floor_preference is not None:
        rf["floor_preference"] = ex.floor_preference
    if ex.vehicle_plate:
        plate = _clean_plate(ex.vehicle_plate)
        if not plate:
            return "That plate doesn't look valid — what's your vehicle plate number?"
        rf["vehicle_plate"] = plate
    if ex.contact_email:
        if not _valid_email(ex.contact_email):
            return "That email doesn't look valid — what's a good contact email?"
        rf["contact_email"] = ex.contact_email.strip()
    return None


def _validate_window(rf: dict) -> Optional[str]:
    """Validate start/end once present; clears invalid values so they're re-asked."""
    start, end = rf.get("start_date_time"), rf.get("end_date_time")
    if start and start <= _now():
        rf.pop("start_date_time", None)
        return "That start time is in the past. When would you like to start?"
    if start and end and end <= start:
        rf.pop("end_date_time", None)
        return "The end time must be after the start time. When should it end?"
    if start and not db.is_open_at(start):
        rf.pop("start_date_time", None)
        return "We're closed at that time. Could you pick a time within opening hours?"
    return None


def _compute_missing(rf: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if not rf.get(f)]


def _ensure_space(rf: dict) -> Optional[str]:
    """Validate the chosen space, or auto-assign one for the window.

    Returns an error message if nothing suitable is available, else ``None``.
    """
    start, end = rf["start_date_time"], rf["end_date_time"]

    if rf.get("space_id"):
        if db.is_space_available(rf["space_id"], start, end):
            return None
        rf["space_id"] = None  # taken; fall through to auto-assign

    floor = rf.get("floor_preference")
    for space in db.get_available_spaces(floor=floor, limit=50):
        if db.is_space_available(space["id"], start, end):
            rf["space_id"] = space["id"]
            return None

    where = f" on floor {floor}" if floor else ""
    return f"Sorry, no spaces are available{where} for that time window."


def _summary(rf: dict) -> str:
    start = rf["start_date_time"].strftime("%Y-%m-%d %H:%M")
    end = rf["end_date_time"].strftime("%Y-%m-%d %H:%M")
    return (
        "Here's your reservation:\n"
        f"  • Space id: {rf['space_id']}\n"
        f"  • From: {start}\n"
        f"  • To:   {end}\n"
        f"  • Plate: {rf['vehicle_plate']}\n"
        f"  • Email: {rf['contact_email']}"
    )


# ---------------------------------------------------------------------------
# Operation handlers
# ---------------------------------------------------------------------------
def _handle_booking(state: ChatState, rf: dict, ex: ExtractedFields) -> dict:
    user_text = state.get("user_input", "")

    # A bare "yes"/"no" answers the confirmation prompt; it never carries new
    # booking details. Skip re-extraction on those turns so a hallucinated date
    # from the extractor can't overwrite already-validated start/end values
    # (which previously tripped the "end must be after start" check on confirm).
    if not (_affirmative(user_text) or _negative(user_text)):
        error = _merge_extracted(rf, ex)
        if error:
            return _respond(rf, error)

        error = _validate_window(rf)
        if error:
            return _respond(rf, error)

    missing = _compute_missing(rf)
    if missing:
        return _respond(rf, _PROMPTS[missing[0]])

    error = _ensure_space(rf)
    if error:
        return _respond(rf, error)

    # Confirmation gate — no write until the user affirms.
    if not rf.get("confirmed"):
        affirmed = ex.confirm is True or _affirmative(state.get("user_input", ""))
        if not affirmed:
            return _respond(
                rf, _summary(rf) + "\n\nShould I confirm this booking? (yes/no)"
            )
        rf["confirmed"] = True

    # Re-check availability right before writing (guard against a race).
    if not db.is_space_available(rf["space_id"], rf["start_date_time"], rf["end_date_time"]):
        rf["space_id"] = None
        rf["confirmed"] = False
        return _respond(
            rf, "That space was just taken. Let me find you another — one moment."
        )

    reservation_id = db.save_reservation(
        user_session_id=_session_id(state),
        space_id=rf["space_id"],
        start_time=rf["start_date_time"],
        end_time=rf["end_date_time"],
        vehicle_plate=rf["vehicle_plate"],
        contact_email=rf["contact_email"],
        status="confirmed",
    )
    rf["reservation_id"] = reservation_id
    return _respond(
        rf,
        f"You're booked! Space {rf['space_id']} is reserved. "
        f"Confirmation id: {reservation_id}.",
    )


def _handle_cancel(state: ChatState, rf: dict, ex: ExtractedFields) -> dict:
    active = [
        r
        for r in db.get_reservations_by_session(_session_id(state))
        if r["status"] in ("pending", "confirmed")
    ]
    if not active:
        return _respond(rf, "I couldn't find an active reservation to cancel.")

    target = active[0]  # most recent active reservation
    if not rf.get("confirmed"):
        affirmed = ex.confirm is True or _affirmative(state.get("user_input", ""))
        if not affirmed:
            start = target["start_time"].strftime("%Y-%m-%d %H:%M")
            return _respond(
                rf,
                f"You have a reservation for space {target['space_id']} starting "
                f"{start}. Cancel it? (yes/no)",
            )
        rf["confirmed"] = True

    db.update_reservation_status(str(target["id"]), "cancelled")
    rf["reservation_id"] = str(target["id"])
    return _respond(rf, "Your reservation has been cancelled.")


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------
def reservation_agent_node(state: ChatState) -> dict:
    logger.debug("reservation_agent (multi-turn): start")
    rf: dict = dict(state.get("reservation_fields") or {})
    ex = _extract(state)

    # Operation is sticky across turns once known; default to booking.
    operation = rf.get("operation") or ex.operation or "book"
    rf["operation"] = operation

    if operation == "cancel":
        return _handle_cancel(state, rf, ex)
    # "modify" is treated as a fresh booking for now (cancel separately first).
    return _handle_booking(state, rf, ex)