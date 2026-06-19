"""Dynamic-query agent node — answers questions about live garage state.

The router sends "live state" questions here (current availability, today's
hours, "is it open right now?"). This node:

    1. lets the LLM pick the right read-only Postgres tool(s) for the question,
    2. executes them against :mod:`data.sql_store.postgres_client`,
    3. feeds the results back to the LLM to synthesise a grounded answer.

Only read-only tools are exposed. Writes (booking / cancelling) belong to the
reservation agent, so they are intentionally not callable from here.
"""

import json
import logging
from datetime import date, datetime, timezone

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from config.configuration import build_chat_model
from core.state import ChatState
from data.sql_store import postgres_client as db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str) -> datetime:
    """Parse an ISO 8601 timestamp; treat a trailing ``Z`` as UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_date(value: str) -> date:
    """Parse the date portion of an ISO 8601 string."""
    return date.fromisoformat(value[:10])


# ---------------------------------------------------------------------------
# Tools — thin, LLM-callable wrappers over the read-only DB helpers. Datetime
# args arrive as ISO strings (that's all the model can emit) and are parsed here.
# ---------------------------------------------------------------------------
@tool
def get_available_spaces(floor: int | None = None, limit: int = 20) -> list[dict]:
    """List parking spaces that are free right now (live ``is_available`` flag).

    Use for "which spots are open?". Pass `floor` only if the user names one.
    """
    return db.get_available_spaces(floor=floor, limit=limit)


@tool
def count_available_spaces(floor: int | None = None) -> list[dict]:
    """Count currently-free spaces grouped by floor.

    Use for "how many free spots (on floor 3)?". Omit `floor` for the whole
    garage.
    """
    return db.count_available_spaces(floor=floor)


@tool
def is_space_available(space_id: int, start_time: str, end_time: str) -> bool:
    """Check whether a specific space is bookable for a future window.

    `start_time` / `end_time` are ISO 8601 timestamps. Returns True if the space
    exists and has no overlapping active reservation.
    """
    return db.is_space_available(space_id, _parse_dt(start_time), _parse_dt(end_time))


@tool
def get_operating_hours(target_date: str) -> dict | None:
    """Get opening/closing hours for a date (ISO 8601), honouring holidays.

    Returns ``{open_time, close_time, is_holiday, is_closed, note}`` or null if
    no rule is defined for that date.
    """
    return db.get_operating_hours(_parse_date(target_date))


@tool
def get_weekly_operating_hours() -> list[dict]:
    """Get the regular weekly schedule (working days and opening/closing hours).

    Use for general "what are your working days/hours?" questions that are not
    tied to a specific date. Returns one row per defined weekday with
    ``{day_of_week, open_time, close_time, is_closed, note}`` where
    ``day_of_week`` is 0 = Sunday … 6 = Saturday.
    """
    return db.get_weekly_operating_hours()


@tool
def is_open_at(when: str) -> bool:
    """Check whether the facility is open at a given instant (ISO 8601)."""
    return db.is_open_at(_parse_dt(when))


TOOL_REGISTRY = {
    "get_available_spaces": get_available_spaces,
    "count_available_spaces": count_available_spaces,
    "is_space_available": is_space_available,
    "get_operating_hours": get_operating_hours,
    "get_weekly_operating_hours": get_weekly_operating_hours,
    "is_open_at": is_open_at,
}

# Forced tool use for the first hop: the router already decided this is a live
# query, so an answer always needs fresh data.
_planner = build_chat_model().bind_tools(
    list(TOOL_REGISTRY.values()), tool_choice="any"
)
# Plain model for turning tool results into a natural-language answer.
_responder = build_chat_model()

_PLANNER_PROMPT = """You answer live questions about a parking facility by
calling tools. The current date and time is {now} (UTC); resolve relative
expressions like "today" or "right now" against it and pass ISO 8601 values.
Pick the tool(s) that fetch exactly the data needed to answer the question."""

_RESPONDER_PROMPT = """You are a parking facility assistant. Using only the tool
results provided, answer the user's question in one or two concise, friendly
sentences. Do not invent data; if the results are empty, say so plainly. Times
are UTC. In schedule results, `day_of_week` is numeric (0 = Sunday, 1 = Monday,
… 6 = Saturday) and `is_closed = true` means the facility is closed that day;
translate these into plain day names in your answer."""


def dynamic_agent_node(state: ChatState) -> dict:
    logger.debug("dynamic_agent (Postgres): start")
    query = state.get("user_input", "")

    messages: list = [
        SystemMessage(content=_PLANNER_PROMPT.format(now=_now().isoformat())),
        HumanMessage(content=query),
    ]
    ai_msg = _planner.invoke(messages)
    messages.append(ai_msg)

    if not ai_msg.tool_calls:
        # tool_choice="any" should prevent this, but degrade gracefully.
        return {"response": "I couldn't look that up just now — could you rephrase?"}

    for call in ai_msg.tool_calls:
        tool_fn = TOOL_REGISTRY.get(call["name"])
        if tool_fn is None:
            logger.warning("unknown tool requested: %s", call["name"])
            continue
        logger.debug("tool=%s args=%s", call["name"], call["args"])
        try:
            result = tool_fn.invoke(call["args"])
            content = json.dumps(result, default=str)
        except Exception as exc:  # surface the failure to the synthesiser
            logger.exception("tool error: %s", call["name"])
            content = json.dumps({"error": str(exc)})
        messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    final = _responder.invoke(
        [SystemMessage(content=_RESPONDER_PROMPT), *messages[1:]]
    )
    response = final.content if isinstance(final, AIMessage) else str(final)

    return {"response": response}