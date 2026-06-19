import json
import logging

from config.configuration import build_chat_model
from core.state import ChatState
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from data.vector_store.weaviate_client import (
    search_facility_info,
    search_parking_details,
)

logger = logging.getLogger(__name__)


@tool
def search_facility_info_tool(
    query: str, category: str | None = None, k: int = 3
) -> list[dict]:
    """Search general facility information: opening hours, amenities, contact details,
    policies, payment methods, how the booking process works.

    Use this tool when the user's question is NOT specifically about parking spots,
    zones, or floors. Pass `category` only if the user explicitly narrows the topic
    (e.g. "hours", "payment").
    """
    response = search_facility_info(query=query, category=category, k=k)
    return [obj.properties for obj in response.objects]


@tool
def search_parking_details_tool(
    query: str,
    floor: int | None = None,
    zone_name: str | None = None,
    k: int = 3,
) -> list[dict]:
    """Search parking-specific details: zones, floors, spot types, pricing per zone,
    accessibility, EV charging spots.

    Use this tool when the user asks specifically about parking spots, zones, or
    floors. Extract `floor` (integer) and `zone_name` (string) only if the user
    explicitly mentions them.
    """
    response = search_parking_details(
        query=query, floor=floor, zone_name=zone_name, k=k
    )
    return [obj.properties for obj in response.objects]


TOOL_REGISTRY = {
    "search_facility_info_tool": search_facility_info_tool,
    "search_parking_details_tool": search_parking_details_tool,
}

# Forced tool use: pick and run a retrieval tool for the question.
_model = build_chat_model().bind_tools(
    list(TOOL_REGISTRY.values()), tool_choice="any"
)
# Plain model that turns the retrieved chunks into a natural-language answer.
_responder = build_chat_model()

SYSTEM_PROMPT = """You are a retrieval agent for a parking facility chatbot.

Pick exactly one tool to answer the user's question:
- `search_parking_details_tool` for questions about parking zones, floors, spot
  types, EV charging, or accessibility.
- `search_facility_info_tool` for general facility questions (hours, amenities,
  payment, policies, how booking works).

Extract filter arguments (category, floor, zone_name) only when the user
explicitly mentions them. Always pass the user's question as `query`.
"""

_RESPONDER_PROMPT = """You are a parking facility assistant. Using only the
retrieved information in the tool results, answer the user's question in one or
two concise, friendly sentences. Do not invent facts; if nothing relevant was
retrieved, say you don't have that information."""


def rag_agent_node(state: ChatState) -> dict:
    logger.debug("rag_agent (Weaviate): start")
    query = state.get("user_input", "")

    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    ai_msg = _model.invoke(messages)
    messages.append(ai_msg)

    retrieved_chunks: list[str] = []
    # Every tool_call must get a matching ToolMessage, or the follow-up LLM call
    # rejects the conversation.
    for call in ai_msg.tool_calls:
        tool_fn = TOOL_REGISTRY.get(call["name"])
        if tool_fn is None:
            logger.warning("unknown tool requested: %s", call["name"])
            messages.append(ToolMessage(content="[]", tool_call_id=call["id"]))
            continue
        logger.debug("tool=%s args=%s", call["name"], call["args"])
        try:
            result = tool_fn.invoke(call["args"])
            retrieved_chunks.extend(str(item) for item in result)
            content = json.dumps(result, default=str)
        except Exception as exc:  # surface the failure to the synthesiser
            logger.exception("tool error: %s", call["name"])
            content = json.dumps({"error": str(exc)})
        messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    final = _responder.invoke(
        [SystemMessage(content=_RESPONDER_PROMPT), *messages[1:]]
    )
    response = final.content if isinstance(final, AIMessage) else str(final)

    return {
        "retrieved_chunks": retrieved_chunks,
        "response": response,
    }