import logging

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from config.configuration import build_chat_model, get_settings
from core.state import ChatState

logger = logging.getLogger(__name__)

# Lighter model with a touch of temperature for friendlier small-talk replies.
_model = build_chat_model(model=get_settings().openai.mini_model, temperature=0.3)

_SYSTEM_PROMPT = """You are the assistant of a parking facility chatbot.
The user's message is OUT OF SCOPE — it is not a parking facility request.

Handle it in ONE short, friendly reply:
- If the message is a greeting or small talk (e.g. "hello", "hi", "hey",
  "good morning", "thanks"), greet them back warmly and introduce yourself as
  the parking reservation chatbot. Do NOT apologize or decline.
- Otherwise (an off-topic question or unrelated request), politely decline in
  one short sentence WITHOUT apologizing harshly.
- In both cases, briefly mention what you CAN help with: facility info (hours,
  zones, amenities), live availability, and booking/canceling/modifying a
  parking spot.
- Reply in the SAME language as the user's message ONLY when that language is
  unambiguous. If the message is too short, ambiguous, or you're not confident
  about the language (e.g. a bare "Hello"), DEFAULT TO ENGLISH.
- Never invent facts about the facility. Do not answer the off-topic question
  even partially.

Example for a greeting like "hello":
"Hello, I'm the parking reservation chatbot. I can help you with facility
information, live availability, and booking, canceling, or modifying a parking
spot."
"""

# Детерминированный fallback на случай сбоя LLM.
_FALLBACK = "Sorry, I can only help with parking facility questions."


def out_of_scope_node(state: ChatState) -> dict:
    logger.debug("out_of_scope: start")
    user_input = state.get("user_input", "")

    try:
        ai = _model.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ]
        )
        response = ai.content if isinstance(ai, AIMessage) else str(ai)
        response = response.strip() or _FALLBACK
    except Exception:
        logger.exception("llm error; using fallback reply")
        response = _FALLBACK

    return {"response": response}