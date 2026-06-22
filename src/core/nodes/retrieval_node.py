"""RAG retrieval node — answers facility/parking questions from Weaviate.

A parking chatbot has only two knowledge indexes (``FacilityInfo`` and
``ParkingDetails``), so instead of having the LLM pick one via tool-calling we
simply query **both** collections and let the responder ground its answer on
the union. This removes the tool-routing failure mode (picking the wrong index
or over-filtering on floor/zone, which dropped relevant chunks) and one LLM
round-trip, at the cost of a couple of cheap extra vector hits.
"""

import json
import logging
from itertools import chain, zip_longest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.configuration import build_chat_model
from core.state import ChatState
from data.vector_store.weaviate_client import (
    search_facility_info,
    search_parking_details,
)

logger = logging.getLogger(__name__)

# How many chunks to pull from each collection.
_K_PER_COLLECTION = 3

# Plain model that turns the retrieved chunks into a natural-language answer.
_responder = build_chat_model()

_RESPONDER_PROMPT = """You are a parking facility assistant. Using only the
retrieved information provided, answer the user's question in one or two concise,
friendly sentences. The retrieved items may include entries unrelated to the
question — ignore those. Do not invent facts; if nothing relevant was retrieved,
say you don't have that information."""


def _search_facility(query: str) -> list[dict]:
    response = search_facility_info(query=query, k=_K_PER_COLLECTION)
    return [obj.properties for obj in response.objects]


def _search_parking(query: str) -> list[dict]:
    response = search_parking_details(query=query, k=_K_PER_COLLECTION)
    return [obj.properties for obj in response.objects]


def _retrieve(query: str) -> list[dict]:
    """Query both collections and merge their hits, interleaved round-robin.

    Each collection is queried independently so a failure in one (e.g. a schema
    issue) still lets the other contribute. Hybrid scores aren't comparable
    across collections, so instead of a global ranking we interleave the two
    ranked lists (facility[0], parking[0], facility[1], …). That keeps both
    collections represented in any top-k prefix, rather than burying the second
    collection's hits after the first's. The two searches are independent and
    could be parallelised, but the dominant cost here is the responder LLM call.
    """
    per_collection: list[list[dict]] = []
    for name, search in (("facility", _search_facility), ("parking", _search_parking)):
        try:
            hits = search(query)
            logger.debug("retrieved %d chunks from %s", len(hits), name)
            per_collection.append(hits)
        except Exception:  # one collection failing must not sink the whole turn
            logger.exception("retrieval failed for %s collection", name)

    # Round-robin interleave, dropping the padding zip_longest inserts for the
    # shorter list.
    interleaved = chain.from_iterable(zip_longest(*per_collection))
    return [hit for hit in interleaved if hit is not None]


def rag_agent_node(state: ChatState) -> dict:
    logger.debug("rag_agent (Weaviate): start")
    query = state.get("user_input", "")

    retrieved = _retrieve(query)
    # str(properties) keeps the same chunk format the evaluation harness expects.
    retrieved_chunks = [str(item) for item in retrieved]

    context = json.dumps(retrieved, default=str)
    final = _responder.invoke(
        [
            SystemMessage(content=_RESPONDER_PROMPT),
            HumanMessage(
                content=f"Question: {query}\n\nRetrieved information:\n{context}"
            ),
        ]
    )
    response = final.content if isinstance(final, AIMessage) else str(final)

    return {
        "retrieved_chunks": retrieved_chunks,
        "response": response,
    }