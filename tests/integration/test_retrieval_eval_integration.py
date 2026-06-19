"""Integration tests for the chatbot's retrieval quality.

These score Recall@k / Precision@k over the bot's **actual retrieval path** —
``core.nodes.retrieval_node.rag_agent_node`` — not the raw vector search. The
node has the LLM pick a tool (facility vs. parking) and its filters, then runs
the search; this test therefore also catches tool-selection / filter mistakes,
which is what really degrades answers for users.

Requires a running, **seeded** Weaviate and a valid ``OPENAI_API_KEY``. When
that environment is unavailable the whole module skips instead of failing.

Run only these::

    uv run pytest -m integration

Skip them in a fast unit run::

    uv run pytest -m "not integration"
"""

import pytest

from evaluation.retrieval_eval import chunk_doc_id, evaluate, load_golden

pytestmark = pytest.mark.integration

# Cut-off and the regression floor for the bot's retrieval path. This is a
# *guard against regressions*, not an aspirational target: the bot currently
# measures ~0.65 mean recall@3 — below the raw index (~0.8+) because the LLM
# tool-selection / filtering / k=3 layer drops some relevant docs. Raise this
# as the retrieval path improves; treat a drop below it as a regression.
K = 5
MIN_MEAN_RECALL = 0.65


@pytest.fixture(scope="module")
def rag_node():
    """Import the bot's RAG node; skip if its deps (Weaviate/key) aren't ready."""
    try:
        from data.vector_store import weaviate_client as wc

        if not wc.get_weaviate_client().is_connected():
            pytest.skip("Weaviate is not reachable")
        from core.nodes.retrieval_node import rag_agent_node
    except Exception as exc:  # missing key, no server, etc.
        pytest.skip(f"chatbot retrieval path is not available: {exc}")
    return rag_agent_node


@pytest.fixture(scope="module")
def golden_cases():
    # Bot auto-routes between collections, so we evaluate every query together.
    data = load_golden()
    return [*data["facility_info"], *data["parking_details"]]


def _retrieve_via_bot(rag_node):
    """Adapt the node to the evaluate() retriever contract.

    Returns the chunks the bot actually retrieved for a query; ``k`` and any
    filter kwargs are ignored because the bot decides those itself.
    """

    def retrieve(query, k, **_):
        result = rag_node({"user_input": query})
        return result.get("retrieved_chunks", [])

    return retrieve


def test_chatbot_retrieval_recall_at_k(rag_node, golden_cases):
    report = evaluate(
        golden_cases,
        retrieve=_retrieve_via_bot(rag_node),
        id_of=chunk_doc_id,
        k=K,
    )
    # per_query is attached to the message so a failure shows which query missed.
    assert report.mean_recall >= MIN_MEAN_RECALL, report.per_query