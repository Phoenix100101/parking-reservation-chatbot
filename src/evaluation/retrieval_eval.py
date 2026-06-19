"""Retrieval-quality metrics: Precision@k and Recall@k.

Design
------
The metric functions (:func:`precision_at_k`, :func:`recall_at_k`) are **pure**:
they take the ordered list of retrieved document ids and the set of relevant
ids and return a float. No Weaviate, no I/O — so they unit-test in isolation.

Turning a live Weaviate result into document ids and scoring a whole golden set
is the job of the ``*_doc_id`` helpers and :func:`evaluate`, which stay thin so
the integration test can drive them against the real index.

Document identity
-----------------
Weaviate assigns each object a random UUID at insert time, which is not stable
across re-seeds, so it is useless as a ground-truth label. Instead we derive a
stable *logical* id from the object's own properties:

* ``FacilityInfo``   → ``source_doc``                (e.g. ``"how_to_book"``)
* ``ParkingDetails`` → ``"{zone_name}|floor{floor}"`` (e.g. ``"Zone C|floor5"``)

The golden dataset labels relevance using these same ids.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_DATASETS_DIR = Path(__file__).resolve().parent / "test_datasets"


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------
def _relevant_hits(top_k: Sequence[str], relevant: set[str]) -> int:
    return sum(1 for doc_id in top_k if doc_id in relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-``k`` retrieved ids that are relevant.

    The denominator is the number of items actually present in the top-k
    (``min(k, len(retrieved))``), so a retriever is not penalised for returning
    fewer than ``k`` candidates. Returns ``0.0`` when nothing is retrieved.

    ``retrieved`` is assumed ordered by descending relevance and free of
    duplicates (true for Weaviate query results).
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return _relevant_hits(top_k, set(relevant)) / len(top_k)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of all relevant ids that appear in the top-``k`` retrieved.

    Raises :class:`ValueError` when ``relevant`` is empty: recall is undefined
    with no ground truth, and a golden query with no relevant docs is a data
    error worth failing loudly on.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("recall@k is undefined when no documents are relevant")
    return _relevant_hits(retrieved[:k], relevant_set) / len(relevant_set)


# ---------------------------------------------------------------------------
# Object → logical-id helpers (one per collection schema)
# ---------------------------------------------------------------------------
def facility_doc_id(properties: dict) -> str:
    """Stable id for a ``FacilityInfo`` object."""
    return str(properties["source_doc"])


def parking_doc_id(properties: dict) -> str:
    """Stable id for a ``ParkingDetails`` object (zone + floor are unique)."""
    return f"{properties['zone_name']}|floor{properties['floor']}"


def chunk_doc_id(chunk: object) -> str:
    """Logical id for one entry of the RAG node's ``retrieved_chunks``.

    The node stores each retrieved object as ``str(properties)``, so we recover
    the dict (``ast.literal_eval`` — safe, only literals) and pick the right id
    by which schema the properties came from. Lets us score the **chatbot's**
    retrieval path, not the raw vector search.
    """
    props = ast.literal_eval(chunk) if isinstance(chunk, str) else chunk
    return facility_doc_id(props) if "source_doc" in props else parking_doc_id(props)


# ---------------------------------------------------------------------------
# Golden-set evaluation harness
# ---------------------------------------------------------------------------
@dataclass
class QueryScore:
    """Per-query scores plus the raw ids, for debugging failed assertions."""

    query: str
    precision: float
    recall: float
    retrieved: list[str]
    relevant: list[str]


@dataclass
class EvalReport:
    """Aggregate scores over a golden set at a fixed ``k``."""

    k: int
    mean_precision: float
    mean_recall: float
    per_query: list[QueryScore] = field(default_factory=list)


def evaluate(
    cases: Iterable[dict],
    retrieve: Callable[..., Iterable[object]],
    id_of: Callable[[object], str],
    k: int,
) -> EvalReport:
    """Score a golden set and return mean Precision@k / Recall@k.

    Parameters
    ----------
    cases:
        Dicts with ``query`` (str) and ``relevant`` (list of ids). Any *other*
        keys are forwarded to ``retrieve`` as keyword arguments (e.g. ``floor``,
        ``category``), so a golden case can exercise the retriever's filters.
    retrieve:
        ``retrieve(query, k=..., **filters) -> objects`` where each object
        exposes ``.properties`` (i.e. a Weaviate response's ``.objects``).
    id_of:
        Maps one retrieved object to its logical id (see ``*_doc_id`` helpers).
    k:
        Cut-off rank for both metrics.
    """
    scores: list[QueryScore] = []
    for raw in cases:
        case = dict(raw)
        query = case.pop("query")
        relevant = case.pop("relevant")
        objects = retrieve(query, k=k, **case)
        retrieved_ids = [id_of(obj) for obj in objects]
        scores.append(
            QueryScore(
                query=query,
                precision=precision_at_k(retrieved_ids, relevant, k),
                recall=recall_at_k(retrieved_ids, relevant, k),
                retrieved=retrieved_ids,
                relevant=list(relevant),
            )
        )

    n = len(scores)
    if n == 0:
        return EvalReport(k=k, mean_precision=0.0, mean_recall=0.0)
    return EvalReport(
        k=k,
        mean_precision=sum(s.precision for s in scores) / n,
        mean_recall=sum(s.recall for s in scores) / n,
        per_query=scores,
    )


def load_golden(name: str = "retrieval_golden.json") -> dict:
    """Load a golden dataset shipped under ``evaluation/test_datasets/``."""
    path = _DATASETS_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))