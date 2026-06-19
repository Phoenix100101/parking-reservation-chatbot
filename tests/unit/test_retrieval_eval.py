"""Unit tests for the pure retrieval metrics and the evaluation harness.

Written with :mod:`unittest`. They have no external dependencies (no Weaviate,
no network), so they run anywhere via either ``python -m unittest`` or ``pytest``.
"""

import unittest

from evaluation.retrieval_eval import (
    EvalReport,
    chunk_doc_id,
    evaluate,
    facility_doc_id,
    parking_doc_id,
    precision_at_k,
    recall_at_k,
)


class PrecisionAtKTests(unittest.TestCase):
    def test_all_relevant(self):
        self.assertEqual(precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3), 1.0)

    def test_none_relevant(self):
        self.assertEqual(precision_at_k(["x", "y"], {"a"}, 2), 0.0)

    def test_partial_match(self):
        # 1 of the top 2 is relevant.
        self.assertEqual(precision_at_k(["a", "x"], {"a", "b"}, 2), 0.5)

    def test_only_top_k_is_considered(self):
        # The relevant item sits at rank 3, outside k=2.
        self.assertEqual(precision_at_k(["x", "y", "a"], {"a"}, 2), 0.0)

    def test_fewer_results_than_k_does_not_dilute(self):
        # Only one result for k=3 → denominator is 1, not 3.
        self.assertEqual(precision_at_k(["a"], {"a"}, 3), 1.0)

    def test_empty_retrieved_is_zero(self):
        self.assertEqual(precision_at_k([], {"a"}, 5), 0.0)

    def test_non_positive_k_raises(self):
        with self.assertRaises(ValueError):
            precision_at_k(["a"], {"a"}, 0)


class RecallAtKTests(unittest.TestCase):
    def test_all_found(self):
        self.assertEqual(recall_at_k(["a", "b"], {"a", "b"}, 2), 1.0)

    def test_partial(self):
        self.assertEqual(recall_at_k(["a", "x"], {"a", "b"}, 2), 0.5)

    def test_relevant_beyond_k_not_counted(self):
        self.assertEqual(recall_at_k(["x", "y", "b"], {"a", "b"}, 2), 0.0)

    def test_capped_by_k(self):
        # Three relevant docs but k=2 caps how many can be found.
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 2), 2 / 3)

    def test_empty_relevant_raises(self):
        with self.assertRaises(ValueError):
            recall_at_k(["a"], set(), 2)

    def test_non_positive_k_raises(self):
        with self.assertRaises(ValueError):
            recall_at_k(["a"], {"a"}, -1)


class DocIdHelperTests(unittest.TestCase):
    def test_facility_doc_id(self):
        self.assertEqual(facility_doc_id({"source_doc": "how_to_book"}), "how_to_book")

    def test_parking_doc_id(self):
        self.assertEqual(
            parking_doc_id({"zone_name": "Zone C", "floor": 5}), "Zone C|floor5"
        )

    def test_chunk_doc_id_parses_facility_chunk(self):
        # The RAG node stores each retrieved object as str(properties).
        chunk = str({"description": "...", "source_doc": "how_to_book"})
        self.assertEqual(chunk_doc_id(chunk), "how_to_book")

    def test_chunk_doc_id_parses_parking_chunk(self):
        chunk = str({"zone_name": "Zone C", "floor": 5, "description": "..."})
        self.assertEqual(chunk_doc_id(chunk), "Zone C|floor5")


class _Obj:
    """Minimal stand-in for a Weaviate object (only ``.properties`` is used)."""

    def __init__(self, properties):
        self.properties = properties


class EvaluateTests(unittest.TestCase):
    def test_aggregates_mean_over_queries(self):
        index = {
            "q1": [_Obj({"id": "a"}), _Obj({"id": "b"})],  # relevant {a} → P=0.5 R=1.0
            "q2": [_Obj({"id": "x"}), _Obj({"id": "y"})],  # relevant {z} → P=0.0 R=0.0
        }
        cases = [
            {"query": "q1", "relevant": ["a"]},
            {"query": "q2", "relevant": ["z"]},
        ]
        report = evaluate(
            cases,
            retrieve=lambda q, k, **kw: index[q][:k],
            id_of=lambda o: o.properties["id"],
            k=2,
        )
        self.assertIsInstance(report, EvalReport)
        self.assertEqual(report.k, 2)
        self.assertAlmostEqual(report.mean_precision, 0.25)
        self.assertAlmostEqual(report.mean_recall, 0.5)
        self.assertEqual(len(report.per_query), 2)

    def test_forwards_filter_kwargs_to_retriever(self):
        captured = {}

        def retrieve(query, k, **kwargs):
            captured.update(kwargs)
            return [_Obj({"id": "a"})]

        evaluate(
            [{"query": "q", "relevant": ["a"], "floor": 5, "zone_name": "Zone C"}],
            retrieve=retrieve,
            id_of=lambda o: o.properties["id"],
            k=1,
        )
        self.assertEqual(captured, {"floor": 5, "zone_name": "Zone C"})

    def test_empty_case_set_returns_zeroed_report(self):
        report = evaluate(
            [], retrieve=lambda q, k, **kw: [], id_of=lambda o: "", k=3
        )
        self.assertEqual((report.mean_precision, report.mean_recall), (0.0, 0.0))
        self.assertEqual(report.per_query, [])


if __name__ == "__main__":
    unittest.main()