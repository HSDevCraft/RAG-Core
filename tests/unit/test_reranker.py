"""
tests/unit/test_reranker.py
────────────────────────────
Unit tests for CrossEncoderReranker, CohereReranker, and LLMReranker.
All external API and model calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from retrieval.reranker import CrossEncoderReranker, LLMReranker
from vector_store.vector_store import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_results(n: int = 5, base_score: float = 0.9) -> list[SearchResult]:
    return [
        SearchResult(
            chunk_id=f"chunk-{i}",
            content=f"Document {i} about refund policies and product returns.",
            score=base_score - i * 0.05,
            metadata={"source": f"doc_{i}.pdf"},
            rank=i,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------
class TestCrossEncoderReranker:
    def _make_reranker_with_mock_model(self, scores: list[float]) -> CrossEncoderReranker:
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = scores
        reranker._batch_size = 16
        reranker._model_name = "mock-cross-encoder"
        return reranker

    def test_rerank_returns_correct_count(self):
        results = _make_results(5)
        reranker = self._make_reranker_with_mock_model([0.9, 0.3, 0.7, 0.5, 0.1])
        reranked = reranker.rerank("refund policy", results)
        assert len(reranked) == 5

    def test_rerank_sorted_by_score_descending(self):
        results = _make_results(4)
        scores = [0.2, 0.9, 0.5, 0.7]
        reranker = self._make_reranker_with_mock_model(scores)
        reranked = reranker.rerank("query", results)
        returned_scores = [r.score for r in reranked]
        assert returned_scores == sorted(returned_scores, reverse=True)

    def test_rerank_best_result_has_rank_zero(self):
        results = _make_results(3)
        scores = [0.3, 0.8, 0.5]
        reranker = self._make_reranker_with_mock_model(scores)
        reranked = reranker.rerank("query", results)
        assert reranked[0].rank == 0

    def test_rerank_updates_ranks_sequentially(self):
        results = _make_results(4)
        scores = [0.4, 0.9, 0.2, 0.6]
        reranker = self._make_reranker_with_mock_model(scores)
        reranked = reranker.rerank("query", results)
        for i, r in enumerate(reranked):
            assert r.rank == i

    def test_rerank_preserves_chunk_ids(self):
        results = _make_results(3)
        original_ids = {r.chunk_id for r in results}
        reranker = self._make_reranker_with_mock_model([0.5, 0.9, 0.3])
        reranked = reranker.rerank("query", results)
        reranked_ids = {r.chunk_id for r in reranked}
        assert original_ids == reranked_ids

    def test_rerank_empty_results_returns_empty(self):
        reranker = self._make_reranker_with_mock_model([])
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_single_result(self):
        results = _make_results(1)
        reranker = self._make_reranker_with_mock_model([0.75])
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 1
        assert reranked[0].score == pytest.approx(0.75)

    def test_rerank_top_k_truncates(self):
        results = _make_results(10)
        scores = list(range(10, 0, -1))
        reranker = self._make_reranker_with_mock_model([float(s) for s in scores])
        reranked = reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3

    def test_model_receives_query_document_pairs(self):
        results = _make_results(3)
        reranker = self._make_reranker_with_mock_model([0.5, 0.7, 0.3])
        query = "specific query"
        reranker.rerank(query, results)
        call_args = reranker._model.predict.call_args[0][0]
        # Should be list of (query, document_content) pairs
        assert len(call_args) == 3
        for pair in call_args:
            assert pair[0] == query
            assert isinstance(pair[1], str)
            assert len(pair[1]) > 0

    def test_model_name_property(self):
        reranker = self._make_reranker_with_mock_model([])
        assert reranker.model_name == "mock-cross-encoder"

    def test_batch_size_limits_predict_calls(self):
        """With batch_size=2 and 5 candidates, predict should be called 3 times."""
        results = _make_results(5)
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = MagicMock()
        reranker._model.predict.side_effect = [
            [0.5, 0.8],    # batch 1
            [0.3, 0.9],    # batch 2
            [0.1],          # batch 3
        ]
        reranker._batch_size = 2
        reranker._model_name = "mock"
        reranked = reranker.rerank("query", results)
        assert reranker._model.predict.call_count == 3
        assert len(reranked) == 5


# ---------------------------------------------------------------------------
# LLMReranker
# ---------------------------------------------------------------------------
class TestLLMReranker:
    def _make_llm_reranker(self, llm_responses: list[str]) -> "LLMReranker":
        """Create an LLMReranker with a mock LLM that returns given responses."""
        reranker = LLMReranker.__new__(LLMReranker)

        responses = iter(llm_responses)

        def _complete(prompt, **kwargs):
            try:
                content = next(responses)
            except StopIteration:
                content = "5"
            return MagicMock(content=content)

        mock_llm = MagicMock()
        mock_llm.complete = _complete
        reranker._llm = mock_llm
        reranker._model_name = "mock-llm-reranker"
        return reranker

    def test_rerank_returns_results(self):
        results = _make_results(3)
        # LLM rates each doc 8, 5, 9 out of 10
        reranker = self._make_llm_reranker(["8", "5", "9"])
        reranked = reranker.rerank("refund policy", results)
        assert len(reranked) == 3

    def test_rerank_sorted_by_llm_score(self):
        results = _make_results(3)
        reranker = self._make_llm_reranker(["3", "9", "6"])
        reranked = reranker.rerank("query", results)
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_handles_non_numeric_response(self):
        """If LLM returns garbage, reranker should not crash."""
        results = _make_results(2)
        reranker = self._make_llm_reranker(["not a number", "8"])
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 2

    def test_rerank_empty_results_returns_empty(self):
        reranker = self._make_llm_reranker([])
        assert reranker.rerank("query", []) == []

    def test_rerank_preserves_content(self):
        results = _make_results(2)
        reranker = self._make_llm_reranker(["7", "4"])
        reranked = reranker.rerank("query", results)
        contents = {r.content for r in reranked}
        originals = {r.content for r in results}
        assert contents == originals


# ---------------------------------------------------------------------------
# CohereReranker
# ---------------------------------------------------------------------------
class TestCohereReranker:
    def test_rerank_calls_cohere_api(self):
        pytest.importorskip("cohere")
        from retrieval.reranker import CohereReranker

        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(index=1, relevance_score=0.95),
            MagicMock(index=0, relevance_score=0.72),
        ]

        with patch("cohere.Client") as MockCohere:
            client_instance = MockCohere.return_value
            client_instance.rerank.return_value = mock_result

            reranker = CohereReranker(api_key="test-key")
            results = _make_results(2)
            reranked = reranker.rerank("query", results)

        assert len(reranked) == 2
        assert reranked[0].score == pytest.approx(0.95)
        assert reranked[1].score == pytest.approx(0.72)

    def test_rerank_chunk_id_mapping_preserved(self):
        pytest.importorskip("cohere")
        from retrieval.reranker import CohereReranker

        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(index=2, relevance_score=0.9),
            MagicMock(index=0, relevance_score=0.5),
            MagicMock(index=1, relevance_score=0.3),
        ]

        with patch("cohere.Client") as MockCohere:
            client_instance = MockCohere.return_value
            client_instance.rerank.return_value = mock_result

            reranker = CohereReranker(api_key="test-key")
            results = _make_results(3)
            reranked = reranker.rerank("query", results)

        # Result at index 2 (score 0.9) should be first
        assert reranked[0].chunk_id == "chunk-2"
        assert reranked[1].chunk_id == "chunk-0"
        assert reranked[2].chunk_id == "chunk-1"
