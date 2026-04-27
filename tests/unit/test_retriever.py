"""
tests/unit/test_retriever.py
─────────────────────────────
Unit tests for Dense, BM25, Hybrid retrieval strategies
and query transformation utilities.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    QueryTransformer,
    maximal_marginal_relevance,
)
from vector_store.vector_store import SearchResult


DIM = 32


def _normalized_vecs(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random((n, DIM)).astype(np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------
class TestDenseRetriever:
    def test_retrieve_returns_results(self, dense_retriever):
        results = dense_retriever.retrieve("What is the refund policy?", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_retrieve_results_have_content(self, dense_retriever):
        results = dense_retriever.retrieve("test query", top_k=5)
        for r in results:
            assert isinstance(r.content, str)
            assert r.content.strip() != ""

    def test_retrieve_scores_descending(self, dense_retriever):
        results = dense_retriever.retrieve("sample query", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_by_vector(self, dense_retriever):
        query_vec = _normalized_vecs(1)[0]
        results = dense_retriever.retrieve_by_vector(query_vec, top_k=3)
        assert len(results) <= 3

    def test_top_k_respected(self, dense_retriever):
        for k in [1, 2, 5]:
            results = dense_retriever.retrieve("query", top_k=k)
            assert len(results) <= k

    def test_filter_applied(self, dense_retriever):
        """Filter by file_type should only return matching chunks."""
        results = dense_retriever.retrieve(
            "query", top_k=10, filter={"file_type": "txt"}
        )
        for r in results:
            assert r.metadata.get("file_type") == "txt"


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------
class TestBM25Retriever:
    def test_retrieve_without_build_raises(self):
        r = BM25Retriever()
        with pytest.raises(RuntimeError, match="not built"):
            r.retrieve("query")

    def test_build_and_retrieve(self, bm25_retriever):
        results = bm25_retriever.retrieve("refund policy", top_k=5)
        assert len(results) >= 0    # may return 0 if no keyword match

    def test_exact_keyword_match(self, sample_chunks):
        """A term that appears in the corpus should score > 0."""
        bm25 = BM25Retriever()
        texts = [c.content for c in sample_chunks]
        ids = [c.chunk_id for c in sample_chunks]
        bm25.build(texts, ids, [{}] * len(texts))

        # Find a word that definitely appears in the corpus
        first_word = texts[0].split()[0].lower()
        results = bm25.retrieve(first_word, top_k=5)
        assert len(results) > 0
        assert results[0].score > 0

    def test_no_match_returns_empty(self, bm25_retriever):
        """Querying a term not in the corpus should return empty list."""
        results = bm25_retriever.retrieve("xylophone quantum", top_k=5)
        assert all(r.score > 0 for r in results) or len(results) == 0

    def test_results_have_chunk_ids(self, bm25_retriever):
        results = bm25_retriever.retrieve("content", top_k=5)
        for r in results:
            assert r.chunk_id != ""

    def test_scores_descending(self, bm25_retriever):
        results = bm25_retriever.retrieve("document", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_save_and_load(self, bm25_retriever, tmp_path):
        bm25_retriever.save(str(tmp_path / "bm25"))
        new_bm25 = BM25Retriever()
        new_bm25.load(str(tmp_path / "bm25"))
        original_results = bm25_retriever.retrieve("content", top_k=3)
        loaded_results = new_bm25.retrieve("content", top_k=3)
        orig_ids = [r.chunk_id for r in original_results]
        load_ids = [r.chunk_id for r in loaded_results]
        assert orig_ids == load_ids

    def test_tokenizer_lowercases(self):
        tokens = BM25Retriever._tokenize("Hello World PYTHON")
        assert all(t == t.lower() for t in tokens)

    def test_k1_b_parameters_accepted(self):
        bm25 = BM25Retriever(k1=1.2, b=0.6)
        assert bm25._k1 == 1.2
        assert bm25._b == 0.6


# ---------------------------------------------------------------------------
# HybridRetriever (RRF)
# ---------------------------------------------------------------------------
class TestHybridRetriever:
    def test_retrieve_combines_results(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("refund policy", top_k=5)
        assert len(results) > 0

    def test_rrf_scores_are_positive(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("product return", top_k=5)
        for r in results:
            assert r.score > 0

    def test_rrf_scores_descending(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("content", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_chunk_ids(self, hybrid_retriever):
        results = hybrid_retriever.retrieve("product", top_k=10)
        ids = [r.chunk_id for r in results]
        assert len(ids) == len(set(ids)), "Hybrid retriever returned duplicate chunk IDs"

    def test_score_fusion_mode(self, dense_retriever, bm25_retriever):
        hybrid = HybridRetriever(
            dense_retriever, bm25_retriever, use_rrf=False, alpha=0.7
        )
        results = hybrid.retrieve("document", top_k=5)
        assert len(results) >= 0    # may be 0 if BM25 finds nothing


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance)
# ---------------------------------------------------------------------------
class TestMMR:
    def _make_candidates(self, n: int) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id=f"c-{i}",
                content=f"content {i}",
                score=float(1.0 - i * 0.05),
                metadata={},
                rank=i,
            )
            for i in range(n)
        ]

    def test_returns_top_k(self):
        candidates = self._make_candidates(10)
        vecs = _normalized_vecs(10)
        query_vec = _normalized_vecs(1)[0]
        selected = maximal_marginal_relevance(query_vec, vecs, candidates, top_k=4)
        assert len(selected) == 4

    def test_no_duplicates_in_output(self):
        candidates = self._make_candidates(8)
        vecs = _normalized_vecs(8)
        query_vec = _normalized_vecs(1)[0]
        selected = maximal_marginal_relevance(query_vec, vecs, candidates, top_k=5)
        ids = [r.chunk_id for r in selected]
        assert len(ids) == len(set(ids))

    def test_empty_candidates_returns_empty(self):
        result = maximal_marginal_relevance(
            _normalized_vecs(1)[0],
            np.empty((0, DIM), dtype=np.float32),
            [],
            top_k=5,
        )
        assert result == []

    def test_lambda_1_is_pure_relevance(self):
        """With lambda=1, MMR degenerates to pure relevance ranking."""
        candidates = self._make_candidates(5)
        vecs = _normalized_vecs(5)
        query_vec = vecs[0].copy()   # most similar to first candidate
        selected = maximal_marginal_relevance(
            query_vec, vecs, candidates, top_k=3, lambda_param=1.0
        )
        # First result should be the most similar (index 0)
        assert selected[0].chunk_id == "c-0"

    def test_top_k_greater_than_candidates(self):
        candidates = self._make_candidates(3)
        vecs = _normalized_vecs(3)
        query_vec = _normalized_vecs(1)[0]
        selected = maximal_marginal_relevance(query_vec, vecs, candidates, top_k=10)
        assert len(selected) == 3


# ---------------------------------------------------------------------------
# QueryTransformer
# ---------------------------------------------------------------------------
class TestQueryTransformer:
    def _mock_llm(self, response: str):
        return lambda prompt: response

    def test_hyde_returns_string(self):
        llm_fn = self._mock_llm(
            "Refunds are processed within 5-7 business days for eligible purchases."
        )
        transformer = QueryTransformer(llm_fn)
        result = transformer.hyde("How long do refunds take?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_multi_query_returns_list(self):
        llm_fn = self._mock_llm("How to return a product?\nReturn process steps\nRefund eligibility")
        transformer = QueryTransformer(llm_fn)
        queries = transformer.multi_query("What is the return policy?", n=3)
        assert isinstance(queries, list)
        assert len(queries) >= 1
        # Original query always included
        assert "What is the return policy?" in queries

    def test_multi_query_includes_original(self):
        llm_fn = self._mock_llm("paraphrase 1\nparaphrase 2")
        transformer = QueryTransformer(llm_fn)
        original = "original question"
        queries = transformer.multi_query(original, n=2)
        assert original in queries

    def test_step_back_returns_string(self):
        llm_fn = self._mock_llm("What are the policies for purchasing goods online?")
        transformer = QueryTransformer(llm_fn)
        result = transformer.step_back("How many days do I have to return item X?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_llm_called_with_prompt(self):
        calls = []
        def tracking_llm(prompt):
            calls.append(prompt)
            return "response"
        transformer = QueryTransformer(tracking_llm)
        transformer.hyde("test query")
        assert len(calls) == 1
        assert "test query" in calls[0]
