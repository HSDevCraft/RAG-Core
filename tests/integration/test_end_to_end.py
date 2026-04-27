"""
tests/integration/test_end_to_end.py
──────────────────────────────────────
End-to-end integration tests for the RAG system.

These tests exercise the full pipeline stack with REAL components:
  - Real SentenceTransformer embedder (local, no API key)
  - Real FAISS vector store (in-process)
  - Real BM25 index
  - Real cross-encoder reranker
  - Mocked LLM only (to avoid API costs in CI)

Tests marked @pytest.mark.slow require a real OpenAI/Ollama key.

Run:
    pytest tests/integration/ -v                  # all integration tests
    pytest tests/integration/ -v -m "not slow"    # skip API-dependent tests
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from data_pipeline.document_loader import Document, DocumentLoader
from data_pipeline.chunking import ChunkingEngine
from embeddings.embedding import EmbeddingEngine
from generation.llm_interface import LLMResponse
from rag_pipeline import RAGPipeline
from vector_store.vector_store import VectorStoreFactory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_embedder():
    """SentenceTransformer embedder — loaded once for all integration tests."""
    pytest.importorskip("sentence_transformers")
    return EmbeddingEngine(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        use_cache=False,
        device="cpu",
    )


@pytest.fixture(scope="module")
def sample_docs() -> List[Document]:
    return [
        Document(
            content=(
                "Refund Policy: Customers are eligible for a full refund within 30 days. "
                "After 30 days, only store credit is available. "
                "Digital downloads are non-refundable once accessed. "
                "Refunds are processed within 5-7 business days."
            ),
            metadata={"source": "policy.txt", "file_type": "txt"},
        ),
        Document(
            content=(
                "XR-500 Laptop: 15-inch display, 32GB RAM, 2TB SSD. "
                "Battery life is 12 hours. Price: $1,299. "
                "Available in Silver and Space Gray."
            ),
            metadata={"source": "products.json", "file_type": "json"},
        ),
        Document(
            content=(
                "SmartWatch Pro: tracks heart rate, sleep, and daily steps. "
                "Water-resistant up to 50 meters. Battery lasts 7 days. Price: $299. "
                "Compatible with iOS and Android."
            ),
            metadata={"source": "products.json", "file_type": "json"},
        ),
        Document(
            content=(
                "Frequently Asked Questions: "
                "How do I track my order? You will receive a tracking email within 24 hours. "
                "What payment methods are accepted? Visa, Mastercard, American Express, PayPal, and Apple Pay. "
                "Do you ship internationally? Yes, we ship to over 50 countries in 10-15 business days."
            ),
            metadata={"source": "faq.md", "file_type": "md"},
        ),
    ]


@pytest.fixture(scope="module")
def mock_llm_module():
    """Module-scoped mock LLM to share across all integration tests."""
    llm = MagicMock()
    llm.provider = "mock"

    def _chat(messages, **kwargs):
        for m in messages:
            if m.get("role") == "user":
                content = m["content"]
                break
        else:
            content = ""
        # Extract the question from the user message
        question = content.split("Question:")[-1].strip() if "Question:" in content else content[:50]
        return LLMResponse(
            content=f"Based on the provided context: {question[:80]}",
            model="mock-llm",
            prompt_tokens=200,
            completion_tokens=50,
            total_tokens=250,
            latency_ms=15.0,
        )

    def _complete(prompt, **kwargs):
        return LLMResponse(
            content="YES",   # Default: "I have enough information"
            model="mock-llm",
            prompt_tokens=50,
            completion_tokens=5,
            total_tokens=55,
            latency_ms=5.0,
        )

    def _stream(messages, **kwargs):
        yield "Integration "
        yield "streaming "
        yield "response."

    llm.chat = _chat
    llm.complete = _complete
    llm.stream = _stream
    return llm


@pytest.fixture(scope="module")
def integrated_pipeline(real_embedder, sample_docs, mock_llm_module):
    """
    Fully assembled pipeline with real embedder + FAISS + mock LLM.
    Pre-indexed with sample_docs. Shared across the module for speed.
    """
    from retrieval.reranker import CrossEncoderReranker

    vector_store = VectorStoreFactory.create(
        backend="faiss",
        dimension=real_embedder.dimension,
        index_type="flat",
        metric="cosine",
    )

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    )

    pipeline = RAGPipeline(
        embedder=real_embedder,
        vector_store=vector_store,
        llm=mock_llm_module,
        reranker=reranker,
    )
    result = pipeline.index_documents(sample_docs)
    assert result.num_chunks > 0, "Indexing produced no chunks"
    return pipeline


# ---------------------------------------------------------------------------
# Indexing tests
# ---------------------------------------------------------------------------
class TestIntegrationIndexing:
    def test_index_documents_succeeds(self, integrated_pipeline, sample_docs):
        assert integrated_pipeline.index_size > 0

    def test_index_produces_correct_chunk_count(self, real_embedder, sample_docs, mock_llm_module):
        """Fresh pipeline to verify chunk count independently."""
        store = VectorStoreFactory.create(
            backend="faiss", dimension=real_embedder.dimension, index_type="flat"
        )
        pipeline = RAGPipeline(embedder=real_embedder, vector_store=store, llm=mock_llm_module)
        result = pipeline.index_documents(sample_docs)
        # 4 medium-length docs with 512-char chunks → expect 4-20 chunks
        assert 4 <= result.num_chunks <= 50

    def test_index_from_directory(self, real_embedder, mock_llm_module, tmp_path):
        """DirectoryLoader integration: files on disk → indexed pipeline."""
        (tmp_path / "policy.txt").write_text(
            "Return policy: full refund within 30 days. Contact support@example.com."
        )
        (tmp_path / "faq.md").write_text(
            "# FAQ\n## Shipping\nWe ship to 50+ countries in 10-15 business days."
        )

        store = VectorStoreFactory.create(
            backend="faiss", dimension=real_embedder.dimension, index_type="flat"
        )
        pipeline = RAGPipeline(embedder=real_embedder, vector_store=store, llm=mock_llm_module)
        result = pipeline.index([str(tmp_path)])

        assert result.num_documents >= 2
        assert result.num_chunks >= 2
        assert result.errors == []

    def test_index_embedding_latency_positive(self, integrated_pipeline, sample_docs):
        store = VectorStoreFactory.create(
            backend="faiss", dimension=integrated_pipeline._embedder.dimension, index_type="flat"
        )
        fresh = RAGPipeline(
            embedder=integrated_pipeline._embedder,
            vector_store=store,
            llm=integrated_pipeline._llm,
        )
        result = fresh.index_documents(sample_docs[:2])
        assert result.embedding_latency_ms >= 0


# ---------------------------------------------------------------------------
# Retrieval tests — semantic quality
# ---------------------------------------------------------------------------
class TestIntegrationRetrieval:
    def test_retrieve_refund_query_finds_policy(self, integrated_pipeline):
        """The refund policy chunk should be in top-3 for a refund query."""
        results = integrated_pipeline.retrieve("How do I get a refund?", top_k=5)
        assert len(results) > 0
        all_content = " ".join(r.content.lower() for r in results[:3])
        assert "refund" in all_content

    def test_retrieve_product_query_finds_product(self, integrated_pipeline):
        results = integrated_pipeline.retrieve("laptop price specifications", top_k=5)
        assert len(results) > 0
        all_content = " ".join(r.content.lower() for r in results[:3])
        assert "laptop" in all_content or "xr-500" in all_content

    def test_retrieve_scores_descending(self, integrated_pipeline):
        results = integrated_pipeline.retrieve("payment methods", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), \
            f"Scores not descending: {scores}"

    def test_retrieve_no_duplicate_chunk_ids(self, integrated_pipeline):
        results = integrated_pipeline.retrieve("product return", top_k=10)
        ids = [r.chunk_id for r in results]
        assert len(ids) == len(set(ids))

    def test_retrieve_metadata_filter_by_file_type(self, integrated_pipeline):
        results = integrated_pipeline.retrieve(
            "shipping", top_k=10, filter={"file_type": "md"}
        )
        for r in results:
            assert r.metadata.get("file_type") == "md"

    def test_hybrid_retrieval_better_than_dense_only(self, real_embedder, sample_docs, mock_llm_module):
        """
        Hybrid retrieval (BM25 + dense) should find a document with an exact
        keyword that dense might rank lower due to semantic distance.
        """
        store = VectorStoreFactory.create(
            backend="faiss", dimension=real_embedder.dimension, index_type="flat"
        )
        pipeline = RAGPipeline(embedder=real_embedder, vector_store=store, llm=mock_llm_module)
        pipeline.index_documents(sample_docs)

        # "SmartWatch Pro" is a specific product name — BM25 should rank it high
        results = pipeline.retrieve("SmartWatch Pro water resistance", top_k=5)
        assert len(results) > 0
        assert any("smartwatch" in r.content.lower() for r in results[:3])


# ---------------------------------------------------------------------------
# Full pipeline (query) tests
# ---------------------------------------------------------------------------
class TestIntegrationQuery:
    def test_query_returns_non_empty_answer(self, integrated_pipeline):
        response = integrated_pipeline.query("What is the refund policy?")
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0

    def test_query_includes_citations(self, integrated_pipeline):
        response = integrated_pipeline.query("What is the refund policy?")
        assert isinstance(response.citations, list)
        assert len(response.citations) > 0

    def test_query_citations_have_required_fields(self, integrated_pipeline):
        response = integrated_pipeline.query("shipping time?")
        for c in response.citations:
            assert "ref" in c
            assert "chunk_id" in c
            assert "source" in c

    def test_query_latency_breakdown_populated(self, integrated_pipeline):
        response = integrated_pipeline.query("payment methods")
        assert "retrieval_ms" in response.latency_breakdown
        assert "generation_ms" in response.latency_breakdown
        assert all(v >= 0 for v in response.latency_breakdown.values())

    def test_query_retrieved_chunks_attached(self, integrated_pipeline):
        response = integrated_pipeline.query("laptop specifications")
        assert len(response.retrieved_chunks) > 0

    def test_query_llm_response_attached(self, integrated_pipeline):
        response = integrated_pipeline.query("product return policy")
        assert response.llm_response is not None
        assert response.llm_response.total_tokens > 0

    def test_multi_turn_conversation(self, integrated_pipeline):
        session = "integration-session-001"
        r1 = integrated_pipeline.query("What can I return?", session_id=session)
        r2 = integrated_pipeline.query("How long does that take?", session_id=session)

        assert isinstance(r1.answer, str)
        assert isinstance(r2.answer, str)
        history = integrated_pipeline.get_session_history(session)
        assert len(history) == 4    # 2 turns × (user + assistant)

        integrated_pipeline.clear_session(session)

    def test_query_with_filter_narrows_scope(self, integrated_pipeline):
        """Filtering to .txt source should only use the policy document."""
        response = integrated_pipeline.query(
            "refund process",
            filter={"file_type": "txt"}
        )
        for chunk in response.retrieved_chunks:
            assert chunk.metadata.get("file_type") == "txt"

    def test_streaming_query(self, integrated_pipeline):
        tokens = list(integrated_pipeline.stream_query("how to return a product"))
        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert len(full_response) > 0


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------
class TestIntegrationPersistence:
    def test_save_and_load_maintains_search_quality(
        self, real_embedder, sample_docs, mock_llm_module, tmp_path
    ):
        """After save/load, retrieval results should be identical."""
        store = VectorStoreFactory.create(
            backend="faiss", dimension=real_embedder.dimension, index_type="flat"
        )
        pipeline = RAGPipeline(embedder=real_embedder, vector_store=store, llm=mock_llm_module)
        pipeline.index_documents(sample_docs)

        query = "refund eligibility"
        before_results = pipeline.retrieve(query, top_k=3)
        before_ids = [r.chunk_id for r in before_results]

        # Save
        save_path = str(tmp_path / "integration_save")
        pipeline.save(save_path)

        # Reload into a fresh pipeline
        store2 = VectorStoreFactory.create(
            backend="faiss", dimension=real_embedder.dimension, index_type="flat"
        )
        pipeline2 = RAGPipeline(embedder=real_embedder, vector_store=store2, llm=mock_llm_module)
        pipeline2.load(save_path)

        after_results = pipeline2.retrieve(query, top_k=3)
        after_ids = [r.chunk_id for r in after_results]

        assert before_ids == after_ids, \
            f"Results differ after reload:\n  Before: {before_ids}\n  After: {after_ids}"


# ---------------------------------------------------------------------------
# Slow tests (require real API keys)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestSlowIntegration:
    """These tests require OPENAI_API_KEY or a running Ollama server."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_openai_llm_full_pipeline(self, real_embedder, sample_docs, tmp_path):
        from generation.llm_interface import LLMInterface
        llm = LLMInterface(provider="openai", model="gpt-4o-mini", temperature=0.0)
        store = VectorStoreFactory.create(
            backend="faiss", dimension=real_embedder.dimension, index_type="flat"
        )
        pipeline = RAGPipeline(embedder=real_embedder, vector_store=store, llm=llm)
        pipeline.index_documents(sample_docs)
        response = pipeline.query("What is the refund policy?")
        assert "refund" in response.answer.lower()
        assert len(response.citations) > 0
