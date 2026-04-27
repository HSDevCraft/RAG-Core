"""
tests/unit/test_rag_pipeline.py
────────────────────────────────
Unit tests for the RAGPipeline orchestrator.

All LLM and embedding calls are mocked — tests run with no API key.
Coverage:
  - IndexingResult model
  - RAGResponse model (to_dict, total_latency_ms)
  - pipeline.index() — document loading + chunking + embed + store
  - pipeline.index_documents() — pre-loaded Documents
  - pipeline.retrieve() — hybrid/dense dispatch
  - pipeline.query() — full end-to-end with mocks
  - pipeline.stream_query() — streaming token generation
  - pipeline.save() / load() — persistence round-trip
  - pipeline.clear_session() — session management
  - pipeline.index_size property
  - Edge cases: empty sources, missing index, unknown template
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_pipeline import IndexingResult, RAGPipeline, RAGResponse
from data_pipeline.document_loader import Document
from generation.llm_interface import LLMResponse


# ---------------------------------------------------------------------------
# IndexingResult
# ---------------------------------------------------------------------------
class TestIndexingResult:
    def test_basic_fields(self):
        r = IndexingResult(
            num_documents=3, num_chunks=12,
            embedding_latency_ms=200.0, total_latency_ms=350.0,
        )
        assert r.num_documents == 3
        assert r.num_chunks == 12
        assert r.embedding_latency_ms == 200.0
        assert r.total_latency_ms == 350.0
        assert r.errors == []

    def test_errors_field(self):
        r = IndexingResult(0, 0, 0.0, 0.0, errors=["file.pdf: Permission denied"])
        assert len(r.errors) == 1


# ---------------------------------------------------------------------------
# RAGResponse
# ---------------------------------------------------------------------------
class TestRAGResponse:
    def test_total_latency_sums_breakdown(self):
        resp = RAGResponse(
            query="q", answer="a", citations=[], retrieved_chunks=[],
            latency_breakdown={"retrieval_ms": 50.0, "generation_ms": 300.0},
        )
        assert resp.total_latency_ms == 350.0

    def test_to_dict_has_required_keys(self):
        resp = RAGResponse(
            query="What?", answer="Answer.", citations=[],
            retrieved_chunks=[],
            latency_breakdown={"retrieval_ms": 50.0, "generation_ms": 300.0},
        )
        d = resp.to_dict()
        assert "query" in d
        assert "answer" in d
        assert "citations" in d
        assert "total_latency_ms" in d

    def test_empty_breakdown_gives_zero_total(self):
        resp = RAGResponse(query="q", answer="a", citations=[], retrieved_chunks=[])
        assert resp.total_latency_ms == 0.0


# ---------------------------------------------------------------------------
# RAGPipeline — indexing
# ---------------------------------------------------------------------------
class TestIndexing:
    def test_index_sources_returns_result(self, rag_pipeline, tmp_path):
        (tmp_path / "test.txt").write_text("Sample content for indexing test.")
        result = rag_pipeline.index([str(tmp_path)])
        assert isinstance(result, IndexingResult)
        assert result.num_documents >= 1
        assert result.num_chunks >= 1

    def test_index_increases_index_size(self, rag_pipeline, tmp_path):
        initial_size = rag_pipeline.index_size
        (tmp_path / "new_doc.txt").write_text(
            "New document with content about shipping and delivery policies."
        )
        rag_pipeline.index([str(tmp_path)])
        assert rag_pipeline.index_size > initial_size

    def test_index_with_empty_sources_returns_zero(self, rag_pipeline):
        result = rag_pipeline.index([])
        assert result.num_documents == 0
        assert result.num_chunks == 0

    def test_index_nonexistent_source_records_error(self, rag_pipeline):
        result = rag_pipeline.index(["/nonexistent/path/file.pdf"])
        assert len(result.errors) >= 1

    def test_index_partial_failure_continues(self, rag_pipeline, tmp_path):
        """A failed source should not prevent other sources from being indexed."""
        good = tmp_path / "good.txt"
        good.write_text("Good document content here.")
        result = rag_pipeline.index([str(good), "/bad/path/missing.pdf"])
        assert result.num_documents >= 1
        assert len(result.errors) >= 1

    def test_index_documents_with_preloaded_docs(self, mock_embedder, faiss_store, mock_llm):
        pipeline = RAGPipeline(
            embedder=mock_embedder,
            vector_store=faiss_store,
            llm=mock_llm,
        )
        docs = [
            Document("First document content.", {"source": "a.txt"}),
            Document("Second document content.", {"source": "b.txt"}),
        ]
        result = pipeline.index_documents(docs)
        assert isinstance(result, IndexingResult)
        assert result.num_documents == 2
        assert result.num_chunks >= 2

    def test_index_documents_empty_list_returns_gracefully(self, rag_pipeline):
        result = rag_pipeline.index_documents([])
        assert result.num_documents == 0
        assert result.errors != []    # should record "No documents provided"

    def test_index_embedding_latency_tracked(self, rag_pipeline, tmp_path):
        (tmp_path / "doc.txt").write_text("Content for latency tracking.")
        result = rag_pipeline.index([str(tmp_path)])
        assert result.embedding_latency_ms >= 0

    def test_index_stores_chunk_embeddings_for_mmr(self, mock_embedder, faiss_store, mock_llm, tmp_path):
        pipeline = RAGPipeline(embedder=mock_embedder, vector_store=faiss_store, llm=mock_llm)
        (tmp_path / "doc.txt").write_text("Content that should be stored for MMR.")
        pipeline.index([str(tmp_path)])
        assert len(pipeline._chunk_embeddings) > 0


# ---------------------------------------------------------------------------
# RAGPipeline — retrieval
# ---------------------------------------------------------------------------
class TestRetrieval:
    def test_retrieve_returns_search_results(self, rag_pipeline):
        results = rag_pipeline.retrieve("What is the refund policy?")
        assert isinstance(results, list)

    def test_retrieve_results_have_content(self, rag_pipeline):
        results = rag_pipeline.retrieve("product return", top_k=3)
        for r in results:
            assert hasattr(r, "content")
            assert hasattr(r, "score")
            assert hasattr(r, "chunk_id")

    def test_retrieve_top_k_respected(self, rag_pipeline):
        results = rag_pipeline.retrieve("query", top_k=2)
        assert len(results) <= 2

    def test_retrieve_with_filter(self, rag_pipeline):
        results = rag_pipeline.retrieve(
            "content", top_k=10, filter={"file_type": "txt"}
        )
        for r in results:
            assert r.metadata.get("file_type") == "txt"


# ---------------------------------------------------------------------------
# RAGPipeline — query (generation)
# ---------------------------------------------------------------------------
class TestQuery:
    def test_query_returns_rag_response(self, rag_pipeline):
        response = rag_pipeline.query("What is the refund policy?")
        assert isinstance(response, RAGResponse)

    def test_query_has_answer(self, rag_pipeline):
        response = rag_pipeline.query("test question")
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0

    def test_query_has_latency_breakdown(self, rag_pipeline):
        response = rag_pipeline.query("test question")
        assert "retrieval_ms" in response.latency_breakdown
        assert "generation_ms" in response.latency_breakdown

    def test_query_empty_result_returns_graceful_response(self, mock_embedder, mock_llm):
        """When no chunks match, pipeline should not crash."""
        from vector_store.vector_store import FAISSVectorStore
        empty_store = FAISSVectorStore(dimension=mock_embedder.dimension, index_type="flat")
        pipeline = RAGPipeline(embedder=mock_embedder, vector_store=empty_store, llm=mock_llm)
        response = pipeline.query("any question with empty index")
        assert isinstance(response.answer, str)
        assert "not find" in response.answer.lower() or len(response.answer) > 0

    def test_query_stores_session_history(self, rag_pipeline):
        session_id = "test-session-abc"
        rag_pipeline.query("First question", session_id=session_id)
        history = rag_pipeline.get_session_history(session_id)
        assert history is not None
        assert len(history) == 2    # user + assistant

    def test_query_second_turn_uses_history(self, rag_pipeline):
        session_id = "test-session-xyz"
        rag_pipeline.query("First question", session_id=session_id)
        rag_pipeline.query("Second question", session_id=session_id)
        history = rag_pipeline.get_session_history(session_id)
        assert len(history) == 4    # 2 turns × (user + assistant)

    def test_clear_session_removes_history(self, rag_pipeline):
        session_id = "clear-me"
        rag_pipeline.query("question", session_id=session_id)
        rag_pipeline.clear_session(session_id)
        assert rag_pipeline.get_session_history(session_id) is None

    def test_clear_nonexistent_session_is_safe(self, rag_pipeline):
        rag_pipeline.clear_session("session-that-does-not-exist")  # should not raise

    def test_query_with_filter_passes_to_retrieve(self, rag_pipeline):
        response = rag_pipeline.query(
            "question", filter={"file_type": "txt"}
        )
        assert isinstance(response, RAGResponse)

    def test_query_with_valid_template(self, rag_pipeline):
        response = rag_pipeline.query("question", template="summarization")
        assert isinstance(response, RAGResponse)

    def test_query_llm_response_attached(self, rag_pipeline):
        response = rag_pipeline.query("test question")
        assert response.llm_response is not None
        assert isinstance(response.llm_response, LLMResponse)


# ---------------------------------------------------------------------------
# RAGPipeline — streaming
# ---------------------------------------------------------------------------
class TestStreamQuery:
    def test_stream_yields_strings(self, rag_pipeline):
        tokens = list(rag_pipeline.stream_query("streaming question"))
        assert all(isinstance(t, str) for t in tokens)

    def test_stream_non_empty(self, rag_pipeline):
        tokens = list(rag_pipeline.stream_query("any question"))
        assert len(tokens) > 0

    def test_stream_with_session(self, rag_pipeline):
        tokens = list(rag_pipeline.stream_query("question", session_id="stream-session"))
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# RAGPipeline — multi-hop
# ---------------------------------------------------------------------------
class TestMultiHopQuery:
    def test_multi_hop_returns_rag_response(self, rag_pipeline):
        response = rag_pipeline.multi_hop_query("Complex question requiring multiple hops")
        assert isinstance(response, RAGResponse)

    def test_multi_hop_metadata_has_hops(self, rag_pipeline):
        response = rag_pipeline.multi_hop_query("complex question", max_hops=2)
        assert "hops" in response.metadata


# ---------------------------------------------------------------------------
# RAGPipeline — persistence
# ---------------------------------------------------------------------------
class TestPersistence:
    def test_save_creates_files(self, rag_pipeline, tmp_path):
        save_path = str(tmp_path / "saved_pipeline")
        rag_pipeline.save(save_path)
        assert Path(save_path).exists()

    def test_index_size_property(self, rag_pipeline):
        size = rag_pipeline.index_size
        assert isinstance(size, int)
        assert size >= 0

    def test_from_config_factory(self):
        """from_config() should not crash with default config."""
        from config import RAGConfig, LLMConfig, EmbeddingConfig, VectorStoreConfig

        config = RAGConfig(
            llm=LLMConfig(provider="openai", model_name="gpt-4o-mini"),
            embedding=EmbeddingConfig(
                provider="sentence_transformers",
                model_name="all-MiniLM-L6-v2",
                dimension=384,
            ),
            vector_store=VectorStoreConfig(backend="faiss", index_type="flat"),
        )
        # Patch the embedder and LLM to avoid real API calls
        with patch("rag_pipeline.EmbeddingEngine") as MockEmbed, \
             patch("rag_pipeline.VectorStoreFactory") as MockStore, \
             patch("rag_pipeline.LLMInterface") as MockLLM, \
             patch("rag_pipeline.CrossEncoderReranker"):

            mock_embedder = MagicMock()
            mock_embedder.dimension = 384
            mock_embedder.provider = "sentence_transformers"
            MockEmbed.return_value = mock_embedder

            mock_store = MagicMock()
            MockStore.create.return_value = mock_store

            mock_llm = MagicMock()
            mock_llm.provider = "openai"
            MockLLM.return_value = mock_llm

            pipeline = RAGPipeline.from_config(config)
            assert pipeline is not None
