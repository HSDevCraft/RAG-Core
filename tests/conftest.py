"""
conftest.py
───────────
Shared pytest fixtures for the entire test suite.

Fixture scopes:
  function  — default; fresh instance per test (safe, isolated)
  module    — shared within a test file (faster, but state can leak)
  session   — shared across all tests (use only for truly expensive setup)

Design rules:
  1. All fixtures that call external APIs must be mocked.
  2. Fixtures that build indexes or load models use session scope where safe.
  3. Every fixture is documented so contributors understand its purpose.
"""

from __future__ import annotations

import os
import tempfile
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Environment guardrails — ensure tests never accidentally call real APIs
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY", "sk-test-00000000000000000000000000000000")
os.environ.setdefault("DISABLE_AUTH", "true")


# ---------------------------------------------------------------------------
# Document fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_text() -> str:
    """Minimal text content for loading / chunking tests."""
    return (
        "Refund Policy\n\n"
        "Customers are eligible for a full refund within 30 days of purchase.\n"
        "After 30 days, only store credit is available.\n"
        "Digital downloads are non-refundable once accessed.\n\n"
        "To initiate a refund, contact support@example.com with your order number.\n"
        "Refunds are processed within 5-7 business days.\n"
        "Original shipping fees are non-refundable.\n\n"
        "Items marked as Final Sale cannot be returned.\n"
        "Custom orders require a 15% restocking fee.\n"
        "Damaged items must be reported within 48 hours of delivery.\n"
    )


@pytest.fixture
def sample_text_file(sample_text, tmp_path):
    """Writes sample_text to a temporary .txt file; yields its path."""
    f = tmp_path / "policy.txt"
    f.write_text(sample_text, encoding="utf-8")
    return str(f)


@pytest.fixture
def sample_json_file(tmp_path):
    """Creates a temporary JSONL file with product records."""
    import json
    records = [
        {"text": "XR-500 Laptop: 15-inch, 32GB RAM, 2TB SSD. Price: $1,299.", "id": 1},
        {"text": "SmartWatch Pro: tracks heart rate and sleep. Price: $299.", "id": 2},
        {"text": "CoolBreeze AC Unit: 500 sq ft, SEER 18. Price: $649.", "id": 3},
    ]
    f = tmp_path / "products.jsonl"
    with f.open("w") as fp:
        for r in records:
            fp.write(json.dumps(r) + "\n")
    return str(f)


@pytest.fixture
def sample_html_file(tmp_path):
    """Creates a minimal HTML file."""
    html = """<!DOCTYPE html>
<html>
<head><title>FAQ</title></head>
<body>
<nav>Navigation menu</nav>
<main>
<h1>Frequently Asked Questions</h1>
<p>How do I return a product? Contact support within 30 days.</p>
<p>What payment methods are accepted? Visa, Mastercard, PayPal.</p>
</main>
<footer>Footer content</footer>
</body>
</html>"""
    f = tmp_path / "faq.html"
    f.write_text(html, encoding="utf-8")
    return str(f)


@pytest.fixture
def sample_documents(sample_text):
    """Returns a list of pre-constructed Document objects."""
    from data_pipeline.document_loader import Document
    return [
        Document(
            content=sample_text,
            metadata={"source": "policy.txt", "file_type": "txt"},
        ),
        Document(
            content="XR-500 Laptop: 15-inch display, 32GB RAM, 2TB SSD. Price: $1,299.",
            metadata={"source": "products.json", "file_type": "json", "id": 1},
        ),
        Document(
            content="SmartWatch Pro tracks heart rate, sleep, and steps. Water-resistant. Price: $299.",
            metadata={"source": "products.json", "file_type": "json", "id": 2},
        ),
    ]


# ---------------------------------------------------------------------------
# Chunking fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_chunks(sample_documents):
    """Returns chunks from sample_documents using the recursive strategy."""
    from data_pipeline.chunking import ChunkingEngine
    engine = ChunkingEngine(strategy="recursive", chunk_size=150, chunk_overlap=20)
    return engine.split(sample_documents)


# ---------------------------------------------------------------------------
# Embedding fixtures
# ---------------------------------------------------------------------------
DIM = 384   # all-MiniLM-L6-v2 dimension


@pytest.fixture(scope="session")
def real_embedder():
    """
    Real SentenceTransformer embedder loaded once per test session.
    Uses all-MiniLM-L6-v2 (small, fast, no GPU needed).
    Skipped if sentence-transformers is not installed.
    """
    pytest.importorskip("sentence_transformers")
    from embeddings.embedding import EmbeddingEngine
    return EmbeddingEngine(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        use_cache=False,
        device="cpu",
    )


@pytest.fixture
def mock_embedder():
    """
    Fast mock embedder that returns deterministic random vectors.
    Use for tests that don't need real semantic similarity.
    """
    embedder = MagicMock()
    embedder.dimension = DIM
    embedder.provider = "mock"

    def _encode(texts, **kwargs):
        from embeddings.embedding import EmbeddingResult
        rng = np.random.default_rng(seed=42)
        vecs = rng.random((len(texts), DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        vecs = vecs / norms
        return EmbeddingResult(
            embeddings=vecs, model="mock", dimension=DIM, latency_ms=1.0
        )

    def _encode_query(query, **kwargs):
        rng = np.random.default_rng(seed=hash(query) % 2**32)
        vec = rng.random(DIM).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)

    embedder.encode = _encode
    embedder.encode_query = _encode_query
    return embedder


@pytest.fixture
def sample_vectors(sample_chunks, mock_embedder):
    """Returns (ids, vectors, texts, metadatas) for the sample chunks."""
    texts = [c.content for c in sample_chunks]
    result = mock_embedder.encode(texts)
    ids = [c.chunk_id for c in sample_chunks]
    metadatas = [c.metadata for c in sample_chunks]
    return ids, result.embeddings, texts, metadatas


# ---------------------------------------------------------------------------
# Vector store fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def faiss_store(sample_vectors):
    """
    Populated in-memory FAISS Flat store.
    Fresh per test (function scope) to prevent state leakage.
    """
    from vector_store.vector_store import FAISSVectorStore
    ids, vectors, texts, metadatas = sample_vectors
    store = FAISSVectorStore(dimension=DIM, index_type="flat", metric="cosine")
    store.add(ids, vectors, texts, metadatas)
    return store


@pytest.fixture
def chroma_store(sample_vectors):
    """Populated ephemeral (in-memory) ChromaDB store."""
    pytest.importorskip("chromadb")
    from vector_store.vector_store import ChromaVectorStore
    ids, vectors, texts, metadatas = sample_vectors

    def clean_meta(m):
        return {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in m.items()}

    store = ChromaVectorStore(
        collection_name="test_collection",
        persist_dir=None,   # ephemeral
    )
    store.add(ids, vectors, texts, [clean_meta(m) for m in metadatas])
    return store


# ---------------------------------------------------------------------------
# Retrieval fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dense_retriever(faiss_store, mock_embedder):
    """Dense retriever backed by in-memory FAISS + mock embedder."""
    from retrieval.retriever import DenseRetriever
    return DenseRetriever(faiss_store, mock_embedder)


@pytest.fixture
def bm25_retriever(sample_chunks):
    """BM25 retriever built from sample chunks."""
    from retrieval.retriever import BM25Retriever
    texts = [c.content for c in sample_chunks]
    ids = [c.chunk_id for c in sample_chunks]
    metas = [c.metadata for c in sample_chunks]
    r = BM25Retriever()
    r.build(texts, ids, metas)
    return r


@pytest.fixture
def hybrid_retriever(dense_retriever, bm25_retriever):
    """Hybrid RRF retriever using mock embedder + BM25."""
    from retrieval.retriever import HybridRetriever
    return HybridRetriever(dense_retriever, bm25_retriever, use_rrf=True)


# ---------------------------------------------------------------------------
# LLM mock fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_llm():
    """
    Mock LLM that returns deterministic answers.
    Prevents any real API calls during tests.
    """
    from generation.llm_interface import LLMResponse

    llm = MagicMock()
    llm.provider = "mock"

    def _chat(messages, **kwargs):
        question = ""
        for m in messages:
            if m.get("role") == "user":
                question = m["content"]
                break
        answer = f"Mock answer for: {question[:50]}"
        return LLMResponse(
            content=answer,
            model="mock-llm",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=10.0,
        )

    def _complete(prompt, **kwargs):
        return LLMResponse(
            content=f"Mock completion: {prompt[:30]}",
            model="mock-llm",
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
            latency_ms=5.0,
        )

    def _stream(messages, **kwargs):
        yield "Mock "
        yield "streaming "
        yield "response."

    llm.chat = _chat
    llm.complete = _complete
    llm.stream = _stream
    return llm


# ---------------------------------------------------------------------------
# Full pipeline fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def rag_pipeline(mock_embedder, faiss_store, mock_llm, sample_documents):
    """
    Fully assembled RAGPipeline with mocked embedder and LLM.
    Index is pre-populated with sample_documents.
    """
    from rag_pipeline import RAGPipeline
    from config import DEFAULT_CONFIG

    pipeline = RAGPipeline(
        embedder=mock_embedder,
        vector_store=faiss_store,
        llm=mock_llm,
        config=DEFAULT_CONFIG,
    )
    # BM25 needs to be built separately since store was pre-populated
    texts = [c for c in [d.content for d in sample_documents]]
    pipeline._bm25.build(
        texts,
        [str(i) for i in range(len(texts))],
        [{} for _ in texts],
    )
    return pipeline


# ---------------------------------------------------------------------------
# API test client fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def api_client(rag_pipeline):
    """
    FastAPI TestClient with the RAG pipeline injected.
    Use for testing HTTP endpoints without starting a real server.
    """
    from fastapi.testclient import TestClient
    import api.main as api_module

    api_module._pipeline = rag_pipeline
    from api.main import app
    return TestClient(app)
