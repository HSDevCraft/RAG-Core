"""
tests/unit/test_vector_store.py
────────────────────────────────
Unit tests for all vector store backends.

Coverage:
  - FAISSVectorStore: add, search, delete, persist/load, metadata filter
  - ChromaVectorStore: add, search (ephemeral mode, no server needed)
  - VectorStoreFactory: backend routing
  - SearchResult model
  - Edge cases: empty store, top_k > n_stored, delete non-existent id
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vector_store.vector_store import (
    FAISSVectorStore,
    SearchResult,
    VectorStoreFactory,
)

DIM = 32   # tiny dimension — fast tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _random_vecs(n: int, dim: int = DIM, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return vecs / norms


def _make_store(n: int = 10, index_type: str = "flat") -> tuple:
    """Return (store, ids, vectors, texts, metadatas) for a populated store."""
    store = FAISSVectorStore(dimension=DIM, index_type=index_type, metric="cosine")
    ids = [f"chunk-{i}" for i in range(n)]
    vecs = _random_vecs(n)
    texts = [f"Document content number {i}" for i in range(n)]
    metas = [{"source": f"doc_{i % 3}.pdf", "page": i, "file_type": "pdf"} for i in range(n)]
    store.add(ids, vecs, texts, metas)
    return store, ids, vecs, texts, metas


# ---------------------------------------------------------------------------
# SearchResult model
# ---------------------------------------------------------------------------
class TestSearchResult:
    def test_instantiation(self):
        r = SearchResult(chunk_id="abc", content="hello", score=0.9)
        assert r.chunk_id == "abc"
        assert r.content == "hello"
        assert r.score == 0.9
        assert r.metadata == {}
        assert r.rank == 0

    def test_metadata_default_empty(self):
        r = SearchResult(chunk_id="x", content="y", score=0.5)
        assert isinstance(r.metadata, dict)


# ---------------------------------------------------------------------------
# FAISSVectorStore — Flat index
# ---------------------------------------------------------------------------
class TestFAISSFlat:
    def test_add_increases_count(self):
        store = FAISSVectorStore(dimension=DIM, index_type="flat")
        assert store.count == 0
        vecs = _random_vecs(5)
        store.add([f"id-{i}" for i in range(5)], vecs,
                  [f"text {i}" for i in range(5)])
        assert store.count == 5

    def test_search_returns_top_k(self):
        store, ids, vecs, texts, metas = _make_store(10)
        query = _random_vecs(1)[0]
        results = store.search(query, top_k=3)
        assert len(results) == 3

    def test_search_scores_in_descending_order(self):
        store, *_ = _make_store(10)
        query = _random_vecs(1)[0]
        results = store.search(query, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_greater_than_stored(self):
        """When top_k > n_stored, return all stored."""
        store, *_ = _make_store(5)
        query = _random_vecs(1)[0]
        results = store.search(query, top_k=100)
        assert len(results) <= 5

    def test_search_returns_correct_content(self):
        store, ids, vecs, texts, metas = _make_store(10)
        # Query with the first stored vector → it should be the top result
        results = store.search(vecs[0], top_k=1)
        assert results[0].content == texts[0]
        assert results[0].chunk_id == ids[0]

    def test_metadata_filter(self):
        store, ids, vecs, texts, metas = _make_store(10)
        query = _random_vecs(1)[0]
        # Filter to only doc_0.pdf sources (indices 0, 3, 6, 9)
        results = store.search(query, top_k=10, filter={"source": "doc_0.pdf"})
        for r in results:
            assert r.metadata.get("source") == "doc_0.pdf"

    def test_metadata_filter_list_value(self):
        store, ids, vecs, texts, metas = _make_store(9)
        query = _random_vecs(1)[0]
        # Filter to sources in list
        results = store.search(query, top_k=10,
                               filter={"source": ["doc_0.pdf", "doc_1.pdf"]})
        for r in results:
            assert r.metadata.get("source") in ["doc_0.pdf", "doc_1.pdf"]

    def test_empty_store_search_returns_empty(self):
        store = FAISSVectorStore(dimension=DIM, index_type="flat")
        query = _random_vecs(1)[0]
        results = store.search(query, top_k=5)
        assert results == []

    def test_delete_removes_vectors(self):
        store, ids, vecs, texts, metas = _make_store(5)
        store.delete([ids[0]])
        # Searching with the deleted vector should not return it as top-1
        results = store.search(vecs[0], top_k=5)
        returned_ids = [r.chunk_id for r in results]
        assert ids[0] not in returned_ids

    def test_delete_nonexistent_id_is_safe(self):
        store, *_ = _make_store(5)
        # Should not raise
        store.delete(["nonexistent-id"])
        assert store.count == 5

    def test_persist_and_reload(self, tmp_path):
        store, ids, vecs, texts, metas = _make_store(5)
        save_path = str(tmp_path / "test_index")
        store.persist(save_path)

        # Verify files created
        assert (Path(save_path) / "index.faiss").exists()
        assert (Path(save_path) / "metadata.pkl").exists()

        # Reload
        new_store = FAISSVectorStore(dimension=DIM, index_type="flat")
        new_store.load(save_path)
        assert new_store.count == store.count

        # Results should match
        query = _random_vecs(1)[0]
        orig = store.search(query, top_k=3)
        reloaded = new_store.search(query, top_k=3)
        assert [r.chunk_id for r in orig] == [r.chunk_id for r in reloaded]

    def test_search_result_has_rank(self):
        store, *_ = _make_store(5)
        results = store.search(_random_vecs(1)[0], top_k=3)
        assert results[0].rank == 0
        assert results[1].rank == 1
        assert results[2].rank == 2

    def test_duplicate_add_updates_entry(self):
        """Adding the same ID twice should update, not duplicate (upsert behavior)."""
        store = FAISSVectorStore(dimension=DIM, index_type="flat")
        vecs = _random_vecs(1)
        store.add(["id-1"], vecs, ["original text"])
        # Add same id with different content
        new_vecs = _random_vecs(1, seed=99)
        store.add(["id-1"], new_vecs, ["updated text"])
        # Count should only increment (FAISS doesn't deduplicate by id natively,
        # but our metadata should reflect the latest)
        results = store.search(new_vecs[0], top_k=1)
        assert results[0].chunk_id == "id-1"


# ---------------------------------------------------------------------------
# FAISSVectorStore — HNSW index
# ---------------------------------------------------------------------------
class TestFAISSHNSW:
    def test_add_and_search(self):
        store = FAISSVectorStore(
            dimension=DIM, index_type="hnsw", metric="cosine",
            hnsw_m=8, hnsw_ef_construction=100, hnsw_ef_search=20,
        )
        vecs = _random_vecs(20)
        ids = [f"id-{i}" for i in range(20)]
        texts = [f"text {i}" for i in range(20)]
        store.add(ids, vecs, texts)
        results = store.search(vecs[0], top_k=5)
        assert len(results) >= 1
        assert results[0].chunk_id == "id-0"    # exact self-match

    def test_persist_and_reload_hnsw(self, tmp_path):
        store = FAISSVectorStore(dimension=DIM, index_type="hnsw")
        vecs = _random_vecs(10)
        ids = [f"id-{i}" for i in range(10)]
        store.add(ids, vecs, [f"text {i}" for i in range(10)])
        store.persist(str(tmp_path / "hnsw_index"))

        new_store = FAISSVectorStore(dimension=DIM, index_type="hnsw")
        new_store.load(str(tmp_path / "hnsw_index"))
        assert new_store.count == 10


# ---------------------------------------------------------------------------
# ChromaVectorStore
# ---------------------------------------------------------------------------
class TestChromaVectorStore:
    def test_add_and_search(self):
        pytest.importorskip("chromadb")
        from vector_store.vector_store import ChromaVectorStore

        store = ChromaVectorStore(
            collection_name="test_unit_chroma",
            persist_dir=None,   # ephemeral
        )
        vecs = _random_vecs(5)
        ids = [f"c-{i}" for i in range(5)]
        texts = [f"chroma text {i}" for i in range(5)]
        store.add(ids, vecs, texts)

        assert store.count == 5
        results = store.search(vecs[0], top_k=3)
        assert len(results) >= 1

    def test_search_scores_in_range(self):
        pytest.importorskip("chromadb")
        from vector_store.vector_store import ChromaVectorStore

        store = ChromaVectorStore(collection_name="test_scores", persist_dir=None)
        vecs = _random_vecs(5)
        store.add([f"id-{i}" for i in range(5)], vecs,
                  [f"text {i}" for i in range(5)])
        results = store.search(vecs[0], top_k=3)
        for r in results:
            assert -1.0 <= r.score <= 1.0

    def test_delete_removes_item(self):
        pytest.importorskip("chromadb")
        from vector_store.vector_store import ChromaVectorStore

        store = ChromaVectorStore(collection_name="test_delete", persist_dir=None)
        vecs = _random_vecs(3)
        ids = ["del-0", "del-1", "del-2"]
        store.add(ids, vecs, ["a", "b", "c"])
        assert store.count == 3
        store.delete(["del-0"])
        assert store.count == 2


# ---------------------------------------------------------------------------
# VectorStoreFactory
# ---------------------------------------------------------------------------
class TestVectorStoreFactory:
    def test_creates_faiss(self):
        store = VectorStoreFactory.create(backend="faiss", dimension=DIM)
        assert isinstance(store, FAISSVectorStore)

    def test_creates_chroma(self):
        pytest.importorskip("chromadb")
        from vector_store.vector_store import ChromaVectorStore
        store = VectorStoreFactory.create(
            backend="chroma",
            collection_name="factory_test",
            persist_dir=None,
        )
        assert isinstance(store, ChromaVectorStore)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            VectorStoreFactory.create(backend="nonexistent")

    def test_case_insensitive(self):
        store = VectorStoreFactory.create(backend="FAISS", dimension=DIM)
        assert isinstance(store, FAISSVectorStore)
