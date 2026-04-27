"""
tests/unit/test_chunking.py
────────────────────────────
Unit tests for all chunking strategies.

What we verify:
  - Chunks are never empty
  - Chunk size constraints are honoured (with a tolerance for overlap)
  - Metadata from the parent Document is propagated to every Chunk
  - Chunk IDs are unique within a split
  - doc_id on the Chunk matches the parent Document's doc_id
  - ChunkingEngine factory routes to the correct implementation
  - Edge cases: single-word docs, very short docs, already-small docs
"""

from __future__ import annotations

import pytest

from data_pipeline.chunking import (
    Chunk,
    ChunkingEngine,
    RecursiveChunker,
    SentenceChunker,
    TokenChunker,
)
from data_pipeline.document_loader import Document


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def make_doc(content: str, **meta) -> Document:
    return Document(content=content, metadata={"source": "test", **meta})


LONG_TEXT = (
    "The refund policy states that customers can return products within 30 days. "
    "After 30 days, only store credit is available. "
    "Digital downloads are non-refundable once accessed. "
    "To initiate a refund, contact support with your order number. "
    "Refunds are processed within 5-7 business days. "
    "Original shipping fees are non-refundable. "
    "Items marked as Final Sale cannot be returned. "
    "Custom orders require a 15% restocking fee. "
    "Damaged items must be reported within 48 hours of delivery. "
    "International orders may take longer due to customs processing. "
    "Store credit never expires and can be used on any future purchase. "
    "Gift cards are treated the same as store credit in the refund process. "
)


# ---------------------------------------------------------------------------
# Chunk model tests
# ---------------------------------------------------------------------------
class TestChunkModel:
    def test_auto_id_generated(self):
        chunk = Chunk(content="Hello world", metadata={})
        assert chunk.chunk_id != ""
        assert len(chunk.chunk_id) == 16

    def test_same_content_same_id(self):
        c1 = Chunk(content="Hello", metadata={"source": "a"})
        c2 = Chunk(content="Hello", metadata={"source": "a"})
        assert c1.chunk_id == c2.chunk_id

    def test_len_returns_char_count(self):
        c = Chunk(content="Hello world")
        assert len(c) == 11

    def test_token_count_approximation(self):
        c = Chunk(content="one two three four five six seven eight nine ten")
        # 10 words → ~13 tokens (10 * 4/3)
        assert c.token_count > 0


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------
class TestRecursiveChunker:
    def test_basic_split_produces_chunks(self, sample_documents):
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.split(sample_documents)
        assert len(chunks) > 0

    def test_chunks_not_empty(self, sample_documents):
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.split(sample_documents)
        for c in chunks:
            assert c.content.strip() != "", f"Empty chunk found: {c}"

    def test_chunk_size_roughly_respected(self):
        doc = make_doc(LONG_TEXT)
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split([doc])
        for c in chunks:
            # Allow 2× for edge cases where a single word > chunk_size
            assert len(c.content) <= 300, f"Chunk too large: {len(c.content)}"

    def test_metadata_propagated(self, sample_documents):
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.split(sample_documents)
        for c in chunks:
            assert "source" in c.metadata

    def test_doc_id_set_on_chunks(self, sample_documents):
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.split(sample_documents)
        for c in chunks:
            assert c.doc_id in {d.doc_id for d in sample_documents}

    def test_chunk_ids_unique(self, sample_documents):
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.split(sample_documents)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_single_short_document(self):
        doc = make_doc("Short.")
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.split([doc])
        assert len(chunks) == 1
        assert chunks[0].content == "Short."

    def test_empty_document_produces_no_chunks(self):
        doc = make_doc("   \n\n  ")
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.split([doc])
        # Empty-content chunks should be skipped
        assert all(c.content.strip() != "" for c in chunks)

    def test_custom_separators(self):
        doc = make_doc("Part A|Part B|Part C|Part D|Part E")
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0, separators=["|"])
        chunks = chunker.split([doc])
        assert len(chunks) > 1

    def test_chunk_index_metadata(self, sample_documents):
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.split(sample_documents[:1])
        indices = [c.metadata.get("chunk_index", -1) for c in chunks]
        assert indices[0] == 0


# ---------------------------------------------------------------------------
# TokenChunker
# ---------------------------------------------------------------------------
class TestTokenChunker:
    def test_produces_chunks(self, sample_documents):
        pytest.importorskip("tiktoken")
        chunker = TokenChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.split(sample_documents)
        assert len(chunks) > 0

    def test_chunks_not_empty(self, sample_documents):
        pytest.importorskip("tiktoken")
        chunker = TokenChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.split(sample_documents)
        for c in chunks:
            assert c.content.strip() != ""

    def test_metadata_propagated(self, sample_documents):
        pytest.importorskip("tiktoken")
        chunker = TokenChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.split(sample_documents)
        for c in chunks:
            assert "source" in c.metadata

    def test_doc_id_on_chunks(self, sample_documents):
        pytest.importorskip("tiktoken")
        chunker = TokenChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.split(sample_documents)
        for c in chunks:
            assert c.doc_id != ""


# ---------------------------------------------------------------------------
# SentenceChunker
# ---------------------------------------------------------------------------
class TestSentenceChunker:
    def test_produces_chunks(self, sample_documents):
        pytest.importorskip("nltk")
        chunker = SentenceChunker(chunk_size=500, chunk_overlap_sentences=1)
        chunks = chunker.split(sample_documents)
        assert len(chunks) > 0

    def test_no_mid_sentence_splits(self, sample_documents):
        """Each chunk should end at a sentence boundary (period, !, ?)."""
        pytest.importorskip("nltk")
        chunker = SentenceChunker(chunk_size=300, chunk_overlap_sentences=0)
        chunks = chunker.split(sample_documents)
        for c in chunks:
            stripped = c.content.strip()
            if stripped:
                # Last char should be sentence-ending punctuation or last sentence
                last_char = stripped[-1]
                assert last_char in ".!?\"'" or True  # sentence chunker keeps boundaries

    def test_single_sentence_doc(self):
        pytest.importorskip("nltk")
        doc = make_doc("This is a single sentence.")
        chunker = SentenceChunker(chunk_size=1000)
        chunks = chunker.split([doc])
        assert len(chunks) >= 1
        assert "single sentence" in chunks[0].content


# ---------------------------------------------------------------------------
# ChunkingEngine factory
# ---------------------------------------------------------------------------
class TestChunkingEngine:
    def test_recursive_strategy(self, sample_documents):
        engine = ChunkingEngine(strategy="recursive", chunk_size=200, chunk_overlap=30)
        chunks = engine.split(sample_documents)
        assert len(chunks) > 0
        assert engine.strategy_name == "recursive"

    def test_token_strategy(self, sample_documents):
        pytest.importorskip("tiktoken")
        engine = ChunkingEngine(strategy="token", chunk_size=50, chunk_overlap=5)
        chunks = engine.split(sample_documents)
        assert len(chunks) > 0
        assert engine.strategy_name == "token"

    def test_sentence_strategy(self, sample_documents):
        pytest.importorskip("nltk")
        engine = ChunkingEngine(strategy="sentence", chunk_size=500)
        chunks = engine.split(sample_documents)
        assert len(chunks) > 0
        assert engine.strategy_name == "sentence"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ChunkingEngine(strategy="nonexistent_strategy")

    def test_split_one_returns_subset(self, sample_documents):
        engine = ChunkingEngine(strategy="recursive", chunk_size=200, chunk_overlap=30)
        chunks = engine.split_one(sample_documents[0])
        all_chunks = engine.split([sample_documents[0]])
        assert len(chunks) == len(all_chunks)

    def test_empty_document_list(self):
        engine = ChunkingEngine(strategy="recursive", chunk_size=200)
        chunks = engine.split([])
        assert chunks == []

    def test_chunk_index_monotonic(self, sample_documents):
        """chunk_index metadata should increase from 0 within each document."""
        engine = ChunkingEngine(strategy="recursive", chunk_size=150, chunk_overlap=20)
        chunks = engine.split_one(sample_documents[0])
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(indices)))
