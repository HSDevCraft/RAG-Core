"""
chunking.py
───────────
Chunking strategies for splitting Documents into retrieval-sized Chunks.

Strategies:
  1. recursive  – LangChain-style recursive character splitting (default)
  2. token      – splits by token count using tiktoken
  3. sentence   – splits on sentence boundaries (NLTK)
  4. semantic   – groups semantically similar sentences (cosine similarity)

Design decisions:
  - Chunk carries a copy of its parent document's metadata plus its own position.
  - Overlap is applied on all fixed-size strategies to reduce boundary effects.
  - Semantic chunking is expensive (O(n) embeddings) – use only for high-value corpora.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from data_pipeline.document_loader import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical chunk model
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0

    def __post_init__(self):
        import hashlib, json
        if not self.chunk_id:
            self.chunk_id = hashlib.sha256(
                (self.content + json.dumps(self.metadata, sort_keys=True)).encode()
            ).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.content)

    @property
    def token_count(self) -> int:
        """Rough token estimate without importing tiktoken every call."""
        return len(self.content.split()) * 4 // 3


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class BaseChunker:
    def split(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            chunks.extend(self._split_doc(doc))
        logger.info("%s produced %d chunks from %d docs",
                    self.__class__.__name__, len(chunks), len(documents))
        return chunks

    def _split_doc(self, doc: Document) -> List[Chunk]:
        raise NotImplementedError

    @staticmethod
    def _make_chunks(texts: List[str], doc: Document) -> List[Chunk]:
        chunks = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            meta = {**doc.metadata, "chunk_index": i, "total_chunks": len(texts)}
            chunks.append(Chunk(
                content=text.strip(),
                metadata=meta,
                doc_id=doc.doc_id,
                chunk_index=i,
            ))
        return chunks


# ---------------------------------------------------------------------------
# 1. Recursive Character Splitter
# ---------------------------------------------------------------------------
class RecursiveChunker(BaseChunker):
    """
    Splits on a hierarchy of separators, falling through to smaller ones
    only when the chunk is still too large.

    This is the go-to strategy for most corpora because it respects natural
    text boundaries (paragraphs → sentences → words) while guaranteeing
    chunk_size compliance.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
        length_fn=len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.length_fn = length_fn

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive helper."""
        final_chunks: List[str] = []

        separator = separators[-1]          # fallback
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []

        for split in splits:
            if self.length_fn(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    final_chunks.extend(self._merge(good_splits, separator))
                    good_splits = []
                # recurse with next-level separators
                remaining = [s for s in separators if s != separator]
                if remaining:
                    final_chunks.extend(self._split_text(split, remaining))
                else:
                    final_chunks.append(split)

        if good_splits:
            final_chunks.extend(self._merge(good_splits, separator))

        return final_chunks

    def _merge(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits back into chunks respecting size + overlap."""
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for split in splits:
            split_len = self.length_fn(split)
            if current_len + split_len + len(separator) > self.chunk_size and current:
                chunk_text = separator.join(current)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                # keep overlap window
                while current and current_len > self.chunk_overlap:
                    current_len -= self.length_fn(current[0]) + len(separator)
                    current.pop(0)

            current.append(split)
            current_len += split_len + len(separator)

        if current:
            chunks.append(separator.join(current))

        return chunks

    def _split_doc(self, doc: Document) -> List[Chunk]:
        texts = self._split_text(doc.content, self.separators)
        return self._make_chunks(texts, doc)


# ---------------------------------------------------------------------------
# 2. Token-based Splitter
# ---------------------------------------------------------------------------
class TokenChunker(BaseChunker):
    """
    Splits by token count using tiktoken.
    Critical when interfacing with models that count tokens, not characters.

    Trade-off: requires tiktoken; slower than char-based splitting.
    """

    def __init__(
        self,
        chunk_size: int = 512,       # in tokens
        chunk_overlap: int = 64,     # in tokens
        model: str = "gpt-4o",
    ):
        try:
            import tiktoken
            self.enc = tiktoken.encoding_for_model(model)
        except Exception:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_doc(self, doc: Document) -> List[Chunk]:
        tokens = self.enc.encode(doc.content)
        texts: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            texts.append(self.enc.decode(chunk_tokens))
            start += self.chunk_size - self.chunk_overlap
        return self._make_chunks(texts, doc)


# ---------------------------------------------------------------------------
# 3. Sentence-based Splitter
# ---------------------------------------------------------------------------
class SentenceChunker(BaseChunker):
    """
    Groups complete sentences up to `chunk_size` characters.
    Avoids mid-sentence cuts; best for QA and fact-verification tasks.

    Requires: nltk punkt tokenizer.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap_sentences: int = 1):
        self.chunk_size = chunk_size
        self.overlap = chunk_overlap_sentences
        self._ensure_nltk()

    @staticmethod
    def _ensure_nltk():
        try:
            import nltk
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            import nltk
            nltk.download("punkt", quiet=True)

    def _tokenize(self, text: str) -> List[str]:
        import nltk
        return nltk.sent_tokenize(text)

    def _split_doc(self, doc: Document) -> List[Chunk]:
        sentences = self._tokenize(doc.content)
        chunks: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > self.chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = current_sentences[-self.overlap:]
                current_len = sum(len(s) for s in current_sentences)

            current_sentences.append(sent)
            current_len += len(sent)

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return self._make_chunks(chunks, doc)


# ---------------------------------------------------------------------------
# 4. Semantic Chunker
# ---------------------------------------------------------------------------
class SemanticChunker(BaseChunker):
    """
    Groups sentences based on semantic similarity.
    Splits when cosine distance between adjacent sentence embeddings exceeds threshold.

    When to use: high-value corpora (legal, medical, financial) where topic boundaries
    matter more than size uniformity.

    Cost: O(n) embedding calls per document — cache embeddings where possible.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        breakpoint_threshold: float = 0.35,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        device: str = "cpu",
    ):
        self.threshold = breakpoint_threshold
        self.min_size = min_chunk_size
        self.max_size = max_chunk_size
        self._model = None
        self._model_name = embedding_model
        self._device = device

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    @staticmethod
    def _cosine_distance(a, b) -> float:
        import numpy as np
        a, b = a / (np.linalg.norm(a) + 1e-10), b / (np.linalg.norm(b) + 1e-10)
        return float(1 - a @ b)

    def _split_doc(self, doc: Document) -> List[Chunk]:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        sentences = nltk.sent_tokenize(doc.content)
        if len(sentences) <= 1:
            return self._make_chunks([doc.content], doc)

        model = self._get_model()
        embeddings = model.encode(sentences, show_progress_bar=False,
                                  normalize_embeddings=True)

        groups: List[List[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            dist = self._cosine_distance(embeddings[i - 1], embeddings[i])
            current_group_text = " ".join(groups[-1])

            if dist > self.threshold or len(current_group_text) > self.max_size:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])

        # merge tiny groups forward
        merged: List[str] = []
        buf = ""
        for grp in groups:
            text = " ".join(grp)
            if len(buf) + len(text) < self.min_size:
                buf += " " + text
            else:
                if buf:
                    merged.append(buf.strip())
                buf = text
        if buf:
            merged.append(buf.strip())

        return self._make_chunks(merged, doc)


# ---------------------------------------------------------------------------
# Chunking Engine — unified façade
# ---------------------------------------------------------------------------
class ChunkingEngine:
    """
    Select and run any chunking strategy.

    Usage:
        engine = ChunkingEngine(strategy="recursive", chunk_size=512, chunk_overlap=64)
        chunks = engine.split(documents)
    """

    _STRATEGY_MAP = {
        "recursive": RecursiveChunker,
        "token":     TokenChunker,
        "sentence":  SentenceChunker,
        "semantic":  SemanticChunker,
    }

    def __init__(self, strategy: str = "recursive", **kwargs):
        if strategy not in self._STRATEGY_MAP:
            raise ValueError(f"Unknown strategy '{strategy}'. "
                             f"Choose from: {list(self._STRATEGY_MAP)}")
        self.strategy = strategy
        self.chunker: BaseChunker = self._STRATEGY_MAP[strategy](**kwargs)

    def split(self, documents: List[Document]) -> List[Chunk]:
        return self.chunker.split(documents)

    def split_one(self, document: Document) -> List[Chunk]:
        return self.chunker._split_doc(document)

    @property
    def strategy_name(self) -> str:
        return self.strategy
