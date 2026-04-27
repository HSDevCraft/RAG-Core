"""
embedding.py
────────────
Unified embedding layer supporting multiple providers.

Providers:
  - sentence_transformers  (local, free, fast on GPU) ← default
  - openai                 (text-embedding-3-large / small, API cost)
  - cohere                 (embed-multilingual-v3.0, API cost)

Design decisions:
  - All providers expose the same interface: encode(texts) → np.ndarray
  - Batch processing with configurable batch_size to avoid OOM / rate-limits
  - Caching via diskcache to skip re-embedding identical texts
  - Normalization is applied by default (required for cosine similarity)
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingResult:
    embeddings: np.ndarray          # shape (n_texts, dim)
    model: str
    dimension: int
    latency_ms: float
    tokens_used: int = 0            # populated only for API providers

    def __repr__(self) -> str:
        return (f"EmbeddingResult(shape={self.embeddings.shape}, "
                f"model={self.model}, latency={self.latency_ms:.1f}ms)")


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class BaseEmbedder:
    def encode(self, texts: List[str], batch_size: int = 64,
               normalize: bool = True) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Single-query optimized path."""
        return self.encode([query], batch_size=1, normalize=normalize)[0]

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Sentence Transformers (local)
# ---------------------------------------------------------------------------
class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local embedding via sentence-transformers.

    Recommended models (accuracy → speed trade-off):
      BAAI/bge-large-en-v1.5   → 1024-dim, best accuracy, ~500ms/batch on CPU
      BAAI/bge-base-en-v1.5    → 768-dim, good accuracy, ~200ms/batch on CPU
      BAAI/bge-small-en-v1.5   → 384-dim, fastest, ~100ms/batch on CPU
      all-MiniLM-L6-v2          → 384-dim, lightweight baseline

    For multilingual:
      BAAI/bge-m3               → 1024-dim, best multilingual
      paraphrase-multilingual-MiniLM-L12-v2
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5",
                 device: str = "cpu", cache_folder: Optional[str] = None):
        self._model_name = model_name
        self._device = device
        self._cache_folder = cache_folder
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer: %s on %s",
                        self._model_name, self._device)
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                cache_folder=self._cache_folder,
            )

    def encode(self, texts: List[str], batch_size: int = 64,
               normalize: bool = True) -> np.ndarray:
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------------------------
# 2. OpenAI Embeddings
# ---------------------------------------------------------------------------
class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI text-embedding-3-large (3072-dim) or text-embedding-3-small (1536-dim).

    Cost comparison (as of 2024):
      text-embedding-3-large → $0.13 / 1M tokens  (best accuracy)
      text-embedding-3-small → $0.02 / 1M tokens  (4x cheaper, still very good)
      text-embedding-ada-002 → $0.10 / 1M tokens  (legacy)

    Rate limits: 1M TPM on Tier 3 → use batch_size=512 and retry with backoff.
    """

    _DIM_MAP = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-3-large",
                 api_key: Optional[str] = None,
                 dimensions: Optional[int] = None):
        self._model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._dimensions = dimensions    # matryoshka truncation for text-embedding-3
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def encode(self, texts: List[str], batch_size: int = 512,
               normalize: bool = True) -> np.ndarray:
        from tenacity import retry, stop_after_attempt, wait_exponential

        client = self._get_client()
        all_embeddings: List[np.ndarray] = []
        total_tokens = 0

        @retry(stop=stop_after_attempt(5),
               wait=wait_exponential(multiplier=1, min=2, max=60))
        def _embed_batch(batch: List[str]):
            kwargs = {"model": self._model_name, "input": batch}
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions
            response = client.embeddings.create(**kwargs)
            vecs = np.array([item.embedding for item in response.data], dtype=np.float32)
            tokens = response.usage.total_tokens
            return vecs, tokens

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            vecs, tokens = _embed_batch(batch)
            total_tokens += tokens
            all_embeddings.append(vecs)
            logger.debug("OpenAI embed batch %d/%d, tokens=%d",
                         i // batch_size + 1, (len(texts) + batch_size - 1) // batch_size,
                         tokens)

        embeddings = np.vstack(all_embeddings)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
            embeddings = embeddings / norms
        return embeddings

    @property
    def dimension(self) -> int:
        if self._dimensions:
            return self._dimensions
        return self._DIM_MAP.get(self._model_name, 1536)

    @property
    def model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------------------------
# 3. Cohere Embeddings
# ---------------------------------------------------------------------------
class CohereEmbedder(BaseEmbedder):
    """
    Cohere embed-english-v3.0 / embed-multilingual-v3.0.
    Supports input_type for search asymmetry (search_query vs search_document).
    """

    def __init__(self, model_name: str = "embed-english-v3.0",
                 api_key: Optional[str] = None):
        self._model_name = model_name
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client(api_key=self._api_key)
        return self._client

    def encode(self, texts: List[str], batch_size: int = 96,
               normalize: bool = True,
               input_type: str = "search_document") -> np.ndarray:
        client = self._get_client()
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            response = client.embed(
                texts=batch,
                model=self._model_name,
                input_type=input_type,
            )
            vecs = np.array(response.embeddings, dtype=np.float32)
            all_embeddings.append(vecs)

        embeddings = np.vstack(all_embeddings)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
            embeddings = embeddings / norms
        return embeddings

    @property
    def dimension(self) -> int:
        return 1024

    @property
    def model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------------------------
# Disk cache decorator
# ---------------------------------------------------------------------------
def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class CachedEmbedder(BaseEmbedder):
    """
    Transparent disk-cache wrapper around any BaseEmbedder.
    Caches individual text → vector mappings to avoid re-embedding.

    Cache hit rate is typically 60-80% in production (repeated queries/chunks).
    """

    def __init__(self, embedder: BaseEmbedder, cache_dir: str = "./.embed_cache"):
        self._embedder = embedder
        try:
            import diskcache
            self._cache = diskcache.Cache(cache_dir)
        except ImportError:
            logger.warning("diskcache not installed; caching disabled.")
            self._cache = {}

    def encode(self, texts: List[str], batch_size: int = 64,
               normalize: bool = True) -> np.ndarray:
        results: dict[int, np.ndarray] = {}
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        key_prefix = self._embedder.model_name + ":"
        for i, text in enumerate(texts):
            cache_key = key_prefix + _text_hash(text)
            vec = self._cache.get(cache_key)
            if vec is not None:
                results[i] = vec
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_embeddings = self._embedder.encode(
                uncached_texts, batch_size=batch_size, normalize=normalize)
            for idx, vec in zip(uncached_indices, new_embeddings):
                cache_key = key_prefix + _text_hash(texts[idx])
                self._cache[cache_key] = vec
                results[idx] = vec

        logger.debug("Cache hit rate: %.1f%%",
                     100 * (1 - len(uncached_texts) / max(len(texts), 1)))
        return np.vstack([results[i] for i in range(len(texts))])

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        return self.encode([query], batch_size=1, normalize=normalize)[0]

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def model_name(self) -> str:
        return self._embedder.model_name


# ---------------------------------------------------------------------------
# Unified façade — EmbeddingEngine
# ---------------------------------------------------------------------------
class EmbeddingEngine:
    """
    Main entry-point for all embedding operations.

    Usage:
        engine = EmbeddingEngine(provider="sentence_transformers",
                                 model_name="BAAI/bge-large-en-v1.5",
                                 use_cache=True)
        vectors = engine.encode(["hello world", "foo bar"])
        query_vec = engine.encode_query("What is RAG?")
    """

    _PROVIDER_MAP = {
        "sentence_transformers": SentenceTransformerEmbedder,
        "openai":                OpenAIEmbedder,
        "cohere":                CohereEmbedder,
    }

    def __init__(
        self,
        provider: str = "sentence_transformers",
        model_name: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = "./.embed_cache",
        device: str = "cpu",
        **kwargs,
    ):
        if provider not in self._PROVIDER_MAP:
            raise ValueError(f"Unknown provider: {provider}. "
                             f"Choose from {list(self._PROVIDER_MAP)}")

        embedder_kwargs = {"device": device, **kwargs} if provider == "sentence_transformers" else kwargs
        if model_name:
            embedder_kwargs["model_name"] = model_name

        base = self._PROVIDER_MAP[provider](**embedder_kwargs)
        self._embedder: BaseEmbedder = (
            CachedEmbedder(base, cache_dir=cache_dir) if use_cache else base
        )
        self._provider = provider

    def encode(self, texts: List[str], batch_size: int = 64,
               normalize: bool = True) -> EmbeddingResult:
        t0 = time.perf_counter()
        vectors = self._embedder.encode(texts, batch_size=batch_size,
                                        normalize=normalize)
        latency = (time.perf_counter() - t0) * 1000
        return EmbeddingResult(
            embeddings=vectors,
            model=self._embedder.model_name,
            dimension=self._embedder.dimension,
            latency_ms=latency,
        )

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Returns a single (dim,) vector — optimized for retrieval hot path."""
        return self._embedder.encode_query(query, normalize=normalize)

    def encode_documents(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Alias with document-optimized settings (normalize=True by default)."""
        return self._embedder.encode(texts, batch_size=batch_size, normalize=True)

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def provider(self) -> str:
        return self._provider
