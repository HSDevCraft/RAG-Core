"""
reranker.py
───────────
Cross-encoder reranking to re-score top-k retrieved candidates.

Why rerank?
  Bi-encoders (used in dense retrieval) embed query and document independently.
  Cross-encoders attend jointly over (query, document) pairs → much higher accuracy
  at the cost of O(k) forward passes.

  Typical pipeline:
    1. Dense/Hybrid retrieval → top-40 candidates  (fast, approximate)
    2. Cross-encoder rerank   → top-5 final        (slow, precise)

Models:
  cross-encoder/ms-marco-MiniLM-L-6-v2   → fast, good quality
  cross-encoder/ms-marco-MiniLM-L-12-v2  → better quality, 2× slower
  BAAI/bge-reranker-large                → state-of-the-art, heavyweight
  Cohere Rerank API                      → managed, excellent, API cost

Production tip: Run reranker async / in a separate thread; parallelize batch.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from vector_store.vector_store import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker (local HuggingFace)
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    """
    Reranks candidates using a cross-encoder model from sentence-transformers.

    Usage:
        reranker = CrossEncoderReranker()
        top5 = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder: %s on %s", self._model_name, self._device)
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                max_length=self._max_length,
            )

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        if not candidates:
            return []
        self._load()

        pairs = [(query, c.content) for c in candidates]

        t0 = time.perf_counter()
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        latency = (time.perf_counter() - t0) * 1000
        logger.debug("CrossEncoder reranked %d candidates in %.1fms", len(pairs), latency)

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        results: List[SearchResult] = []
        limit = top_k or len(scored)
        for rank, (score, candidate) in enumerate(scored[:limit]):
            results.append(SearchResult(
                chunk_id=candidate.chunk_id,
                content=candidate.content,
                score=float(score),
                metadata=candidate.metadata,
                doc_id=candidate.doc_id,
                rank=rank,
            ))
        return results


# ---------------------------------------------------------------------------
# Cohere Reranker (managed API)
# ---------------------------------------------------------------------------
class CohereReranker:
    """
    Cohere Rerank API — state-of-the-art quality, pay-per-call.

    Cost: ~$2 / 1000 reranking calls (1 call = 1 query × N documents).
    Best for: production where quality > cost, multilingual use cases.

    Model options:
      rerank-english-v3.0     (English only, best accuracy)
      rerank-multilingual-v3.0 (supports 100+ languages)
      rerank-english-v2.0     (legacy, cheaper)
    """

    def __init__(
        self,
        model_name: str = "rerank-english-v3.0",
        api_key: Optional[str] = None,
        top_k_default: int = 5,
    ):
        self._model = model_name
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        self._top_k_default = top_k_default
        self._client = None

    def _get_client(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client(api_key=self._api_key)
        return self._client

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        if not candidates:
            return []

        client = self._get_client()
        docs = [c.content for c in candidates]
        limit = top_k or self._top_k_default

        response = client.rerank(
            model=self._model,
            query=query,
            documents=docs,
            top_n=limit,
        )

        results: List[SearchResult] = []
        for rank, result in enumerate(response.results):
            original = candidates[result.index]
            results.append(SearchResult(
                chunk_id=original.chunk_id,
                content=original.content,
                score=float(result.relevance_score),
                metadata=original.metadata,
                doc_id=original.doc_id,
                rank=rank,
            ))

        logger.debug("CohereReranker: %d → %d results", len(candidates), len(results))
        return results


# ---------------------------------------------------------------------------
# LLM-based Reranker (zero-shot, flexible but expensive)
# ---------------------------------------------------------------------------
class LLMReranker:
    """
    Uses an LLM to score relevance of each candidate on a 1-10 scale.
    Very accurate but expensive (1 LLM call per candidate).
    Use only for high-value offline reranking, not online serving.
    """

    PROMPT_TEMPLATE = """You are an expert information retrieval system.
Score the relevance of the following document to the query on a scale of 1 to 10.
Output ONLY the integer score.

Query: {query}

Document:
{document}

Relevance score (1-10):"""

    def __init__(self, llm_fn):
        """llm_fn: callable(prompt: str) → str"""
        self._llm = llm_fn

    def _score_one(self, query: str, content: str) -> float:
        prompt = self.PROMPT_TEMPLATE.format(query=query, document=content[:1500])
        response = self._llm(prompt)
        try:
            score = float("".join(c for c in response if c.isdigit() or c == "."))
            return min(max(score, 1.0), 10.0)
        except (ValueError, TypeError):
            return 5.0

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        scored: List[Tuple[float, SearchResult]] = []
        for c in candidates:
            score = self._score_one(query, c.content)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        limit = top_k or len(scored)

        return [
            SearchResult(
                chunk_id=c.chunk_id,
                content=c.content,
                score=score,
                metadata=c.metadata,
                rank=rank,
            )
            for rank, (score, c) in enumerate(scored[:limit])
        ]


# ---------------------------------------------------------------------------
# Reranker Factory
# ---------------------------------------------------------------------------
class RerankerFactory:
    """Instantiate a reranker by name."""

    @staticmethod
    def create(backend: str = "cross_encoder", **kwargs):
        if backend == "cross_encoder":
            return CrossEncoderReranker(**kwargs)
        elif backend == "cohere":
            return CohereReranker(**kwargs)
        elif backend == "llm":
            if "llm_fn" not in kwargs:
                raise ValueError("llm_fn is required for LLMReranker")
            return LLMReranker(kwargs["llm_fn"])
        else:
            raise ValueError(f"Unknown reranker backend: {backend}")
