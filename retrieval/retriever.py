"""
retriever.py
────────────
Retrieval layer — dense, sparse (BM25), hybrid, and query transformation.

Architecture:
  DenseRetriever     → pure vector ANN search
  BM25Retriever      → lexical inverted-index search
  HybridRetriever    → Reciprocal Rank Fusion of dense + BM25

Query transformations:
  - HyDE  (Hypothetical Document Embeddings): generate a fake answer, embed it
  - MultiQuery: generate N paraphrases, union results
  - StepBack: generate a more abstract question to broaden retrieval

MMR (Maximal Marginal Relevance): balance relevance vs diversity in final set.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vector_store.vector_store import BaseVectorStore, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dense Retriever
# ---------------------------------------------------------------------------
class DenseRetriever:
    """
    Pure vector similarity search.

    Query → embed → ANN search → top-k results.
    Best for semantic / paraphrase matching.
    Weakness: keyword mismatch, rare terms, acronyms.
    """

    def __init__(self, vector_store: BaseVectorStore, embedder):
        self._store = vector_store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 10,
                 filter: Optional[dict] = None) -> List[SearchResult]:
        query_vec = self._embedder.encode_query(query)
        results = self._store.search(query_vec, top_k=top_k, filter=filter)
        logger.debug("DenseRetriever: %d results for query='%s'", len(results), query[:60])
        return results

    def retrieve_by_vector(self, query_vec: np.ndarray, top_k: int = 10,
                           filter: Optional[dict] = None) -> List[SearchResult]:
        return self._store.search(query_vec, top_k=top_k, filter=filter)


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------
class BM25Retriever:
    """
    Sparse BM25 retrieval using rank_bm25.

    BM25 parameters:
      k1 = 1.5  (term frequency saturation; higher → TF has more impact)
      b  = 0.75 (length normalization; b=0 → no normalization)

    Best for: exact keyword matches, rare terms, code snippets, product IDs.
    Weakness: semantic gap — "automobile" vs "car" are unrelated.

    Build corpus from the same chunks used to build the vector store.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._bm25 = None
        self._corpus: List[str] = []
        self._chunk_ids: List[str] = []
        self._metadatas: List[dict] = []

    def build(self, texts: List[str], chunk_ids: List[str],
              metadatas: Optional[List[dict]] = None) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Install rank-bm25: pip install rank-bm25")
        from rank_bm25 import BM25Okapi

        self._corpus = texts
        self._chunk_ids = chunk_ids
        self._metadatas = metadatas or [{} for _ in texts]
        tokenized = [self._tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized, k1=self._k1, b=self._b)
        logger.info("BM25 index built with %d documents", len(texts))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                break
            results.append(SearchResult(
                chunk_id=self._chunk_ids[idx],
                content=self._corpus[idx],
                score=float(scores[idx]),
                metadata=self._metadatas[idx],
                rank=rank,
            ))
        return results

    def save(self, path: str) -> None:
        import pickle
        from pathlib import Path
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "bm25.pkl", "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "corpus": self._corpus,
                "chunk_ids": self._chunk_ids,
                "metadatas": self._metadatas,
            }, f)

    def load(self, path: str) -> None:
        import pickle
        from pathlib import Path
        with open(Path(path) / "bm25.pkl", "rb") as f:
            state = pickle.load(f)
        self._bm25 = state["bm25"]
        self._corpus = state["corpus"]
        self._chunk_ids = state["chunk_ids"]
        self._metadatas = state["metadatas"]
        logger.info("BM25 index loaded from %s (%d docs)", path, len(self._corpus))


# ---------------------------------------------------------------------------
# Hybrid Retriever — RRF fusion
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    Hybrid retrieval: BM25 + Dense ANN fused via Reciprocal Rank Fusion (RRF).

    RRF formula:  score(d) = Σ  1 / (k + rank_i(d))
    where k=60 (empirically strong default from Cormack et al., 2009).

    Why RRF over score normalization?
    - Score scales differ wildly between BM25 and cosine similarity.
    - RRF only uses rank positions → scale-invariant.
    - Empirically matches or beats weighted score combination on BEIR.

    alpha parameter (0-1): if you want weighted combination instead of RRF,
    set use_rrf=False and alpha controls dense weight.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        use_rrf: bool = True,
        alpha: float = 0.7,          # dense weight when use_rrf=False
    ):
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._rrf_k = rrf_k
        self._use_rrf = use_rrf
        self._alpha = alpha

    def retrieve(self, query: str, top_k: int = 10,
                 filter: Optional[dict] = None) -> List[SearchResult]:
        n_candidates = top_k * 3

        dense_results = self._dense.retrieve(query, top_k=n_candidates, filter=filter)
        bm25_results = self._bm25.retrieve(query, top_k=n_candidates)

        if self._use_rrf:
            return self._rrf_fusion(dense_results, bm25_results, top_k)
        else:
            return self._score_fusion(dense_results, bm25_results, top_k)

    def _rrf_fusion(self, dense: List[SearchResult],
                    bm25: List[SearchResult], top_k: int) -> List[SearchResult]:
        k = self._rrf_k
        scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}

        for rank, r in enumerate(dense, start=1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1 / (k + rank)
            result_map[r.chunk_id] = r

        for rank, r in enumerate(bm25, start=1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1 / (k + rank)
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        results: List[SearchResult] = []
        for rank, cid in enumerate(sorted_ids):
            r = result_map[cid]
            results.append(SearchResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=scores[cid],
                metadata=r.metadata,
                rank=rank,
            ))
        logger.debug("HybridRetriever RRF: %d results", len(results))
        return results

    def _score_fusion(self, dense: List[SearchResult],
                      bm25: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Normalize scores then combine: final = alpha*dense + (1-alpha)*bm25."""
        def normalize(results: List[SearchResult]) -> Dict[str, float]:
            if not results:
                return {}
            max_s = max(r.score for r in results) + 1e-10
            return {r.chunk_id: r.score / max_s for r in results}

        d_scores = normalize(dense)
        b_scores = normalize(bm25)
        result_map = {r.chunk_id: r for r in dense}
        result_map.update({r.chunk_id: r for r in bm25 if r.chunk_id not in result_map})

        all_ids = set(d_scores) | set(b_scores)
        combined = {cid: self._alpha * d_scores.get(cid, 0)
                    + (1 - self._alpha) * b_scores.get(cid, 0)
                    for cid in all_ids}

        sorted_ids = sorted(combined, key=combined.__getitem__, reverse=True)[:top_k]
        return [
            SearchResult(
                chunk_id=cid,
                content=result_map[cid].content,
                score=combined[cid],
                metadata=result_map[cid].metadata,
                rank=rank,
            )
            for rank, cid in enumerate(sorted_ids)
        ]


# ---------------------------------------------------------------------------
# Query Transformation Utilities
# ---------------------------------------------------------------------------
class QueryTransformer:
    """
    Query transformations to improve retrieval recall.

    1. HyDE   - embed a hypothetical answer instead of the question
    2. MultiQuery - generate N paraphrases, union results
    3. StepBack  - raise abstraction level of the question
    """

    def __init__(self, llm_fn):
        """
        llm_fn: callable(prompt: str) → str
        Inject any LLM (OpenAI, Ollama, etc.)
        """
        self._llm = llm_fn

    def hyde(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        prompt = (
            "Write a short, factual paragraph that directly answers the question below. "
            "Do not include the question itself. Be specific and detailed.\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        return self._llm(prompt)

    def multi_query(self, query: str, n: int = 3) -> List[str]:
        """Generate n paraphrases of the query for ensemble retrieval."""
        prompt = (
            f"Generate {n} different versions of the following question. "
            "Each version should ask for the same information but use different wording. "
            "Output ONLY the questions, one per line, no numbering.\n\n"
            f"Original question: {query}"
        )
        raw = self._llm(prompt)
        queries = [q.strip() for q in raw.strip().split("\n") if q.strip()]
        return [query] + queries[:n]      # include original

    def step_back(self, query: str) -> str:
        """Generate a broader, more abstract version of the question."""
        prompt = (
            "Rewrite the following specific question as a broader, more general question "
            "that would help retrieve relevant background information.\n\n"
            f"Specific question: {query}\n\nBroader question:"
        )
        return self._llm(prompt).strip()


# ---------------------------------------------------------------------------
# MMR — Maximal Marginal Relevance
# ---------------------------------------------------------------------------
def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: List[SearchResult],
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> List[SearchResult]:
    """
    MMR greedily selects candidates that are relevant to the query
    but dissimilar to already-selected items.

    score(d) = λ·sim(d, q) − (1−λ)·max_{s∈S} sim(d, s)

    lambda_param=1 → pure relevance, lambda_param=0 → pure diversity.
    """
    if not candidates:
        return []

    q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    vecs = candidate_embeddings / (
        np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
    )
    relevance = vecs @ q

    selected_indices: List[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        if not selected_indices:
            best = int(np.argmax(relevance))
            selected_indices.append(best)
            remaining.remove(best)
            continue

        sel_vecs = vecs[selected_indices]
        scores = []
        for idx in remaining:
            rel = float(relevance[idx])
            redundancy = float(np.max(vecs[idx] @ sel_vecs.T))
            mmr_score = lambda_param * rel - (1 - lambda_param) * redundancy
            scores.append((idx, mmr_score))

        best_idx = max(scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]
