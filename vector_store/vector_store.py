"""
vector_store.py
───────────────
Vector database abstraction layer.

Backends:
  - FAISS      (in-process, blazing fast, production-proven at Meta scale)
  - ChromaDB   (embedded / client-server, metadata filtering, easy dev setup)
  - Pinecone   (managed cloud, serverless, zero ops overhead)

Index types and trade-offs:
  ┌──────────┬────────────────┬─────────────────┬────────────────────┐
  │ Index    │ Build time     │ Query latency   │ Memory             │
  ├──────────┼────────────────┼─────────────────┼────────────────────┤
  │ Flat     │ O(1)           │ O(n) — exact    │ n × dim × 4 bytes  │
  │ IVF      │ O(n log n)     │ O(√n)           │ Flat + centroids   │
  │ HNSW     │ O(n log n)     │ O(log n)        │ ~2× Flat           │
  └──────────┴────────────────┴─────────────────┴────────────────────┘

Similarity metrics:
  cosine  →  normalized vectors, inner-product ≡ cosine
  ip      →  raw inner product (use when embeddings are already normalized)
  l2      →  Euclidean distance (less common for text)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval result type
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    rank: int = 0


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, ids: List[str], embeddings: np.ndarray,
            documents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """Insert vectors and their payloads."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 10,
               filter: Optional[dict] = None) -> List[SearchResult]:
        """ANN search; returns top_k results with scores."""

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Remove vectors by ID."""

    @abstractmethod
    def persist(self, path: str) -> None:
        """Persist index to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""

    @property
    @abstractmethod
    def count(self) -> int:
        """Number of vectors stored."""


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------
class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store.

    Index strategy selection:
      n < 10_000   → IndexFlatIP  (exact, no training needed)
      n < 1_000_000 → IndexIVFFlat (nlist=128, nprobe=16)
      n > 1_000_000 → IndexHNSWFlat (M=16, ef=200) — no training, scalable

    Production note: FAISS does not support native metadata filtering.
    Metadata is stored in a parallel dict keyed by internal FAISS id.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "hnsw",       # "flat" | "ivf" | "hnsw"
        metric: str = "cosine",          # "cosine" | "ip" | "l2"
        nlist: int = 128,                # IVF: number of centroids
        nprobe: int = 16,                # IVF: cells to search
        hnsw_m: int = 16,               # HNSW: connections per layer
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
    ):
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError("Install faiss: pip install faiss-cpu")

        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self._index = None
        self._id_map: Dict[int, str] = {}          # faiss internal id → chunk_id
        self._reverse_map: Dict[str, int] = {}     # chunk_id → faiss internal id
        self._documents: Dict[str, str] = {}       # chunk_id → content
        self._metadatas: Dict[str, dict] = {}      # chunk_id → metadata
        self._next_id = 0

        self._nlist = nlist
        self._nprobe = nprobe
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search

        self._build_index()

    def _build_index(self):
        faiss = self._faiss
        d = self.dimension

        if self.metric == "l2":
            metric_flag = faiss.METRIC_L2
        else:
            metric_flag = faiss.METRIC_INNER_PRODUCT   # cosine on normalized vecs

        if self.index_type == "flat":
            if self.metric == "l2":
                self._index = faiss.IndexFlatL2(d)
            else:
                self._index = faiss.IndexFlatIP(d)

        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFFlat(quantizer, d, self._nlist, metric_flag)
            self._index.nprobe = self._nprobe

        elif self.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(d, self._hnsw_m, metric_flag)
            self._index.hnsw.efConstruction = self._hnsw_ef_construction
            self._index.hnsw.efSearch = self._hnsw_ef_search
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        logger.info("FAISS index built: type=%s metric=%s dim=%d",
                    self.index_type, self.metric, d)

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
            return (embeddings / norms).astype(np.float32)
        return embeddings.astype(np.float32)

    def add(self, ids: List[str], embeddings: np.ndarray,
            documents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings must have the same length")

        vecs = self._normalize(embeddings)

        # IVF requires training
        if self.index_type == "ivf" and not self._index.is_trained:
            if len(vecs) < self._nlist:
                raise ValueError(
                    f"IVF requires at least {self._nlist} vectors for training; "
                    f"got {len(vecs)}. Use index_type='flat' for small datasets."
                )
            logger.info("Training IVF index on %d vectors...", len(vecs))
            self._index.train(vecs)

        faiss_ids = np.arange(self._next_id, self._next_id + len(ids), dtype=np.int64)
        self._index.add_with_ids(vecs, faiss_ids)

        for i, chunk_id in enumerate(ids):
            internal = int(faiss_ids[i])
            self._id_map[internal] = chunk_id
            self._reverse_map[chunk_id] = internal
            self._documents[chunk_id] = documents[i]
            if metadatas:
                self._metadatas[chunk_id] = metadatas[i]

        self._next_id += len(ids)
        logger.info("FAISS: added %d vectors. Total: %d", len(ids), self._next_id)

    def search(self, query_embedding: np.ndarray, top_k: int = 10,
               filter: Optional[dict] = None) -> List[SearchResult]:
        query = self._normalize(query_embedding.reshape(1, -1))
        scores, faiss_ids = self._index.search(query, top_k * 2 if filter else top_k)

        results: List[SearchResult] = []
        for rank, (score, fid) in enumerate(zip(scores[0], faiss_ids[0])):
            if fid == -1:
                continue
            chunk_id = self._id_map.get(int(fid))
            if chunk_id is None:
                continue
            meta = self._metadatas.get(chunk_id, {})

            # client-side metadata filter (FAISS lacks native support)
            if filter and not self._match_filter(meta, filter):
                continue

            results.append(SearchResult(
                chunk_id=chunk_id,
                content=self._documents.get(chunk_id, ""),
                score=float(score),
                metadata=meta,
                rank=rank,
            ))

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _match_filter(metadata: dict, filter_dict: dict) -> bool:
        for k, v in filter_dict.items():
            if isinstance(v, list):
                if metadata.get(k) not in v:
                    return False
            elif metadata.get(k) != v:
                return False
        return True

    def delete(self, ids: List[str]) -> None:
        internal_ids = np.array(
            [self._reverse_map[cid] for cid in ids if cid in self._reverse_map],
            dtype=np.int64,
        )
        if len(internal_ids) > 0:
            self._index.remove_ids(internal_ids)
        for cid in ids:
            fid = self._reverse_map.pop(cid, None)
            if fid is not None:
                self._id_map.pop(fid, None)
            self._documents.pop(cid, None)
            self._metadatas.pop(cid, None)

    def persist(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(p / "index.faiss"))
        with open(p / "metadata.pkl", "wb") as f:
            pickle.dump({
                "id_map": self._id_map,
                "reverse_map": self._reverse_map,
                "documents": self._documents,
                "metadatas": self._metadatas,
                "next_id": self._next_id,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
            }, f)
        logger.info("FAISS index persisted to %s", path)

    def load(self, path: str) -> None:
        p = Path(path)
        self._index = self._faiss.read_index(str(p / "index.faiss"))
        with open(p / "metadata.pkl", "rb") as f:
            state = pickle.load(f)
        self._id_map = state["id_map"]
        self._reverse_map = state["reverse_map"]
        self._documents = state["documents"]
        self._metadatas = state["metadatas"]
        self._next_id = state["next_id"]
        logger.info("FAISS index loaded from %s (%d vectors)", path, self._next_id)

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index else 0


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------
class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-backed store. Supports:
      - Embedded mode (in-process, dev/testing)
      - Client-server mode (production)
      - Native metadata filtering via `where` clauses

    Trade-off vs FAISS:
      + Better developer experience, REST API, persistent by default
      + Native metadata filtering
      - Slower for very large (>10M) collections
      - Extra process / network hop in server mode
    """

    def __init__(self, collection_name: str = "rag_collection",
                 persist_dir: Optional[str] = "./chroma_data",
                 host: Optional[str] = None,
                 port: int = 8000,
                 embedding_function=None):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")

        import chromadb

        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        elif persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function,
        )
        self._collection_name = collection_name
        logger.info("ChromaDB collection '%s' ready (%d items)",
                    collection_name, self._collection.count())

    def add(self, ids: List[str], embeddings: np.ndarray,
            documents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        batch_size = 5000       # Chroma batching limit
        for i in range(0, len(ids), batch_size):
            sl = slice(i, i + batch_size)
            self._collection.upsert(
                ids=ids[sl],
                embeddings=embeddings[sl].tolist(),
                documents=documents[sl],
                metadatas=(metadatas[sl] if metadatas else [{}] * len(ids[sl])),
            )
        logger.info("ChromaDB: upserted %d vectors. Total: %d",
                    len(ids), self._collection.count())

    def search(self, query_embedding: np.ndarray, top_k: int = 10,
               filter: Optional[dict] = None) -> List[SearchResult]:
        kwargs: dict = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter:
            kwargs["where"] = filter

        response = self._collection.query(**kwargs)
        results: List[SearchResult] = []
        ids = response["ids"][0]
        docs = response["documents"][0]
        metas = response["metadatas"][0]
        dists = response["distances"][0]

        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
            score = 1 - dist    # Chroma returns distance, convert to similarity
            results.append(SearchResult(
                chunk_id=cid,
                content=doc,
                score=score,
                metadata=meta or {},
                rank=rank,
            ))
        return results

    def delete(self, ids: List[str]) -> None:
        self._collection.delete(ids=ids)

    def persist(self, path: str) -> None:
        logger.info("ChromaDB with PersistentClient auto-persists. Path: %s", path)

    def load(self, path: str) -> None:
        logger.info("ChromaDB: collection loaded from persistent storage at %s", path)

    @property
    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# Pinecone backend (stub with full interface)
# ---------------------------------------------------------------------------
class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone serverless vector store.

    When to use Pinecone:
      - You want zero infra management
      - Dataset > 10M vectors (Pinecone scales horizontally)
      - Multi-tenant SaaS applications (namespace isolation)

    Cost note: ~$0.096 / 1M reads (serverless), ~$70/mo (pod-based starter)
    """

    def __init__(self, index_name: str, dimension: int,
                 api_key: Optional[str] = None, environment: str = "us-east-1-aws",
                 namespace: str = ""):
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError:
            raise ImportError("Install pinecone: pip install pinecone-client")

        from pinecone import Pinecone, ServerlessSpec

        self._pc = Pinecone(api_key=api_key or os.getenv("PINECONE_API_KEY"))
        self._dimension = dimension
        self._namespace = namespace

        existing = [idx.name for idx in self._pc.list_indexes()]
        if index_name not in existing:
            self._pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("Pinecone: created index '%s'", index_name)

        self._index = self._pc.Index(index_name)

    def add(self, ids: List[str], embeddings: np.ndarray,
            documents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            sl = slice(i, i + batch_size)
            vectors = []
            for j, (cid, vec) in enumerate(zip(ids[sl], embeddings[sl])):
                meta = (metadatas[i + j] if metadatas else {})
                meta["_content"] = documents[i + j][:500]   # Pinecone metadata size limit
                vectors.append({"id": cid, "values": vec.tolist(), "metadata": meta})
            self._index.upsert(vectors=vectors, namespace=self._namespace)

        logger.info("Pinecone: upserted %d vectors", len(ids))

    def search(self, query_embedding: np.ndarray, top_k: int = 10,
               filter: Optional[dict] = None) -> List[SearchResult]:
        kwargs: dict = {
            "vector": query_embedding.tolist(),
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self._namespace,
        }
        if filter:
            kwargs["filter"] = filter

        response = self._index.query(**kwargs)
        results: List[SearchResult] = []
        for rank, match in enumerate(response.matches):
            meta = match.metadata or {}
            content = meta.pop("_content", "")
            results.append(SearchResult(
                chunk_id=match.id,
                content=content,
                score=float(match.score),
                metadata=meta,
                rank=rank,
            ))
        return results

    def delete(self, ids: List[str]) -> None:
        self._index.delete(ids=ids, namespace=self._namespace)

    def persist(self, path: str) -> None:
        logger.info("Pinecone is cloud-managed; no local persist needed.")

    def load(self, path: str) -> None:
        logger.info("Pinecone index loaded from cloud.")

    @property
    def count(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.total_vector_count


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
class VectorStoreFactory:
    """
    Instantiate the right vector store from config.

    Usage:
        store = VectorStoreFactory.create(backend="faiss", dimension=1024)
        store = VectorStoreFactory.create(backend="chroma", collection_name="docs")
        store = VectorStoreFactory.create(backend="pinecone", index_name="prod", dimension=1024)
    """

    @staticmethod
    def create(backend: str = "faiss", **kwargs) -> BaseVectorStore:
        backend = backend.lower()
        if backend == "faiss":
            return FAISSVectorStore(**kwargs)
        elif backend == "chroma":
            return ChromaVectorStore(**kwargs)
        elif backend == "pinecone":
            return PineconeVectorStore(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose from faiss, chroma, pinecone")
