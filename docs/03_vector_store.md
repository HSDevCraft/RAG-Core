# Module 03 — Vector Store
### `vector_store/vector_store.py`

---

## Table of Contents
1. [What Is a Vector Store?](#1-what-is-a-vector-store)
2. [SearchResult Model](#2-searchresult-model)
3. [BaseVectorStore Interface](#3-basevectorstore-interface)
4. [FAISSVectorStore](#4-faissvectorstore)
5. [ChromaVectorStore](#5-chromavectorstore)
6. [PineconeVectorStore](#6-pineconevectorstore)
7. [VectorStoreFactory](#7-vectorstorefactory)
8. [Index Types Compared](#8-index-types-compared)
9. [Metadata Filtering](#9-metadata-filtering)
10. [Persistence (Save / Load)](#10-persistence-save--load)
11. [Quick Reference](#11-quick-reference)
12. [Common Pitfalls](#12-common-pitfalls)

---

## 1. What Is a Vector Store?

A vector store is a database optimised for **approximate nearest-neighbour (ANN)**
search — finding the `k` vectors most similar to a query vector.

```
Query vector: [0.12, -0.34, 0.89, …]
                        │
                        ▼
              ┌─────────────────────┐
              │    Vector Index     │
              │  (10M stored vecs)  │
              └─────────────────────┘
                        │
                        ▼ top-k nearest neighbors
              chunk-042  score=0.923
              chunk-189  score=0.901
              chunk-007  score=0.887
```

**Why approximate?** Exact brute-force search over millions of 1024-dim vectors
is O(n × d) — too slow. ANN trades a small accuracy loss (< 1%) for 100–1000×
speedup.

---

## 2. SearchResult Model

Every search returns a list of `SearchResult` objects:

```python
@dataclass
class SearchResult:
    chunk_id:  str              # links to the original Chunk
    content:   str              # the text of the chunk
    score:     float            # cosine similarity (0–1; higher = more relevant)
    metadata:  Dict[str, Any]   # source, file_type, page, …
    rank:      int = 0          # position in result list (0 = best)
```

```python
results = store.search(query_vector, top_k=5)
for r in results:
    print(f"[{r.rank}] {r.score:.3f}  {r.chunk_id}  {r.content[:60]}")
    print(f"       source={r.metadata['source']}, page={r.metadata.get('page')}")
```

---

## 3. BaseVectorStore Interface

All backends implement the same 5 methods:

```python
class BaseVectorStore(ABC):

    def add(
        self,
        ids:       List[str],           # chunk IDs
        vectors:   np.ndarray,          # (n, dim) float32
        texts:     List[str],           # raw text for retrieval
        metadatas: List[Dict] = None,   # optional per-chunk metadata
    ) -> None: ...

    def search(
        self,
        query_vector: np.ndarray,       # (dim,) float32
        top_k:        int = 10,
        filter:       Dict = None,      # metadata filter (optional)
    ) -> List[SearchResult]: ...

    def delete(self, ids: List[str]) -> None: ...

    def persist(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...

    @property
    def count(self) -> int: ...         # number of indexed vectors
```

---

## 4. FAISSVectorStore

FAISS (Facebook AI Similarity Search) is a pure in-process C++ library —
**no server, no network, maximum speed**.

### 4.1 Creation

```python
from vector_store import FAISSVectorStore

store = FAISSVectorStore(
    dimension=384,          # must match your embedding model
    index_type="hnsw",      # "flat" | "ivf" | "hnsw"
    metric="cosine",        # "cosine" | "l2" | "ip" (inner product)
    hnsw_m=32,              # HNSW: connections per node
    hnsw_ef_construction=200,
    hnsw_ef_search=50,
)
```

### 4.2 FAISS Index Types

```
flat   ─── Brute force. 100% exact. Slow at scale.
           Best for: < 50K vectors, dev/testing.

ivf    ─── Inverted File. Clusters vectors; search only nearest clusters.
           Requires: training on corpus before use.
           Best for: 50K–10M vectors.

hnsw   ─── Hierarchical Navigable Small World graph.
           Fastest at query time; no training needed.
           Best for: production, > 10K vectors, latency-critical.
```

#### IVF Setup (requires training)
```python
store = FAISSVectorStore(
    dimension=384,
    index_type="ivf",
    ivf_nlist=256,      # number of clusters (√n_vectors rule of thumb)
    ivf_nprobe=32,      # clusters to search at query time (more = slower but more accurate)
)
# Training happens automatically on first .add() call
```

#### HNSW Setup (recommended for production)
```python
store = FAISSVectorStore(
    dimension=384,
    index_type="hnsw",
    hnsw_m=32,                  # degree of graph (16–64; 32 is balanced)
    hnsw_ef_construction=200,   # quality of build (higher = better but slower build)
    hnsw_ef_search=50,          # quality of search (higher = better but slower query)
)
```

### 4.3 Add & Search

```python
# Add
store.add(
    ids=["chunk-0", "chunk-1", "chunk-2"],
    vectors=np.array([[...], [...], [...]]),   # (3, 384) float32
    texts=["text 0", "text 1", "text 2"],
    metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}, {"source": "a.pdf"}],
)

# Search
results = store.search(query_vec, top_k=5)
# results is sorted by score descending

# With filter
results = store.search(
    query_vec, top_k=10,
    filter={"source": "a.pdf"}           # only from this file
)
results = store.search(
    query_vec, top_k=10,
    filter={"source": ["a.pdf", "b.pdf"]}  # from either file (list = OR)
)
```

---

## 5. ChromaVectorStore

ChromaDB stores vectors AND metadata in a single SQLite database.
Supports **rich metadata filtering** natively.

### 5.1 Modes

```python
from vector_store import ChromaVectorStore

# Mode 1: Ephemeral (in-memory, tests only)
store = ChromaVectorStore(
    collection_name="rag_chunks",
    persist_dir=None,
)

# Mode 2: Persistent (on-disk SQLite)
store = ChromaVectorStore(
    collection_name="rag_chunks",
    persist_dir="./chroma_data",
)

# Mode 3: Client-server (Docker container)
store = ChromaVectorStore(
    collection_name="rag_chunks",
    host="chroma",
    port=8000,
)
```

### 5.2 Rich Filtering

ChromaDB supports operators that FAISS doesn't:

```python
# Equality
results = store.search(q, filter={"file_type": "pdf"})

# Range (not available in FAISS Flat)
results = store.search(q, filter={"page": {"$gte": 5, "$lte": 10}})

# Contains
results = store.search(q, filter={"source": {"$contains": "policy"}})

# AND / OR
results = store.search(q, filter={
    "$and": [
        {"file_type": "pdf"},
        {"page": {"$gte": 1}}
    ]
})
```

### 5.3 Metadata Constraint

ChromaDB metadata values must be `str | int | float | bool`.
The store auto-converts other types to `str`:

```python
# This dict will be automatically cleaned before adding
metadata = {"source": "a.pdf", "tags": ["refund", "policy"]}
# → {"source": "a.pdf", "tags": "['refund', 'policy']"}  (list → str)
```

---

## 6. PineconeVectorStore

Managed cloud vector database. Zero infrastructure, scales automatically.

```python
from vector_store.vector_store import PineconeVectorStore

store = PineconeVectorStore(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name="rag-production",
    dimension=384,
    metric="cosine",
    environment="us-east-1-aws",
)
```

**When to use Pinecone**:
- Team doesn't want to manage FAISS persistence
- Corpus > 10M vectors (FAISS starts requiring large RAM)
- Multi-region deployment
- Real-time updates at high ingestion rate

---

## 7. VectorStoreFactory

```python
from vector_store import VectorStoreFactory

# FAISS (default for most use cases)
store = VectorStoreFactory.create(
    backend="faiss",
    dimension=384,
    index_type="hnsw",
    metric="cosine",
    persist_dir="./vector_store_data",
)

# Chroma
store = VectorStoreFactory.create(
    backend="chroma",
    collection_name="my_collection",
    persist_dir="./chroma_data",
)

# Pinecone
store = VectorStoreFactory.create(
    backend="pinecone",
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name="rag-prod",
    dimension=1024,
)
```

---

## 8. Index Types Compared

| Feature | Flat (Exact) | IVF | HNSW |
|---|---|---|---|
| **Search accuracy** | 100% | 95–99% | 97–99% |
| **Query latency** | O(n) — slow | O(√n) — fast | O(log n) — fastest |
| **Build time** | Instant | Requires training | Medium |
| **Memory usage** | n × dim × 4 bytes | Slightly more | + graph (~10%) |
| **Delete support** | ✅ | ✅ | ⚠️ (mark-delete) |
| **Best for** | < 50K vectors | 50K–10M | > 10K, production |

### Memory Estimate

```
n vectors × dim × 4 bytes = RAM needed

384-dim model, 1M vectors:
  = 1,000,000 × 384 × 4 = 1.5 GB

1024-dim model, 1M vectors:
  = 1,000,000 × 1024 × 4 = 4 GB

HNSW overhead: ~10% extra for graph structure
IVF overhead:  negligible
```

---

## 9. Metadata Filtering

FAISS doesn't support native metadata filtering — it's implemented via
post-processing in our store wrapper:

```python
# FAISS filter flow:
# 1. Search top_k × filter_factor candidates  (over-fetch)
# 2. Apply Python-level metadata filter
# 3. Return top_k matching results

store.search(query, top_k=5, filter={"file_type": "pdf"})
# internally searches top 50 (top_k × 10), then filters
```

ChromaDB filters at the DB level (more efficient for large corpora).

**Supported filter operators** (FAISS wrapper):
```python
{"key": "value"}             # equality
{"key": ["v1", "v2"]}       # list → any match (OR)
{"key": {"$gte": 5}}        # ≥   (numeric)
{"key": {"$lte": 10}}       # ≤   (numeric)
{"key": {"$gt": 5}}         # >
{"key": {"$lt": 10}}        # <
```

---

## 10. Persistence (Save / Load)

```python
# Save index + metadata to disk
store.persist("./vector_store_data/production")
# Creates:
#   ./vector_store_data/production/index.faiss
#   ./vector_store_data/production/metadata.pkl

# Load on restart (no re-indexing needed)
store = FAISSVectorStore(dimension=384, index_type="hnsw")
store.load("./vector_store_data/production")
print(store.count)   # all vectors restored
```

**In RAGPipeline**: persistence is handled automatically via
`pipeline.save(path)` / `pipeline.load(path)`, which also saves the BM25 index.

---

## 11. Quick Reference

```python
from vector_store import VectorStoreFactory, FAISSVectorStore

# ── Create ───────────────────────────────────────────────────────────────
store = FAISSVectorStore(dimension=384, index_type="hnsw", metric="cosine")

# ── Add vectors ──────────────────────────────────────────────────────────
store.add(ids, vectors, texts, metadatas)

# ── Search ───────────────────────────────────────────────────────────────
results = store.search(query_vec, top_k=10)
results = store.search(query_vec, top_k=10, filter={"file_type": "pdf"})

# ── Delete ───────────────────────────────────────────────────────────────
store.delete(["chunk-5", "chunk-12"])

# ── Count ────────────────────────────────────────────────────────────────
print(store.count)    # number of indexed vectors

# ── Persist ──────────────────────────────────────────────────────────────
store.persist("./my_index")
store.load("./my_index")   # restore on next run

# ── Backend selection quick guide ─────────────────────────────────────────
# Development / small corpus    → FAISSVectorStore, index_type="flat"
# Production (single node)      → FAISSVectorStore, index_type="hnsw"
# Rich metadata filters needed  → ChromaVectorStore
# Managed / serverless          → PineconeVectorStore
# Multi-model experiments       → VectorStoreFactory.create(backend=...)
```

---

## 12. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| **Wrong dimension** | FAISS `AssertionError` | `store.dimension == engine.dimension` |
| **IVF not trained** | `"Index not trained"` error | Add ≥ nlist × 39 vectors before search |
| **Forgetting to persist** | Index lost on restart | Call `store.persist()` or `pipeline.save()` |
| **HNSW delete** | Score oddities after delete | HNSW uses mark-delete; rebuild periodically |
| **Chroma metadata type** | `TypeError` on add | Auto-convert non-primitive types to str |
| **Flat on large corpus** | Search takes > 1s | Switch to HNSW for > 50K vectors |
| **IVF nprobe too low** | Poor recall (< 90%) | Increase nprobe (try nlist/8 → nlist/4) |
| **Filter returns empty** | Zero results even with data | Check metadata key spelling exactly |
