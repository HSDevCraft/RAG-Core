# Module 02 — Embeddings
### `embeddings/embedding.py`

---

## Table of Contents
1. [What Are Embeddings?](#1-what-are-embeddings)
2. [EmbeddingEngine Overview](#2-embeddingengine-overview)
3. [Provider Implementations](#3-provider-implementations)
4. [EmbeddingResult Model](#4-embeddingresult-model)
5. [Caching Layer](#5-caching-layer)
6. [Batching & Normalization](#6-batching--normalization)
7. [Model Selection Guide](#7-model-selection-guide)
8. [Quick Reference](#8-quick-reference)
9. [Common Pitfalls](#9-common-pitfalls)

---

## 1. What Are Embeddings?

An **embedding** is a fixed-length float vector that encodes the *semantic meaning*
of text. Two texts with similar meaning have vectors close together in
high-dimensional space (measured by cosine similarity).

```
"What is the refund policy?"   →  [0.12, -0.34, 0.89, …]  (384 dims)
"How do I return a product?"   →  [0.11, -0.35, 0.87, …]  (384 dims)
                                   ↑ very close → high cosine similarity

"The weather is sunny today"   →  [0.98,  0.12, -0.45, …]
                                   ↑ far away → low cosine similarity
```

### Cosine Similarity Formula

```
similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)

Range: -1 (opposite) → 0 (orthogonal) → 1 (identical)
Threshold for "relevant": typically > 0.7
```

---

## 2. EmbeddingEngine Overview

```python
from embeddings import EmbeddingEngine

engine = EmbeddingEngine(
    provider="sentence_transformers",   # "sentence_transformers" | "openai" | "cohere"
    model_name="all-MiniLM-L6-v2",
    device="cpu",                       # "cpu" | "cuda" | "mps"
    use_cache=True,                     # disk cache for repeated texts
    cache_dir="./.embed_cache",
    normalize=True,                     # L2-normalize → cosine = dot product
)
```

### Class Hierarchy

```
EmbeddingEngine                    ← public façade
├── SentenceTransformerEmbedder    ← local, free, fast
├── OpenAIEmbedder                 ← API, paid, high quality
└── CohereEmbedder                 ← API, paid, multilingual
```

### Core Methods

```python
# Encode a list of texts (chunks/documents)
result: EmbeddingResult = engine.encode(
    texts=["text 1", "text 2", "text 3"],
    batch_size=64,          # process this many at once
    show_progress=True,
)

# Encode a single query (optimised path — no caching needed)
vec: np.ndarray = engine.encode_query("What is the refund policy?")

# Inspect
print(engine.dimension)    # 384  (model-specific)
print(engine.provider)     # "sentence_transformers"
```

---

## 3. Provider Implementations

### 3.1 SentenceTransformerEmbedder — Local, Free

```python
engine = EmbeddingEngine(
    provider="sentence_transformers",
    model_name="all-MiniLM-L6-v2",   # 384-dim, 80MB, very fast
    device="cpu",
)
```

**Under the hood**:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts, batch_size=64, normalize_embeddings=True)
```

First run downloads the model (~80MB) to `~/.cache/huggingface/`.
Subsequent runs load from cache (~0.5s startup).

### 3.2 OpenAIEmbedder — API-based

```python
engine = EmbeddingEngine(
    provider="openai",
    model_name="text-embedding-3-small",   # 1536-dim
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

**API call**:
```python
response = openai_client.embeddings.create(
    input=texts,
    model="text-embedding-3-small",
)
vectors = [item.embedding for item in response.data]
```

**Limits**: max 8191 tokens per text, max 2048 texts per batch.

### 3.3 CohereEmbedder — Multilingual

```python
engine = EmbeddingEngine(
    provider="cohere",
    model_name="embed-multilingual-v3.0",   # 1024-dim, 100+ languages
    api_key=os.getenv("COHERE_API_KEY"),
    input_type="search_document",           # "search_document" | "search_query"
)
```

> ⚠️ **Cohere asymmetric embeddings**: documents and queries must use different
> `input_type` values. The engine handles this automatically — `encode()` uses
> `"search_document"`, `encode_query()` uses `"search_query"`.

---

## 4. EmbeddingResult Model

```python
@dataclass
class EmbeddingResult:
    embeddings:  np.ndarray   # shape: (n_texts, dimension) — float32
    model:       str          # model name used
    dimension:   int          # vector size (e.g., 384)
    latency_ms:  float        # total time to embed all texts

# Usage
result = engine.encode(["text a", "text b"])
print(result.embeddings.shape)   # (2, 384)
print(result.embeddings.dtype)   # float32
print(result.latency_ms)         # 45.3
```

Vectors are always **L2-normalised** by default so cosine similarity equals
the dot product — useful for FAISS IVF and HNSW indexes.

---

## 5. Caching Layer

Re-embedding the same text is wasteful. The cache stores
`(model_name, text)` → `vector` on disk.

```
First call:  text → embed (150ms) → save to .embed_cache/
Second call: text → load from .embed_cache/ (0.2ms) → return
```

**Implementation** (disk cache via `diskcache`):
```python
cache_key = sha256(f"{model_name}:{text}").hexdigest()[:16]
cached = disk_cache.get(cache_key)
if cached is not None:
    return cached                           # cache HIT
vector = model.encode(text)
disk_cache.set(cache_key, vector, expire=86400)   # cache MISS → store
return vector
```

**When caching matters**:
- Re-indexing a corpus after config changes → 100× speedup
- High-traffic API where many users ask the same questions
- Development: avoid API costs while iterating

**Disable for benchmarking**:
```python
engine = EmbeddingEngine(..., use_cache=False)
```

---

## 6. Batching & Normalization

### Batching

GPU throughput scales with batch size. The engine processes texts in
`batch_size` chunks to avoid OOM:

```
texts = 10,000 chunks, batch_size = 64

Batch 0:  texts[  0: 64]  → GPU forward pass → 64 vectors
Batch 1:  texts[ 64:128]  → GPU forward pass → 64 vectors
…
Batch 156: texts[9984:10000] → 16 vectors

Total = 157 GPU calls instead of 10,000
```

### L2 Normalization

```python
# Raw vector
v = np.array([3.0, 4.0])   # ‖v‖ = 5

# Normalized
v_norm = v / np.linalg.norm(v)   # [0.6, 0.8]   ‖v_norm‖ = 1

# Benefit: cosine_similarity(a, b) = dot(a, b)  when both normalized
# → enables fast matrix multiplication instead of expensive trig operations
```

All vectors from `EmbeddingEngine` are normalized unless you set
`normalize=False` (not recommended unless using L2 distance metrics).

---

## 7. Model Selection Guide

| Model | Dim | Size | Speed | Quality | Use case |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 80MB | ⚡⚡⚡ | ★★★ | Development, small corpus |
| `all-mpnet-base-v2` | 768 | 420MB | ⚡⚡ | ★★★★ | Production, English |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3GB | ⚡ | ★★★★★ | Best local English |
| `text-embedding-3-small` | 1536 | API | ⚡⚡ | ★★★★ | OpenAI, cost-effective |
| `text-embedding-3-large` | 3072 | API | ⚡ | ★★★★★ | OpenAI, highest quality |
| `embed-multilingual-v3.0` | 1024 | API | ⚡⚡ | ★★★★★ | Non-English, multilingual |

### Decision Tree

```
Is the corpus in English only?
    YES → Do you have GPU?
        YES → BAAI/bge-large-en-v1.5  (best local)
        NO  → all-mpnet-base-v2       (CPU production)
    NO  → embed-multilingual-v3.0    (Cohere)

Is cost a hard constraint?
    YES → SentenceTransformer (free, local)
    NO  → text-embedding-3-small (OpenAI, best value)

Is latency < 50ms per query critical?
    YES → all-MiniLM-L6-v2 (384-dim, fastest)
    NO  → any of the above
```

---

## 8. Quick Reference

```python
from embeddings import EmbeddingEngine

# ── Local (no API key, recommended for dev) ──────────────────────────────
engine = EmbeddingEngine("sentence_transformers", "all-MiniLM-L6-v2")

# ── OpenAI ───────────────────────────────────────────────────────────────
engine = EmbeddingEngine("openai", "text-embedding-3-small",
                         api_key=os.getenv("OPENAI_API_KEY"))

# ── Cohere (multilingual) ─────────────────────────────────────────────────
engine = EmbeddingEngine("cohere", "embed-multilingual-v3.0",
                         api_key=os.getenv("COHERE_API_KEY"))

# ── Embed chunks (bulk) ──────────────────────────────────────────────────
result = engine.encode([c.content for c in chunks], batch_size=128)
vectors = result.embeddings          # np.ndarray (n, dim)

# ── Embed a query (single) ───────────────────────────────────────────────
q_vec = engine.encode_query("What is the return policy?")
# q_vec.shape → (384,)

# ── Similarity between two texts ─────────────────────────────────────────
v1 = engine.encode_query("refund")
v2 = engine.encode_query("return policy")
similarity = float(np.dot(v1, v2))    # cosine sim (both normalized)
print(f"Similarity: {similarity:.3f}")  # e.g., 0.823

# ── Dimension ────────────────────────────────────────────────────────────
print(engine.dimension)   # pass to VectorStoreFactory.create(dimension=...)
```

---

## 9. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| **Mismatched dimensions** | FAISS `AssertionError` on add | Check `engine.dimension` == `store.dimension` |
| **Query vs document asymmetry** | Cohere results poor | Engine handles automatically via `input_type` |
| **Not normalizing** | Cosine search gives wrong results with dot product | Keep `normalize=True` (default) |
| **Caching stale embeddings** | Old model embeddings mixed with new | Clear `.embed_cache/` after model change |
| **OOM on large batch** | CUDA out of memory | Reduce `batch_size` (try 32 or 16) |
| **GPU not used** | Slow encoding (CPU speeds) | Set `device="cuda"` explicitly |
| **OpenAI rate limit** | `RateLimitError` during indexing | Reduce `batch_size`; add retry with backoff |
