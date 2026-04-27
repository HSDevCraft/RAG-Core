# Module 09 — Cache & Configuration
### `utils/cache.py` · `config.py`

---

## Table of Contents
1. [Why Cache in a RAG System?](#1-why-cache-in-a-rag-system)
2. [Cache Backends](#2-cache-backends)
3. [Cache Interface](#3-cache-interface)
4. [RedisCache](#4-rediscache)
5. [DiskCache](#5-diskcache)
6. [NullCache](#6-nullcache)
7. [Cache Key Helpers](#7-cache-key-helpers)
8. [Factory & Auto-configure](#8-factory--auto-configure)
9. [Configuration System](#9-configuration-system)
10. [Config Dataclasses Reference](#10-config-dataclasses-reference)
11. [Quick Reference](#11-quick-reference)
12. [Common Pitfalls](#12-common-pitfalls)

---

## 1. Why Cache in a RAG System?

RAG has three expensive operations — all three benefit from caching:

| Operation | Typical latency | Cache benefit |
|---|---|---|
| Embedding a query | 5–50ms (local), 50–200ms (API) | High — same query asked often |
| LLM generation | 500–5000ms | Very high — identical queries are common |
| Retrieval (FAISS) | 1–20ms | Low — already fast; cache if top-K is stable |

```
Without cache:          With cache:
User asks Q1 → 2000ms   User asks Q1 → 2000ms  (miss)
User asks Q1 → 2000ms   User asks Q1 → 2ms      (hit!)
User asks Q1 → 2000ms   User asks Q1 → 2ms      (hit!)
```

At scale (1000 users, 20% repeated queries):
- Without cache: 1000 × 2000ms × 1.0 = 2000 seconds of LLM time
- With cache:    1000 × 2000ms × 0.8 = 1600 seconds  (+ 200 × 2ms)
- **Savings: 20% latency reduction + 20% cost reduction**

---

## 2. Cache Backends

```
CacheBackend (ABC)
├── RedisCache    ← distributed; shared across all API workers; production
├── DiskCache     ← local SQLite; single-node; development / single-server
└── NullCache     ← no-op; for testing and cache-disabled mode
```

**Choosing a backend**:

```
Multiple API workers (Docker/K8s) → RedisCache
Single server, simple deployment  → DiskCache
Testing / benchmarking            → NullCache
Dev machine, no Redis             → DiskCache (zero setup)
```

---

## 3. Cache Interface

All three backends share the same interface:

```python
class CacheBackend(ABC):

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None on miss/expired."""

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value. TTL in seconds (None = no expiry)."""

    def delete(self, key: str) -> bool:
        """Remove a key. Returns True if it existed."""

    def clear(self) -> int:
        """Delete all keys. Returns count deleted."""

    def exists(self, key: str) -> bool:
        """True if key exists and has not expired."""

    def get_or_set(self, key: str, factory, ttl=None) -> Any:
        """Cache-aside: return cached value or compute + cache it."""

    def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get. Returns only hits."""

    def mset(self, items: Dict[str, Any], ttl=None) -> None:
        """Batch set."""
```

### Cache-Aside Pattern (most common)

```python
# Without cache-aside
query_key = query_cache_key(query)
cached = cache.get(query_key)
if cached:
    return cached
result = pipeline.query(query)
cache.set(query_key, result, ttl=3600)
return result

# With cache-aside helper (one liner)
result = cache.get_or_set(
    key=query_cache_key(query),
    factory=lambda: pipeline.query(query),
    ttl=3600,
)
```

---

## 4. RedisCache

Distributed cache — essential for multi-worker deployments.

```python
from utils.cache import RedisCache

cache = RedisCache(
    url="redis://localhost:6379/0",   # database 0
    prefix="rag:",                    # all keys prefixed → "rag:a3f1b2c9"
    max_connections=20,               # connection pool size
    socket_timeout=2.0,               # fail fast on network issues
)
```

### Serialization

Values are serialized with `pickle` — supports **any Python object**:
- `str`, `int`, `float`, `bool`
- `np.ndarray` (embedding vectors)
- `RAGResponse` (full pipeline output)
- `List[SearchResult]` (retrieval results)

```python
import numpy as np

# Store an embedding vector
vec = np.random.rand(384).astype(np.float32)
cache.set("embed:abc123", vec, ttl=86400)  # 24 hours

# Retrieve it
vec2 = cache.get("embed:abc123")
assert np.allclose(vec, vec2)
```

### TTL Strategy

```python
# Query results: cache for 1 hour (answers may go stale if index updated)
cache.set(query_cache_key(q), response, ttl=3600)

# Embedding vectors: cache for 24 hours (stable unless model changes)
cache.set(embedding_cache_key(text, model), vector, ttl=86400)

# BM25 index: don't cache (too large; load from disk directly)

# Health check results: cache for 30 seconds
cache.set("health:status", health_data, ttl=30)
```

### Redis Namespacing

Use `prefix` to avoid collisions with other apps on the same Redis instance:

```python
rag_cache  = RedisCache(url="redis://...", prefix="rag:v1:")
auth_cache = RedisCache(url="redis://...", prefix="auth:")
api_cache  = RedisCache(url="redis://...", prefix="api:v2:")
```

### Inspecting Redis State

```python
info = cache.info()
print(info)
# {
#   "used_memory_human": "45.2M",
#   "connected_clients": 3,
#   "redis_version": "7.2.4",
#   "uptime_in_seconds": 86400,
#   "keyspace": {"keys": 1523, "expires": 1200}
# }

ttl = cache.ttl_remaining("rag:a3f1b2c9")  # seconds until expiry
```

---

## 5. DiskCache

Local persistent cache — no setup needed, works out-of-the-box.

```python
from utils.cache import DiskCache

cache = DiskCache(
    cache_dir="./.rag_cache",      # SQLite + blobs stored here
    size_limit=1_073_741_824,      # 1 GB max (LRU eviction when exceeded)
    default_ttl=3600,              # default 1 hour if TTL not specified
)
```

### Storage Structure

```
.rag_cache/
├── cache.db          ← SQLite: keys, metadata, expiry
├── 00/               ← blob files (auto-sharded)
│   ├── 00abc123...
│   └── 00def456...
├── 01/
└── …
```

### Size Management

```python
print(cache.size_bytes)     # current size in bytes
print(cache.key_count)      # number of cached items
cache.evict_expired()       # manually remove expired keys (auto on get too)
```

**LRU eviction** — when `size_limit` is exceeded, the **least recently used**
entries are evicted automatically on the next write.

---

## 6. NullCache

Zero-overhead no-op. Every `get()` returns `None`, every `set()` is silently discarded.

```python
from utils.cache import NullCache

cache = NullCache()  # no constructor args

# Perfect for:
# 1. Tests — ensures no shared state between test cases
# 2. Benchmarking — measure raw pipeline latency without cache interference
# 3. Feature flag — disable caching without changing application code
```

---

## 7. Cache Key Helpers

Deterministic, collision-resistant keys using SHA-256:

```python
from utils.cache import query_cache_key, embedding_cache_key, chunk_cache_key

# For query result caching
key = query_cache_key(
    query="What is the refund policy?",
    filter={"file_type": "pdf"},
    template="with_citations",
)
# → "a3f1b2c9d4e57812"  (SHA-256 prefix, 16 hex chars)

# For embedding caching
key = embedding_cache_key(
    text="Customers are eligible for a full refund within 30 days.",
    model_name="all-MiniLM-L6-v2",
)
# → "f7e21b8c3d094a61"

# For chunk caching
key = chunk_cache_key(doc_id="abc123", chunk_index=7)
# → "2d3a4f5b6c7e8d9f"
```

**Why not use the raw string as a key?**
- Queries can be 4096 chars — Redis keys should be short
- Consistent length regardless of input size
- Avoids special characters in key names (colons, spaces, etc.)

---

## 8. Factory & Auto-configure

### `build_cache()` — explicit

```python
from utils.cache import build_cache

# Production
cache = build_cache("redis", url="redis://prod-redis:6379/1", prefix="rag:v2:")

# Development
cache = build_cache("disk", cache_dir="./.dev_cache", size_limit=200*1024*1024)

# Testing
cache = build_cache("null")

# Redis with DiskCache fallback (if Redis unreachable)
cache = build_cache("redis", url="redis://maybe-down:6379")
# → If Redis unavailable: automatically falls back to DiskCache
```

### `cache_from_env()` — automatic

```python
from utils.cache import cache_from_env

# Reads: REDIS_URL, CACHE_DIR, CACHE_SIZE_MB, CACHE_DEFAULT_TTL
cache = cache_from_env()
```

```bash
# .env settings
REDIS_URL=redis://localhost:6379/0     # → RedisCache
CACHE_PREFIX=rag:                      # → key prefix
CACHE_DIR=./.rag_cache                 # → used if no REDIS_URL
CACHE_SIZE_MB=512                      # → disk cache limit
CACHE_DEFAULT_TTL=3600                 # → 1 hour default TTL
```

**Priority**:
1. `REDIS_URL` set → `RedisCache`
2. `CACHE_DIR` set (or default) → `DiskCache`
3. Neither → `NullCache`

---

## 9. Configuration System

All system parameters are centralised in `config.py` as Python dataclasses.
No YAML parsing, no magic — just typed Python.

```python
# config.py
@dataclass
class RAGConfig:
    llm:          LLMConfig
    embedding:    EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval:    RetrievalConfig
    generation:   GenerationConfig
    evaluation:   EvaluationConfig
    observability: ObservabilityConfig

    # API
    api_host:    str = "0.0.0.0"
    api_port:    int = 8080
    api_workers: int = 4
```

### Loading Config

```python
from config import DEFAULT_CONFIG, RAGConfig

# Use defaults (suitable for dev)
pipeline = RAGPipeline.from_config(DEFAULT_CONFIG)

# Override specific fields
from dataclasses import replace
prod_config = replace(
    DEFAULT_CONFIG,
    llm=replace(DEFAULT_CONFIG.llm, model_name="gpt-4o", temperature=0.0),
    vector_store=replace(DEFAULT_CONFIG.vector_store, index_type="hnsw"),
)
pipeline = RAGPipeline.from_config(prod_config)
```

---

## 10. Config Dataclasses Reference

### LLMConfig

```python
@dataclass
class LLMConfig:
    provider:     str   = "openai"        # "openai"|"azure"|"ollama"|"anthropic"|"huggingface"
    model_name:   str   = "gpt-4o-mini"
    temperature:  float = 0.1             # 0.0 = deterministic, 1.0 = creative
    max_tokens:   int   = 1024
    top_p:        float = 0.95
    api_key:      str   = ""              # reads from OPENAI_API_KEY if empty
    base_url:     Optional[str] = None    # Ollama: "http://localhost:11434"
    fallback_provider: Optional[str] = None
    fallback_model:    Optional[str] = None
```

### EmbeddingConfig

```python
@dataclass
class EmbeddingConfig:
    provider:   str = "sentence_transformers"
    model_name: str = "all-MiniLM-L6-v2"
    dimension:  int = 384
    device:     str = "cpu"              # "cpu" | "cuda" | "mps"
    batch_size: int = 64
    use_cache:  bool = True
    cache_dir:  str = "./.embed_cache"
```

### VectorStoreConfig

```python
@dataclass
class VectorStoreConfig:
    backend:     str = "faiss"           # "faiss" | "chroma" | "pinecone"
    index_type:  str = "hnsw"            # "flat" | "ivf" | "hnsw"
    metric:      str = "cosine"          # "cosine" | "l2" | "ip"
    persist_dir: str = "./vector_store_data"

    # HNSW
    hnsw_m:               int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search:       int = 50

    # IVF
    ivf_nlist:  int = 256
    ivf_nprobe: int = 32

    # Chroma
    collection_name: str = "rag_chunks"

    # Pinecone
    pinecone_api_key:    str = ""
    pinecone_index_name: str = "rag-production"
```

### RetrievalConfig

```python
@dataclass
class RetrievalConfig:
    top_k:              int   = 10
    use_hybrid:         bool  = True
    alpha:              float = 0.7      # dense weight in score fusion
    use_rrf:            bool  = True     # True = RRF; False = score fusion
    rrf_k:              int   = 60
    use_reranker:       bool  = False
    reranker_model:     str   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int  = 32
    use_mmr:            bool  = False
    mmr_lambda:         float = 0.7
    bm25_k1:            float = 1.5
    bm25_b:             float = 0.75
```

### GenerationConfig

```python
@dataclass
class GenerationConfig:
    template:           str = "default"
    max_context_tokens: int = 3000
    citation_style:     str = "inline"   # "inline" | "footnote" | "none"
    include_metadata:   bool = True
```

### ObservabilityConfig

```python
@dataclass
class ObservabilityConfig:
    log_level:          str  = "INFO"
    enable_prometheus:  bool = True
    prometheus_port:    int  = 8001
    enable_langfuse:    bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
```

---

## 11. Quick Reference

```python
# ── Cache setup ───────────────────────────────────────────────────────────
from utils.cache import build_cache, cache_from_env, query_cache_key

cache = build_cache("redis", url="redis://localhost:6379")  # production
cache = build_cache("disk",  cache_dir="./.cache")           # development
cache = build_cache("null")                                   # testing
cache = cache_from_env()                                      # auto (reads .env)

# ── Cache operations ──────────────────────────────────────────────────────
cache.set("key", value, ttl=3600)      # store with 1-hour TTL
value = cache.get("key")              # None on miss
cache.delete("key")
cache.clear()                          # wipe all keys in namespace
cache.exists("key")                    # bool

# ── Cache-aside (recommended) ─────────────────────────────────────────────
result = cache.get_or_set(
    key=query_cache_key("What is the refund policy?"),
    factory=lambda: pipeline.query("What is the refund policy?"),
    ttl=3600,
)

# ── Config ────────────────────────────────────────────────────────────────
from config import DEFAULT_CONFIG, RAGConfig, LLMConfig
from dataclasses import replace

# Swap just the LLM for production
config = replace(DEFAULT_CONFIG,
    llm=replace(DEFAULT_CONFIG.llm,
        provider="openai", model_name="gpt-4o", temperature=0.0
    )
)

# Use in pipeline
from rag_pipeline import RAGPipeline
pipeline = RAGPipeline.from_config(config)
```

---

## 12. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| **Caching stale query results** | New documents not reflected | Invalidate cache on index update or use short TTL (< 1h) |
| **Caching with wrong key** | Filters ignored — wrong results returned | Always include `filter` in `query_cache_key()` |
| **Redis OOM** | Redis runs out of memory | Set `maxmemory` + `allkeys-lru` policy in Redis config |
| **Pickle security risk** | Deserialising attacker-controlled data | Only cache your own outputs; never cache user inputs |
| **DiskCache grows unbounded** | Disk fills up | Set `size_limit`; run `evict_expired()` periodically |
| **Multi-worker cache miss storm** | All workers miss simultaneously → LLM hammered | Add jitter to TTL: `ttl = 3600 + random.randint(-300, 300)` |
| **Config not environment-aware** | Dev config used in prod | Use `cache_from_env()` and read `os.getenv()` in config |
| **Wrong dimension in VectorStoreConfig** | FAISS assertion error | Must match your embedding model's output dimension |
