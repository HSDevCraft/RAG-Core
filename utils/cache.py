"""
utils/cache.py
──────────────
Standalone cache utility for the RAG system.

Backends:
  RedisCache   — distributed, shared across API workers (production default)
  DiskCache    — local persistent cache via diskcache (single-node)
  NullCache    — no-op cache for testing / cache-disabled mode

Design decisions:
  - All backends expose the same interface: get / set / delete / clear
  - Values are serialized with pickle (supports arbitrary Python objects incl. RAGResponse)
  - TTL (time-to-live) is per-key, not global — fine-grained expiry control
  - Cache key helpers for query result caching and embedding caching
  - Thread-safe: Redis uses pipelining, diskcache handles concurrent access natively

Usage:
    from utils.cache import build_cache

    cache = build_cache(backend="redis", url="redis://localhost:6379/0")
    cache.set("key", value, ttl=3600)
    value = cache.get("key")   # None on miss
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class CacheBackend(ABC):
    """Abstract base for all cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Return the cached value or None on miss / expired."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value under key with optional TTL in seconds. Returns True on success."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove a key. Returns True if it existed."""

    @abstractmethod
    def clear(self) -> int:
        """Remove all keys. Returns count of deleted keys."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if key exists and has not expired."""

    def get_or_set(self, key: str, factory, ttl: Optional[int] = None) -> Any:
        """
        Cache-aside pattern: return cached value or compute and cache it.

        Args:
            key: Cache key.
            factory: Zero-argument callable that produces the value on miss.
            ttl: Optional TTL in seconds.

        Returns:
            Cached or freshly computed value.

        Example:
            result = cache.get_or_set(
                "query:abc123",
                lambda: pipeline.query("What is RAG?"),
                ttl=3600,
            )
        """
        cached = self.get(key)
        if cached is not None:
            logger.debug("Cache HIT: %s", key)
            return cached

        logger.debug("Cache MISS: %s", key)
        value = factory()
        self.set(key, value, ttl=ttl)
        return value

    def mget(self, keys: list[str]) -> dict[str, Any]:
        """Batch get. Returns dict of key → value (only hits)."""
        return {k: v for k in keys if (v := self.get(k)) is not None}

    def mset(self, items: dict[str, Any], ttl: Optional[int] = None) -> None:
        """Batch set."""
        for key, value in items.items():
            self.set(key, value, ttl=ttl)


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------
class RedisCache(CacheBackend):
    """
    Redis-backed distributed cache.

    Best for production deployments where multiple API workers share state.
    Uses pickle serialization — supports any Python object including numpy arrays.

    Connection options:
      - URL:      redis://[:password@]host[:port][/db]
      - Sentinel: redis+sentinel://host:port/master_name
      - Cluster:  redis+cluster://host:port

    Args:
        url: Redis connection URL. Default: redis://localhost:6379/0
        prefix: Key prefix for namespacing (e.g. "rag:" prevents collision with
                other apps using the same Redis instance).
        max_connections: Connection pool size.
        socket_timeout: Network timeout in seconds.
        decode_errors: How to handle decode errors ('strict', 'replace').

    Example:
        cache = RedisCache(url="redis://prod-redis:6379/1", prefix="rag:v2:")
        cache.set("query:abc", response, ttl=3600)
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "rag:",
        max_connections: int = 20,
        socket_timeout: float = 2.0,
    ):
        try:
            import redis
        except ImportError:
            raise ImportError("Install redis: pip install redis")

        import redis as redis_lib
        self._client = redis_lib.from_url(
            url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            decode_responses=False,     # we handle binary pickle manually
        )
        self._prefix = prefix
        self._url = url

        # Validate connection
        try:
            self._client.ping()
            logger.info("RedisCache connected to %s with prefix '%s'", url, prefix)
        except Exception as e:
            logger.warning("RedisCache connection failed: %s (using NullCache fallback)", e)
            raise

    def _key(self, key: str) -> str:
        return self._prefix + key

    def get(self, key: str) -> Optional[Any]:
        try:
            raw = self._client.get(self._key(key))
            if raw is None:
                return None
            return pickle.loads(raw)
        except Exception as e:
            logger.warning("RedisCache.get failed for key '%s': %s", key, e)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            if ttl:
                return bool(self._client.setex(self._key(key), ttl, raw))
            return bool(self._client.set(self._key(key), raw))
        except Exception as e:
            logger.warning("RedisCache.set failed for key '%s': %s", key, e)
            return False

    def delete(self, key: str) -> bool:
        try:
            return bool(self._client.delete(self._key(key)))
        except Exception as e:
            logger.warning("RedisCache.delete failed: %s", e)
            return False

    def clear(self) -> int:
        """Delete all keys matching this cache's prefix."""
        try:
            pattern = self._prefix + "*"
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning("RedisCache.clear failed: %s", e)
            return 0

    def exists(self, key: str) -> bool:
        try:
            return bool(self._client.exists(self._key(key)))
        except Exception:
            return False

    def ttl_remaining(self, key: str) -> int:
        """Return seconds until expiry, -1 if no TTL, -2 if key missing."""
        return self._client.ttl(self._key(key))

    def info(self) -> dict:
        """Return Redis server info (memory, connected_clients, etc.)."""
        try:
            info = self._client.info()
            return {
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "redis_version": info.get("redis_version"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
                "keyspace": info.get("db0"),
            }
        except Exception as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# DiskCache backend
# ---------------------------------------------------------------------------
class DiskCache(CacheBackend):
    """
    Local on-disk cache via diskcache.

    Best for single-node deployments or development.
    Persistent across restarts. Thread-safe. Process-safe.
    Supports concurrent reads/writes from multiple threads/processes.

    Storage: SQLite + file system under cache_dir.
    Default size limit: 1GB (evicts LRU when exceeded).

    Args:
        cache_dir: Directory for SQLite + blob files.
        size_limit: Maximum cache size in bytes (default: 1GB).
        default_ttl: Default TTL in seconds (None = no expiry).

    Example:
        cache = DiskCache(cache_dir="./.query_cache", size_limit=512*1024*1024)
        cache.set("embed:abc", vector, ttl=86400)
    """

    def __init__(
        self,
        cache_dir: str = "./.rag_cache",
        size_limit: int = 1_073_741_824,  # 1 GB
        default_ttl: Optional[int] = None,
    ):
        try:
            import diskcache
        except ImportError:
            raise ImportError("Install diskcache: pip install diskcache")

        import diskcache as dc
        self._cache = dc.Cache(
            directory=cache_dir,
            size_limit=size_limit,
        )
        self._default_ttl = default_ttl
        logger.info("DiskCache initialized at '%s' (limit=%dMB)", cache_dir,
                    size_limit // 1_048_576)

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key, default=None)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        expire = ttl or self._default_ttl
        try:
            self._cache.set(key, value, expire=expire)
            return True
        except Exception as e:
            logger.warning("DiskCache.set failed: %s", e)
            return False

    def delete(self, key: str) -> bool:
        return self._cache.delete(key)

    def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count

    def exists(self, key: str) -> bool:
        return key in self._cache

    @property
    def size_bytes(self) -> int:
        """Current total size of cached data in bytes."""
        return self._cache.volume()

    @property
    def key_count(self) -> int:
        return len(self._cache)

    def evict_expired(self) -> int:
        """Manually trigger eviction of expired entries."""
        return self._cache.expire()


# ---------------------------------------------------------------------------
# NullCache backend (no-op)
# ---------------------------------------------------------------------------
class NullCache(CacheBackend):
    """
    No-op cache — every get() misses, set() is ignored.

    Use cases:
      - Testing (ensures tests don't share state through cache)
      - Disabling cache in config without code changes
      - Benchmarking raw pipeline latency without cache interference

    Thread-safe by nature (no state).
    """

    def get(self, key: str) -> Optional[Any]:
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        return True

    def delete(self, key: str) -> bool:
        return False

    def clear(self) -> int:
        return 0

    def exists(self, key: str) -> bool:
        return False


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------
def query_cache_key(query: str, filter: Optional[dict] = None,
                    template: Optional[str] = None) -> str:
    """
    Build a deterministic cache key for a RAG query.

    Incorporates: query text + metadata filter + template name.
    Excludes: session_id (session state is not cached at query level).

    Args:
        query: User query string.
        filter: Metadata filter dict (e.g. {'file_type': 'pdf'}).
        template: Prompt template name.

    Returns:
        16-character hex string (SHA-256 prefix).

    Example:
        key = query_cache_key("What is the refund policy?", filter={"source": "policy.pdf"})
        # → "a3f1b2c9d4e57812"
    """
    payload = {
        "query": query.strip().lower(),
        "filter": filter or {},
        "template": template or "default",
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def embedding_cache_key(text: str, model_name: str) -> str:
    """
    Build a deterministic cache key for an embedding.

    Args:
        text: The text to be embedded.
        model_name: Name of the embedding model.

    Returns:
        16-character hex string.
    """
    raw = f"{model_name}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_cache_key(doc_id: str, chunk_index: int) -> str:
    """Cache key for a specific chunk within a document."""
    return hashlib.sha256(f"{doc_id}:{chunk_index}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_cache(
    backend: str = "null",
    **kwargs,
) -> CacheBackend:
    """
    Factory function — instantiate a cache backend by name.

    Args:
        backend: One of "redis", "disk", "null".
        **kwargs: Passed to the backend constructor.

    Returns:
        CacheBackend instance.

    Examples:
        # Production: Redis cache
        cache = build_cache("redis", url="redis://prod-redis:6379/1", prefix="rag:v2:")

        # Development: local disk cache
        cache = build_cache("disk", cache_dir="./.dev_cache", size_limit=100*1024*1024)

        # Testing: no-op
        cache = build_cache("null")

        # Auto: use Redis if URL available, else disk
        cache = build_cache(
            "redis" if os.getenv("REDIS_URL") else "disk",
            url=os.getenv("REDIS_URL", ""),
            cache_dir="./.rag_cache",
        )
    """
    backend = backend.lower()

    if backend == "redis":
        try:
            return RedisCache(**kwargs)
        except Exception as e:
            logger.warning(
                "RedisCache init failed (%s) — falling back to DiskCache", e
            )
            disk_kwargs = {k: v for k, v in kwargs.items()
                          if k in ("cache_dir", "size_limit", "default_ttl")}
            return DiskCache(**disk_kwargs)

    elif backend == "disk":
        return DiskCache(**kwargs)

    elif backend == "null":
        return NullCache()

    else:
        raise ValueError(
            f"Unknown cache backend: '{backend}'. Choose from: redis, disk, null"
        )


# ---------------------------------------------------------------------------
# Auto-configure from environment
# ---------------------------------------------------------------------------
def cache_from_env() -> CacheBackend:
    """
    Build a cache backend from environment variables.

    Priority:
      1. If REDIS_URL is set → RedisCache
      2. If CACHE_DIR is set → DiskCache
      3. Otherwise → NullCache

    Environment variables:
      REDIS_URL        — Redis connection URL
      CACHE_PREFIX     — Redis key prefix (default: "rag:")
      CACHE_DIR        — Disk cache directory (default: "./.rag_cache")
      CACHE_SIZE_MB    — Disk cache size limit in MB (default: 1024)
      CACHE_DEFAULT_TTL — Default TTL in seconds (default: 3600)

    Example:
        # In your startup code:
        cache = cache_from_env()
        pipeline = RAGPipeline(..., cache=cache)
    """
    redis_url = os.getenv("REDIS_URL")
    cache_dir = os.getenv("CACHE_DIR", "./.rag_cache")
    size_mb = int(os.getenv("CACHE_SIZE_MB", "1024"))
    default_ttl = int(os.getenv("CACHE_DEFAULT_TTL", "3600"))

    if redis_url:
        return build_cache(
            "redis",
            url=redis_url,
            prefix=os.getenv("CACHE_PREFIX", "rag:"),
        )
    elif os.path.exists(cache_dir) or os.getenv("CACHE_DIR"):
        return build_cache(
            "disk",
            cache_dir=cache_dir,
            size_limit=size_mb * 1_048_576,
            default_ttl=default_ttl,
        )
    else:
        logger.info("No cache backend configured — using NullCache")
        return NullCache()
