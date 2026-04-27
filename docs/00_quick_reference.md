# RAG System — Quick Reference Cheat Sheet
> One page covering every module. For deep-dives see `docs/01_*.md` → `docs/09_*.md`.

---

## System Architecture at a Glance

```
 Raw Sources (PDF, HTML, JSON, TXT, DOCX, URL, SQL)
         │
         ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  DATA PIPELINE                                                  │
 │  DocumentLoader → [Document] → ChunkingEngine → [Chunk]        │
 └─────────────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  INDEXING                                                       │
 │  EmbeddingEngine → vectors → VectorStore (FAISS/Chroma)        │
 │                           → BM25Index                          │
 └─────────────────────────────────────────────────────────────────┘
         │
  Query ─┘
         ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  RETRIEVAL                                                      │
 │  Dense + BM25 → HybridRetriever (RRF) → [Optional: Reranker]  │
 └─────────────────────────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  GENERATION                                                     │
 │  ContextBuilder + PromptBuilder → LLMInterface → LLMResponse   │
 └─────────────────────────────────────────────────────────────────┘
         │
         ▼
    RAGResponse  (answer + citations + latency + cost)
```

---

## Module 01 — Data Pipeline

```python
from data_pipeline import DocumentLoader, ChunkingEngine

# Load any source (auto-detect type)
docs = DocumentLoader().load("./docs/")           # directory
docs = DocumentLoader().load("./policy.pdf")      # single file
docs = DocumentLoader().load("https://example.com/faq")  # URL

# Chunk strategies: "recursive" | "token" | "sentence" | "semantic"
chunks = ChunkingEngine(strategy="recursive", chunk_size=512, chunk_overlap=64).split(docs)

# Key chunk fields
chunk.chunk_id       # unique ID
chunk.doc_id         # parent document ID
chunk.content        # text slice
chunk.chunk_index    # position (0-based)
chunk.metadata       # inherits doc metadata + chunk_index
```

**Chunk size rules of thumb**:

| LLM | chunk_size | chunk_overlap |
|---|---|---|
| GPT-4o-mini (128K) | 512 | 64 |
| Llama 3.1 8B (8K) | 256 | 32 |
| Claude 3 (200K) | 1024 | 128 |

---

## Module 02 — Embeddings

```python
from embeddings import EmbeddingEngine

# Providers: "sentence_transformers" | "openai" | "cohere"
engine = EmbeddingEngine("sentence_transformers", "all-MiniLM-L6-v2")  # free, local
engine = EmbeddingEngine("openai", "text-embedding-3-small", api_key=...)

# Embed chunks (bulk)
result = engine.encode([c.content for c in chunks], batch_size=64)
vectors = result.embeddings    # np.ndarray (n, dim), float32, L2-normalised

# Embed a query (single)
q_vec = engine.encode_query("What is the refund policy?")

print(engine.dimension)   # pass this to VectorStoreFactory
```

**Model quick pick**:
- **Fast / free**: `all-MiniLM-L6-v2` (384-dim, 80MB)
- **Best local**: `BAAI/bge-large-en-v1.5` (1024-dim)
- **Best API**: `text-embedding-3-small` (OpenAI, 1536-dim)
- **Multilingual**: `embed-multilingual-v3.0` (Cohere, 1024-dim)

---

## Module 03 — Vector Store

```python
from vector_store import VectorStoreFactory, FAISSVectorStore

# Create: "faiss" | "chroma" | "pinecone"
store = VectorStoreFactory.create("faiss", dimension=384, index_type="hnsw")

# FAISS index types: "flat" (exact) | "ivf" (fast, needs training) | "hnsw" (fastest)

# Add
store.add(ids, vectors, texts, metadatas)

# Search
results = store.search(q_vec, top_k=10)
results = store.search(q_vec, top_k=10, filter={"source": "policy.pdf"})
# filter: {"key": "val"} | {"key": ["v1","v2"]} | {"key": {"$gte": 5}}

# Persist
store.persist("./my_index")
store.load("./my_index")   # restore without re-indexing

print(store.count)    # total indexed vectors
```

**SearchResult fields**: `chunk_id`, `content`, `score` (0–1), `metadata`, `rank`

---

## Module 04 — Retrieval

```python
from retrieval import DenseRetriever, BM25Retriever, HybridRetriever
from retrieval import CrossEncoderReranker, QueryTransformer, maximal_marginal_relevance

# Dense (semantic)
dense   = DenseRetriever(store, embedder)
results = dense.retrieve("query", top_k=10)

# BM25 (keyword)
bm25 = BM25Retriever()
bm25.build([c.content for c in chunks], [c.chunk_id for c in chunks], [c.metadata for c in chunks])
results = bm25.retrieve("exact term", top_k=10)
bm25.save("./bm25"); bm25.load("./bm25")   # persist

# Hybrid (recommended for production)
hybrid  = HybridRetriever(dense, bm25, use_rrf=True)   # RRF fusion
results = hybrid.retrieve("query", top_k=10)

# Reranker (precision boost)
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked  = reranker.rerank("query", results, top_k=5)

# Query transforms
t = QueryTransformer(lambda p: llm.complete(p).content)
t.hyde("query")              # → hypothetical answer text
t.multi_query("query", n=3) # → [original + 2 paraphrases]
t.step_back("specific q")   # → general question

# MMR (diversity)
diverse = maximal_marginal_relevance(q_vec, candidates, cand_vecs, top_k=5, lambda_param=0.7)
```

**Retrieval strategy cheat sheet**:

| Scenario | Use |
|---|---|
| Semantic / paraphrase queries | Dense |
| Exact product codes, IDs, names | BM25 |
| General production | Hybrid (RRF) |
| Precision critical | + CrossEncoder reranker |
| Diverse results needed | + MMR |
| Terse or ambiguous queries | + HyDE |

---

## Module 05 — Generation

```python
from generation import PromptBuilder, LLMInterface, ConversationHistory

# Templates: "default"|"with_citations"|"conversational"|"summarization"
#            "structured_output"|"chain_of_thought"|"safety_strict"
builder = PromptBuilder(template="with_citations", max_context_tokens=3000)

# Providers: "openai"|"azure"|"ollama"|"anthropic"|"huggingface"
llm = LLMInterface(provider="openai", model="gpt-4o-mini", temperature=0.1)

# Single-turn
messages, citations = builder.build("What is the refund policy?", search_results)
response = llm.chat(messages)
print(response.content)      # answer text
print(response.total_tokens) # tokens used
print(response.cost_usd)     # $ cost

# Multi-turn
history = ConversationHistory(max_tokens=2000)
history.add("user", "Q1"); history.add("assistant", "A1")
messages, _ = builder.build("Q2", results, history=history)

# Streaming
for token in llm.stream(messages):
    print(token, end="", flush=True)

# JSON mode
response = llm.chat(messages, json_mode=True)
data = json.loads(response.content)
```

---

## Module 06 — RAG Pipeline (Orchestrator)

```python
from rag_pipeline import RAGPipeline
from config import DEFAULT_CONFIG

pipeline = RAGPipeline.from_config(DEFAULT_CONFIG)

# Index
pipeline.index(["./docs/", "./data.json"])       # from file paths
pipeline.index_documents(list_of_documents)       # from pre-loaded docs

# Query
response = pipeline.query("What is the refund policy?")
response = pipeline.query("q", top_k=10, template="with_citations")
response = pipeline.query("q", session_id="user-123")        # multi-turn
response = pipeline.query("q", filter={"file_type": "pdf"})
response = pipeline.query("q", use_mmr=True, mmr_lambda=0.7)

# Stream
for token in pipeline.stream_query("q", session_id="user-abc"):
    print(token, end="")

# Advanced
pipeline.multi_hop_query("complex q", max_hops=3)
pipeline.agentic_query("research q", max_iterations=5)

# Sessions
pipeline.get_session_history("user-123")
pipeline.clear_session("user-123")

# Persist
pipeline.save("./pipeline_state")
pipeline.load("./pipeline_state")
print(pipeline.index_size)    # chunk count

# RAGResponse fields
response.answer           # str
response.citations        # [{"ref":1,"chunk_id":…,"source":…,"score":…}]
response.retrieved_chunks # List[SearchResult]
response.llm_response     # LLMResponse (tokens, cost, latency)
response.latency_breakdown # {"retrieval_ms":…, "generation_ms":…}
response.total_latency_ms # sum of breakdown
```

---

## Module 07 — Evaluation

```python
from evaluation import RAGEvaluator, EvaluationSample

evaluator = RAGEvaluator(
    llm=llm, embedder=embedder,
    metrics=["faithfulness", "answer_relevancy", "rouge_l", "token_f1"],
    thresholds={"faithfulness": 0.8, "answer_relevancy": 0.7,
                "rouge_l": 0.5,      "token_f1": 0.6},
)

sample = EvaluationSample(
    question="What is the refund policy?",
    ground_truth="30-day full refund.",
    answer=pipeline.query("What is the refund policy?").answer,
    contexts=[r.content for r in pipeline.retrieve("What is the refund policy?")],
)

result = evaluator.evaluate_sample(sample)
print(result.metrics["faithfulness"].score)   # 0.0–1.0
print(result.metrics["faithfulness"].passed)  # bool

report = evaluator.evaluate_dataset(samples)  # list of EvaluationSample
print(report.summary)       # {"faithfulness": 0.87, …}
print(len(report.failures)) # samples below threshold
report.save("report.json")
```

**Metric interpretation**:

| Metric | Catches | Reference needed? |
|---|---|---|
| Faithfulness | Hallucinations | No |
| Answer Relevancy | Off-topic answers | No |
| Context Precision | Noisy retrieval | No |
| Context Recall | Missing information | Yes |
| Answer Correctness | Factual errors | Yes |
| ROUGE-L | Wording drift | Yes |
| Token F1 | Token-level accuracy | Yes |

---

## Module 08 — API (FastAPI)

```bash
# Start
make api                   # dev (hot-reload)
make api-prod WORKERS=4    # production

# Endpoints
GET  /health               # no auth; for load balancer probes
POST /index                # index documents
POST /query                # sync query → full answer
POST /query/stream         # SSE streaming query
DELETE /sessions/{id}      # clear conversation history
GET  /metrics              # Prometheus metrics (port 8001)
GET  /docs                 # Swagger UI

# Auth: X-API-Key header
curl -H "X-API-Key: dev-key-12345" ...

# Query example
curl -X POST http://localhost:8080/query \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the refund policy?","top_k":10}'
```

**Response codes**: `200 OK` | `400 Blocked (injection)` | `403 Bad key` | `422 Validation` | `429 Rate limit` | `500 Error`

---

## Module 09 — Cache & Config

```python
from utils.cache import build_cache, cache_from_env, query_cache_key

# Backends: "redis" (distributed) | "disk" (local) | "null" (testing)
cache = build_cache("redis", url="redis://localhost:6379", prefix="rag:")
cache = build_cache("disk",  cache_dir="./.cache", size_limit=500*1024*1024)
cache = build_cache("null")
cache = cache_from_env()   # reads REDIS_URL / CACHE_DIR from .env

# Operations (same API for all backends)
cache.set("key", value, ttl=3600)     # store (TTL in seconds)
cache.get("key")                       # → value or None
cache.get_or_set("key", factory_fn, ttl=3600)  # cache-aside
cache.delete("key"); cache.clear()

# Key helpers (deterministic SHA-256 prefix)
key = query_cache_key("What is the refund policy?", filter={"source":"a.pdf"})
key = embedding_cache_key("text content", "all-MiniLM-L6-v2")
```

```python
from config import DEFAULT_CONFIG, RAGConfig, LLMConfig
from dataclasses import replace

# Override specific config values
config = replace(DEFAULT_CONFIG,
    llm=replace(DEFAULT_CONFIG.llm, model_name="gpt-4o", temperature=0.0),
)
pipeline = RAGPipeline.from_config(config)
```

---

## End-to-End: Full Working Example

```python
import os
from rag_pipeline import RAGPipeline
from config import DEFAULT_CONFIG
from dataclasses import replace

# 1. Build pipeline
config   = replace(DEFAULT_CONFIG)
pipeline = RAGPipeline.from_config(config)

# 2. Index your documents
result = pipeline.index(["./knowledge_base/"])
print(f"Indexed: {result.num_chunks} chunks from {result.num_documents} docs")

# 3. Query
response = pipeline.query(
    "What is the refund policy?",
    top_k=10,
    template="with_citations",
    session_id="demo-session",
)
print(response.answer)
for c in response.citations:
    print(f"  [{c['ref']}] {c['source']}  (score={c['score']:.3f})")

# 4. Follow-up (uses conversation history)
response2 = pipeline.query(
    "What about digital downloads?",
    session_id="demo-session",
)
print(response2.answer)

# 5. Save index (avoid re-indexing on next run)
pipeline.save("./pipeline_state")

# 6. On next startup — just load
pipeline2 = RAGPipeline.from_config(config)
pipeline2.load("./pipeline_state")
```

---

## Common Issues → Quick Fixes

| Problem | Likely cause | Fix |
|---|---|---|
| Empty results from query | Index not built | Call `pipeline.index()` first |
| `AssertionError` in FAISS | Dimension mismatch | Check `engine.dimension == store.dimension` |
| `RuntimeError: BM25 not built` | BM25 not initialised | Pipeline builds it in `index()`; or call `bm25.build()` |
| High latency (> 5s) | No cache + slow LLM | Add cache; use `gpt-4o-mini` over `gpt-4o` |
| Hallucinated answer | Retrieval missing relevant chunk | Increase `top_k`; switch to Hybrid |
| Answer off-topic | Wrong template | Use `"with_citations"` or `"safety_strict"` |
| Session history leaking between users | Shared session_id | Use unique `session_id` per user |
| 403 from API | Missing `X-API-Key` header | Add `-H "X-API-Key: dev-key-12345"` |
| OOM on large corpus | Batch too large | Reduce `batch_size` in `index()` |
| Index lost after restart | No `save()` call | Call `pipeline.save()` after indexing |

---

## File Map

```
rag-system/
├── config.py                    ← all configuration dataclasses
├── rag_pipeline.py              ← master orchestrator
├── data_pipeline/
│   ├── document_loader.py       ← load any file format
│   └── chunking.py              ← split into chunks
├── embeddings/
│   └── embedding.py             ← embed text → vectors
├── vector_store/
│   └── vector_store.py          ← FAISS / ChromaDB / Pinecone
├── retrieval/
│   ├── retriever.py             ← Dense / BM25 / Hybrid / MMR / HyDE
│   └── reranker.py              ← CrossEncoder / Cohere / LLM reranker
├── generation/
│   ├── prompt_builder.py        ← 7 templates + context assembly
│   └── llm_interface.py         ← OpenAI / Azure / Ollama / Anthropic / HF
├── evaluation/
│   └── evaluator.py             ← faithfulness / ROUGE / RAGAS
├── api/
│   └── main.py                  ← FastAPI: /index /query /health
├── utils/
│   └── cache.py                 ← Redis / Disk / Null cache
├── tests/
│   ├── conftest.py              ← shared fixtures
│   ├── unit/                    ← 10 test files, ~400 tests
│   └── integration/             ← end-to-end tests
└── docs/
    ├── 00_quick_reference.md    ← this file
    ├── 01_data_pipeline.md
    ├── 02_embeddings.md
    ├── 03_vector_store.md
    ├── 04_retrieval.md
    ├── 05_generation.md
    ├── 06_rag_pipeline.md
    ├── 07_evaluation.md
    ├── 08_api.md
    └── 09_cache_and_config.md
```
