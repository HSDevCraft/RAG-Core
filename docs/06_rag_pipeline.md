# Module 06 — RAG Pipeline Orchestrator
### `rag_pipeline.py`

---

## Table of Contents
1. [What the Pipeline Does](#1-what-the-pipeline-does)
2. [Key Data Models](#2-key-data-models)
3. [Construction](#3-construction)
4. [Indexing](#4-indexing)
5. [Querying](#5-querying)
6. [Streaming](#6-streaming)
7. [Multi-Hop Retrieval](#7-multi-hop-retrieval)
8. [Agentic RAG (ReAct)](#8-agentic-rag-react)
9. [Session Management](#9-session-management)
10. [Persistence](#10-persistence)
11. [Full Data-Flow Diagram](#11-full-data-flow-diagram)
12. [Quick Reference](#12-quick-reference)
13. [Common Pitfalls](#13-common-pitfalls)

---

## 1. What the Pipeline Does

`RAGPipeline` is the **master orchestrator**. It wires every module together
and exposes a clean four-method API: `index`, `index_documents`, `retrieve`,
`query`.

```
User calls pipeline.query("What is the refund policy?")
                    │
    ┌───────────────┼───────────────────────────────┐
    │               │                               │
    ▼               ▼                               ▼
embed query    hybrid retrieve               build prompt
    │          (dense + BM25)                       │
    ▼               │                               ▼
    └───────────────┤                        call LLM
                    │                               │
                    ▼                               ▼
              [reranker]                    LLMResponse
                    │                               │
                    └───────────────────────────────┘
                                    │
                                    ▼
                              RAGResponse
                  answer + citations + latency breakdown
```

---

## 2. Key Data Models

### IndexingResult

```python
@dataclass
class IndexingResult:
    num_documents:        int
    num_chunks:           int
    embedding_latency_ms: float
    total_latency_ms:     float
    errors:               List[str] = field(default_factory=list)
```

### RAGResponse

```python
@dataclass
class RAGResponse:
    query:            str
    answer:           str               # the LLM-generated text
    citations:        List[Dict]        # [{"ref":1, "chunk_id":…, "source":…}]
    retrieved_chunks: List[SearchResult]
    llm_response:     LLMResponse       # tokens, cost, latency
    latency_breakdown: Dict[str, float] # {"retrieval_ms":…, "generation_ms":…}
    session_id:       Optional[str]
    metadata:         Dict = {}         # e.g. {"hops": 2} for multi-hop

    @property
    def total_latency_ms(self) -> float:
        return sum(self.latency_breakdown.values())

    def to_dict(self) -> dict: ...      # JSON-serialisable
```

---

## 3. Construction

### 3.1 Manual (explicit components)

```python
from rag_pipeline import RAGPipeline
from embeddings import EmbeddingEngine
from vector_store import VectorStoreFactory
from generation import LLMInterface
from retrieval import CrossEncoderReranker

embedder = EmbeddingEngine("sentence_transformers", "all-MiniLM-L6-v2")
store    = VectorStoreFactory.create("faiss", dimension=384, index_type="hnsw")
llm      = LLMInterface("openai", "gpt-4o-mini", temperature=0.1)
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

pipeline = RAGPipeline(
    embedder=     embedder,
    vector_store= store,
    llm=          llm,
    reranker=     reranker,  # optional
    config=       my_config, # optional; uses DEFAULT_CONFIG if omitted
)
```

### 3.2 From Config (recommended for production)

```python
from rag_pipeline import RAGPipeline
from config import RAGConfig, LLMConfig, EmbeddingConfig, VectorStoreConfig, RetrievalConfig

config = RAGConfig(
    llm=LLMConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_tokens=1024,
    ),
    embedding=EmbeddingConfig(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        dimension=384,
    ),
    vector_store=VectorStoreConfig(
        backend="faiss",
        index_type="hnsw",
        persist_dir="./vector_store_data",
    ),
    retrieval=RetrievalConfig(
        top_k=10,
        use_reranker=True,
        use_hybrid=True,
        use_mmr=False,
    ),
)

pipeline = RAGPipeline.from_config(config)
```

---

## 4. Indexing

### 4.1 From File Paths / Directories / URLs

```python
result = pipeline.index(
    sources=["./docs/", "./pdfs/policy.pdf", "https://example.com/faq"],
    loader_kwargs={"recursive": True, "max_files": 500},
    batch_size=128,
)

print(result.num_documents)        # 47
print(result.num_chunks)           # 312
print(result.embedding_latency_ms) # 3420.5
print(result.errors)               # [] (or list of failed sources)
```

**What happens inside `index()`**:
```
1. For each source → DocumentLoader.load()   → [Document]
2. ChunkingEngine.split([Document])          → [Chunk]
3. EmbeddingEngine.encode([chunk.content])   → vectors
4. VectorStore.add(ids, vectors, texts, metas)
5. BM25Retriever.build(texts, ids, metas)
6. Store chunk_id → vector mapping (for MMR)
```

### 4.2 From Pre-loaded Documents

```python
from data_pipeline import Document

# Use when you load documents from a database, API, or custom source
docs = [
    Document("Product A: wireless earbuds…", {"source": "catalog.db", "id": 101}),
    Document("Return policy: 30 days…",      {"source": "policy.db",  "id": 202}),
]

result = pipeline.index_documents(docs, batch_size=64)
# Skips DocumentLoader; goes straight to ChunkingEngine → Embed → Store
```

> **Bug Note**: In v1, `index_documents()` had a premature `return` making the
> implementation unreachable. **Fixed in v1.1** — always use the latest version.

### 4.3 Incremental Indexing

```python
# Index new documents without rebuilding the entire index
new_docs = [...load new files...]
pipeline.index_documents(new_docs)   # adds to existing index
pipeline.save("./pipeline_state")    # persist updated state
```

---

## 5. Querying

### 5.1 Basic Query

```python
response = pipeline.query("What is the refund policy?")

print(response.answer)
# "Based on the provided context, customers are eligible for a full
#  refund within 30 days of purchase. After that period, only store
#  credit is available. Digital downloads are non-refundable once accessed."

print(response.citations)
# [{"ref": 1, "chunk_id": "a3f1b2c9", "source": "policy.pdf", "score": 0.923}]

print(response.total_latency_ms)   # 1850.3
```

### 5.2 Full Query Options

```python
response = pipeline.query(
    query=     "What is the refund policy?",
    top_k=     10,               # retrieve this many chunks
    filter=    {"file_type": "pdf"},  # limit to PDF sources only
    template=  "with_citations", # prompt template
    session_id="user-123-session-abc",  # for multi-turn history
    use_hyde=  False,            # HyDE query transformation
    use_mmr=   True,             # MMR diversity in results
    mmr_lambda=0.7,
)
```

### 5.3 What Happens Inside `query()`

```
1. [Optional] transform query (HyDE / MultiQuery)
2. embed query → query_vec
3. DenseRetriever.retrieve(query_vec, top_k)
4. BM25Retriever.retrieve(query_text, top_k)
5. HybridRetriever.fuse(dense_results, bm25_results)
6. [Optional] MMR re-ranking for diversity
7. [Optional] CrossEncoderReranker.rerank(query, candidates)
8. ContextBuilder.build(top_results) → (context_str, citations)
9. PromptBuilder.build(query, results, history) → messages
10. LLMInterface.chat(messages) → LLMResponse
11. Update ConversationHistory (if session_id provided)
12. Return RAGResponse
```

### 5.4 Empty Index Behaviour

```python
# Pipeline with no documents indexed
response = pipeline.query("any question")
# answer = "I could not find relevant information to answer your question."
# citations = []
# retrieved_chunks = []
```

---

## 6. Streaming

Returns tokens as they are generated — suitable for SSE / WebSocket UIs.

```python
# Synchronous generator
for token in pipeline.stream_query("What is the return window?"):
    print(token, end="", flush=True)
print()

# With options
for token in pipeline.stream_query(
    query="refund details",
    session_id="user-123",
    top_k=5,
    template="conversational",
):
    print(token, end="", flush=True)
```

**In FastAPI (SSE)**:
```python
from fastapi.responses import StreamingResponse

@app.post("/query/stream")
async def stream(req: QueryRequest):
    async def generate():
        for token in pipeline.stream_query(req.query, session_id=req.session_id):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 7. Multi-Hop Retrieval

Some questions require chaining multiple retrieval steps. Example:

```
Q: "What is the refund policy for the XR-500 Laptop?"

Hop 1: retrieve "XR-500 Laptop" → find product details, category="Electronics"
Hop 2: retrieve "Electronics refund policy" → find category-specific policy
Hop 3: (if needed) retrieve "exceptions for accessories"

Combined context → LLM generates final answer
```

```python
response = pipeline.multi_hop_query(
    query="What is the refund policy for the XR-500 Laptop?",
    max_hops=3,
    top_k_per_hop=5,
)

print(response.answer)
print(response.metadata["hops"])    # number of hops actually performed (1–3)
print(response.metadata["sub_queries"])  # intermediate queries generated
```

**Algorithm**:
```
1. Generate sub-query from original question (LLM)
2. Retrieve for sub-query
3. Check: "Do I have enough info to answer?" (LLM)
   → YES: synthesise final answer
   → NO:  generate next sub-query using accumulated context, go to step 2
4. After max_hops: synthesise with available context
```

---

## 8. Agentic RAG (ReAct)

ReAct (Reason + Act) combines reasoning and tool use. The LLM decides which
tool to call (search, calculator, API) at each step.

```
Loop:
  Thought: "I need to find the refund policy"
  Action:  search("refund policy")
  Observation: [retrieved chunks]

  Thought: "I need the product price too"
  Action:  search("XR-500 laptop price")
  Observation: [price chunks]

  Thought: "I have enough info now"
  Action:  finish("Based on context: …")
```

```python
response = pipeline.agentic_query(
    query="Compare the refund policies for laptops vs smartwatches",
    max_iterations=5,
    tools=["search", "calculate"],   # available tools
)
```

> ℹ️ Agentic RAG is best for complex research tasks. For simple QA, standard
> `query()` is faster and more reliable.

---

## 9. Session Management

Sessions maintain per-user conversation history.

```python
# Turn 1
r1 = pipeline.query("What is the refund policy?", session_id="user-abc")

# Turn 2 — LLM sees previous Q&A in context
r2 = pipeline.query("What about for digital items?", session_id="user-abc")

# Inspect history
history = pipeline.get_session_history("user-abc")
print(len(history))   # 4 (2 user + 2 assistant messages)

# Clear (on logout / session timeout)
pipeline.clear_session("user-abc")
```

**Session isolation**: each `session_id` gets its own `ConversationHistory`.
Sessions are stored in-memory (`Dict[str, ConversationHistory]`). For
multi-worker deployments, use Redis-backed session storage.

---

## 10. Persistence

```python
# Save everything: vector index + BM25 index + chunk embeddings (for MMR)
pipeline.save("./pipeline_state/production_v1")
# Creates:
#   pipeline_state/production_v1/vector_store/   (FAISS files)
#   pipeline_state/production_v1/bm25/           (pickle)
#   pipeline_state/production_v1/chunk_embs.pkl
#   pipeline_state/production_v1/metadata.json   (config snapshot)

# Load on restart (no re-indexing needed)
pipeline.load("./pipeline_state/production_v1")
print(pipeline.index_size)   # number of indexed chunks
```

**In production** — save after every index batch:
```python
result = pipeline.index(["./new_docs/"])
pipeline.save("./pipeline_state/production_v1")   # overwrite
```

---

## 11. Full Data-Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       RAGPipeline                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  INDEXING PATH                                          │   │
│  │                                                         │   │
│  │  sources / docs                                         │   │
│  │      │                                                  │   │
│  │      ▼                                                  │   │
│  │  DocumentLoader ──▶ [Document] ──▶ ChunkingEngine       │   │
│  │                                         │               │   │
│  │                                         ▼               │   │
│  │                                    [Chunk]              │   │
│  │                                    │       │            │   │
│  │                                    ▼       ▼            │   │
│  │                              EmbeddingEngine           │   │
│  │                                    │                    │   │
│  │                          ┌─────────┴──────────┐        │   │
│  │                          ▼                    ▼         │   │
│  │                    VectorStore           BM25Index      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  QUERY PATH                                             │   │
│  │                                                         │   │
│  │  query_text                                             │   │
│  │      │                                                  │   │
│  │      ├──▶ [HyDE / MultiQuery transform]                │   │
│  │      │                                                  │   │
│  │      ▼                                                  │   │
│  │  EmbeddingEngine ──▶ query_vec                         │   │
│  │      │                   │                              │   │
│  │      ▼                   ▼                              │   │
│  │  BM25Retriever    DenseRetriever                        │   │
│  │      │                   │                              │   │
│  │      └──────────┬────────┘                             │   │
│  │                 │                                       │   │
│  │                 ▼                                       │   │
│  │          HybridRetriever (RRF)                          │   │
│  │                 │                                       │   │
│  │                 ├──▶ [MMR diversity filter]            │   │
│  │                 │                                       │   │
│  │                 ▼                                       │   │
│  │          CrossEncoderReranker                           │   │
│  │                 │                                       │   │
│  │                 ▼                                       │   │
│  │    ContextBuilder ──▶ context_str + citations           │   │
│  │                 │                                       │   │
│  │                 ▼                                       │   │
│  │         PromptBuilder ──▶ messages                     │   │
│  │         + ConversationHistory                           │   │
│  │                 │                                       │   │
│  │                 ▼                                       │   │
│  │          LLMInterface ──▶ LLMResponse                  │   │
│  │                 │                                       │   │
│  │                 ▼                                       │   │
│  │            RAGResponse                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Quick Reference

```python
from rag_pipeline import RAGPipeline
from config import DEFAULT_CONFIG

# ── Create ───────────────────────────────────────────────────────────────
pipeline = RAGPipeline.from_config(DEFAULT_CONFIG)

# ── Index ────────────────────────────────────────────────────────────────
result = pipeline.index(["./docs/", "./data.json"])
result = pipeline.index_documents(list_of_documents)
print(pipeline.index_size)          # total indexed chunks

# ── Query ────────────────────────────────────────────────────────────────
response = pipeline.query("question here")
response = pipeline.query("question", top_k=10, template="with_citations")
response = pipeline.query("question", session_id="user-123")
response = pipeline.query("question", filter={"source": "policy.pdf"})
response = pipeline.query("question", use_mmr=True, mmr_lambda=0.7)

# ── Stream ───────────────────────────────────────────────────────────────
for token in pipeline.stream_query("question"):
    print(token, end="")

# ── Advanced ─────────────────────────────────────────────────────────────
response = pipeline.multi_hop_query("complex question", max_hops=3)
response = pipeline.agentic_query("research question", max_iterations=5)

# ── Sessions ─────────────────────────────────────────────────────────────
history = pipeline.get_session_history("user-123")
pipeline.clear_session("user-123")

# ── Persistence ──────────────────────────────────────────────────────────
pipeline.save("./pipeline_state")
pipeline.load("./pipeline_state")
```

---

## 13. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| **Querying before indexing** | Empty results, no errors | Always call `index()` or `load()` first |
| **Not saving after index** | Re-indexes on every restart | Call `pipeline.save()` after indexing |
| **Session not cleared** | Memory grows unbounded | Clear sessions on user logout / TTL |
| **index_documents empty list** | Silently returns 0 chunks | Validate document list before calling |
| **top_k too high with reranker** | Slow reranker (> 1 second) | `top_k=10` for reranker input; retrieve top-50 first |
| **Multi-hop on simple queries** | 3× LLM calls for a 1-hop answer | Use `query()` for simple QA; multi-hop for complex |
| **Missing filter key in metadata** | Filter silently returns nothing | Log `chunk.metadata` to verify key names |
| **Wrong template for streaming** | Structured JSON mid-stream | Use `default` or `conversational` for streaming |
