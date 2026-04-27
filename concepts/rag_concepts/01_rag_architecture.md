# RAG Architecture — System Design & Data Flow

## Conceptual Foundation

RAG (Retrieval-Augmented Generation) combines the **parametric knowledge** of large language models with **non-parametric knowledge** retrieved from external sources. This hybrid approach addresses the key limitations of pure LLMs: knowledge cutoffs, hallucinations, and inability to access private/recent information.

**Core insight**: Instead of requiring the LLM to memorize all facts, RAG dynamically retrieves relevant information and provides it as context, enabling accurate answers about information the LLM was never trained on.

### The RAG Paradigm

```
Traditional LLM:  Query → LLM → Answer (limited by training data)
RAG System:       Query → Retrieval → Context + Query → LLM → Answer
```

**Key advantages**:
- **Fresh information** — access to recently updated documents
- **Private knowledge** — company-specific or personal data
- **Reduced hallucination** — grounded in retrieved evidence
- **Explainable** — can cite sources for claims
- **Updatable** — change knowledge by updating document store

---

## Mathematical Formulation

### Probabilistic Framework

Traditional language model:
```
P(answer | query) = LLM_θ(query)
```

RAG formulation:
```
P(answer | query) = Σ_d P(answer | query, d) × P(d | query)
                  = Σ_d LLM_θ(query, d) × Retriever_φ(d | query)
```

Where:
- `d` represents retrieved documents/chunks
- `θ` are LLM parameters (fixed during inference)
- `φ` are retrieval model parameters

### Two-Stage Optimization

**Stage 1: Retrieval** (maximize relevance)
```
φ* = argmax_φ Σ_i log P(d_i^+ | query_i)
```
Where `d_i^+` are relevant documents for query `i`

**Stage 2: Generation** (maximize answer quality)
```
θ* = argmax_θ Σ_i log P(answer_i | query_i, retrieved_docs_i)
```

**End-to-end optimization** (joint training):
```
(θ*, φ*) = argmax_{θ,φ} Σ_i log P(answer_i | query_i) 
```

---

## Implementation Architecture

### System Overview

Your RAG system (`rag_pipeline.py`) implements a **modular orchestrator pattern**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAGPipeline Orchestrator                     │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Data Pipeline  │  │   Vector Store  │  │   Generation    │ │
│  │                 │  │                 │  │                 │ │
│  │  DocumentLoader │  │  FAISSVector    │  │  PromptBuilder  │ │
│  │  ChunkingEngine │  │  ChromaDB       │  │  LLMInterface   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│            │                     │                     │        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Embeddings    │  │   Retrieval     │  │   Evaluation    │ │
│  │                 │  │                 │  │                 │ │
│  │  EmbeddingEngine│  │  DenseRetriever │  │  RAGEvaluator   │ │
│  │  (ST/OpenAI)    │  │  BM25Retriever  │  │  LLMJudge       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow — Indexing Path

```python
# From rag_pipeline.py: index() method
def index(self, sources: List[str]) -> IndexingResult:
    """
    sources → DocumentLoader → [Document] → ChunkingEngine → [Chunk]
                                                    ↓
    EmbeddingEngine.encode([chunk.content]) → vectors (np.ndarray)
                                                    ↓
    VectorStore.add(chunk_ids, vectors, texts, metadatas)
                                                    ↓ 
    BM25Retriever.build(texts, ids, metadatas)  ← for hybrid search
    """
```

**Step-by-step breakdown**:

1. **Document Loading**:
   ```python
   loader = DocumentLoader()
   docs = []
   for source in sources:
       if os.path.isdir(source):
           docs.extend(loader.load_directory(source, recursive=True))
       elif source.startswith(('http://', 'https://')):
           docs.append(loader.load_url(source))
       else:
           docs.append(loader.load_file(source))
   ```

2. **Chunking**:
   ```python
   chunker = ChunkingEngine(strategy="recursive", chunk_size=512, overlap=64)
   chunks = chunker.split(docs)
   # Each chunk: Chunk(chunk_id, doc_id, content, metadata, chunk_index)
   ```

3. **Embedding**:
   ```python
   embedder = EmbeddingEngine("sentence_transformers", "all-MiniLM-L6-v2")
   texts = [chunk.content for chunk in chunks]
   result = embedder.encode(texts, batch_size=64)
   vectors = result.embeddings  # shape: (n_chunks, embedding_dim)
   ```

4. **Vector Storage**:
   ```python
   vector_store.add(
       ids=[c.chunk_id for c in chunks],
       vectors=vectors,
       texts=texts,
       metadatas=[c.metadata for c in chunks]
   )
   ```

5. **BM25 Index** (for hybrid retrieval):
   ```python
   bm25_retriever.build(texts, [c.chunk_id for c in chunks], [c.metadata for c in chunks])
   ```

### Data Flow — Query Path

```python
# From rag_pipeline.py: query() method
def query(self, query: str, **kwargs) -> RAGResponse:
    """
    query → [query_transform] → embed_query → retrieval → reranking
                                                    ↓
    context_construction → prompt_building → LLM → RAGResponse
    """
```

**Detailed query flow**:

1. **Query Preprocessing** (optional):
   ```python
   if use_hyde:
       # Generate hypothetical answer, use for retrieval
       hyde_prompt = f"Answer this question: {query}"
       hypothetical_answer = llm.complete(hyde_prompt).content
       search_query = hypothetical_answer
   else:
       search_query = query
   ```

2. **Query Embedding**:
   ```python
   query_vector = embedder.encode_query(search_query)
   # shape: (embedding_dim,)
   ```

3. **Retrieval** (hybrid by default):
   ```python
   # Dense retrieval
   dense_results = dense_retriever.retrieve(query_vector, top_k=20)
   
   # Sparse retrieval  
   bm25_results = bm25_retriever.retrieve(query, top_k=20)
   
   # Hybrid fusion (RRF)
   fused_results = hybrid_retriever.fuse(dense_results, bm25_results, top_k=10)
   ```

4. **Reranking** (optional):
   ```python
   if use_reranker:
       reranked = cross_encoder_reranker.rerank(query, fused_results, top_k=5)
   else:
       reranked = fused_results[:top_k]
   ```

5. **Context Construction**:
   ```python
   context_str, citations = context_builder.build(
       search_results=reranked,
       max_tokens=3000,
       citation_style="inline"
   )
   ```

6. **Prompt Building**:
   ```python
   messages, _ = prompt_builder.build(
       question=query,
       retrieved_results=reranked,
       history=conversation_history  # if session_id provided
   )
   ```

7. **LLM Generation**:
   ```python
   llm_response = llm.chat(messages)
   ```

8. **Response Assembly**:
   ```python
   return RAGResponse(
       query=query,
       answer=llm_response.content,
       citations=citations,
       retrieved_chunks=reranked,
       llm_response=llm_response,
       latency_breakdown=timing_dict
   )
   ```

---

## Comparative Analysis

### RAG vs Alternatives

| Approach | Knowledge Source | Update Frequency | Accuracy | Explainability |
|---|---|---|---|---|
| **Pure LLM** | Training data | Never (fixed) | Medium | Low |
| **Fine-tuned LLM** | Training + domain data | Rare (expensive) | High | Low |
| **RAG** | External documents | Real-time | High | High |
| **Function calling** | APIs/databases | Real-time | Very high | High |

### Architecture Patterns

**Naive RAG** (simple retrieval + generation):
```
Query → Retrieval → Top-K → Concat → LLM → Answer
```
✅ Simple to implement
❌ Poor retrieval quality, no context optimization

**Advanced RAG** (your implementation):
```
Query → [Transform] → Hybrid Retrieval → Rerank → Context Build → LLM → Answer
```
✅ Better retrieval, optimized context
✅ Multi-turn conversations
✅ Explainable with citations

**Modular RAG** (agent-based):
```
Query → Plan → [Retrieve → Reason → Act]* → Synthesize → Answer  
```
✅ Complex reasoning capability
❌ Higher latency, more failure modes

### Configuration Strategies

**Development** (fast iteration):
```python
config = RAGConfig(
    embedding=EmbeddingConfig(provider="sentence_transformers"),  # local
    vector_store=VectorStoreConfig(backend="faiss", index_type="flat"),  # exact search
    llm=LLMConfig(provider="ollama", model="llama3.1:8b"),  # local
    retrieval=RetrievalConfig(use_reranker=False)  # faster
)
```

**Production** (quality optimized):
```python
config = RAGConfig(
    embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
    vector_store=VectorStoreConfig(backend="faiss", index_type="hnsw"),  # fast ANN
    llm=LLMConfig(provider="openai", model="gpt-4o-mini"),
    retrieval=RetrievalConfig(use_reranker=True, use_hybrid=True)  # best quality
)
```

---

## Practical Guidelines

### Performance Optimization

**Indexing**:
- **Batch processing**: Use `batch_size=128` for embedding to optimize GPU utilization
- **Incremental updates**: Call `index_documents()` for new docs without rebuilding
- **Persistence**: Always `save()` after indexing to avoid recomputation

**Querying**:
- **Caching**: Enable embedding and response caching for repeated queries
- **Top-K tuning**: Start with `top_k=10`, increase if missing relevant info
- **Streaming**: Use `stream_query()` for better user experience

### Quality Tuning

**Chunk size optimization**:
```python
# Start with these, tune based on your domain
chunk_size = 512    # 1-2 paragraphs, good for most content
overlap = 64        # 12.5% overlap, prevents boundary issues

# For technical docs: smaller chunks (256)
# For narrative text: larger chunks (1024)
```

**Retrieval balance**:
```python
# Hybrid retrieval alpha (dense vs BM25 weight)
alpha = 0.7   # 70% dense, 30% BM25 (good default)
alpha = 0.9   # More semantic, less keyword matching
alpha = 0.5   # Balanced semantic + keyword
```

### Common Failure Modes

1. **Empty retrievals**: Query doesn't match indexed content
   - **Debug**: Check embeddings similarity manually
   - **Fix**: Improve query preprocessing, expand chunk coverage

2. **Irrelevant retrievals**: High similarity but wrong context
   - **Debug**: Examine top-10 results before reranking  
   - **Fix**: Enable reranker, improve chunking boundaries

3. **Context overflow**: Retrieved content exceeds LLM context window
   - **Debug**: Monitor `total_latency_ms` for sudden spikes
   - **Fix**: Reduce `max_context_tokens`, improve retrieval precision

4. **Hallucination despite context**: LLM ignores provided information
   - **Debug**: Check prompt template, context construction
   - **Fix**: Use stricter templates (`safety_strict`), better LLM

### Monitoring & Observability

**Key metrics to track**:
```python
# Retrieval quality
retrieval_precision = relevant_retrieved / total_retrieved
retrieval_recall = relevant_retrieved / total_relevant

# Generation quality  
faithfulness = claims_supported_by_context / total_claims
answer_relevancy = addresses_query_fully(answer, query)

# System performance
p95_latency = np.percentile(response_times, 95)
cache_hit_rate = cache_hits / total_queries
cost_per_query = (embedding_cost + llm_cost) / num_queries
```

**Implementation**:
```python
# In your RAGResponse
response.latency_breakdown = {
    "retrieval_ms": retrieval_time,
    "generation_ms": generation_time,
    "total_ms": total_time
}

# Track in production
metrics.histogram("rag.latency.retrieval", retrieval_time)
metrics.histogram("rag.latency.generation", generation_time)  
metrics.counter("rag.queries.total").inc()
```

---

## Advanced Patterns

Your implementation includes several advanced RAG patterns:

### Multi-Hop Retrieval
```python
def multi_hop_query(self, query: str, max_hops: int = 3) -> RAGResponse:
    """
    Hop 1: Query → Initial retrieval → Check sufficiency
    Hop 2: Generate sub-query → Additional retrieval  
    Hop 3: Synthesize accumulated context → Final answer
    """
```

### Agentic RAG (ReAct)
```python
def agentic_query(self, query: str, max_iterations: int = 5) -> RAGResponse:
    """
    Loop:
      Thought: Analyze what information is needed
      Action: Choose tool (search, calculate, etc.)  
      Observation: Process tool output
      [Repeat until sufficient information gathered]
    Final: Generate comprehensive answer
    """
```

### Session Management
```python
# Multi-turn conversation with history
session_history = ConversationHistory(max_tokens=2000)
response1 = pipeline.query("What is the refund policy?", session_id="user-123")
response2 = pipeline.query("What about digital items?", session_id="user-123")
# LLM sees both exchanges for context
```

**Next concept**: Document Processing — how loading and chunking strategies affect RAG quality
