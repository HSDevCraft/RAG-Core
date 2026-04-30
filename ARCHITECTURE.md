# RAG System — Architecture Reference

## 1. System Overview

### Problem Definition
Large Language Models (LLMs) have static knowledge frozen at training time and cannot access private enterprise data. RAG solves this by:
- Retrieving relevant documents at inference time
- Injecting them as context into the LLM prompt
- Grounding answers in verifiable, citable sources

### Enterprise Use Cases
| Use Case | Description | Scale |
|---|---|---|
| Internal KB QA | Employee Q&A over policies, runbooks, wikis | 100K docs |
| Legal Document Search | Contract analysis, compliance Q&A | 500K docs |
| Customer Support | Auto-answer from product docs + FAQs | 1M queries/day |
| Code Assistant | Q&A over internal codebases + docs | 5M chunks |
| Research Synthesis | Literature review over scientific papers | 10M papers |

---

## 2. System Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        RAG SYSTEM ARCHITECTURE                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌─────────────────────── INDEXING PIPELINE ──────────────────────────┐ ║
║  │                                                                     │ ║
║  │  [Raw Sources]                                                      │ ║
║  │  PDF/HTML/DOCX/JSON/DB/URL                                          │ ║
║  │       │                                                             │ ║
║  │       ▼                                                             │ ║
║  │  [DocumentLoader] ──► [Cleaner] ──► [Documents]                    │ ║
║  │                                          │                          │ ║
║  │       ┌─────────────────────────────────┘                          │ ║
║  │       ▼                                                             │ ║
║  │  [ChunkingEngine]                                                   │ ║
║  │   Recursive │ Token │ Sentence │ Semantic                          │ ║
║  │       │                                                             │ ║
║  │       ▼                                                             │ ║
║  │  [EmbeddingEngine] ──────────────────────── Cache (diskcache)      │ ║
║  │   ST / OpenAI / Cohere                                              │ ║
║  │       │                            │                               │ ║
║  │       ▼                            ▼                               │ ║
║  │  [VectorStore]              [BM25 Index]                           │ ║
║  │  FAISS/Chroma/Pinecone      rank_bm25                              │ ║
║  └─────────────────────────────────────────────────────────────────────┘ ║
║                                                                          ║
║  ┌─────────────────────── RETRIEVAL PIPELINE ─────────────────────────┐ ║
║  │                                                                     │ ║
║  │  [User Query]                                                       │ ║
║  │       │                                                             │ ║
║  │       ├──► [QueryTransformer] (HyDE / MultiQuery / StepBack)       │ ║
║  │       │                                                             │ ║
║  │       ├──► [DenseRetriever]  ──► VectorStore ANN                   │ ║
║  │       │         │                                                   │ ║
║  │       └──► [BM25Retriever]   ──► BM25 Index                       │ ║
║  │                 │                                                   │ ║
║  │         [HybridRetriever] (RRF Fusion)                             │ ║
║  │                 │                                                   │ ║
║  │         [CrossEncoderReranker]                                      │ ║
║  │                 │                                                   │ ║
║  │         [MMR Diversity Filter]                                      │ ║
║  │                 │                                                   │ ║
║  │         [top-k Chunks]                                              │ ║
║  └─────────────────────────────────────────────────────────────────────┘ ║
║                                                                          ║
║  ┌─────────────────────── GENERATION PIPELINE ────────────────────────┐ ║
║  │                                                                     │ ║
║  │  [top-k Chunks] + [Query] + [Session History]                       │ ║
║  │       │                                                             │ ║
║  │       ▼                                                             │ ║
║  │  [PromptBuilder]                                                    │ ║
║  │   Template + Context injection + Citation markers + Token budget    │ ║
║  │       │                                                             │ ║
║  │       ▼                                                             │ ║
║  │  [LLMInterface]                                                     │ ║
║  │   OpenAI / Azure / Ollama / Anthropic / HuggingFace                │ ║
║  │       │                                                             │ ║
║  │       ▼                                                             │ ║
║  │  [RAGResponse]                                                      │ ║
║  │   answer + citations + latency breakdown + token usage             │ ║
║  └─────────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 3. Component-Level Design Decisions

### 3.1 Data Ingestion Layer

**Decision**: Adapter pattern per format → unified `Document` type
- Each loader is thin (< 50 LOC) delegating to proven libs (pypdf, bs4, python-docx)
- `Document.doc_id` is content-hash based → idempotent re-ingestion
- `DirectoryLoader` dispatches by extension → zero config for mixed corpora

**Chunking strategy selection**:
```
IF corpus is structured (legal, scientific) → semantic chunking
IF latency matters (real-time indexing)     → recursive chunking (default)
IF model token limit is strict              → token chunking
IF QA requires full sentences               → sentence chunking
```

**Chunk size guidelines**:
| Use Case | Chunk Size | Overlap |
|---|---|---|
| Dense retrieval + GPT-4 | 512 tokens | 64 tokens |
| Dense retrieval + Llama 8B | 256 tokens | 32 tokens |
| Long-context models (128K) | 2048 tokens | 128 tokens |
| Code retrieval | 200–500 chars | 0 (function boundaries) |

---

### 3.2 Embedding Layer

**Asymmetric embeddings**: Query and document can use different instructions.
BGE models support this via prefix:
- Document: `"Represent this sentence: "`
- Query: `"Represent this question for retrieving relevant passages: "`

**Matryoshka embeddings** (text-embedding-3-*): Can truncate to 256/512 dims
for 2-4× speed at ~3% accuracy loss. Useful when latency > accuracy.

**Cost model** (1M document chunks, 500 tokens avg):
| Model | Total tokens | Cost |
|---|---|---|
| all-MiniLM-L6-v2 | 500M | $0 (local) |
| text-embedding-3-small | 500M | $10 |
| text-embedding-3-large | 500M | $65 |
| Cohere embed-v3 | 500M | ~$50 |

---

### 3.3 Vector Database

**FAISS (recommended for most use cases)**:
- No network hop → lowest possible latency
- Proven at Meta/industry scale (billions of vectors)
- Limitation: manual metadata filtering (client-side)
- HNSW: O(log n) search, 95-99% recall at ef=50

**Chroma (recommended for dev/mid-scale)**:
- Native metadata filtering with `where` clauses
- REST API for microservice architectures
- Built-in persistence

**Pinecone (recommended for enterprise SaaS)**:
- Zero infrastructure management
- Namespace isolation for multi-tenancy
- Hybrid search built-in (sparse + dense)
- Cost: ~$96/1M reads (serverless)

**Index Selection Logic**:
```
n_vectors < 10,000      → IndexFlatIP (exact, fast build)
n_vectors < 1,000,000   → IndexIVFFlat (nlist=128)
n_vectors >= 1,000,000  → IndexHNSWFlat (M=16, ef=200)
memory constrained      → Add IndexPQ (4x compression, ~10% accuracy loss)
```

---

### 3.4 Retrieval Layer

**Hybrid Search (RRF)**: Reciprocal Rank Fusion
```
score(doc) = Σᵢ 1 / (k + rankᵢ(doc))    where k=60
```
- Scale-invariant: BM25 scores (0-∞) and cosine scores (0-1) fuse correctly
- Empirically outperforms score normalization on BEIR benchmark

**Cross-Encoder vs Bi-Encoder trade-off**:
```
Bi-encoder:   O(1) query time, pre-computed doc embeddings
              Approximate — documents/query embedded independently
              
Cross-encoder: O(k) query time (k = candidate set size)
               Exact — attends over (query, doc) jointly
               ~15-20% better MRR@10 than bi-encoder alone
```

**Production pattern**: Bi-encoder retrieval of top-40 → cross-encoder rerank to top-5
- Retrieval: ~10ms (HNSW)
- Reranking: ~50ms (MiniLM L6 on CPU for 40 candidates)
- Total added latency: ~60ms for substantial accuracy gain

---

### 3.5 Generation Layer

**Token budget management**:
```
Total context window  =  system_prompt  +  context  +  answer_reserved
                      e.g. 8192         =  500       +  6692  +  1000

Context budget = floor(available_tokens / avg_chunk_tokens) chunks
```

**Prompt injection defense** (implemented in `api/main.py`):
1. Regex detection of known injection patterns
2. Reject with 400 before reaching LLM
3. System prompt with role anchoring ("You are a professional assistant...")
4. Output sanitization (PII scrubbing before returning)

**Guardrails taxonomy**:
| Type | Implementation | Layer |
|---|---|---|
| Input guardrails | Regex + LLM classifier | API middleware |
| Output guardrails | PII scrubbing, content filter | Post-generation |
| Prompt defense | Role anchoring + injection detection | Prompt template |
| Citation grounding | Require [N] references in answer | Prompt instruction |

---

### 3.6 Evaluation Layer

**Evaluation pyramid**:
```
            ┌─────────────────┐
            │   Human Eval    │  ← Gold standard, expensive, slow
            ├─────────────────┤
            │  LLM-as-Judge   │  ← Automated, GPT-4 class, moderate cost
            ├─────────────────┤
            │  RAGAS Metrics  │  ← Automated, end-to-end, reproducible
            ├─────────────────┤
            │ String Metrics  │  ← ROUGE/F1, free, fast, limited
            └─────────────────┘
```

**Key metrics and what they catch**:
| Metric | What it measures | Failure it catches |
|---|---|---|
| Faithfulness | Claims grounded in context? | Hallucination |
| Answer Relevancy | Does answer address question? | Off-topic / verbose |
| Context Precision | Retrieved chunks relevant? | Poor retrieval precision |
| Context Recall | All relevant info retrieved? | Low recall / incomplete |
| Answer Correctness | Matches reference answer? | Factual errors |

---

## 4. Advanced RAG Patterns

### 4.1 Multi-Hop Retrieval
```
Q: "Who leads the company that acquired OpenAI's biggest competitor?"

Hop 1: Retrieve → "Anthropic is a leading AI company"
Hop 2: Retrieve → "Amazon invested $4B in Anthropic"  
Hop 3: Retrieve → "Amazon CEO is Andy Jassy"
Generate: "Andy Jassy (Amazon CEO) leads the company..."
```

**When to use**: Questions requiring 2+ inference steps across documents.
**Implementation**: Iterative retrieve-reason-retrieve loop with LLM as planner.
**Risk**: Latency grows linearly with hops; cap at 3-5.

### 4.2 Agentic RAG (ReAct)
```
Thought: I need to find the refund policy and the product price.
Action: retrieve("refund policy")
Observation: "Full refund within 30 days..."
Thought: Now I need the product price.
Action: retrieve("SmartWatch Pro price")
Observation: "SmartWatch Pro costs $299"
Thought: I have all information needed.
Action: finish("You can return the SmartWatch Pro ($299) within 30 days for a full refund.")
```

**When to use**: Complex queries requiring multiple retrieval steps + tool use.

### 4.3 Graph RAG
Builds a knowledge graph from documents; retrieves subgraphs around entities.
**When to use**: Entity-centric queries, relationship discovery, large corpora.
**Implementation**: Extract entities/relations → Neo4j or NetworkX → graph traversal.

### 4.4 Streaming RAG
```python
for token in pipeline.stream_query(question):
    send_to_client(token)   # SSE or WebSocket
```
**When to use**: UI where perceived latency matters. Reduces time-to-first-token.

### 4.5 Memory-Augmented RAG
Session memory (`ConversationHistory`) + episodic memory (past Q&A stored in vector DB).
```
Short-term:  ConversationHistory (last N turns in-context)
Long-term:   Store past Q&A pairs as documents; retrieve on new queries
```

### 4.6 Tool-Augmented RAG
```python
tools = {
    "retrieve": lambda q: pipeline.retrieve(q),
    "calculate": lambda expr: eval(expr),         # math
    "search_web": lambda q: web_search_api(q),    # live web
    "sql_query": lambda q: db.execute(q),         # structured data
}
response = pipeline.agentic_query(question, tools=tools)
```

---

## 5. Production Considerations

### Caching Strategy
```
L1: In-process query cache (dict, TTL=60s) — eliminates duplicate requests
L2: Redis cache (TTL=3600s) — shared across workers/replicas
L3: Embedding cache (diskcache) — avoid re-embedding same chunks
L4: BM25 index (persisted to disk) — avoid full rebuild on restart
```

### Scaling Architecture
```
                    [Load Balancer]
                    /      |      \
              [API-1]  [API-2]  [API-3]   ← stateless FastAPI workers
                    \      |      /
                     [Redis Cache]         ← shared query cache
                         |
                  [Vector Store]           ← FAISS (per-process) OR
                                           ← Chroma/Pinecone (shared)
                         |
                  [Embedding Service]      ← separate GPU service if needed
```

**Horizontal scaling**: API layer is stateless → scale with more workers/pods.
**FAISS scaling**: Each worker loads its own FAISS index (memory-mapped). For > 100GB indices, use Pinecone or Weaviate.

### Cost Optimization
| Lever | Saving | Trade-off |
|---|---|---|
| Local embeddings (BGE) vs OpenAI | 100% embed cost | Slight accuracy loss |
| gpt-4o-mini vs gpt-4o | 90% LLM cost | ~5% accuracy loss |
| Redis query cache | 60-80% LLM calls | Stale responses |
| Matryoshka truncation (256-dim) | 4× vector storage | ~3% recall loss |
| Smaller reranker (MiniLM-L6) | 2× speed | ~5% rerank accuracy |

### Security Checklist
- [ ] API key authentication on all endpoints
- [ ] Input sanitization (prompt injection detection)
- [ ] PII scrubbing on outputs
- [ ] Document-level access control (per-user namespace in vector DB)
- [ ] Audit logs for all queries (who asked what, when)
- [ ] Rate limiting (10 req/s per API key)
- [ ] TLS termination at load balancer
- [ ] Secrets in environment variables, never in code

---

## 6. Deployment

### Cloud Deployment Options

**AWS**:
```
ECS Fargate     → API service (auto-scaling)
ElastiCache     → Redis cache
S3              → FAISS index persistence
ECR             → Docker image registry
ALB             → Load balancer + TLS
CloudWatch      → Logging + alerts
```

**GCP**:
```
Cloud Run       → Serverless API (scales to zero)
Memorystore     → Redis
Cloud Storage   → Index persistence
Artifact Registry → Docker images
Cloud Load Balancing → TLS
Cloud Monitoring → Observability
```

**Azure**:
```
Container Apps  → API service
Azure Cache     → Redis
Blob Storage    → Index persistence
Azure OpenAI    → Enterprise LLM endpoint
Application Gateway → Load balancer
Monitor         → Logging
```

---

## 7. RAG Failure Modes & Mitigations

| Failure | Root Cause | Mitigation |
|---|---|---|
| Hallucination | No relevant chunks retrieved | Faithfulness check + "I don't know" fallback |
| Missing context | Chunking splits related info | Larger chunks + overlap |
| Stale answers | Index not refreshed | Scheduled re-indexing + source timestamps |
| Wrong document | Query too ambiguous | HyDE + MultiQuery |
| PII leakage | Sensitive data in chunks | PII detection before indexing |
| Prompt injection | Malicious user input | Input sanitization middleware |
| Slow response | Large context + slow LLM | Caching + gpt-4o-mini + streaming |
| High cost | Many tokens per query | Smaller context window + local LLM |
| Multi-hop failure | Single retrieval step | Multi-hop retrieval pipeline |
| Duplicate chunks | Same doc indexed twice | Content-hash deduplication |

---

## 8. CI/CD Pipeline

### 8.1 Pipeline Overview

```
Developer push / PR
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    GitHub Actions CI                           │
│                                                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Lint &  │   │  Unit    │   │  Docker  │   │ Publish  │  │
│  │  Type    │──▶│  Tests   │──▶│  Build + │──▶│ to GHCR  │  │
│  │  Check   │   │  3.10-12 │   │  Trivy   │   │ (on tag) │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│       │               │                                       │
│     ruff            pytest                                    │
│     black           coverage                                  │
│     mypy            codecov                                   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼ (on git tag v*.*.*)
┌───────────────────────────────────────────────────────────────┐
│                    Release Workflow                            │
│                                                               │
│   git-cliff changelog ──▶ GitHub Release ──▶ GHCR image tag  │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    Deployment Targets                          │
│                                                               │
│   Docker Compose (single node)                                │
│   ┌──────────┐  ┌───────┐  ┌──────────┐  ┌────────┐         │
│   │ rag-api  │  │ Redis │  │  Chroma  │  │Grafana │         │
│   │ :8080    │  │ :6379 │  │  :8000   │  │ :3001  │         │
│   └──────────┘  └───────┘  └──────────┘  └────────┘         │
│                                                               │
│   Kubernetes (production)                                     │
│   ┌─────────────────────────────────────────────────┐        │
│   │  Deployment (rag-api, replicas=3)               │        │
│   │  HPA (CPU ≥ 70% → scale up, max=10)             │        │
│   │  Service (ClusterIP) + Ingress (nginx/ALB)       │        │
│   │  ConfigMap (config.yaml) + Secret (api-keys)     │        │
│   └─────────────────────────────────────────────────┘        │
└───────────────────────────────────────────────────────────────┘
```

### 8.2 Branch Strategy

```
main          ← production-ready; protected; CI must pass
  │
develop       ← integration branch; merges to main via PR
  │
feature/*     ← feature branches; PR into develop
bugfix/*      ← bug fixes; PR into develop (or main for hotfixes)
release/v*.*  ← release prep; bumps version, updates changelog
```

### 8.3 CI Job Details

| Job | Trigger | Runtime | Key actions |
|---|---|---|---|
| `lint` | push + PR | ~30s | ruff, black --check, mypy |
| `test-unit` | push + PR | ~2min | pytest tests/unit/, codecov |
| `test-integration` | push to main/develop | ~5min | pytest tests/integration/ -m "not slow" |
| `docker` | push + PR | ~4min | docker build, trivy CVE scan |
| `publish` | git tag v*.*.* | ~5min | push image to ghcr.io |
| `release` | git tag v*.*.* | ~30s | git-cliff changelog, GitHub Release |

### 8.4 Quality Gates

A PR cannot be merged to `main` unless:
- ✅ All unit tests pass on Python 3.10, 3.11, 3.12
- ✅ Code coverage ≥ 75% (enforced via `fail_under` in `pyproject.toml`)
- ✅ `ruff` reports zero violations
- ✅ `black --check` passes (no unformatted code)
- ✅ Docker image builds successfully
- ✅ Trivy reports zero CRITICAL CVEs

---

## 9. Testing Strategy

### 9.1 Test Pyramid

```
                    ┌──────────┐
                    │  E2E /   │  ← Slow; require running server
                    │ Contract │    pytest tests/integration/ -m slow
                    └──────────┘
                  ┌──────────────┐
                  │ Integration  │  ← Real embedder + FAISS + mock LLM
                  │    Tests     │    pytest tests/integration/
                  └──────────────┘
            ┌──────────────────────┐
            │      Unit Tests      │  ← Fully mocked; fast; no API key
            │  (all components)    │    pytest tests/unit/
            └──────────────────────┘
```

### 9.2 Test Suite Structure

```
tests/
├── conftest.py               ← Shared fixtures (mock_embedder, mock_llm,
│                                faiss_store, rag_pipeline, api_client)
├── unit/
│   ├── test_document_loader.py  ← Document model, loaders, DirectoryLoader
│   ├── test_chunking.py         ← Recursive, Token, Sentence chunkers
│   ├── test_vector_store.py     ← FAISS (Flat/HNSW), Chroma, SearchResult
│   ├── test_retriever.py        ← Dense, BM25, Hybrid, MMR, QueryTransformer
│   ├── test_reranker.py         ← CrossEncoder, LLMReranker, CohereReranker
│   ├── test_prompt_builder.py   ← Templates, ContextBuilder, ConversationHistory
│   ├── test_llm_interface.py    ← OpenAI, Ollama, LLMResponse, fallback logic
│   ├── test_rag_pipeline.py     ← index, index_documents, query, stream, persistence
│   └── test_api.py              ← FastAPI endpoints via TestClient
└── integration/
    └── test_end_to_end.py       ← Real embedder + FAISS; mock LLM only
```

### 9.3 Mocking Strategy

**What we mock:**
- LLM API calls (`openai.OpenAI`, `requests.post` for Ollama)
- Cohere API calls
- Embedding model loading (in unit tests; real model in integration)
- External HTTP requests in loaders

**What we do NOT mock in unit tests:**
- FAISS index operations (pure C++, no network, fast)
- BM25 ranking algorithm (pure Python)
- Chunking logic (pure string operations)
- PromptBuilder template rendering

**What we never mock:**
- Data structures / dataclasses
- Pure utility functions
- Type validation logic

### 9.4 Fixture Scopes

| Fixture | Scope | Why |
|---|---|---|
| `mock_embedder` | function | Mock state is cheap; fresh per test prevents leakage |
| `mock_llm` | function | Same reasoning |
| `faiss_store` | function | Pre-populated with sample_vectors; isolated |
| `rag_pipeline` | function | Assembled from function-scoped components |
| `real_embedder` | session | Model load takes ~2s; safe to share (read-only) |
| `integrated_pipeline` | module | Index build takes ~1s; shared across integration tests |
| `api_client` | function | New TestClient per test; prevents session state bleed |

### 9.5 Coverage Targets

| Module | Target | Current strategy |
|---|---|---|
| `data_pipeline/` | ≥ 85% | Full unit tests for all loaders + chunkers |
| `embeddings/` | ≥ 80% | Mock provider calls; test cache, batching, normalization |
| `vector_store/` | ≥ 90% | FAISS Flat + HNSW + Chroma covered; persist/reload |
| `retrieval/` | ≥ 85% | Dense, BM25, Hybrid, MMR, QueryTransformer |
| `generation/` | ≥ 85% | All 7 templates, ContextBuilder token budget, history |
| `rag_pipeline.py` | ≥ 80% | Index, query, stream, multi-hop, session management |
| `api/main.py` | ≥ 75% | All endpoints via TestClient; auth + error paths |
| `utils/cache.py` | ≥ 80% | NullCache + DiskCache; Redis mocked |

### 9.6 Running Tests

```bash
# Fast feedback (unit only, ~30s)
make test-fast

# Full suite with coverage report
make test

# HTML coverage report
make test-cov
open htmlcov/index.html

# Integration tests only
make test-integration

# Specific file
pytest tests/unit/test_retriever.py -v

# Specific test
pytest tests/unit/test_vector_store.py::TestFAISSFlat::test_search_returns_top_k -v

# Skip slow API tests
pytest tests/ -m "not slow" -v

# Run with real OpenAI key
OPENAI_API_KEY=sk-... pytest tests/ -m slow -v
```
