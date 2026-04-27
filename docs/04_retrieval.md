# Module 04 — Retrieval
### `retrieval/retriever.py` · `retrieval/reranker.py`

---

## Table of Contents
1. [Retrieval Overview](#1-retrieval-overview)
2. [DenseRetriever](#2-denseretriever)
3. [BM25Retriever](#3-bm25retriever)
4. [HybridRetriever (RRF Fusion)](#4-hybridretriever-rrf-fusion)
5. [Query Transformation](#5-query-transformation)
6. [Maximal Marginal Relevance (MMR)](#6-maximal-marginal-relevance-mmr)
7. [Reranking](#7-reranking)
8. [Full Retrieval Pipeline](#8-full-retrieval-pipeline)
9. [Quick Reference](#9-quick-reference)
10. [Common Pitfalls](#10-common-pitfalls)

---

## 1. Retrieval Overview

Retrieval is the most critical stage — garbage in, garbage out applies
directly here. A bad retrieval means a bad (or hallucinated) answer even with
a perfect LLM.

```
Query
  │
  ├── Dense Retriever     (semantic similarity via embeddings)
  │         │
  ├── BM25 Retriever      (keyword matching via TF-IDF)
  │         │
  └── Hybrid Retriever ───┤ RRF Fusion
              │
              ▼
        Reranker         (CrossEncoder re-scores top-k)
              │
              ▼
      Top-K results      → PromptBuilder
```

**Retrieval strategies at a glance**:

| Strategy | Signal | Strong at | Weak at |
|---|---|---|---|
| Dense (ANN) | Semantic meaning | Paraphrases, concepts | Rare keywords, IDs |
| BM25 | Keyword frequency | Exact terms, codes, names | Synonyms, context |
| Hybrid (RRF) | Both combined | General production use | Tuning alpha parameter |
| Reranker | Deep cross-attention | Precision boost | Latency (+100–300ms) |

---

## 2. DenseRetriever

Uses the embedding engine to encode the query, then searches the vector store.

```python
from retrieval import DenseRetriever

retriever = DenseRetriever(
    vector_store=faiss_store,     # any BaseVectorStore
    embedder=embedding_engine,    # EmbeddingEngine
)

results = retriever.retrieve(
    query="What is the refund policy?",
    top_k=10,
    filter={"file_type": "pdf"},  # optional metadata filter
)
# Returns List[SearchResult] sorted by cosine similarity descending
```

**Retrieve by pre-computed vector** (when query is already embedded):
```python
query_vec = embedder.encode_query("refund policy")
results = retriever.retrieve_by_vector(query_vec, top_k=5)
```

### How It Works

```
Query text
    │
    ▼  encode_query()
Query vector [0.12, -0.34, …]          (1 × dim)
    │
    ▼  FAISS HNSW ANN search
Candidate vectors with scores
    │
    ▼  metadata filter (post-processing)
Top-K SearchResults
```

---

## 3. BM25Retriever

BM25 (Best Matching 25) is the gold-standard sparse retrieval algorithm.
It scores documents based on term frequency, inverse document frequency,
and document length normalization.

### BM25 Formula

```
                    tf(t,d) × (k1 + 1)
BM25(t,d) = IDF(t) × ────────────────────────────────────
                    tf(t,d) + k1 × (1 - b + b × |d|/avgdl)

where:
  tf(t,d)  = frequency of term t in document d
  IDF(t)   = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
  N        = total number of documents
  df(t)    = documents containing term t
  |d|      = document length
  avgdl    = average document length
  k1       = term saturation (default 1.5)
  b        = length normalization (default 0.75)
```

```python
from retrieval import BM25Retriever

retriever = BM25Retriever(k1=1.5, b=0.75)

# Must build index from corpus before querying
retriever.build(
    texts=     [c.content   for c in chunks],
    ids=       [c.chunk_id  for c in chunks],
    metadatas= [c.metadata  for c in chunks],
)

results = retriever.retrieve("refund policy 30 days", top_k=10)
```

### Save & Load BM25 Index

```python
# Save to disk (pickle)
retriever.save("./bm25_data/index")

# Load on restart (much faster than rebuild)
new_retriever = BM25Retriever()
new_retriever.load("./bm25_data/index")
```

### When BM25 Outperforms Dense

```
Query: "SKU-48291 refund"        → BM25 wins (exact product code)
Query: "What is your policy?"   → Dense wins (semantic matching)
Query: "30-day return window"    → Both similar
```

---

## 4. HybridRetriever (RRF Fusion)

Combines dense and sparse signals. Consistently outperforms either alone.

### Reciprocal Rank Fusion (RRF)

```
                         1
RRF_score(d) = Σ ─────────────────
                  rank_in_list + k

where k = 60 (constant smooths rank differences)

Example:
  Dense results:  chunk-A rank=1, chunk-B rank=3, chunk-C rank=7
  BM25 results:   chunk-B rank=1, chunk-D rank=2, chunk-A rank=5

  RRF scores:
    chunk-A: 1/(1+60) + 1/(5+60)  = 0.0164 + 0.0154 = 0.0318  ← merged top
    chunk-B: 1/(3+60) + 1/(1+60)  = 0.0159 + 0.0164 = 0.0323
    chunk-C: 1/(7+60)             = 0.0149
    chunk-D: 1/(2+60)             = 0.0161
```

RRF is **rank-based** (not score-based), so it's robust to score scale
differences between dense (cosine 0–1) and BM25 (unbounded positive).

```python
from retrieval import HybridRetriever, DenseRetriever, BM25Retriever

dense = DenseRetriever(vector_store, embedder)
bm25  = BM25Retriever()
bm25.build(texts, ids, metadatas)

# Mode 1: RRF fusion (recommended)
hybrid = HybridRetriever(dense, bm25, use_rrf=True, rrf_k=60)

# Mode 2: Score fusion with alpha weighting
hybrid = HybridRetriever(dense, bm25, use_rrf=False, alpha=0.7)
# alpha=0.7 → 70% dense score + 30% BM25 score

results = hybrid.retrieve("refund policy", top_k=10)
```

### Alpha Tuning Guide (score fusion)

| alpha | Bias | When to use |
|---|---|---|
| 1.0 | Pure dense | Semantic / conceptual queries |
| 0.7 | Dense-heavy | General purpose (start here) |
| 0.5 | Balanced | Mixed query types |
| 0.3 | BM25-heavy | Keyword / product code queries |
| 0.0 | Pure BM25 | Exact term matching only |

---

## 5. Query Transformation

Improves retrieval recall by generating better search queries from the original.

### 5.1 HyDE — Hypothetical Document Embeddings

Instead of embedding the query directly (which may be terse), generate a
hypothetical answer and embed that. The answer is similar to real answers
in the corpus.

```
Query: "What is the refund policy?"
            │
            ▼  LLM generates hypothetical answer
"Customers can return products within 30 days of purchase for a full
refund. After that, store credit is available. Digital downloads are
non-refundable once accessed."
            │
            ▼  embed the hypothetical answer
[0.34, -0.12, 0.78, …]   ← much closer to actual policy chunks
```

```python
from retrieval import QueryTransformer

llm_fn = lambda prompt: llm.complete(prompt).content

transformer = QueryTransformer(llm_fn)

# HyDE
hyde_text = transformer.hyde("What is the refund policy?")
hyde_vec  = embedder.encode_query(hyde_text)
results   = dense_retriever.retrieve_by_vector(hyde_vec)

# In pipeline (automatic)
results = pipeline.retrieve(query, use_hyde=True)
```

### 5.2 Multi-Query

Generate N paraphrases, retrieve for each, deduplicate, take union.
Increases recall by covering different phrasings.

```python
queries = transformer.multi_query(
    "What is the refund policy?",
    n=3
)
# Returns:
# ["What is the refund policy?",           ← original always included
#  "How do I return a purchased item?",
#  "What are the return conditions?"]

# Retrieve for each, merge results
all_results = {}
for q in queries:
    for r in dense.retrieve(q, top_k=5):
        all_results[r.chunk_id] = r   # deduplicate by chunk_id
final = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
```

### 5.3 StepBack

Reformulates a specific question into a more general one to retrieve
background knowledge first.

```python
general = transformer.step_back(
    "How many days do I have to return item SKU-48291?"
)
# → "What is the general product return policy?"
results = dense.retrieve(general, top_k=5)
```

---

## 6. Maximal Marginal Relevance (MMR)

Addresses the **redundancy problem**: top-K results often contain near-duplicate
chunks from the same section. MMR trades off relevance vs. diversity.

### Formula

```
MMR(d) = λ × sim(d, query) − (1 − λ) × max_{d'∈S} sim(d, d')

where:
  λ (lambda)    = relevance weight (0=pure diversity, 1=pure relevance)
  S             = already selected documents
  sim(d, d')    = cosine similarity between candidate and selected
```

```python
from retrieval import maximal_marginal_relevance

# First retrieve a large candidate set
candidates = dense.retrieve(query, top_k=50)
candidate_vecs = np.array([
    chunk_embeddings[r.chunk_id] for r in candidates
])
query_vec = embedder.encode_query(query)

# Then re-rank for diversity
diverse_results = maximal_marginal_relevance(
    query_vec=   query_vec,
    candidates=  candidates,
    vectors=     candidate_vecs,
    top_k=       10,
    lambda_param=0.7,     # 0.7 → mostly relevance, some diversity
)
```

**Effect of lambda**:
```
lambda=1.0: chunk-A(0.95), chunk-B(0.93), chunk-C(0.91)  ← same section!
lambda=0.7: chunk-A(0.95), chunk-D(0.82), chunk-F(0.79)  ← diverse sources
lambda=0.0: maximally spread across topics                 ← often too scattered
```

In the pipeline: `pipeline.retrieve(query, use_mmr=True, mmr_lambda=0.7)`

---

## 7. Reranking

Reranking applies a more expensive but more accurate model to the top-K
candidates retrieved by the fast retriever.

```
Fast retriever: top-50 candidates in ~10ms  (ANN search)
       │
       ▼
Reranker: re-scores all 50 pairs (query, chunk) in ~200ms
       │
       ▼
Top-10 reranked results  (better precision than retriever alone)
```

### 7.1 CrossEncoderReranker (Local)

Cross-encoders see both query and document together — much more accurate
than bi-encoders (which encode independently).

```python
from retrieval import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu",
    batch_size=32,
)

candidates = dense.retrieve(query, top_k=50)
reranked   = reranker.rerank(query, candidates, top_k=10)
```

**Bi-encoder vs cross-encoder**:
```
Bi-encoder (retriever):  encode(query) + encode(doc) → cosine  [FAST]
Cross-encoder (reranker): encode(query + doc) → score           [ACCURATE]
```

### 7.2 CohereReranker (API)

```python
from retrieval.reranker import CohereReranker

reranker = CohereReranker(
    api_key=os.getenv("COHERE_API_KEY"),
    model="rerank-english-v3.0",
)
reranked = reranker.rerank(query, candidates, top_k=10)
```

### 7.3 LLMReranker (LLM-as-judge)

Asks the LLM to rate each (query, document) pair 0–10 for relevance.
Most accurate but slowest (n LLM calls).

```python
from retrieval.reranker import LLMReranker

reranker = LLMReranker(llm=llm_interface)
reranked = reranker.rerank(query, candidates[:10], top_k=5)
```

---

## 8. Full Retrieval Pipeline

```python
# Example: production hybrid + reranker pipeline

# 1. Transform query
hyde_text = transformer.hyde(query)

# 2. Retrieve candidates (large candidate set)
dense_candidates = dense.retrieve(hyde_text, top_k=50)
bm25_candidates  = bm25.retrieve(query,      top_k=50)

# 3. Fuse
all_candidates = hybrid.fuse(dense_candidates, bm25_candidates)[:50]

# 4. MMR for diversity
candidate_vecs = np.array([chunk_embs[r.chunk_id] for r in all_candidates])
query_vec      = embedder.encode_query(query)
diverse        = maximal_marginal_relevance(query_vec, all_candidates, candidate_vecs, top_k=20)

# 5. Rerank
final = reranker.rerank(query, diverse, top_k=5)
```

In practice, the `RAGPipeline` handles all of this internally.

---

## 9. Quick Reference

```python
from retrieval import DenseRetriever, BM25Retriever, HybridRetriever
from retrieval import CrossEncoderReranker, QueryTransformer, maximal_marginal_relevance

# ── Dense ────────────────────────────────────────────────────────────────
dense   = DenseRetriever(store, embedder)
results = dense.retrieve("query", top_k=10)
results = dense.retrieve("query", top_k=10, filter={"source": "policy.pdf"})

# ── BM25 ─────────────────────────────────────────────────────────────────
bm25 = BM25Retriever()
bm25.build([c.content for c in chunks], [c.chunk_id for c in chunks], [c.metadata for c in chunks])
results = bm25.retrieve("exact keyword query", top_k=10)
bm25.save("./bm25_data"); bm25.load("./bm25_data")

# ── Hybrid ───────────────────────────────────────────────────────────────
hybrid  = HybridRetriever(dense, bm25, use_rrf=True)
results = hybrid.retrieve("mixed query", top_k=10)

# ── Rerank ───────────────────────────────────────────────────────────────
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank("query", results, top_k=5)

# ── Query transform ──────────────────────────────────────────────────────
t = QueryTransformer(lambda p: llm.complete(p).content)
t.hyde("query")           # → hypothetical document text
t.multi_query("query", 3) # → [original, paraphrase1, paraphrase2]
t.step_back("specific q") # → general question

# ── MMR ──────────────────────────────────────────────────────────────────
diverse = maximal_marginal_relevance(q_vec, candidates, cand_vecs, top_k=5, lambda_param=0.7)
```

---

## 10. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| **BM25 not built** | `RuntimeError: index not built` | Call `.build()` before `.retrieve()` |
| **Dense missing keywords** | "SKU-12345" not found | Add BM25 via Hybrid retriever |
| **BM25 missing synonyms** | "return" not found for "refund" | Add Dense via Hybrid retriever |
| **top_k too small** | Good chunk not in context | Retrieve top-50, rerank to top-5 |
| **Reranker on all docs** | Reranking takes > 1 second | Only rerank top-50 candidates |
| **HyDE hallucination** | Hypothetical answer is wrong domain | Use MultiQuery as fallback |
| **MMR lambda=0** | Results from random topics | Use lambda ≥ 0.5 for RAG |
| **No dedup in multi-query** | Same chunk returned 3× | Dedup by chunk_id before reranking |
