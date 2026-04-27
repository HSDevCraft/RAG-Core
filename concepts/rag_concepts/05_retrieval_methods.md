# Retrieval Methods — Dense, Sparse & Hybrid Search

## Conceptual Foundation

Retrieval is the **heart of RAG systems** — it determines which information reaches the LLM for answer generation. Different retrieval methods capture different aspects of relevance:

- **Dense retrieval**: Captures **semantic similarity** using learned embeddings
- **Sparse retrieval**: Captures **lexical matching** using statistical term weighting  
- **Hybrid retrieval**: Combines both approaches for **comprehensive coverage**

**Key insight**: No single retrieval method is universally best. Dense excels at paraphrase/concept matching, sparse excels at exact term/entity matching, and hybrid provides the robustness needed for production systems.

### The Retrieval Spectrum

```
Query: "What's the return policy?"

Dense → Finds: "refund procedures", "money-back guarantee", "exchange terms"
        (semantic/conceptual matches)

Sparse → Finds: "return policy", "returns and exchanges", "return window"  
         (lexical/keyword matches)

Hybrid → Finds: Best of both approaches, ranked by fusion algorithm
```

---

## Mathematical Formulation

### Dense Retrieval (Semantic)

**Bi-encoder architecture**:
```
score(q, d) = sim(E_q(q), E_d(d))
```

Where:
- `E_q`: Query encoder (transforms query to vector)
- `E_d`: Document encoder (transforms document to vector)  
- `sim`: Similarity function (cosine, dot-product)

**Training objective** (contrastive learning):
```
L = -log(exp(score(q, d⁺)) / (exp(score(q, d⁺)) + Σᵢ exp(score(q, dᵢ⁻))))
```

Where `d⁺` is relevant document, `dᵢ⁻` are negative examples.

### Sparse Retrieval (Lexical)

**BM25 scoring formula**:
```
BM25(q, d) = Σₜ∈q IDF(t) × (tf(t,d) × (k₁ + 1)) / (tf(t,d) + k₁ × (1 - b + b × |d|/avgdl))
```

Where:
- `tf(t,d)`: Term frequency of `t` in document `d`
- `IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))`: Inverse document frequency
- `|d|`: Document length, `avgdl`: Average document length
- `k₁ = 1.2`, `b = 0.75`: Tuning parameters

### Hybrid Fusion

**Reciprocal Rank Fusion (RRF)**:
```
RRF(d) = Σᵣ∈R 1/(k + rank_r(d))
```

Where `R` is set of rankers, `rank_r(d)` is rank of document `d` in ranker `r`, `k=60` (constant).

**Score Normalization Fusion**:
```
hybrid_score(d) = α × normalize(dense_score(d)) + (1-α) × normalize(sparse_score(d))
```

---

## Implementation Details

Your system (`retrieval/retriever.py`) implements multiple retrieval strategies with a unified interface.

### Dense Retrieval

```python
class DenseRetriever:
    def __init__(self, vector_store: BaseVectorStore, embedder: EmbeddingEngine):
        self.vector_store = vector_store
        self.embedder = embedder
        
    def retrieve(self, query: Union[str, np.ndarray], top_k: int = 10,
                 filter: Optional[Dict] = None) -> List[SearchResult]:
        """
        Semantic similarity search using embeddings
        """
        # Convert query to vector if string
        if isinstance(query, str):
            query_vector = self.embedder.encode_query(query)
        else:
            query_vector = query
            
        # Search vector store
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filter=filter
        )
        
        return results
```

**Advantages**:
- Finds conceptually similar content even with different wording
- Handles synonyms, paraphrases, and multilingual queries
- Works well for broad, conceptual questions

**Limitations**:
- May miss exact keyword/entity matches
- Embedding quality determines ceiling performance
- Can retrieve semantically similar but factually irrelevant content

### Sparse Retrieval (BM25)

```python
class BM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1          # Term frequency saturation parameter
        self.b = b            # Length normalization parameter
        self.corpus = []      # Document texts
        self.corpus_ids = []  # Document IDs
        self.tokenizer = None # Text tokenizer
        self.bm25 = None      # BM25 index
        
    def build(self, texts: List[str], ids: List[str], metadatas: List[Dict]):
        """Build BM25 index from document corpus"""
        from rank_bm25 import BM25Okapi
        import nltk
        from nltk.corpus import stopwords
        
        # Download stopwords if needed
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            
        self.corpus = texts
        self.corpus_ids = ids
        self.metadatas = metadatas
        
        # Tokenize and remove stopwords
        tokenized_corpus = []
        for text in texts:
            tokens = self._tokenize(text.lower())
            # Remove stopwords and short tokens
            filtered_tokens = [token for token in tokens 
                             if token not in stop_words and len(token) > 2]
            tokenized_corpus.append(filtered_tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization with basic normalization"""
        import re
        
        # Replace punctuation with spaces, split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        return tokens
        
    def retrieve(self, query: str, top_k: int = 10, 
                 filter: Optional[Dict] = None) -> List[SearchResult]:
        """Keyword-based retrieval using BM25 scoring"""
        
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build() first.")
            
        # Tokenize query
        query_tokens = self._tokenize(query.lower())
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only include docs with non-zero scores
                result = SearchResult(
                    chunk_id=self.corpus_ids[idx],
                    content=self.corpus[idx],
                    score=scores[idx],
                    metadata=self.metadatas[idx],
                    rank=rank
                )
                
                # Apply metadata filter if provided
                if filter is None or self._matches_filter(result.metadata, filter):
                    results.append(result)
        
        return results
        
    def _matches_filter(self, metadata: Dict, filter_criteria: Dict) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
```

**BM25 Parameter Tuning**:
```python
def tune_bm25_parameters(queries: List[str], relevant_docs: List[List[int]], 
                        corpus: List[str]) -> Dict[str, float]:
    """Find optimal k1 and b parameters via grid search"""
    
    param_grid = {
        'k1': [0.5, 0.9, 1.2, 1.5, 2.0],
        'b': [0.0, 0.25, 0.5, 0.75, 1.0]
    }
    
    best_params = {'k1': 1.2, 'b': 0.75}
    best_score = 0
    
    for k1 in param_grid['k1']:
        for b in param_grid['b']:
            bm25 = BM25Retriever(k1=k1, b=b)
            bm25.build(corpus, list(range(len(corpus))), [{}]*len(corpus))
            
            # Evaluate on query set
            total_score = 0
            for query, relevant in zip(queries, relevant_docs):
                results = bm25.retrieve(query, top_k=10)
                retrieved_ids = [r.chunk_id for r in results]
                
                # Calculate precision@10
                precision = len(set(retrieved_ids) & set(relevant)) / len(retrieved_ids)
                total_score += precision
            
            avg_score = total_score / len(queries)
            if avg_score > best_score:
                best_score = avg_score
                best_params = {'k1': k1, 'b': b, 'score': avg_score}
    
    return best_params
```

### Hybrid Retrieval

```python
class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever,
                 alpha: float = 0.7, use_rrf: bool = True, rrf_k: int = 60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.alpha = alpha      # Weight for dense vs sparse (0.7 = 70% dense)
        self.use_rrf = use_rrf  # Use RRF vs score normalization
        self.rrf_k = rrf_k      # RRF constant
        
    def retrieve(self, query: str, top_k: int = 10, 
                 filter: Optional[Dict] = None) -> List[SearchResult]:
        """Hybrid retrieval combining dense and sparse methods"""
        
        # Retrieve from both methods (get more candidates for fusion)
        candidate_k = min(top_k * 3, 50)  # Get 3x candidates for better fusion
        
        dense_results = self.dense.retrieve(query, top_k=candidate_k, filter=filter)
        sparse_results = self.bm25.retrieve(query, top_k=candidate_k, filter=filter)
        
        if self.use_rrf:
            return self._fuse_with_rrf(dense_results, sparse_results, top_k)
        else:
            return self._fuse_with_score_normalization(dense_results, sparse_results, top_k)
    
    def _fuse_with_rrf(self, dense_results: List[SearchResult], 
                       sparse_results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Reciprocal Rank Fusion"""
        
        # Create rank mappings
        dense_ranks = {r.chunk_id: i for i, r in enumerate(dense_results)}
        sparse_ranks = {r.chunk_id: i for i, r in enumerate(sparse_results)}
        
        # Combine all unique chunk IDs
        all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        chunk_id_to_result = {}
        
        for chunk_id in all_chunk_ids:
            rrf_score = 0
            
            # Add dense contribution
            if chunk_id in dense_ranks:
                rrf_score += 1 / (self.rrf_k + dense_ranks[chunk_id] + 1)
                chunk_id_to_result[chunk_id] = next(r for r in dense_results if r.chunk_id == chunk_id)
            
            # Add sparse contribution  
            if chunk_id in sparse_ranks:
                rrf_score += 1 / (self.rrf_k + sparse_ranks[chunk_id] + 1)
                if chunk_id not in chunk_id_to_result:
                    chunk_id_to_result[chunk_id] = next(r for r in sparse_results if r.chunk_id == chunk_id)
            
            rrf_scores[chunk_id] = rrf_score
        
        # Sort by RRF score and return top-k
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        fused_results = []
        for rank, (chunk_id, rrf_score) in enumerate(sorted_chunks):
            result = chunk_id_to_result[chunk_id]
            result.score = rrf_score  # Replace with RRF score
            result.rank = rank
            fused_results.append(result)
        
        return fused_results
    
    def _fuse_with_score_normalization(self, dense_results: List[SearchResult],
                                      sparse_results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Score normalization fusion"""
        
        # Normalize scores to [0, 1] range
        dense_scores = [r.score for r in dense_results]
        sparse_scores = [r.score for r in sparse_results]
        
        dense_min, dense_max = min(dense_scores), max(dense_scores)
        sparse_min, sparse_max = min(sparse_scores), max(sparse_scores)
        
        # Avoid division by zero
        dense_range = max(dense_max - dense_min, 1e-8)
        sparse_range = max(sparse_max - sparse_min, 1e-8)
        
        # Create score mappings
        dense_norm_scores = {r.chunk_id: (r.score - dense_min) / dense_range 
                            for r in dense_results}
        sparse_norm_scores = {r.chunk_id: (r.score - sparse_min) / sparse_range 
                             for r in sparse_results}
        
        # Combine scores
        all_chunk_ids = set(dense_norm_scores.keys()) | set(sparse_norm_scores.keys())
        combined_scores = {}
        chunk_id_to_result = {}
        
        for chunk_id in all_chunk_ids:
            dense_score = dense_norm_scores.get(chunk_id, 0)
            sparse_score = sparse_norm_scores.get(chunk_id, 0)
            
            combined_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
            combined_scores[chunk_id] = combined_score
            
            # Store result object (prefer dense if available)
            if chunk_id in dense_norm_scores:
                chunk_id_to_result[chunk_id] = next(r for r in dense_results if r.chunk_id == chunk_id)
            else:
                chunk_id_to_result[chunk_id] = next(r for r in sparse_results if r.chunk_id == chunk_id)
        
        # Sort and return top-k
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        fused_results = []
        for rank, (chunk_id, combined_score) in enumerate(sorted_chunks):
            result = chunk_id_to_result[chunk_id]
            result.score = combined_score
            result.rank = rank
            fused_results.append(result)
        
        return fused_results
```

### Maximal Marginal Relevance (MMR)

MMR addresses the diversity problem — avoiding redundant results by balancing relevance and diversity.

```python
def maximal_marginal_relevance(query_embedding: np.ndarray, 
                              candidates: List[SearchResult],
                              candidate_embeddings: np.ndarray,
                              top_k: int, lambda_param: float = 0.7) -> List[SearchResult]:
    """
    Select diverse subset of candidates using MMR
    
    Args:
        query_embedding: Query vector
        candidates: Candidate search results
        candidate_embeddings: Embeddings for candidates  
        top_k: Number of results to return
        lambda_param: Relevance vs diversity trade-off (0.7 = 70% relevance, 30% diversity)
    """
    
    if len(candidates) <= top_k:
        return candidates
    
    # Compute query-candidate similarities (relevance)
    query_similarities = cosine_similarity(
        query_embedding.reshape(1, -1), 
        candidate_embeddings
    )[0]
    
    # Compute candidate-candidate similarities (for diversity)
    candidate_similarities = cosine_similarity(candidate_embeddings)
    
    # MMR selection algorithm
    selected_indices = []
    remaining_indices = list(range(len(candidates)))
    
    # Select first item (highest query similarity)
    first_idx = np.argmax(query_similarities)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively select remaining items
    while len(selected_indices) < top_k and remaining_indices:
        mmr_scores = []
        
        for i in remaining_indices:
            # Relevance component: similarity to query
            relevance = query_similarities[i]
            
            # Diversity component: max similarity to already selected items
            max_similarity_to_selected = max(
                candidate_similarities[i][j] for j in selected_indices
            ) if selected_indices else 0
            
            # MMR formula
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity_to_selected
            mmr_scores.append((i, mmr_score))
        
        # Select item with highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    # Return selected results with updated ranks
    selected_results = []
    for rank, idx in enumerate(selected_indices):
        result = candidates[idx]
        result.rank = rank
        selected_results.append(result)
    
    return selected_results
```

**MMR Usage Example**:
```python
# Get initial candidates
candidates = hybrid_retriever.retrieve(query, top_k=20)

# Get embeddings for MMR (need to re-embed candidates)
candidate_texts = [c.content for c in candidates]
candidate_embeddings = embedder.encode(candidate_texts).embeddings
query_embedding = embedder.encode_query(query)

# Apply MMR for diversity
diverse_results = maximal_marginal_relevance(
    query_embedding=query_embedding,
    candidates=candidates,
    candidate_embeddings=candidate_embeddings,
    top_k=10,
    lambda_param=0.7  # 70% relevance, 30% diversity
)
```

---

## Comparative Analysis

### Retrieval Method Performance

| Method | Precision | Recall | Latency | Best Use Case |
|---|---|---|---|---|
| **Dense only** | Medium | High | Fast | Conceptual queries, paraphrases |
| **Sparse only** | High | Medium | Very Fast | Keyword queries, entity lookup |
| **Hybrid (RRF)** | High | High | Medium | General purpose, production |
| **Hybrid + MMR** | High | High | Slow | Avoiding redundant results |

### Fusion Algorithm Comparison

**Reciprocal Rank Fusion**:
✅ Robust to score scale differences  
✅ No parameter tuning needed  
✅ Handles missing candidates gracefully  
❌ Ignores magnitude of score differences

**Score Normalization**:
✅ Preserves score magnitude information  
✅ Simple and interpretable  
❌ Sensitive to outlier scores  
❌ Requires careful normalization

**Selection criteria**:
```python
# Use RRF when:
use_rrf = (
    score_distributions_vary or      # Different score ranges/scales
    missing_candidates_common or     # Not all methods return same docs  
    robustness_preferred            # Want consistent performance
)

# Use score normalization when:
use_score_norm = (
    score_magnitudes_meaningful and  # Scores have semantic meaning
    similar_score_distributions and  # Methods return similar score ranges
    fine_grained_control_needed     # Want to tune alpha parameter
)
```

### Parameter Sensitivity Analysis

**Alpha (dense/sparse weight) impact**:
```python
def analyze_alpha_sensitivity(queries: List[str], ground_truth: List[List[int]]):
    """Test hybrid performance across different alpha values"""
    
    alphas = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
    results = []
    
    for alpha in alphas:
        hybrid = HybridRetriever(dense, bm25, alpha=alpha, use_rrf=False)
        
        total_precision = 0
        for query, relevant in zip(queries, ground_truth):
            retrieved = hybrid.retrieve(query, top_k=10)
            retrieved_ids = [r.chunk_id for r in retrieved]
            precision = len(set(retrieved_ids) & set(relevant)) / 10
            total_precision += precision
        
        avg_precision = total_precision / len(queries)
        results.append((alpha, avg_precision))
    
    # Find optimal alpha
    best_alpha, best_precision = max(results, key=lambda x: x[1])
    return best_alpha, results

# Typical results show optimal alpha around 0.6-0.8 for most domains
```

---

## Practical Guidelines

### Retrieval Pipeline Configuration

**Development (fast iteration)**:
```python
retrieval_config = RetrievalConfig(
    top_k=5,
    use_hybrid=False,      # Dense only for speed
    use_reranker=False,
    use_mmr=False
)
```

**Production (balanced)**:
```python
retrieval_config = RetrievalConfig(
    top_k=10,
    use_hybrid=True,
    alpha=0.7,             # 70% dense, 30% sparse
    use_rrf=True,          # Robust fusion
    use_reranker=True,     # Precision boost
    use_mmr=False          # Skip for speed
)
```

**High-recall scenarios**:
```python
retrieval_config = RetrievalConfig(
    top_k=20,              # More candidates
    use_hybrid=True,
    alpha=0.5,             # Balanced dense/sparse
    use_rrf=True,
    use_mmr=True,          # Diversity important
    mmr_lambda=0.6         # Favor diversity slightly
)
```

### Query Preprocessing

Different retrieval methods benefit from different query preprocessing:

```python
def preprocess_for_dense(query: str) -> str:
    """Optimize query for dense retrieval"""
    # Keep natural language, remove stop words cautiously
    query = query.strip().lower()
    
    # Expand abbreviations that might not be in embeddings
    expansions = {
        'refund': 'refund return money back',
        'API': 'API application programming interface',
        'ML': 'ML machine learning artificial intelligence'
    }
    
    for abbrev, expansion in expansions.items():
        if abbrev.lower() in query.lower():
            query = query.replace(abbrev.lower(), expansion)
    
    return query

def preprocess_for_sparse(query: str) -> str:
    """Optimize query for BM25 retrieval"""
    # Emphasize keywords, remove question words
    import re
    
    # Remove question words that don't add value
    stop_words = {'what', 'how', 'when', 'where', 'why', 'which', 'who', 'is', 'are', 'the', 'a', 'an'}
    tokens = [t for t in query.lower().split() if t not in stop_words]
    
    # Add important term variants
    enhanced_tokens = tokens.copy()
    for token in tokens:
        if token in ['return', 'refund']:
            enhanced_tokens.extend(['return', 'refund', 'exchange'])
        elif token in ['policy', 'rule']:
            enhanced_tokens.extend(['policy', 'rule', 'guideline', 'procedure'])
    
    return ' '.join(enhanced_tokens)
```

### Performance Optimization

**Async retrieval for hybrid**:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncHybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def retrieve_async(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Run dense and sparse retrieval in parallel"""
        
        loop = asyncio.get_event_loop()
        
        # Execute both retrievals concurrently
        dense_task = loop.run_in_executor(
            self.executor, self.dense.retrieve, query, top_k * 2
        )
        sparse_task = loop.run_in_executor(
            self.executor, self.bm25.retrieve, query, top_k * 2  
        )
        
        # Wait for both to complete
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        # Fuse results
        return self._fuse_with_rrf(dense_results, sparse_results, top_k)
```

**Caching retrieval results**:
```python
from utils.cache import CacheBackend, query_cache_key

class CachedRetriever:
    def __init__(self, base_retriever, cache: CacheBackend, ttl: int = 3600):
        self.retriever = base_retriever
        self.cache = cache
        self.ttl = ttl
    
    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[SearchResult]:
        """Cache retrieval results to avoid repeated computation"""
        
        # Create cache key including all parameters
        cache_key = f"retrieval:{query_cache_key(query)}:k={top_k}:{hash(str(kwargs))}"
        
        # Try cache first
        cached_results = self.cache.get(cache_key)
        if cached_results is not None:
            return cached_results
        
        # Compute and cache
        results = self.retriever.retrieve(query, top_k, **kwargs)
        self.cache.set(cache_key, results, ttl=self.ttl)
        
        return results
```

### Quality Assessment

**Retrieval evaluation metrics**:
```python
def evaluate_retrieval_quality(retriever, queries: List[str], 
                              ground_truth: List[List[str]], k: int = 10) -> Dict:
    """Comprehensive retrieval evaluation"""
    
    metrics = {
        'precision_at_k': [],
        'recall_at_k': [],
        'mrr': [],  # Mean Reciprocal Rank
        'ndcg_at_k': []  # Normalized Discounted Cumulative Gain
    }
    
    for query, relevant_chunks in zip(queries, ground_truth):
        results = retriever.retrieve(query, top_k=k)
        retrieved_chunks = [r.chunk_id for r in results]
        
        # Precision@K
        relevant_retrieved = set(retrieved_chunks) & set(relevant_chunks)
        precision = len(relevant_retrieved) / len(retrieved_chunks)
        metrics['precision_at_k'].append(precision)
        
        # Recall@K  
        recall = len(relevant_retrieved) / len(relevant_chunks) if relevant_chunks else 0
        metrics['recall_at_k'].append(recall)
        
        # MRR
        mrr = 0
        for i, chunk_id in enumerate(retrieved_chunks):
            if chunk_id in relevant_chunks:
                mrr = 1 / (i + 1)
                break
        metrics['mrr'].append(mrr)
        
        # NDCG@K (simplified binary relevance)
        dcg = sum(1 / np.log2(i + 2) for i, chunk_id in enumerate(retrieved_chunks) 
                 if chunk_id in relevant_chunks)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_chunks), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        metrics['ndcg_at_k'].append(ndcg)
    
    # Return average metrics
    return {metric: np.mean(values) for metric, values in metrics.items()}
```

### Common Issues & Solutions

**Issue**: Poor hybrid performance despite good individual retrievers
```python
# Problem: Score distributions don't align well
# Solution: Analyze and normalize score distributions

def analyze_score_distributions(dense_results: List[SearchResult], 
                               sparse_results: List[SearchResult]):
    """Analyze score distributions for fusion optimization"""
    
    dense_scores = [r.score for r in dense_results]
    sparse_scores = [r.score for r in sparse_results]
    
    analysis = {
        'dense': {
            'mean': np.mean(dense_scores),
            'std': np.std(dense_scores),
            'min': np.min(dense_scores),
            'max': np.max(dense_scores)
        },
        'sparse': {
            'mean': np.mean(sparse_scores),
            'std': np.std(sparse_scores), 
            'min': np.min(sparse_scores),
            'max': np.max(sparse_scores)
        }
    }
    
    # Check for problematic distributions
    if analysis['dense']['std'] > 2 * analysis['sparse']['std']:
        print("Warning: Dense scores have high variance, consider RRF fusion")
    
    if analysis['dense']['max'] > 10 * analysis['sparse']['max']:
        print("Warning: Score scales very different, normalize before fusion")
    
    return analysis
```

**Issue**: BM25 returns too many irrelevant results
```python
# Problem: Poor tokenization or inadequate stopword removal
# Solution: Improve text preprocessing

def improve_bm25_preprocessing(text: str) -> List[str]:
    """Enhanced preprocessing for better BM25 performance"""
    import re
    from nltk.stem import PorterStemmer
    
    stemmer = PorterStemmer()
    
    # 1. Normalize text
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', 'NUM', text)    # Replace numbers with token
    
    # 2. Tokenize
    tokens = text.split()
    
    # 3. Remove stopwords (expanded list)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 
                'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did'}
    
    # 4. Filter tokens
    filtered_tokens = []
    for token in tokens:
        if (len(token) > 2 and                    # Min length
            token not in stopwords and            # Not stopword
            not token.isdigit() and              # Not pure number
            re.match(r'^[a-zA-Z]', token)):      # Starts with letter
            
            # Apply stemming
            stemmed = stemmer.stem(token)
            filtered_tokens.append(stemmed)
    
    return filtered_tokens
```

**Issue**: Dense retrieval missing obvious keyword matches
```python
# Problem: Query-document vocabulary mismatch
# Solution: Query expansion with domain-specific terms

def expand_query_with_synonyms(query: str, domain_synonyms: Dict[str, List[str]]) -> str:
    """Expand query with domain-specific synonyms"""
    
    expanded_terms = query.split()
    
    for term in query.split():
        term_lower = term.lower()
        if term_lower in domain_synonyms:
            # Add synonyms with lower weight
            synonyms = domain_synonyms[term_lower]
            expanded_terms.extend(synonyms)
    
    # Join with original query
    expanded_query = ' '.join(expanded_terms)
    return expanded_query

# Domain-specific synonym dictionary
DOMAIN_SYNONYMS = {
    'refund': ['return', 'money-back', 'reimbursement', 'repayment'],
    'policy': ['rule', 'guideline', 'procedure', 'terms'],
    'customer': ['client', 'user', 'buyer', 'consumer'],
    'issue': ['problem', 'bug', 'error', 'fault', 'defect']
}
```

**Next concept**: Query Transformation — HyDE, multi-query, and step-back techniques for query enhancement
