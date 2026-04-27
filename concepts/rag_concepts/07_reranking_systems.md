# Reranking Systems — Cross-Encoder & LLM-Based Precision

## Conceptual Foundation

Reranking addresses the **precision vs recall trade-off** in retrieval systems. Initial retrieval casts a wide net (high recall) but may include many irrelevant results (low precision). Reranking applies more sophisticated — but computationally expensive — models to refine the ranking of candidates.

**Key insight**: Two-stage retrieval is more efficient than single-stage precision. First stage retrieves candidates quickly (bi-encoder), second stage ranks them precisely (cross-encoder or LLM).

### The Precision Bottleneck

```
Stage 1: Bi-encoder retrieval
Query: "refund policy for digital products"
Retrieved: 50 candidates in 10ms
Issues: Some results about "digital marketing" or "product reviews"

Stage 2: Cross-encoder reranking  
Input: Query + each candidate text
Output: Precise relevance score for each pair
Result: 10 highly relevant results in 100ms total
```

**Why not use cross-encoders directly?** O(n) complexity — would require n forward passes for n documents, too slow for large corpora.

---

## Mathematical Formulation

### Bi-encoder vs Cross-encoder

**Bi-encoder** (fast, used in first stage):
```
score(q, d) = sim(E_q(q), E_d(d))
```
Embeddings computed independently, similarity is dot product.

**Cross-encoder** (precise, used in reranking):
```
score(q, d) = CrossEncoder([q; d])
```
Query and document processed jointly, can model complex interactions.

### Learning to Rank

**Pairwise ranking loss**:
```
L = Σᵢ Σⱼ max(0, margin - score(q, d⁺ᵢ) + score(q, d⁻ⱼ))
```
Where d⁺ are relevant documents, d⁻ are irrelevant.

**Listwise ranking loss** (e.g., ListNet):
```
L = -Σᵢ P(πᵢ | q) log P̂(πᵢ | q)
```
Where π are ranking permutations.

### LLM-based Scoring

**Pointwise scoring**:
```
relevance = LLM("Rate relevance of document to query on 1-10 scale:\nQuery: {q}\nDocument: {d}\nScore:")
```

**Pairwise comparison**:
```
preference = LLM("Which document is more relevant to the query?\nQuery: {q}\nA: {d₁}\nB: {d₂}\nAnswer: A or B")
```

---

## Implementation Details

Your system (`retrieval/reranker.py`) implements multiple reranking approaches with a unified interface.

### CrossEncoder Reranker

```python
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = "cpu", batch_size: int = 32):
        """
        Cross-encoder model for precise relevance scoring
        
        Popular models:
        - ms-marco-MiniLM-L-6-v2: Fast, good quality (80MB)
        - ms-marco-MiniLM-L-12-v2: Better quality, slower (110MB)  
        - ms-marco-TinyBERT-L-2-v2: Fastest (17MB)
        - ms-marco-distilbert-base-v4: Balanced (250MB)
        """
        from sentence_transformers import CrossEncoder
        
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        
    def rerank(self, query: str, candidates: List[SearchResult], 
               top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Rerank candidates using cross-encoder model
        
        Args:
            query: Search query
            candidates: Initial retrieval results
            top_k: Number of results to return (None = all)
        """
        
        if not candidates:
            return []
        
        # Prepare query-document pairs for cross-encoder
        pairs = [(query, candidate.content) for candidate in candidates]
        
        # Compute cross-encoder scores in batches
        scores = self.model.predict(pairs, batch_size=self.batch_size, 
                                  show_progress_bar=len(pairs) > 100)
        
        # Update candidates with new scores
        reranked_candidates = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            # Create new result with cross-encoder score
            reranked_result = SearchResult(
                chunk_id=candidate.chunk_id,
                content=candidate.content,
                score=float(score),  # Cross-encoder score
                metadata=candidate.metadata,
                rank=i  # Will be updated after sorting
            )
            reranked_candidates.append(reranked_result)
        
        # Sort by cross-encoder score (descending)
        reranked_candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks and apply top_k limit
        final_results = []
        limit = top_k if top_k is not None else len(reranked_candidates)
        
        for rank, candidate in enumerate(reranked_candidates[:limit]):
            candidate.rank = rank
            final_results.append(candidate)
        
        return final_results
```

**Cross-encoder Model Selection**:
```python
def select_cross_encoder_model(latency_budget_ms: int, quality_target: str) -> str:
    """Select appropriate cross-encoder model based on requirements"""
    
    model_specs = {
        "ms-marco-TinyBERT-L-2-v2": {
            "size_mb": 17,
            "latency_per_pair_ms": 1.2,
            "ndcg_at_10": 0.325,
            "use_case": "high_throughput"
        },
        "ms-marco-MiniLM-L-6-v2": {
            "size_mb": 80, 
            "latency_per_pair_ms": 3.5,
            "ndcg_at_10": 0.342,
            "use_case": "balanced"
        },
        "ms-marco-MiniLM-L-12-v2": {
            "size_mb": 110,
            "latency_per_pair_ms": 6.8,
            "ndcg_at_10": 0.350,
            "use_case": "quality_focused"
        },
        "ms-marco-distilbert-base-v4": {
            "size_mb": 250,
            "latency_per_pair_ms": 12.0,
            "ndcg_at_10": 0.367,
            "use_case": "maximum_quality"
        }
    }
    
    # Filter by latency budget (assume 10 candidates to rerank)
    viable_models = {}
    for model, specs in model_specs.items():
        total_latency = specs["latency_per_pair_ms"] * 10  # 10 candidates
        if total_latency <= latency_budget_ms:
            viable_models[model] = specs
    
    if not viable_models:
        return "ms-marco-TinyBERT-L-2-v2"  # Fastest fallback
    
    # Select based on quality target
    if quality_target == "maximum":
        return max(viable_models.keys(), key=lambda x: viable_models[x]["ndcg_at_10"])
    elif quality_target == "minimum":
        return min(viable_models.keys(), key=lambda x: viable_models[x]["ndcg_at_10"])
    else:  # balanced
        return "ms-marco-MiniLM-L-6-v2" if "ms-marco-MiniLM-L-6-v2" in viable_models else list(viable_models.keys())[0]
```

### LLM Reranker

```python
class LLMReranker:
    def __init__(self, llm: LLMInterface, scoring_method: str = "pointwise"):
        """
        LLM-based reranking using language model as judge
        
        Args:
            llm: Language model interface
            scoring_method: "pointwise"|"pairwise"|"listwise"
        """
        self.llm = llm
        self.scoring_method = scoring_method
        
    def rerank(self, query: str, candidates: List[SearchResult],
               top_k: Optional[int] = None) -> List[SearchResult]:
        """Rerank using LLM as judge"""
        
        if not candidates:
            return []
        
        if self.scoring_method == "pointwise":
            return self._pointwise_rerank(query, candidates, top_k)
        elif self.scoring_method == "pairwise":
            return self._pairwise_rerank(query, candidates, top_k)
        elif self.scoring_method == "listwise":
            return self._listwise_rerank(query, candidates, top_k)
        
    def _pointwise_rerank(self, query: str, candidates: List[SearchResult],
                         top_k: Optional[int]) -> List[SearchResult]:
        """Score each document independently"""
        
        scored_candidates = []
        
        for candidate in candidates:
            # Create scoring prompt
            prompt = f"""Rate the relevance of this document to the given query on a scale from 1 to 10.

Query: {query}

Document: {candidate.content[:1000]}...

Consider:
- How well does the document answer the query?
- Is the information directly relevant?
- Does it provide useful details?

Return only a single number from 1 to 10.
Score:"""

            try:
                response = self.llm.complete(prompt)
                score_text = response.content.strip()
                
                # Extract numeric score
                score_match = re.search(r'\b([1-9]|10)\b', score_text)
                if score_match:
                    score = float(score_match.group(1))
                else:
                    score = 5.0  # Default fallback
                
                # Normalize to [0, 1]
                normalized_score = (score - 1) / 9
                
                scored_result = SearchResult(
                    chunk_id=candidate.chunk_id,
                    content=candidate.content,
                    score=normalized_score,
                    metadata=candidate.metadata,
                    rank=0  # Will be updated
                )
                scored_candidates.append(scored_result)
                
            except Exception as e:
                # Fallback to original score on error
                candidate.score = candidate.score * 0.5  # Penalize failed scoring
                scored_candidates.append(candidate)
        
        # Sort by LLM scores
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks and apply limit
        limit = top_k if top_k is not None else len(scored_candidates)
        for rank, candidate in enumerate(scored_candidates[:limit]):
            candidate.rank = rank
        
        return scored_candidates[:limit]
    
    def _pairwise_rerank(self, query: str, candidates: List[SearchResult],
                        top_k: Optional[int]) -> List[SearchResult]:
        """Compare documents pairwise to establish ranking"""
        
        n = len(candidates)
        if n <= 1:
            return candidates
        
        # Create pairwise comparison matrix
        win_matrix = np.zeros((n, n))
        
        # Compare each pair
        for i in range(n):
            for j in range(i + 1, n):
                winner = self._compare_pair(query, candidates[i], candidates[j])
                if winner == 'A':
                    win_matrix[i][j] = 1
                    win_matrix[j][i] = 0
                elif winner == 'B':
                    win_matrix[i][j] = 0
                    win_matrix[j][i] = 1
                else:  # Tie
                    win_matrix[i][j] = 0.5
                    win_matrix[j][i] = 0.5
        
        # Calculate win rates (score for each candidate)
        win_rates = np.sum(win_matrix, axis=1) / (n - 1)
        
        # Update scores and sort
        for i, candidate in enumerate(candidates):
            candidate.score = win_rates[i]
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks and apply limit
        limit = top_k if top_k is not None else len(candidates)
        for rank, candidate in enumerate(candidates[:limit]):
            candidate.rank = rank
        
        return candidates[:limit]
    
    def _compare_pair(self, query: str, doc_a: SearchResult, doc_b: SearchResult) -> str:
        """Compare two documents for relevance to query"""
        
        prompt = f"""Compare the relevance of these two documents to the given query.

Query: {query}

Document A: {doc_a.content[:500]}...

Document B: {doc_b.content[:500]}...

Which document is more relevant to answering the query? Consider accuracy, completeness, and directness of the answer.

Respond with exactly one letter: A, B, or T (for tie).
Answer:"""
        
        try:
            response = self.llm.complete(prompt)
            answer = response.content.strip().upper()
            
            if 'A' in answer and 'B' not in answer:
                return 'A'
            elif 'B' in answer and 'A' not in answer:
                return 'B'
            else:
                return 'T'  # Tie or unclear
                
        except Exception:
            return 'T'  # Default to tie on error
```

### Cohere Reranker

```python
class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        """
        Cohere's specialized reranking API
        
        Models:
        - rerank-english-v3.0: Latest English model
        - rerank-multilingual-v3.0: Multilingual support
        - rerank-english-v2.0: Previous generation
        """
        import cohere
        
        self.client = cohere.Client(api_key)
        self.model = model
        
    def rerank(self, query: str, candidates: List[SearchResult],
               top_k: Optional[int] = None, return_documents: bool = True) -> List[SearchResult]:
        """
        Rerank using Cohere's reranking API
        
        Args:
            query: Search query
            candidates: Documents to rerank
            top_k: Number of results to return
            return_documents: Whether to return full document content
        """
        
        if not candidates:
            return []
        
        # Prepare documents for Cohere API
        documents = [candidate.content for candidate in candidates]
        
        try:
            # Call Cohere rerank API
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_k=top_k or len(documents),
                return_documents=return_documents
            )
            
            # Process results
            reranked_results = []
            for rank, result in enumerate(response.results):
                original_candidate = candidates[result.index]
                
                reranked_result = SearchResult(
                    chunk_id=original_candidate.chunk_id,
                    content=original_candidate.content,
                    score=result.relevance_score,  # Cohere's relevance score
                    metadata=original_candidate.metadata,
                    rank=rank
                )
                reranked_results.append(reranked_result)
            
            return reranked_results
            
        except Exception as e:
            # Fallback to original ranking on API error
            logger.warning(f"Cohere reranking failed: {e}")
            limit = top_k if top_k is not None else len(candidates)
            return candidates[:limit]
```

---

## Comparative Analysis

### Reranking Method Comparison

| Method | Latency | Quality | Cost | Best Use Case |
|---|---|---|---|---|
| **CrossEncoder** | 50-200ms | High | Free (compute) | Production, balanced needs |
| **LLM Pointwise** | 1-5s | Very High | $0.001-0.01 per query | High-value queries |
| **LLM Pairwise** | 5-30s | Highest | $0.01-0.10 per query | Critical applications |
| **Cohere API** | 100-500ms | Very High | $0.002 per 1K docs | Managed solution |

### Quality vs Speed Trade-offs

```python
def select_reranker_by_requirements(latency_budget_ms: int, quality_level: str, 
                                   cost_budget_per_query: float) -> str:
    """Select optimal reranker based on constraints"""
    
    reranker_specs = {
        "cross_encoder_tiny": {"latency": 20, "quality": 0.78, "cost": 0.0},
        "cross_encoder_mini": {"latency": 80, "quality": 0.82, "cost": 0.0},
        "cross_encoder_base": {"latency": 200, "quality": 0.85, "cost": 0.0},
        "cohere_rerank": {"latency": 300, "quality": 0.88, "cost": 0.002},
        "llm_pointwise": {"latency": 2000, "quality": 0.90, "cost": 0.005},
        "llm_pairwise": {"latency": 15000, "quality": 0.95, "cost": 0.050}
    }
    
    # Filter by constraints
    viable_options = []
    for name, specs in reranker_specs.items():
        if (specs["latency"] <= latency_budget_ms and 
            specs["cost"] <= cost_budget_per_query):
            viable_options.append((name, specs))
    
    if not viable_options:
        return "cross_encoder_tiny"  # Fastest/cheapest fallback
    
    # Select by quality requirement
    quality_thresholds = {
        "minimum": 0.75,
        "good": 0.80, 
        "high": 0.85,
        "maximum": 0.90
    }
    
    threshold = quality_thresholds.get(quality_level, 0.80)
    qualified = [(name, specs) for name, specs in viable_options 
                if specs["quality"] >= threshold]
    
    if qualified:
        # Return fastest among qualified options
        return min(qualified, key=lambda x: x[1]["latency"])[0]
    else:
        # Return best quality within constraints
        return max(viable_options, key=lambda x: x[1]["quality"])[0]
```

---

## Practical Guidelines

### Reranking Pipeline Integration

```python
class AdaptiveReranker:
    def __init__(self, cross_encoder: CrossEncoderReranker, 
                 llm_reranker: LLMReranker, cohere_reranker: CohereReranker):
        self.cross_encoder = cross_encoder
        self.llm_reranker = llm_reranker  
        self.cohere_reranker = cohere_reranker
        
    def rerank(self, query: str, candidates: List[SearchResult], 
               quality_mode: str = "balanced") -> List[SearchResult]:
        """Adaptive reranking based on query characteristics and requirements"""
        
        # Analyze query complexity and importance
        query_analysis = self._analyze_query(query)
        
        if quality_mode == "fast":
            # Use cross-encoder for speed
            return self.cross_encoder.rerank(query, candidates, top_k=10)
            
        elif quality_mode == "balanced":
            if query_analysis["complexity"] > 0.7:
                # Complex query → use LLM reranker
                return self.llm_reranker.rerank(query, candidates, top_k=10)
            else:
                # Simple query → cross-encoder sufficient
                return self.cross_encoder.rerank(query, candidates, top_k=10)
                
        elif quality_mode == "premium":
            # Two-stage reranking for maximum quality
            
            # Stage 1: Cross-encoder to reduce candidate set
            stage1_results = self.cross_encoder.rerank(query, candidates, top_k=20)
            
            # Stage 2: LLM reranker for final precision
            return self.llm_reranker.rerank(query, stage1_results, top_k=10)
    
    def _analyze_query(self, query: str) -> Dict[str, float]:
        """Analyze query characteristics to guide reranker selection"""
        
        analysis = {"complexity": 0.0, "ambiguity": 0.0}
        
        tokens = query.lower().split()
        
        # Complexity indicators
        complex_patterns = ["how to", "what are the steps", "explain", "compare", "analyze"]
        analysis["complexity"] = sum(1 for pattern in complex_patterns 
                                   if pattern in query.lower()) / len(complex_patterns)
        
        # Length-based complexity
        if len(tokens) > 10:
            analysis["complexity"] += 0.3
        
        # Ambiguity indicators  
        ambiguous_terms = ["best", "good", "better", "issue", "problem", "help"]
        analysis["ambiguity"] = sum(1 for term in ambiguous_terms 
                                  if term in tokens) / len(ambiguous_terms)
        
        return analysis
```

### Performance Optimization

**Batch processing for cross-encoders**:
```python
class BatchedCrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 32, max_candidates: int = 50):
        self.reranker = CrossEncoderReranker(model_name, batch_size=batch_size)
        self.max_candidates = max_candidates
        
    def rerank_multiple_queries(self, query_candidate_pairs: List[Tuple[str, List[SearchResult]]]) -> List[List[SearchResult]]:
        """Efficiently rerank multiple queries in batch"""
        
        # Collect all query-document pairs
        all_pairs = []
        pair_to_query_idx = []
        query_results = [[] for _ in range(len(query_candidate_pairs))]
        
        for query_idx, (query, candidates) in enumerate(query_candidate_pairs):
            # Limit candidates to avoid memory issues
            limited_candidates = candidates[:self.max_candidates]
            
            for candidate in limited_candidates:
                all_pairs.append((query, candidate.content))
                pair_to_query_idx.append((query_idx, candidate))
        
        # Batch score all pairs
        scores = self.reranker.model.predict(all_pairs, 
                                           batch_size=self.reranker.batch_size,
                                           show_progress_bar=True)
        
        # Distribute scores back to respective queries
        for (query_idx, candidate), score in zip(pair_to_query_idx, scores):
            candidate.score = float(score)
            query_results[query_idx].append(candidate)
        
        # Sort each query's results
        for results in query_results:
            results.sort(key=lambda x: x.score, reverse=True)
            for rank, result in enumerate(results):
                result.rank = rank
        
        return query_results
```

**Caching reranking results**:
```python
class CachedReranker:
    def __init__(self, base_reranker, cache: CacheBackend):
        self.reranker = base_reranker
        self.cache = cache
        self.ttl = 3600  # 1 hour cache
        
    def rerank(self, query: str, candidates: List[SearchResult], **kwargs) -> List[SearchResult]:
        """Cache reranking results by query + candidate set hash"""
        
        # Create cache key from query and candidate IDs (order matters)
        candidate_ids = [c.chunk_id for c in candidates]
        cache_key = f"rerank:{hash(query)}:{hash(tuple(candidate_ids))}:{hash(str(kwargs))}"
        
        # Check cache
        cached_results = self.cache.get(cache_key)
        if cached_results is not None:
            return cached_results
        
        # Compute and cache
        results = self.reranker.rerank(query, candidates, **kwargs)
        self.cache.set(cache_key, results, ttl=self.ttl)
        
        return results
```

### Quality Assessment

**Reranking effectiveness metrics**:
```python
def evaluate_reranking_effectiveness(original_results: List[SearchResult],
                                   reranked_results: List[SearchResult],
                                   ground_truth_relevant: Set[str]) -> Dict[str, float]:
    """Measure improvement from reranking"""
    
    def compute_metrics(results: List[SearchResult], relevant: Set[str]) -> Dict:
        retrieved_ids = [r.chunk_id for r in results[:10]]  # Top-10
        
        relevant_retrieved = set(retrieved_ids) & relevant
        
        precision_at_10 = len(relevant_retrieved) / 10
        recall_at_10 = len(relevant_retrieved) / len(relevant) if relevant else 0
        
        # Mean Reciprocal Rank
        mrr = 0
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in relevant:
                mrr = 1 / (i + 1)
                break
        
        return {"precision": precision_at_10, "recall": recall_at_10, "mrr": mrr}
    
    original_metrics = compute_metrics(original_results, ground_truth_relevant)
    reranked_metrics = compute_metrics(reranked_results, ground_truth_relevant)
    
    improvements = {}
    for metric in ["precision", "recall", "mrr"]:
        original_val = original_metrics[metric]
        reranked_val = reranked_metrics[metric]
        
        if original_val > 0:
            improvement = (reranked_val - original_val) / original_val
        else:
            improvement = reranked_val  # Improvement from zero
        
        improvements[f"{metric}_improvement"] = improvement
    
    improvements["original_metrics"] = original_metrics
    improvements["reranked_metrics"] = reranked_metrics
    
    return improvements
```

### Common Issues & Solutions

**Issue**: Cross-encoder slower than expected
```python
# Problem: Large batch size causing GPU memory issues
# Solution: Adaptive batch sizing

def optimize_cross_encoder_batch_size(model: CrossEncoder, 
                                     sample_pairs: List[Tuple[str, str]]) -> int:
    """Find optimal batch size for available GPU memory"""
    
    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    optimal_batch_size = 1
    
    for batch_size in batch_sizes:
        try:
            # Test batch processing
            test_pairs = sample_pairs[:batch_size]
            start_time = time.time()
            _ = model.predict(test_pairs, batch_size=batch_size)
            elapsed = time.time() - start_time
            
            # Check if processing was successful and reasonably fast
            if elapsed < 10.0:  # Less than 10 seconds per batch
                optimal_batch_size = batch_size
            else:
                break  # Too slow, use previous batch size
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break  # GPU memory limit reached
            else:
                raise  # Other error
    
    return optimal_batch_size
```

**Issue**: LLM reranker producing inconsistent scores
```python
# Problem: Temperature too high, non-deterministic scoring
# Solution: Use low temperature and structured prompts

def create_consistent_scoring_prompt(query: str, document: str) -> str:
    """Create prompt that encourages consistent LLM scoring"""
    
    prompt = f"""You are an expert relevance assessor. Rate how well this document answers the given query.

Query: {query}

Document: {document}

Scoring criteria:
- 9-10: Document directly and completely answers the query
- 7-8: Document mostly answers the query with good detail
- 5-6: Document partially answers or is somewhat relevant
- 3-4: Document has minimal relevance to the query
- 1-2: Document is not relevant to the query

Consider only factual relevance, not writing quality.

Think step by step:
1. What is the query asking for?
2. What information does the document provide?
3. How well do they match?

Score (1-10):"""

    return prompt

# Use with temperature=0.1 for consistency
llm_config = LLMConfig(temperature=0.1, max_tokens=50)
```

**Issue**: Reranking doesn't improve precision
```python
# Problem: Initial retrieval quality too low to benefit from reranking
# Solution: Improve first-stage retrieval before reranking

def diagnose_reranking_issues(query: str, original_results: List[SearchResult],
                             reranked_results: List[SearchResult]) -> Dict[str, str]:
    """Diagnose why reranking isn't helping"""
    
    issues = {}
    
    # Check if original results contain any relevant documents
    # (Simplified: assume documents with score > 0.7 are relevant)
    high_scoring_original = [r for r in original_results if r.score > 0.7]
    
    if len(high_scoring_original) == 0:
        issues["poor_recall"] = "No relevant documents in original results. Improve first-stage retrieval (chunking, embeddings, query processing)."
    
    # Check score distribution
    original_scores = [r.score for r in original_results]
    score_range = max(original_scores) - min(original_scores)
    
    if score_range < 0.1:
        issues["poor_discrimination"] = "Original scores too similar. Reranker cannot distinguish between candidates. Check embedding model quality."
    
    # Check for position changes after reranking
    original_order = [r.chunk_id for r in original_results[:10]]
    reranked_order = [r.chunk_id for r in reranked_results[:10]]
    
    changes = sum(1 for i, (orig, rerank) in enumerate(zip(original_order, reranked_order)) 
                 if orig != rerank)
    
    if changes < 3:
        issues["minimal_reordering"] = "Reranker made few changes. May indicate reranker and first-stage retriever are too similar, or reranker confidence is low."
    
    return issues
```

**Next concept**: Context Construction — prompt building, citation systems, and token management strategies
