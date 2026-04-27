# Query Transformation — HyDE, Multi-Query & Step-Back

## Conceptual Foundation

Query transformation addresses a fundamental challenge in RAG: **query-document mismatch**. User queries are often brief, ambiguous, or use different vocabulary than the documents they seek. Query transformation techniques generate alternative formulations that increase the likelihood of retrieving relevant information.

**Key insight**: Instead of relying on a single query formulation, generate multiple perspectives on the same information need. This creates a **retrieval safety net** — if one formulation fails, others may succeed.

### The Query Gap Problem

```
User query:     "How to get money back?"
Document text:  "Refund procedures require submitting Form RMA-105..."

Problem: No lexical overlap, embeddings may not bridge the gap
Solution: Transform query → "What are the refund procedures?" 
                          → "How do I submit a return request?"
                          → "What forms are needed for returns?"
```

**Transformation strategies**:
- **HyDE**: Generate hypothetical document, use for retrieval  
- **Multi-Query**: Generate query variations, retrieve separately
- **Step-Back**: Abstract to broader concepts, then retrieve
- **Chain-of-Verification**: Generate, verify, refine query

---

## Mathematical Formulation

### HyDE (Hypothetical Document Embeddings)

Traditional retrieval: 
```
sim(query, doc) = cos(E(query), E(doc))
```

HyDE approach:
```
hypothetical_doc = LLM(query)
sim(query, doc) = cos(E(hypothetical_doc), E(doc))
```

**Intuition**: LLM-generated hypothetical documents are more similar to real documents than queries are.

### Multi-Query Expansion

Generate query set: `Q = {q₁, q₂, ..., qₙ}`
Retrieve for each: `Rᵢ = retrieve(qᵢ, k)`
Aggregate results: `R_final = aggregate(R₁, R₂, ..., Rₙ)`

**Aggregation strategies**:
- **Union**: R_final = R₁ ∪ R₂ ∪ ... ∪ Rₙ
- **Intersection**: R_final = R₁ ∩ R₂ ∩ ... ∩ Rₙ  
- **Ranked Fusion**: RRF across all result sets

### Step-Back Prompting

```
original_query → abstract_query → retrieve(abstract) → filter_by_original
```

**Abstraction function**:
```
abstract(q) = "What are the general principles behind: " + q
```

---

## Implementation Details

Your system (`retrieval/retriever.py`) includes a `QueryTransformer` class with multiple transformation techniques.

### HyDE Implementation

```python
class QueryTransformer:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    def hyde(self, query: str, document_type: str = "general") -> str:
        """
        Generate hypothetical document that would answer the query
        
        Args:
            query: User's question
            document_type: "general"|"technical"|"policy"|"faq"
        """
        
        # Template based on document type
        templates = {
            "general": """Please write a passage to answer the question: "{query}"
            
            Write as if you are a knowledgeable author writing a section of a document that would comprehensively answer this question. Include specific details and examples where appropriate.
            
            Passage:""",
            
            "technical": """Please write a technical documentation section that answers: "{query}"
            
            Write in the style of professional technical documentation with clear explanations, steps, and requirements. Include technical terms and specific details.
            
            Documentation:""",
            
            "policy": """Please write a policy document section that addresses: "{query}"
            
            Write in formal policy language with clear rules, procedures, and requirements. Include specific conditions and exceptions.
            
            Policy Section:""",
            
            "faq": """Please write an FAQ entry that answers: "{query}"
            
            Format as a clear, direct answer that would appear in a frequently asked questions section.
            
            Answer:"""
        }
        
        prompt = templates.get(document_type, templates["general"]).format(query=query)
        
        response = self.llm.complete(prompt)
        hypothetical_doc = response.content.strip()
        
        # Clean up the response (remove any prefixes the LLM might add)
        lines = hypothetical_doc.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Answer:', 'Response:', 'Passage:')):
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)

# Usage example
transformer = QueryTransformer(llm_interface)
hypothetical_doc = transformer.hyde("What is the refund policy?", "policy")

# Use hypothetical document for retrieval instead of original query
results = dense_retriever.retrieve(hypothetical_doc, top_k=10)
```

**HyDE Prompt Engineering**:
```python
def create_domain_specific_hyde_prompt(query: str, domain: str) -> str:
    """Create specialized HyDE prompts for different domains"""
    
    domain_instructions = {
        "legal": {
            "style": "formal legal language with citations and precedents",
            "structure": "definitions, rules, exceptions, enforcement",
            "examples": "case references, statutory citations"
        },
        "technical": {
            "style": "clear technical writing with step-by-step procedures", 
            "structure": "overview, requirements, implementation, troubleshooting",
            "examples": "code snippets, configuration examples, error scenarios"
        },
        "customer_service": {
            "style": "friendly, helpful customer service language",
            "structure": "direct answer, steps to take, contact information",
            "examples": "specific timelines, contact methods, escalation paths"
        }
    }
    
    if domain not in domain_instructions:
        domain = "general"
        
    instructions = domain_instructions.get(domain, {
        "style": "clear, informative writing",
        "structure": "direct answer with supporting details",
        "examples": "relevant examples and context"
    })
    
    prompt = f"""Write a document section that would answer: "{query}"

Style: {instructions['style']}
Structure: {instructions['structure']}  
Include: {instructions['examples']}

Document section:"""
    
    return prompt
```

### Multi-Query Generation

```python
def multi_query(self, query: str, num_queries: int = 3, 
                diversity_mode: str = "paraphrase") -> List[str]:
    """
    Generate multiple query variations
    
    Args:
        query: Original query
        num_queries: Number of variations to generate
        diversity_mode: "paraphrase"|"perspective"|"granularity"
    """
    
    if diversity_mode == "paraphrase":
        prompt = f"""Generate {num_queries} different ways to ask the same question: "{query}"

        Make the questions semantically equivalent but use different words and phrasings. Focus on:
        - Different vocabulary choices
        - Alternative sentence structures  
        - Different levels of formality
        
        Questions:
        1."""
        
    elif diversity_mode == "perspective":
        prompt = f"""Generate {num_queries} different perspectives on this question: "{query}"

        Create questions that approach the same information need from different angles:
        - Different user roles or contexts
        - Different aspects of the same topic
        - Different levels of detail
        
        Questions:
        1."""
        
    elif diversity_mode == "granularity":
        prompt = f"""Generate {num_queries} questions at different levels of detail for: "{query}"

        Create:
        - 1 broad, general version
        - 1 specific, detailed version  
        - 1 intermediate version
        
        Questions:
        1."""
    
    response = self.llm.complete(prompt)
    
    # Parse generated queries
    lines = response.content.strip().split('\n')
    queries = []
    
    for line in lines:
        line = line.strip()
        # Remove numbering and extract question
        if re.match(r'^\d+\.', line):
            question = re.sub(r'^\d+\.\s*', '', line)
            if question and question != query:  # Avoid duplicates
                queries.append(question)
    
    # Ensure we have the requested number (fill with original if needed)
    while len(queries) < num_queries:
        queries.append(query)
    
    return queries[:num_queries]

# Usage example
queries = transformer.multi_query(
    "How do I return a product?", 
    num_queries=3, 
    diversity_mode="perspective"
)
# Returns: [
#   "How do I return a product?",
#   "What is the process for sending back an item?", 
#   "What are the steps to get a refund for a purchase?"
# ]
```

### Step-Back Prompting

```python
def step_back(self, query: str, abstraction_level: str = "principle") -> str:
    """
    Generate broader, more abstract version of the query
    
    Args:
        query: Specific query
        abstraction_level: "principle"|"category"|"domain"
    """
    
    if abstraction_level == "principle":
        prompt = f"""What general principle or concept is behind this specific question: "{query}"?

        Think about what broader topic or principle this question relates to. Rephrase as a question about the general concept rather than the specific case.
        
        For example:
        - "How do I reset my password?" → "How do authentication and access recovery systems work?"
        - "What's the refund policy for digital products?" → "How do return policies work for different product types?"
        
        General principle question:"""
        
    elif abstraction_level == "category":
        prompt = f"""What category or type of question is this: "{query}"?

        Identify the general category and rephrase as a question about that category.
        
        Category question:"""
        
    elif abstraction_level == "domain":
        prompt = f"""What domain or field does this question belong to: "{query}"?

        Identify the broader domain and rephrase as a question about that domain.
        
        Domain question:"""
    
    response = self.llm.complete(prompt)
    abstract_query = response.content.strip()
    
    # Clean up response
    abstract_query = re.sub(r'^(General principle question:|Category question:|Domain question:)\s*', '', abstract_query)
    
    return abstract_query

# Usage in retrieval pipeline
def step_back_retrieval(self, query: str, top_k: int = 10) -> List[SearchResult]:
    """Retrieve using step-back prompting strategy"""
    
    # Step 1: Generate abstract query
    abstract_query = self.transformer.step_back(query, "principle")
    
    # Step 2: Retrieve using abstract query (get more candidates)
    abstract_results = self.dense_retriever.retrieve(abstract_query, top_k=top_k*2)
    
    # Step 3: Filter/rerank by relevance to original query
    original_embedding = self.embedder.encode_query(query)
    result_embeddings = self.embedder.encode([r.content for r in abstract_results])
    
    # Compute relevance to original query
    similarities = cosine_similarity([original_embedding], result_embeddings.embeddings)[0]
    
    # Re-score results (combine abstract retrieval rank + original query relevance)
    for i, result in enumerate(abstract_results):
        abstract_score = (len(abstract_results) - i) / len(abstract_results)  # Rank-based score
        relevance_score = similarities[i]
        result.score = 0.6 * abstract_score + 0.4 * relevance_score
    
    # Sort by combined score and return top-k
    abstract_results.sort(key=lambda x: x.score, reverse=True)
    return abstract_results[:top_k]
```

### Chain-of-Verification (CoVe)

```python
def chain_of_verification(self, query: str) -> str:
    """
    Generate and verify query formulation
    
    Process:
    1. Generate initial query reformulation
    2. Identify potential issues with the reformulation
    3. Generate improved version
    """
    
    # Step 1: Initial reformulation
    initial_prompt = f"""Reformulate this query to be more precise and specific: "{query}"

    Consider:
    - What specific information is the user really seeking?
    - What context might be missing?
    - How can this be made more searchable?
    
    Reformulated query:"""
    
    initial_response = self.llm.complete(initial_prompt)
    reformulated = initial_response.content.strip()
    
    # Step 2: Verification and critique
    verify_prompt = f"""Original query: "{query}"
    Reformulated query: "{reformulated}"
    
    Analyze the reformulated query:
    1. Does it capture the original intent?
    2. Is it more specific and searchable?
    3. What potential issues or ambiguities remain?
    4. What improvements could be made?
    
    Analysis:"""
    
    verification = self.llm.complete(verify_prompt)
    
    # Step 3: Final improvement
    improve_prompt = f"""Based on this analysis: {verification.content}
    
    Create a final, improved version of the query that addresses any identified issues.
    
    Original: "{query}"
    Previous reformulation: "{reformulated}"
    
    Final improved query:"""
    
    final_response = self.llm.complete(improve_prompt)
    final_query = final_response.content.strip()
    
    return final_query
```

---

## Comparative Analysis

### Transformation Method Performance

| Method | Best For | Retrieval Improvement | Computational Cost | Failure Modes |
|---|---|---|---|---|
| **HyDE** | Conceptual queries | 15-30% | Medium (1 LLM call) | Hallucinated details |
| **Multi-Query** | Ambiguous queries | 10-25% | High (3-5 LLM calls) | Query drift |
| **Step-Back** | Specific queries | 5-20% | Medium (1 LLM call) | Over-generalization |
| **CoVe** | Complex queries | 20-35% | High (3 LLM calls) | Recursive refinement |

### Use Case Selection

**HyDE works best for**:
```python
hyde_suitable_queries = [
    "What causes server errors?",          # Conceptual, needs explanation
    "How to optimize database performance?", # Technical, needs detailed answer
    "What are security best practices?"     # Broad knowledge question
]
```

**Multi-Query works best for**:
```python
multi_query_suitable = [
    "API issues",                          # Ambiguous, multiple interpretations
    "Payment problems",                    # Could mean many different things  
    "Login not working"                    # Various possible causes
]
```

**Step-Back works best for**:
```python
step_back_suitable = [
    "Why is rate limit 429 error happening?",     # Specific → general principles
    "How to fix SSL certificate expired error?",  # Specific → certificate management
    "What does 'insufficient permissions' mean?"  # Specific → access control concepts
]
```

---

## Practical Guidelines

### Query Analysis & Selection

```python
def analyze_query_characteristics(query: str) -> Dict[str, float]:
    """Analyze query to recommend appropriate transformation"""
    
    characteristics = {
        'specificity': 0.0,    # How specific vs general
        'ambiguity': 0.0,      # Multiple possible interpretations
        'completeness': 0.0,   # Has sufficient context
        'technicality': 0.0    # Technical vs natural language
    }
    
    tokens = query.lower().split()
    
    # Specificity indicators
    specific_markers = ['error', 'code', 'version', 'exactly', 'precisely']
    characteristics['specificity'] = sum(1 for marker in specific_markers if marker in tokens) / len(specific_markers)
    
    # Ambiguity indicators (short queries, vague terms)
    if len(tokens) <= 3:
        characteristics['ambiguity'] += 0.5
    
    vague_terms = ['issue', 'problem', 'not working', 'broken', 'help']
    characteristics['ambiguity'] += sum(1 for term in vague_terms if term in query.lower()) / len(vague_terms)
    
    # Completeness (has context words)
    context_words = ['when', 'where', 'how', 'why', 'what', 'which']
    characteristics['completeness'] = sum(1 for word in context_words if word in tokens) / len(context_words)
    
    # Technicality
    technical_patterns = ['API', 'HTTP', 'SSL', 'JSON', 'SQL', 'error code', 'exception']
    characteristics['technicality'] = sum(1 for pattern in technical_patterns if pattern in query) / len(technical_patterns)
    
    return characteristics

def recommend_transformation_strategy(query: str) -> List[str]:
    """Recommend best transformation strategies for query"""
    
    chars = analyze_query_characteristics(query)
    recommendations = []
    
    # HyDE for conceptual queries
    if chars['specificity'] < 0.3 and chars['completeness'] > 0.2:
        recommendations.append('hyde')
    
    # Multi-Query for ambiguous queries
    if chars['ambiguity'] > 0.4:
        recommendations.append('multi_query')
    
    # Step-Back for overly specific queries
    if chars['specificity'] > 0.6 and chars['technicality'] > 0.3:
        recommendations.append('step_back')
    
    # CoVe for complex, incomplete queries
    if chars['completeness'] < 0.3 and len(query.split()) > 5:
        recommendations.append('chain_of_verification')
    
    # Default to multi-query if no clear recommendation
    if not recommendations:
        recommendations.append('multi_query')
    
    return recommendations
```

### Aggregation Strategies

When using multiple query transformations, results need intelligent aggregation:

```python
def aggregate_multi_transformation_results(result_sets: List[List[SearchResult]], 
                                         strategy: str = "weighted_rrf") -> List[SearchResult]:
    """Aggregate results from multiple query transformations"""
    
    if strategy == "weighted_rrf":
        # Give different weights to different transformation methods
        weights = {
            'original': 1.0,
            'hyde': 0.8,
            'multi_query': 0.6,
            'step_back': 0.4
        }
        
        chunk_scores = defaultdict(float)
        chunk_to_result = {}
        
        for method, results in result_sets.items():
            weight = weights.get(method, 0.5)
            
            for rank, result in enumerate(results):
                rrf_score = weight / (60 + rank + 1)  # Weighted RRF
                chunk_scores[result.chunk_id] += rrf_score
                
                if result.chunk_id not in chunk_to_result:
                    chunk_to_result[result.chunk_id] = result
        
        # Sort by aggregated score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        aggregated_results = []
        for rank, (chunk_id, score) in enumerate(sorted_chunks):
            result = chunk_to_result[chunk_id]
            result.score = score
            result.rank = rank
            aggregated_results.append(result)
        
        return aggregated_results
    
    elif strategy == "intersection":
        # Only return documents that appear in multiple result sets
        chunk_counts = defaultdict(int)
        chunk_to_result = {}
        
        for results in result_sets.values():
            seen_chunks = set()
            for result in results:
                if result.chunk_id not in seen_chunks:  # Avoid double-counting from same method
                    chunk_counts[result.chunk_id] += 1
                    chunk_to_result[result.chunk_id] = result
                    seen_chunks.add(result.chunk_id)
        
        # Require appearance in at least 2 methods
        min_appearances = max(2, len(result_sets) // 2)
        filtered_chunks = {chunk_id: count for chunk_id, count in chunk_counts.items() 
                          if count >= min_appearances}
        
        # Sort by appearance count, then by best score
        intersected_results = []
        for chunk_id in sorted(filtered_chunks.keys(), key=lambda x: filtered_chunks[x], reverse=True):
            result = chunk_to_result[chunk_id]
            result.score = filtered_chunks[chunk_id]  # Use appearance count as score
            intersected_results.append(result)
        
        return intersected_results
```

### Performance Optimization

**Parallel transformation execution**:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncQueryTransformer:
    def __init__(self, llm: LLMInterface, max_workers: int = 3):
        self.transformer = QueryTransformer(llm)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def parallel_transform(self, query: str, methods: List[str]) -> Dict[str, str]:
        """Execute multiple transformation methods in parallel"""
        
        tasks = {}
        loop = asyncio.get_event_loop()
        
        for method in methods:
            if method == 'hyde':
                task = loop.run_in_executor(self.executor, self.transformer.hyde, query)
            elif method == 'step_back':
                task = loop.run_in_executor(self.executor, self.transformer.step_back, query)
            elif method == 'multi_query':
                task = loop.run_in_executor(self.executor, self.transformer.multi_query, query, 3)
            
            tasks[method] = task
        
        # Wait for all transformations to complete
        results = await asyncio.gather(*tasks.values())
        
        return dict(zip(methods, results))
```

**Caching transformations**:
```python
class CachedQueryTransformer:
    def __init__(self, base_transformer: QueryTransformer, cache: CacheBackend):
        self.transformer = base_transformer
        self.cache = cache
        self.ttl = 86400  # 24 hours
    
    def hyde(self, query: str, document_type: str = "general") -> str:
        cache_key = f"hyde:{hash(query)}:{document_type}"
        
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        result = self.transformer.hyde(query, document_type)
        self.cache.set(cache_key, result, ttl=self.ttl)
        
        return result
    
    # Similar caching for other methods...
```

### Quality Assessment

**Transformation quality metrics**:
```python
def evaluate_transformation_quality(original_query: str, transformed_query: str,
                                  retrieval_results: List[SearchResult]) -> Dict[str, float]:
    """Assess quality of query transformation"""
    
    # 1. Semantic similarity (should be related but not identical)
    original_embedding = embedder.encode_query(original_query)
    transformed_embedding = embedder.encode_query(transformed_query)
    semantic_similarity = cosine_similarity([original_embedding], [transformed_embedding])[0][0]
    
    # 2. Lexical diversity (should use different words)
    original_tokens = set(original_query.lower().split())
    transformed_tokens = set(transformed_query.lower().split())
    lexical_overlap = len(original_tokens & transformed_tokens) / len(original_tokens | transformed_tokens)
    
    # 3. Retrieval improvement (compare to baseline)
    baseline_results = retriever.retrieve(original_query, top_k=10)
    baseline_relevance = assess_relevance(baseline_results, ground_truth)
    transformed_relevance = assess_relevance(retrieval_results, ground_truth)
    
    improvement = (transformed_relevance - baseline_relevance) / baseline_relevance if baseline_relevance > 0 else 0
    
    return {
        'semantic_similarity': semantic_similarity,
        'lexical_diversity': 1 - lexical_overlap,
        'retrieval_improvement': improvement,
        'overall_quality': (semantic_similarity * 0.3 + 
                           (1 - lexical_overlap) * 0.3 + 
                           max(0, improvement) * 0.4)
    }
```

### Common Issues & Solutions

**Issue**: HyDE generates hallucinated facts
```python
# Problem: LLM invents specific details not in knowledge base
# Solution: Use more general, principle-based HyDE prompts

def safe_hyde_prompt(query: str) -> str:
    """Generate HyDE prompt that avoids hallucination"""
    
    prompt = f"""Write a general explanation that would help someone understand: "{query}"

    Focus on:
    - General principles and concepts
    - Common patterns and approaches  
    - Typical considerations and factors
    
    Avoid:
    - Specific numbers, dates, or names
    - Definitive claims about specific products/services
    - Detailed procedures without verification
    
    Explanation:"""
    
    return prompt
```

**Issue**: Multi-query generates too similar variations
```python
# Problem: Generated queries are near-duplicates
# Solution: Enforce diversity constraints

def diverse_multi_query(self, query: str, num_queries: int = 3) -> List[str]:
    """Generate diverse query variations with similarity constraints"""
    
    generated_queries = [query]  # Start with original
    embeddings = [self.embedder.encode_query(query)]
    
    max_attempts = 10
    similarity_threshold = 0.8  # Reject if >80% similar to existing
    
    while len(generated_queries) < num_queries and max_attempts > 0:
        # Generate candidate
        candidate = self._generate_single_variation(query, generated_queries)
        candidate_embedding = self.embedder.encode_query(candidate)
        
        # Check similarity to existing queries
        similarities = cosine_similarity([candidate_embedding], embeddings)[0]
        max_similarity = max(similarities)
        
        if max_similarity < similarity_threshold:
            generated_queries.append(candidate)
            embeddings.append(candidate_embedding)
        
        max_attempts -= 1
    
    return generated_queries
```

**Issue**: Step-back becomes too abstract
```python
# Problem: Abstract query loses connection to original intent
# Solution: Controlled abstraction with relevance check

def controlled_step_back(self, query: str, max_abstraction: float = 0.5) -> str:
    """Generate step-back query with abstraction limits"""
    
    abstract_query = self.step_back(query)
    
    # Check if abstraction went too far
    original_embedding = self.embedder.encode_query(query)
    abstract_embedding = self.embedder.encode_query(abstract_query)
    
    similarity = cosine_similarity([original_embedding], [abstract_embedding])[0][0]
    
    if similarity < max_abstraction:
        # Too abstract, generate intermediate abstraction
        intermediate_prompt = f"""The question "{query}" is related to the general topic "{abstract_query}".
        
        Create a question that is more general than the original but more specific than the abstract version.
        
        Intermediate question:"""
        
        response = self.llm.complete(intermediate_prompt)
        return response.content.strip()
    
    return abstract_query
```

**Next concept**: Reranking Systems — cross-encoder models and LLM-based reranking for precision improvement
