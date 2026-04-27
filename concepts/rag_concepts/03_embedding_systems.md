# Embedding Systems — Dense Representations & Similarity

## Conceptual Foundation

Embeddings are **dense vector representations** that capture semantic meaning in high-dimensional space. They transform discrete text into continuous vectors where **similar meanings cluster together geometrically**. This is the core enabler of semantic search in RAG systems.

**Key insight**: Traditional keyword search finds exact matches, but embeddings find **conceptual matches**. A query about "refund policy" can retrieve documents mentioning "return procedures" or "money-back guarantee" because their embeddings are geometrically close.

### The Semantic Space

```
Traditional Search:  "refund policy" → exact string match only
Embedding Search:    "refund policy" → [0.23, -0.15, 0.87, ...] → similar vectors
                     "return procedure" → [0.21, -0.18, 0.82, ...] ← cosine similarity: 0.94
```

**Mathematical foundation**: Embeddings map text to points in ℝᵈ where semantic similarity corresponds to geometric proximity.

---

## Mathematical Formulation

### Embedding Function

An embedding function `E: Text → ℝᵈ` maps text to d-dimensional vectors:
```
E("refund policy") = [e₁, e₂, ..., eₑ] where eᵢ ∈ ℝ
```

### Similarity Metrics

**Cosine Similarity** (most common):
```
cos_sim(u, v) = (u · v) / (||u|| × ||v||) = Σᵢ(uᵢvᵢ) / (√Σᵢuᵢ² × √Σᵢvᵢ²)
```

Range: [-1, 1], where:
- 1.0 = identical direction (perfect match)
- 0.0 = orthogonal (unrelated)  
- -1.0 = opposite direction (antonyms)

**Dot Product** (for normalized vectors):
```
dot(u, v) = Σᵢ(uᵢvᵢ)
```
When ||u|| = ||v|| = 1, dot product equals cosine similarity.

**Euclidean Distance**:
```
L2(u, v) = √Σᵢ(uᵢ - vᵢ)²
```
Range: [0, ∞], where 0 = identical, larger = more different.

**Manhattan Distance**:
```
L1(u, v) = Σᵢ|uᵢ - vᵢ|
```

### Vector Normalization

**L2 Normalization** (unit vectors):
```
û = u / ||u|| = u / √Σᵢuᵢ²
```

**Why normalize?** Makes cosine similarity equivalent to dot product:
```
cos_sim(û, v̂) = û · v̂  (when both are unit vectors)
```

This enables faster similarity computation in vector databases.

---

## Implementation Details

Your system (`embeddings/embedding.py`) provides a unified interface over multiple embedding providers.

### EmbeddingEngine Architecture

```python
@dataclass 
class EmbeddingResult:
    embeddings: np.ndarray      # Shape: (n_texts, embedding_dim)
    model_name: str             # "all-MiniLM-L6-v2"
    dimension: int              # 384, 512, 1536, etc.
    normalized: bool            # True if L2-normalized
    
class EmbeddingEngine(ABC):
    def encode(self, texts: List[str], batch_size: int = 64) -> EmbeddingResult
    def encode_query(self, query: str) -> np.ndarray
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray
```

### Provider Implementations

#### SentenceTransformers (Local)
```python
class SentenceTransformerEmbedder(EmbeddingEngine):
    """
    Library: sentence-transformers
    Execution: Local GPU/CPU
    Models: 100+ pre-trained models on HuggingFace
    Cost: Free (only compute)
    """
```

**Implementation highlights**:
```python
def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    
    self.model = SentenceTransformer(model_name, device=device)
    self.dimension = self.model.get_sentence_embedding_dimension()
    
def encode(self, texts: List[str], batch_size: int = 64) -> EmbeddingResult:
    # Automatic batching for memory efficiency
    embeddings = self.model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # L2 normalize
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True
    )
    
    return EmbeddingResult(
        embeddings=embeddings,
        model_name=self.model_name,
        dimension=self.dimension,
        normalized=True
    )
```

**Popular models**:
| Model | Dimensions | Size | Performance | Use Case |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast | General purpose |
| `all-mpnet-base-v2` | 768 | 420MB | Better | Quality over speed |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.2GB | Best | Production |
| `intfloat/e5-large-v2` | 1024 | 1.2GB | SOTA | Research |

#### OpenAI Embeddings (API)
```python
class OpenAIEmbedder(EmbeddingEngine):
    """
    Provider: OpenAI API
    Models: text-embedding-3-small, text-embedding-3-large, ada-002
    Execution: API call
    Cost: $0.02-$0.13 per 1M tokens
    """
```

**Implementation**:
```python
def encode(self, texts: List[str], batch_size: int = 64) -> EmbeddingResult:
    import openai
    
    client = openai.OpenAI(api_key=self.api_key)
    all_embeddings = []
    
    # Process in batches (OpenAI limit: 8192 texts per request)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        response = client.embeddings.create(
            model=self.model_name,
            input=batch,
            encoding_format="float"  # vs base64 for smaller payloads
        )
        
        batch_embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(batch_embeddings)
    
    embeddings = np.array(all_embeddings, dtype=np.float32)
    
    # OpenAI embeddings are pre-normalized
    return EmbeddingResult(
        embeddings=embeddings,
        model_name=self.model_name, 
        dimension=len(embeddings[0]),
        normalized=True
    )
```

**OpenAI models comparison**:
| Model | Dimensions | Cost ($/1M tokens) | MTEB Score | Notes |
|---|---|---|---|---|
| `text-embedding-3-small` | 1536 | $0.02 | 62.3 | Best value |
| `text-embedding-3-large` | 3072 | $0.13 | 64.6 | Highest quality |
| `text-embedding-ada-002` | 1536 | $0.10 | 61.0 | Legacy |

#### Cohere Embeddings (API)
```python
class CohereEmbedder(EmbeddingEngine):
    """
    Provider: Cohere API
    Models: embed-english-v3.0, embed-multilingual-v3.0
    Strengths: Multilingual support, semantic search optimized
    Cost: $0.10 per 1M tokens
    """
```

**Multilingual example**:
```python
def encode_multilingual(self, texts: List[str], input_type: str = "search_document"):
    """
    input_type: "search_document" | "search_query" | "classification" | "clustering"
    Cohere optimizes embeddings based on intended use
    """
    import cohere
    
    co = cohere.Client(self.api_key)
    
    response = co.embed(
        texts=texts,
        model=self.model_name,
        input_type=input_type,  # Task-specific optimization
        embedding_types=['float']
    )
    
    return np.array(response.embeddings.float, dtype=np.float32)
```

### Caching Layer

Embedding computation is expensive — caching is essential for production.

```python
class CachedEmbeddingEngine:
    def __init__(self, base_engine: EmbeddingEngine, cache: CacheBackend):
        self.engine = base_engine
        self.cache = cache
        
    def encode(self, texts: List[str], **kwargs) -> EmbeddingResult:
        # Check cache for each text
        cache_keys = [embedding_cache_key(text, self.engine.model_name) 
                     for text in texts]
        cached_embeddings = self.cache.mget(cache_keys)
        
        # Find uncached texts
        uncached_indices = []
        uncached_texts = []
        for i, (text, key) in enumerate(zip(texts, cache_keys)):
            if key not in cached_embeddings:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Compute embeddings for uncached texts
        if uncached_texts:
            result = self.engine.encode(uncached_texts, **kwargs)
            
            # Cache new embeddings
            cache_items = {}
            for i, (idx, embedding) in enumerate(zip(uncached_indices, result.embeddings)):
                cache_items[cache_keys[idx]] = embedding
            self.cache.mset(cache_items, ttl=86400)  # 24 hour TTL
        
        # Assemble final result
        all_embeddings = np.zeros((len(texts), self.engine.dimension))
        for i, key in enumerate(cache_keys):
            if key in cached_embeddings:
                all_embeddings[i] = cached_embeddings[key]
            else:
                # Find in newly computed results
                uncached_pos = uncached_indices.index(i)
                all_embeddings[i] = result.embeddings[uncached_pos]
        
        return EmbeddingResult(all_embeddings, result.model_name, 
                              result.dimension, result.normalized)
```

---

## Comparative Analysis

### Local vs API Embeddings

| Factor | SentenceTransformers | OpenAI API | Cohere API |
|---|---|---|---|
| **Cost** | Free (compute only) | $0.02-$0.13/1M tokens | $0.10/1M tokens |
| **Latency** | 5-50ms (local) | 50-200ms (network) | 50-200ms (network) |
| **Privacy** | Full control | Data sent to OpenAI | Data sent to Cohere |  
| **Quality** | Model-dependent | Very high | Very high |
| **Availability** | Always available | API rate limits | API rate limits |
| **Multilingual** | Limited models | Good | Excellent |

### Model Selection Guide

**Development/Prototyping**:
```python
# Fast iteration, no API dependencies
embedding_config = EmbeddingConfig(
    provider="sentence_transformers",
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)
```

**Production (Quality-first)**:
```python  
# Best quality, acceptable cost
embedding_config = EmbeddingConfig(
    provider="openai", 
    model_name="text-embedding-3-small"
)
```

**Production (Cost-sensitive)**:
```python
# Good quality, lower cost
embedding_config = EmbeddingConfig(
    provider="sentence_transformers",
    model_name="BAAI/bge-large-en-v1.5",
    device="cuda"  # GPU acceleration
)
```

**Multilingual**:
```python
embedding_config = EmbeddingConfig(
    provider="cohere",
    model_name="embed-multilingual-v3.0"
)
```

### Embedding Dimensions Impact

| Dimensions | Memory (1M vectors) | Search Speed | Quality | Notes |
|---|---|---|---|---|
| 384 | 1.5GB | Fast | Good | all-MiniLM-L6-v2 |
| 768 | 3.0GB | Medium | Better | all-mpnet-base-v2 |
| 1024 | 4.0GB | Medium | High | BGE-large |
| 1536 | 6.0GB | Slower | High | OpenAI-3-small |
| 3072 | 12.0GB | Slow | Highest | OpenAI-3-large |

**Recommendation**: Start with 384-768 dimensions, scale up if quality insufficient.

---

## Practical Guidelines

### Batch Size Optimization

**Memory constraints**:
```python
def estimate_memory_usage(batch_size: int, max_seq_len: int, embedding_dim: int) -> int:
    """Estimate GPU memory for embedding batch"""
    
    # Input tokens: batch_size × max_seq_len × vocab_size (sparse)
    input_memory = batch_size * max_seq_len * 4  # 4 bytes per token ID
    
    # Hidden states: batch_size × max_seq_len × hidden_dim  
    hidden_memory = batch_size * max_seq_len * embedding_dim * 4  # float32
    
    # Output embeddings: batch_size × embedding_dim
    output_memory = batch_size * embedding_dim * 4
    
    # Add 20% overhead for activation, gradients, etc.
    total_mb = (input_memory + hidden_memory + output_memory) * 1.2 / (1024 * 1024)
    
    return int(total_mb)

# Example usage
gpu_memory_gb = 8  # RTX 3070
max_batch_memory_mb = gpu_memory_gb * 1024 * 0.8  # Leave 20% headroom

optimal_batch_size = 1
while estimate_memory_usage(optimal_batch_size + 8, 512, 768) < max_batch_memory_mb:
    optimal_batch_size += 8

print(f"Optimal batch size: {optimal_batch_size}")  # Often 32-128
```

### Text Preprocessing

**Standard preprocessing pipeline**:
```python
def preprocess_for_embedding(text: str) -> str:
    """Prepare text for embedding computation"""
    
    # 1. Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 2. Handle special characters (keep meaningful punctuation)
    text = re.sub(r'[^\w\s\.\!\?\-\,\:\;\(\)]', '', text)
    
    # 3. Normalize case (model-dependent)
    if not self.model_is_case_sensitive:
        text = text.lower()
    
    # 4. Truncate to model's max sequence length
    if self.max_seq_length:
        tokens = text.split()[:self.max_seq_length - 2]  # Leave room for [CLS]/[SEP]
        text = ' '.join(tokens)
    
    return text
```

**Query vs document preprocessing**:
```python
def preprocess_query(query: str) -> str:
    """Query-specific preprocessing"""
    query = preprocess_for_embedding(query)
    
    # Remove question words that don't add semantic value
    stop_words = {'what', 'how', 'when', 'where', 'why', 'which', 'who'}
    tokens = [t for t in query.split() if t.lower() not in stop_words]
    
    return ' '.join(tokens)

def preprocess_document(text: str) -> str:
    """Document-specific preprocessing"""  
    text = preprocess_for_embedding(text)
    
    # Keep more context for documents
    # Remove only administrative text
    text = re.sub(r'Page \d+ of \d+', '', text)  # Page numbers
    text = re.sub(r'Copyright \d{4}', '', text)  # Copyright notices
    
    return text
```

### Quality Assessment

**Embedding quality metrics**:
```python
def assess_embedding_quality(embeddings: np.ndarray, texts: List[str]) -> Dict[str, float]:
    """Compute quality metrics for embeddings"""
    
    # 1. Average pairwise similarity (lower is better for diversity)
    similarities = cosine_similarity(embeddings)
    np.fill_diagonal(similarities, 0)  # Exclude self-similarity
    avg_similarity = similarities.mean()
    
    # 2. Embedding variance (higher is better for expressiveness)
    embedding_variance = embeddings.var(axis=0).mean()
    
    # 3. Dimension utilization (fraction of dimensions with significant variance)
    dim_std = embeddings.std(axis=0)
    utilized_dims = (dim_std > dim_std.mean() * 0.1).sum() / len(dim_std)
    
    # 4. Semantic coherence (similar texts should have similar embeddings)
    semantic_coherence = measure_semantic_coherence(embeddings, texts)
    
    return {
        'avg_pairwise_similarity': avg_similarity,
        'embedding_variance': embedding_variance,
        'dimension_utilization': utilized_dims,
        'semantic_coherence': semantic_coherence
    }

def measure_semantic_coherence(embeddings: np.ndarray, texts: List[str]) -> float:
    """Measure if semantically similar texts have similar embeddings"""
    
    # Simple heuristic: texts with shared keywords should be similar
    coherence_scores = []
    
    for i in range(len(texts)):
        for j in range(i + 1, min(i + 50, len(texts))):  # Sample for efficiency
            text_similarity = jaccard_similarity(set(texts[i].split()), set(texts[j].split()))
            embedding_similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0,0]
            
            coherence_scores.append(text_similarity * embedding_similarity)
    
    return np.mean(coherence_scores)
```

### Performance Optimization

**GPU utilization**:
```python
def optimize_gpu_usage(embedder: SentenceTransformerEmbedder, texts: List[str]):
    """Maximize GPU throughput for embedding computation"""
    
    # 1. Sort texts by length for better batching
    indexed_texts = [(i, text) for i, text in enumerate(texts)]
    indexed_texts.sort(key=lambda x: len(x[1]))
    
    # 2. Dynamic batching based on sequence length
    embeddings_result = [None] * len(texts)
    
    i = 0
    while i < len(indexed_texts):
        # Estimate batch size for current sequence length
        current_length = len(indexed_texts[i][1].split())
        estimated_batch_size = min(64, max(1, 1024 // current_length))
        
        # Collect batch of similar-length sequences
        batch_indices = []
        batch_texts = []
        j = i
        
        while j < len(indexed_texts) and len(batch_texts) < estimated_batch_size:
            idx, text = indexed_texts[j]
            if len(text.split()) <= current_length * 1.2:  # 20% length tolerance
                batch_indices.append(idx)
                batch_texts.append(text)
                j += 1
            else:
                break
        
        # Process batch
        batch_embeddings = embedder.encode(batch_texts)
        
        # Store results in original order
        for orig_idx, embedding in zip(batch_indices, batch_embeddings.embeddings):
            embeddings_result[orig_idx] = embedding
        
        i = j
    
    return np.array(embeddings_result)
```

**Memory-efficient processing**:
```python
def embed_large_corpus(texts: List[str], embedder: EmbeddingEngine, 
                       max_memory_mb: int = 4096) -> np.ndarray:
    """Embed corpus larger than memory with disk streaming"""
    import tempfile
    import h5py
    
    # Estimate memory usage
    sample_embedding = embedder.encode([texts[0]])
    embedding_dim = sample_embedding.dimension
    bytes_per_embedding = embedding_dim * 4  # float32
    
    # Calculate chunk size to stay within memory limit
    max_embeddings_in_memory = (max_memory_mb * 1024 * 1024) // bytes_per_embedding
    chunk_size = min(max_embeddings_in_memory, 10000)
    
    # Create temporary HDF5 file for streaming storage
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp_file:
        with h5py.File(tmp_file.name, 'w') as h5f:
            # Create dataset with known final shape
            embeddings_dataset = h5f.create_dataset(
                'embeddings', 
                shape=(len(texts), embedding_dim),
                dtype=np.float32,
                chunks=True,  # Enable compression
                compression='gzip'
            )
            
            # Process in chunks
            for start in range(0, len(texts), chunk_size):
                end = min(start + chunk_size, len(texts))
                chunk_texts = texts[start:end]
                
                chunk_embeddings = embedder.encode(chunk_texts)
                embeddings_dataset[start:end] = chunk_embeddings.embeddings
                
                # Force garbage collection
                del chunk_embeddings
                gc.collect()
            
            # Load final result (or return HDF5 handle for very large datasets)
            return embeddings_dataset[:]
```

### Common Issues & Solutions

**Issue**: Inconsistent similarity scores across batches
```python
# Problem: Different normalization between batches
# Solution: Always normalize consistently
def ensure_consistent_normalization(embeddings: np.ndarray) -> np.ndarray:
    """Ensure all embeddings are L2-normalized consistently"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)  # Avoid division by zero
    return embeddings / norms
```

**Issue**: Poor similarity for short texts
```python
# Problem: Short texts have sparse signal
# Solution: Context expansion
def expand_short_text(text: str, context_docs: List[str], min_length: int = 50) -> str:
    """Expand short texts with relevant context"""
    if len(text) >= min_length:
        return text
    
    # Find most similar document for context
    text_embedding = embedder.encode([text])
    context_embeddings = embedder.encode(context_docs)
    
    similarities = cosine_similarity(text_embedding, context_embeddings)[0]
    best_context_idx = similarities.argmax()
    
    # Append relevant sentence from context
    context_sentences = context_docs[best_context_idx].split('.')
    expanded_text = f"{text}. Context: {context_sentences[0].strip()}."
    
    return expanded_text
```

**Issue**: Embedding drift across model versions
```python
# Problem: Different model versions produce incompatible embeddings
# Solution: Version tracking and migration
@dataclass
class VersionedEmbeddingResult:
    embeddings: np.ndarray
    model_name: str
    model_version: str  # Track exact version
    embedding_hash: str  # Hash of first few embeddings for compatibility check
    
def migrate_embeddings(old_embeddings: np.ndarray, 
                      old_model: str, new_model: str) -> np.ndarray:
    """Migrate embeddings between model versions"""
    
    # Option 1: Linear transformation (if models are similar)
    if model_similarity_score(old_model, new_model) > 0.8:
        transformation_matrix = learn_transformation(old_embeddings, new_embeddings_sample)
        return old_embeddings @ transformation_matrix
    
    # Option 2: Re-embed everything (if models are different)
    else:
        logger.warning(f"Re-embedding required: {old_model} → {new_model}")
        return None  # Signal full re-embedding needed
```

**Next concept**: Vector Storage — indexing algorithms and similarity search optimization
