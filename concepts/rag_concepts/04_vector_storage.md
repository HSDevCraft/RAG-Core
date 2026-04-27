# Vector Storage — Indexing Algorithms & Similarity Search

## Conceptual Foundation

Vector storage systems enable **fast similarity search** over millions or billions of high-dimensional embeddings. The core challenge: find the k most similar vectors to a query vector without computing distances to every vector in the database.

**Key insight**: Exact nearest neighbor search is O(n) — impractical for large datasets. Vector databases use **Approximate Nearest Neighbor (ANN)** algorithms that trade small accuracy losses for dramatic speedup (sub-millisecond search over millions of vectors).

### The Curse of Dimensionality

In high-dimensional spaces, **all distances become approximately equal**:
```
As d → ∞, distance_max/distance_min → 1
```

This makes traditional tree-based indexes (like k-d trees) ineffective. Vector databases use specialized data structures that exploit **local clustering** and **dimensionality reduction**.

---

## Mathematical Formulation

### Similarity Search Problem

Given:
- Database vectors: **D** = {v₁, v₂, ..., vₙ} where vᵢ ∈ ℝᵈ
- Query vector: **q** ∈ ℝᵈ
- Distance function: **dist(q, vᵢ)**

Find: **k-NN(q)** = {vᵢ₁, vᵢ₂, ..., vᵢₖ} such that
```
dist(q, vᵢ₁) ≤ dist(q, vᵢ₂) ≤ ... ≤ dist(q, vᵢₖ) ≤ dist(q, vⱼ) ∀j ∉ {i₁,...,iₖ}
```

### ANN Trade-off

**Approximation factor α ≥ 1**:
ANN returns vectors where dist(q, vᵢ) ≤ α × dist(q, v*), where v* is the true nearest neighbor.

**Success probability δ**:
P(ANN returns α-approximate result) ≥ δ

**Quality measures**:
- **Recall@k**: |true_top_k ∩ returned_top_k| / k
- **Precision@k**: |relevant_returned ∩ returned_top_k| / k

---

## Implementation Details

Your system (`vector_store/vector_store.py`) provides a unified interface over multiple vector storage backends.

### BaseVectorStore Interface

```python
@dataclass
class SearchResult:
    chunk_id: str           # Unique identifier
    content: str            # Original text
    score: float           # Similarity score [0,1]
    metadata: Dict         # Associated metadata
    rank: int             # Position in results (0-based)

class BaseVectorStore(ABC):
    def add(self, ids: List[str], vectors: np.ndarray, 
            texts: List[str], metadatas: List[Dict]) -> None
    
    def search(self, query_vector: np.ndarray, top_k: int,
               filter: Optional[Dict] = None) -> List[SearchResult]
    
    def delete(self, ids: List[str]) -> int
    def update(self, ids: List[str], vectors: np.ndarray, 
               texts: List[str], metadatas: List[Dict]) -> None
    
    def persist(self, path: str) -> None
    def load(self, path: str) -> None
```

### FAISS Implementation

FAISS (Facebook AI Similarity Search) provides highly optimized similarity search with multiple indexing algorithms.

```python
class FAISSVectorStore(BaseVectorStore):
    def __init__(self, dimension: int, index_type: str = "hnsw",
                 metric: str = "cosine"):
        """
        index_type: "flat" | "ivf" | "hnsw"
        metric: "cosine" | "l2" | "ip" (inner product)
        """
        self.dimension = dimension
        self.index = self._create_index(index_type, metric)
        self.id_mapping = {}  # faiss_id ↔ chunk_id
        self.texts = {}       # chunk_id → text
        self.metadatas = {}   # chunk_id → metadata
```

#### Flat Index (Exact Search)
```python
def _create_flat_index(self, metric: str) -> faiss.Index:
    """
    Brute-force exact search
    Time: O(n×d) per query
    Memory: O(n×d)
    Use case: Small datasets (< 10K vectors), highest accuracy needed
    """
    if metric == "cosine":
        # Cosine = inner product on normalized vectors
        return faiss.IndexFlatIP(self.dimension)
    elif metric == "l2":
        return faiss.IndexFlatL2(self.dimension)
    elif metric == "ip":
        return faiss.IndexFlatIP(self.dimension)
```

#### IVF Index (Inverted File)
```python
def _create_ivf_index(self, metric: str, nlist: int = 256) -> faiss.Index:
    """
    Cluster vectors into nlist centroids, search only relevant clusters
    Training: K-means clustering on sample vectors
    Query: Find nearest centroids, search only those clusters
    
    Parameters:
    - nlist: Number of clusters (√n to n/39 rule of thumb)
    - nprobe: Number of clusters to search (1 to nlist)
    
    Time: O(nprobe × n/nlist × d) ≈ O(√n × d) 
    Memory: O(n×d + nlist×d)
    """
    if metric == "cosine":
        quantizer = faiss.IndexFlatIP(self.dimension)
        return faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    elif metric == "l2":
        quantizer = faiss.IndexFlatL2(self.dimension)
        return faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)

def train_ivf_index(self, training_vectors: np.ndarray):
    """IVF requires training on representative sample"""
    if not self.index.is_trained:
        # Need at least 39*nlist training vectors for good clustering
        min_training_size = 39 * self.nlist
        if len(training_vectors) < min_training_size:
            # Augment with random vectors if insufficient data
            random_vectors = np.random.randn(min_training_size - len(training_vectors), 
                                           self.dimension).astype(np.float32)
            training_vectors = np.vstack([training_vectors, random_vectors])
        
        self.index.train(training_vectors)
```

#### HNSW Index (Hierarchical Navigable Small World)
```python
def _create_hnsw_index(self, M: int = 32, ef_construction: int = 200) -> faiss.Index:
    """
    Graph-based index with hierarchical layers
    Construction: Build multi-layer graph with decreasing connectivity
    Query: Navigate from top layer to bottom, following closest neighbors
    
    Parameters:
    - M: Number of connections per node (16-64, higher = better recall)
    - ef_construction: Search width during construction (100-800)
    - ef_search: Search width during query (set via nprobe)
    
    Time: O(log n × M × d)
    Memory: O(n × M × log n)
    Quality: Highest recall for fast queries
    """
    index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.ef_construction = ef_construction
    return index

def configure_search_params(self, ef_search: int = 50):
    """Configure search-time parameters"""
    if isinstance(self.index, faiss.IndexHNSWFlat):
        self.index.hnsw.ef_search = ef_search
    elif isinstance(self.index, faiss.IndexIVFFlat):
        self.index.nprobe = min(ef_search, self.index.nlist)
```

**HNSW Algorithm Details**:
```
Layer Selection: level ∈ [0, ⌊-ln(unif(0,1)) × ml⌋] where ml = 1/ln(2)
Graph Structure: 
- Layer L: M connections per node
- Layer 0: 2×M connections per node (densest)
- Higher layers: Exponentially fewer nodes

Search Algorithm:
1. Start at entry point (highest layer)
2. Greedily navigate to query's nearest neighbor on current layer  
3. Descend to next layer, continue until layer 0
4. Return k closest nodes from layer 0
```

### Memory Estimates

| Index Type | Memory Formula | 100K vectors (384-dim) | 1M vectors (768-dim) |
|---|---|---|---|
| **Flat** | n × d × 4 bytes | 154 MB | 3.1 GB |
| **IVF** | n × d × 4 + nlist × d × 4 | 154 MB + 0.4 MB | 3.1 GB + 0.8 MB |
| **HNSW** | n × (d × 4 + M × 8 × levels) | ~200 MB | ~4.5 GB |

### ChromaDB Implementation

ChromaDB focuses on ease of use with built-in embedding, persistence, and rich metadata filtering.

```python
class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str = "rag_chunks", 
                 persist_directory: str = "./chroma_db"):
        import chromadb
        from chromadb.config import Settings
        
        # Persistent client with local storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # cosine similarity
        )
```

**Rich Metadata Filtering**:
```python
def search(self, query_vector: np.ndarray, top_k: int = 10, 
           filter: Optional[Dict] = None) -> List[SearchResult]:
    """
    ChromaDB supports complex metadata filters:
    
    filter = {
        "source": "policy.pdf",                    # Exact match
        "page": {"$gte": 5, "$lte": 10},          # Range
        "tags": {"$contains": "refund"},          # Array contains  
        "file_type": {"$in": ["pdf", "docx"]},   # One of values
        "$and": [{"year": 2024}, {"dept": "legal"}]  # Boolean logic
    }
    """
    
    # Convert numpy to list (ChromaDB requirement)
    query_embeddings = [query_vector.tolist()]
    
    results = self.collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        where=filter  # Rich filtering support
    )
    
    return self._convert_to_search_results(results)

def add_with_metadata(self, vectors: np.ndarray, texts: List[str], 
                     metadatas: List[Dict], ids: List[str]):
    """ChromaDB automatically handles ID conflicts and updates"""
    
    self.collection.add(
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    # ChromaDB automatically persists changes
```

**Collection Management**:
```python
def create_collection_with_schema(self, schema: Dict):
    """Define collection schema for better performance"""
    
    collection = self.client.create_collection(
        name="structured_docs",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M": 32,                    # HNSW connectivity
            "hnsw:ef_construction": 200,     # Build-time search width
            "hnsw:ef_search": 50,            # Query-time search width
            "hnsw:max_elements": 1000000,    # Pre-allocate space
            "description": "Legal documents with structured metadata"
        }
    )
    
    return collection
```

### Pinecone Implementation (Managed Service)

```python
class PineconeVectorStore(BaseVectorStore):
    def __init__(self, index_name: str, dimension: int, 
                 metric: str = "cosine", environment: str = "us-west1-gcp"):
        import pinecone
        
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=environment)
        
        # Create index if doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                pods=1,                  # Start with 1 pod
                replicas=1,             # No replication initially
                pod_type="p1.x1"       # Performance tier
            )
        
        self.index = pinecone.Index(index_name)

def upsert_with_namespaces(self, vectors: np.ndarray, texts: List[str],
                          metadatas: List[Dict], ids: List[str],
                          namespace: str = "default"):
    """
    Pinecone namespaces enable multi-tenancy in single index
    Use cases: Different users, document collections, experiments
    """
    
    # Convert to Pinecone format
    vectors_to_upsert = []
    for i, (vector, text, metadata, id_) in enumerate(zip(vectors, texts, metadatas, ids)):
        vectors_to_upsert.append({
            "id": id_,
            "values": vector.tolist(),
            "metadata": {**metadata, "text": text}  # Store text in metadata
        })
    
    # Batch upsert (Pinecone supports up to 100 vectors per request)
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        self.index.upsert(vectors=batch, namespace=namespace)

def search_with_hybrid_scoring(self, query_vector: np.ndarray, 
                              sparse_vector: Dict, alpha: float = 0.7,
                              top_k: int = 10) -> List[SearchResult]:
    """
    Pinecone hybrid search: combine dense + sparse signals
    alpha: weight for dense vs sparse (0.7 = 70% dense, 30% sparse)
    """
    
    results = self.index.query(
        vector=query_vector.tolist(),
        sparse_vector=sparse_vector,  # {"indices": [1,2,3], "values": [0.1,0.2,0.3]}
        alpha=alpha,
        top_k=top_k,
        include_metadata=True
    )
    
    return self._convert_pinecone_results(results)
```

---

## Comparative Analysis

### Backend Selection Guide

| Factor | FAISS | ChromaDB | Pinecone |
|---|---|---|---|
| **Setup** | Complex | Simple | Managed |
| **Performance** | Fastest | Good | Very Good |
| **Scalability** | Manual sharding | Single-node | Auto-scaling |
| **Metadata filtering** | Limited | Rich | Rich |
| **Cost** | Free | Free | Usage-based |
| **Latency** | <1ms | 1-5ms | 5-50ms |
| **Persistence** | Manual | Automatic | Managed |
| **Multi-tenancy** | Manual | Collections | Namespaces |

### Index Type Selection

**Development/Testing**:
```python
# Fast setup, exact results
config = VectorStoreConfig(
    backend="faiss",
    index_type="flat",
    persist_dir="./dev_vector_store"
)
```

**Production (Medium Scale: 10K-1M vectors)**:
```python
# Best balance of speed and accuracy
config = VectorStoreConfig(
    backend="faiss", 
    index_type="hnsw",
    hnsw_m=32,
    hnsw_ef_construction=200,
    hnsw_ef_search=50
)
```

**Production (Large Scale: 1M+ vectors)**:
```python
# Memory-efficient for massive datasets
config = VectorStoreConfig(
    backend="faiss",
    index_type="ivf", 
    ivf_nlist=4096,  # √n clusters
    ivf_nprobe=64    # Search 64 clusters
)
```

**Rich Metadata Queries**:
```python
# Complex filtering requirements
config = VectorStoreConfig(
    backend="chroma",
    collection_name="structured_docs"
)
```

---

## Practical Guidelines

### Index Tuning

**HNSW Parameter Optimization**:
```python
def tune_hnsw_params(vectors: np.ndarray, queries: np.ndarray, 
                    ground_truth: List[List[int]]) -> Dict:
    """Find optimal HNSW parameters via grid search"""
    
    param_grid = {
        'M': [16, 32, 48, 64],
        'ef_construction': [100, 200, 400, 800],
        'ef_search': [16, 32, 64, 128, 256]
    }
    
    best_params = {}
    best_recall = 0
    
    for M in param_grid['M']:
        for ef_cons in param_grid['ef_construction']:
            # Build index with these parameters
            index = faiss.IndexHNSWFlat(vectors.shape[1], M)
            index.hnsw.ef_construction = ef_cons
            index.add(vectors)
            
            for ef_search in param_grid['ef_search']:
                index.hnsw.ef_search = ef_search
                
                # Measure recall@10
                recall = evaluate_recall(index, queries, ground_truth, k=10)
                
                if recall > best_recall:
                    best_recall = recall
                    best_params = {
                        'M': M, 'ef_construction': ef_cons, 
                        'ef_search': ef_search, 'recall': recall
                    }
    
    return best_params

def evaluate_recall(index: faiss.Index, queries: np.ndarray, 
                   ground_truth: List[List[int]], k: int = 10) -> float:
    """Compute recall@k for query set"""
    
    _, retrieved_ids = index.search(queries, k)
    
    total_recall = 0
    for i, true_neighbors in enumerate(ground_truth):
        retrieved = set(retrieved_ids[i])
        true_set = set(true_neighbors[:k])
        recall = len(retrieved & true_set) / len(true_set)
        total_recall += recall
    
    return total_recall / len(ground_truth)
```

**IVF Parameter Tuning**:
```python
def optimize_ivf_params(n_vectors: int) -> Dict:
    """Rule-based IVF parameter selection"""
    
    # nlist: Number of clusters (√n to n/39)
    nlist_options = [int(np.sqrt(n_vectors)), n_vectors // 39, n_vectors // 100]
    nlist = max(1, min(nlist_options))  # Clamp to reasonable range
    
    # nprobe: Clusters to search (log₂(nlist) to nlist/4)
    min_nprobe = max(1, int(np.log2(nlist)))
    max_nprobe = max(min_nprobe, nlist // 4)
    
    return {
        'nlist': nlist,
        'nprobe_range': (min_nprobe, max_nprobe),
        'recommended_nprobe': min(32, max_nprobe)  # Good default
    }

# Example usage
n_docs = 500_000
params = optimize_ivf_params(n_docs)
# {'nlist': 12804, 'nprobe_range': (13, 3201), 'recommended_nprobe': 32}
```

### Memory Management

**Index Size Estimation**:
```python
def estimate_index_memory(n_vectors: int, dimension: int, index_type: str) -> Dict:
    """Estimate memory requirements for different index types"""
    
    vector_size_gb = (n_vectors * dimension * 4) / (1024**3)  # float32
    
    estimates = {
        'vectors_gb': vector_size_gb,
        'flat_gb': vector_size_gb,  # Just store all vectors
        'ivf_gb': vector_size_gb * 1.1,  # 10% overhead for quantizer
        'hnsw_gb': vector_size_gb * (1.5 + 0.1 * np.log(n_vectors)),  # Graph overhead
    }
    
    return estimates

# Production planning
memory_est = estimate_index_memory(1_000_000, 768, 'hnsw')
# {'vectors_gb': 2.86, 'flat_gb': 2.86, 'ivf_gb': 3.15, 'hnsw_gb': 4.74}
```

**Batch Processing for Large Datasets**:
```python
def batch_index_construction(vectors: np.ndarray, batch_size: int = 50000) -> faiss.Index:
    """Build index incrementally to manage memory"""
    
    index = faiss.IndexHNSWFlat(vectors.shape[1], 32)
    
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        batch = vectors[start:end].copy()  # Ensure contiguous memory
        
        index.add(batch)
        
        # Clear batch from memory
        del batch
        gc.collect()
        
        print(f"Added batch {start//batch_size + 1}, total vectors: {index.ntotal}")
    
    return index
```

### Performance Optimization

**Query Batching**:
```python
def batch_search_optimization(index: faiss.Index, queries: np.ndarray, 
                             k: int, batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Optimize search throughput with batching"""
    
    all_distances = []
    all_indices = []
    
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        query_batch = queries[start:end]
        
        # Batch search is more efficient than individual queries
        distances, indices = index.search(query_batch, k)
        
        all_distances.append(distances)
        all_indices.append(indices)
    
    return np.vstack(all_distances), np.vstack(all_indices)
```

**Index Warming**:
```python
def warm_up_index(index: faiss.Index, dimension: int, n_warmup: int = 1000):
    """Pre-warm index caches with random queries"""
    
    warmup_queries = np.random.randn(n_warmup, dimension).astype(np.float32)
    
    # Normalize if using cosine similarity
    faiss.normalize_L2(warmup_queries)
    
    # Perform throwaway searches to warm CPU caches
    for i in range(0, n_warmup, 100):
        batch = warmup_queries[i:i+100]
        _ = index.search(batch, 10)
    
    print(f"Index warmed up with {n_warmup} queries")
```

### Monitoring & Observability

**Index Health Metrics**:
```python
def compute_index_health_metrics(index: faiss.Index, 
                               sample_queries: np.ndarray) -> Dict:
    """Monitor index performance and quality"""
    
    metrics = {}
    
    # 1. Search latency distribution
    latencies = []
    for query in sample_queries:
        start_time = time.perf_counter()
        _ = index.search(query.reshape(1, -1), 10)
        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(latency_ms)
    
    metrics['latency_p50'] = np.percentile(latencies, 50)
    metrics['latency_p95'] = np.percentile(latencies, 95)
    metrics['latency_p99'] = np.percentile(latencies, 99)
    
    # 2. Index utilization
    if hasattr(index, 'ntotal'):
        metrics['total_vectors'] = index.ntotal
    
    # 3. Memory usage
    if hasattr(index, 'storage_size'):
        metrics['storage_mb'] = index.storage_size() / (1024 * 1024)
    
    return metrics
```

### Common Issues & Solutions

**Issue**: Poor recall on HNSW index
```python
# Problem: ef_search too low, M too low
# Solution: Increase search parameters
def fix_hnsw_recall(index: faiss.IndexHNSWFlat, target_recall: float = 0.95):
    """Automatically tune HNSW for target recall"""
    
    current_ef = index.hnsw.ef_search
    
    # Binary search for optimal ef_search
    low, high = current_ef, 512
    best_ef = current_ef
    
    while low <= high:
        mid = (low + high) // 2
        index.hnsw.ef_search = mid
        
        # Test recall on small sample
        recall = measure_sample_recall(index)
        
        if recall >= target_recall:
            best_ef = mid
            high = mid - 1
        else:
            low = mid + 1
    
    index.hnsw.ef_search = best_ef
    print(f"Optimized ef_search: {best_ef}, recall: {measure_sample_recall(index):.3f}")
```

**Issue**: Index too large for memory
```python
# Problem: Cannot fit full index in RAM
# Solution: Use quantized index or disk-based storage
def create_memory_efficient_index(vectors: np.ndarray) -> faiss.Index:
    """Create compressed index for memory-constrained environments"""
    
    # Product Quantization: compress vectors to reduce memory
    dimension = vectors.shape[1]
    
    # Use 8-bit quantization (8x compression)
    m = dimension // 8  # Number of subquantizers
    nbits = 8          # Bits per subquantizer
    
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, 256, m, nbits)
    
    # Train quantizer
    training_sample = vectors[:max(10000, len(vectors)//10)]  
    index.train(training_sample)
    index.add(vectors)
    
    print(f"Original size: {vectors.nbytes / (1024**2):.1f} MB")
    print(f"Compressed size: ~{vectors.nbytes / (8 * 1024**2):.1f} MB")
    
    return index
```

**Issue**: Slow metadata filtering
```python
# Problem: Filtering after vector search is inefficient
# Solution: Pre-filter vector space or use hybrid approach
def create_filtered_index(vectors: np.ndarray, metadatas: List[Dict],
                         filter_key: str) -> Dict[str, faiss.Index]:
    """Create separate indexes for each filter value"""
    
    # Group vectors by filter value
    groups = defaultdict(list)
    for i, metadata in enumerate(metadatas):
        filter_value = metadata.get(filter_key, 'unknown')
        groups[filter_value].append(i)
    
    # Create separate FAISS index for each group
    filtered_indexes = {}
    for filter_value, indices in groups.items():
        group_vectors = vectors[indices]
        
        index = faiss.IndexHNSWFlat(vectors.shape[1], 32)
        index.add(group_vectors)
        
        filtered_indexes[filter_value] = {
            'index': index,
            'original_indices': indices
        }
    
    return filtered_indexes
```

**Next concept**: Retrieval Methods — dense, sparse, hybrid search strategies and fusion algorithms
