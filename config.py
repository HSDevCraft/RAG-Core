"""
Central configuration for the RAG system.
All tunable parameters, model names, and environment bindings live here.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# LLM / Embedding providers
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    provider: Literal["openai", "azure_openai", "ollama", "huggingface"] = "openai"
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout: int = 60
    # Azure-specific
    azure_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version: str = "2024-02-01"
    # Ollama / local
    ollama_base_url: str = "http://localhost:11434"
    # HuggingFace
    hf_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    hf_device: str = "cuda"


@dataclass
class EmbeddingConfig:
    provider: Literal["openai", "sentence_transformers", "cohere"] = "sentence_transformers"
    model_name: str = "BAAI/bge-large-en-v1.5"                 # default ST model
    openai_model: str = "text-embedding-3-large"                # 3072-dim
    dimension: int = 1024                                        # BAAI/bge-large
    batch_size: int = 64
    normalize: bool = True
    device: str = "cpu"                                          # "cuda" for GPU


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
@dataclass
class ChunkingConfig:
    strategy: Literal["recursive", "token", "semantic", "sentence"] = "recursive"
    chunk_size: int = 512          # tokens or chars depending on strategy
    chunk_overlap: int = 64
    min_chunk_size: int = 100
    separators: list = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------
@dataclass
class VectorStoreConfig:
    backend: Literal["faiss", "chroma", "pinecone", "weaviate"] = "faiss"
    persist_dir: str = "./vector_store_data"
    index_type: Literal["flat", "ivf", "hnsw"] = "hnsw"
    metric: Literal["cosine", "ip", "l2"] = "cosine"
    # FAISS IVF
    faiss_nlist: int = 128
    faiss_nprobe: int = 16
    # HNSW
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    # Pinecone
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "rag-index"
    # Chroma
    chroma_collection_name: str = "rag_collection"
    chroma_host: str = "localhost"
    chroma_port: int = 8000


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
@dataclass
class RetrievalConfig:
    top_k: int = 10                         # candidates before rerank
    final_top_k: int = 5                    # after reranking
    use_hybrid: bool = True                  # BM25 + dense
    hybrid_alpha: float = 0.7               # weight for dense score
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32
    mmr_lambda: float = 0.5                 # diversity trade-off
    use_mmr: bool = False


# ---------------------------------------------------------------------------
# Generation / Prompting
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    system_prompt_template: str = "default"
    max_context_tokens: int = 3000
    citation_style: Literal["inline", "footnote", "none"] = "inline"
    streaming: bool = False
    use_guardrails: bool = True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvaluationConfig:
    metrics: list = field(default_factory=lambda: [
        "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "answer_correctness"
    ])
    judge_model: str = "gpt-4o"
    sample_size: int = 100


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
@dataclass
class ObservabilityConfig:
    enable_tracing: bool = True
    log_level: str = "INFO"
    langfuse_public_key: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host: str = "https://cloud.langfuse.com"
    enable_prometheus: bool = True
    prometheus_port: int = 8001


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------
@dataclass
class RAGConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 4

    # Keys (pulled from env)
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    cohere_api_key: Optional[str] = field(default_factory=lambda: os.getenv("COHERE_API_KEY"))


# Singleton default config (override per environment)
DEFAULT_CONFIG = RAGConfig()
