# Changelog

All notable changes to the RAG System are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `pyproject.toml` with full tool configuration (ruff, black, mypy, pytest)
- `Makefile` with developer convenience targets
- `CONTRIBUTING.md` with code standards and module guide
- GitHub issue templates (bug report, feature request)
- Pull request template with checklist
- `monitoring/prometheus.yml` for Prometheus scraping config
- `utils/cache.py` — standalone Redis + disk cache utility
- Complete test suite (`tests/unit/` + `tests/integration/`)
- `CHANGELOG.md` (this file)

### Fixed
- `rag_pipeline.index_documents()` — removed premature `return` that made the actual implementation unreachable dead code

---

## [1.0.0] — 2024-01-01

### Added
- Full end-to-end RAG pipeline (`rag_pipeline.py`)
- **Data ingestion**: PDF, HTML, DOCX, JSON/JSONL, plain text, Markdown, SQL, URL loaders
- **Chunking**: Recursive (default), Token-based (tiktoken), Sentence-based (NLTK), Semantic (cosine-breakpoint)
- **Embeddings**: SentenceTransformers (local), OpenAI text-embedding-3, Cohere embed-v3; disk-cache wrapper
- **Vector stores**: FAISS (Flat/IVF/HNSW), ChromaDB (embedded + client-server), Pinecone (managed)
- **Retrieval**: Dense ANN, BM25 sparse, Hybrid RRF fusion, HyDE, MultiQuery, StepBack, MMR
- **Reranking**: CrossEncoder (ms-marco-MiniLM), Cohere Rerank API, LLM-as-judge reranker
- **Generation**: 7 prompt templates, token-budget context injection, citation formatting, conversation history
- **LLM providers**: OpenAI, Azure OpenAI, Ollama (local), Anthropic Claude, HuggingFace Transformers
- **Advanced RAG**: Multi-hop retrieval, Agentic RAG (ReAct), streaming, memory-augmented, tool-augmented
- **Evaluation**: LLM-judge (faithfulness, relevancy, precision), ROUGE-L, token F1, RAGAS integration, benchmark loader
- **FastAPI service**: Auth, SSE streaming, prompt injection defense, PII scrubbing, Prometheus metrics
- **6 Jupyter notebooks**: Ingestion → Embedding → Vector DB → Retrieval → Pipeline → Evaluation
- **Docker**: Multi-stage Dockerfile + docker-compose with Redis, ChromaDB, Prometheus, Grafana
- `ARCHITECTURE.md`: Full design decisions, ASCII diagrams, failure mode matrix
- `README.md`: Quick start, API reference, performance benchmarks
