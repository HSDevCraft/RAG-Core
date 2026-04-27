# Contributing to RAG System

Thank you for your interest in contributing! This document outlines the process
for contributing code, documentation, and bug reports.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Module Guide for Contributors](#module-guide-for-contributors)

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/rag-system.git
   cd rag-system
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/original/rag-system.git
   ```

---

## Development Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install with dev extras
pip install -e ".[dev]"

# Or install dev dependencies directly
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov mypy ruff black

# Copy env template
cp .env.example .env
# Add OPENAI_API_KEY (or use Ollama — free)

# Verify installation
make test-fast
```

---

## Code Standards

### Style
- **Formatter**: `black` (line length 99)
- **Linter**: `ruff` (replaces flake8 + isort)
- **Type checker**: `mypy --strict`

Run all checks at once:
```bash
make lint
```

### Docstrings
Every public class, method, and function **must** have a Google-style docstring:

```python
def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
    """Retrieve relevant chunks for a query.

    Args:
        query: Natural language query string.
        top_k: Maximum number of results to return.

    Returns:
        List of SearchResult sorted by relevance score descending.

    Raises:
        RuntimeError: If the index has not been built yet.

    Example:
        >>> results = retriever.retrieve("What is the refund policy?", top_k=5)
        >>> print(results[0].score)
        0.923
    """
```

### Type Hints
All function signatures must have full type annotations:
```python
# Good
def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray: ...

# Bad
def encode(self, texts, batch_size=64): ...
```

### Logging
Use the module-level `logger`, not `print()`:
```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Detailed trace: %s", variable)
logger.info("High-level event: %d items processed", count)
logger.warning("Non-fatal issue: %s", msg)
logger.error("Recoverable error: %s", exc)
```

---

## Testing

### Test Structure
```
tests/
├── unit/
│   ├── test_document_loader.py
│   ├── test_chunking.py
│   ├── test_embedding.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_prompt_builder.py
│   ├── test_llm_interface.py
│   └── test_rag_pipeline.py
├── integration/
│   └── test_end_to_end.py
└── conftest.py
```

### Running Tests
```bash
make test          # full suite with coverage
make test-fast     # unit tests only (no API calls)
make test-cov      # HTML coverage report
```

### Writing Tests
- Use `pytest` fixtures (see `tests/conftest.py`)
- Mock all external API calls — tests must run without an API key
- Each test must be independent (no shared mutable state)
- Aim for > 80% coverage on all new modules

```python
def test_recursive_chunker_respects_size(sample_documents):
    """Chunks must never exceed chunk_size characters."""
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.split(sample_documents)
    for chunk in chunks:
        assert len(chunk.content) <= 300, f"Chunk too large: {len(chunk.content)}"
```

---

## Submitting Changes

### Branch Naming
```
feature/add-graph-rag
bugfix/fix-ivf-training-crash
docs/update-embedding-guide
refactor/extract-cache-module
```

### Commit Messages (Conventional Commits)
```
feat(retrieval): add ColBERT late-interaction reranker
fix(vector_store): resolve IVF training assertion for small datasets
docs(readme): add Azure OpenAI setup instructions
test(chunking): add semantic chunker edge case tests
refactor(pipeline): extract _embed_and_store as reusable method
perf(embedding): add batch prefetch to reduce GPU idle time
```

### Pull Request Checklist

Before opening a PR, ensure:
- [ ] `make lint` passes (no errors)
- [ ] `make test` passes (all green)
- [ ] New code has docstrings and type hints
- [ ] New features have corresponding tests
- [ ] `ARCHITECTURE.md` updated if design changes
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`
- [ ] No secrets or API keys committed

---

## Module Guide for Contributors

### Adding a new Document Loader
1. Create a class in `data_pipeline/document_loader.py` extending `BaseLoader`
2. Implement `load(self) -> List[Document]`
3. Register in `DirectoryLoader.EXTENSION_MAP` and `DocumentLoader._detect_type`
4. Add a unit test in `tests/unit/test_document_loader.py`

### Adding a new Embedding Provider
1. Create a class in `embeddings/embedding.py` extending `BaseEmbedder`
2. Implement `encode()`, `dimension`, and `model_name`
3. Register in `EmbeddingEngine._PROVIDER_MAP`
4. Add tests mocking the provider's API calls

### Adding a new Vector Store Backend
1. Create a class in `vector_store/vector_store.py` extending `BaseVectorStore`
2. Implement all 5 abstract methods: `add`, `search`, `delete`, `persist`, `load`
3. Register in `VectorStoreFactory.create`
4. Add integration tests (can use EphemeralClient for Chroma)

### Adding a new Prompt Template
Templates live in `generation/prompt_builder.py` as `PromptTemplate` dataclass instances
in the `TEMPLATES` dict. No subclassing required:
```python
TEMPLATES["my_template"] = PromptTemplate(
    name="my_template",
    description="...",
    system_message="...",
    user_template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
)
```

---

## Questions?

Open a [GitHub Discussion](https://github.com/original/rag-system/discussions) or
file an [Issue](https://github.com/original/rag-system/issues/new/choose).
