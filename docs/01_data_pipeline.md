# Module 01 — Data Pipeline
### `data_pipeline/document_loader.py` · `data_pipeline/chunking.py`

---

## Table of Contents
1. [Why a Data Pipeline?](#1-why-a-data-pipeline)
2. [Document Model](#2-document-model)
3. [Document Loaders](#3-document-loaders)
4. [Chunking Strategies](#4-chunking-strategies)
5. [Chunk Model](#5-chunk-model)
6. [ChunkingEngine Factory](#6-chunkingengine-factory)
7. [Data Flow Diagram](#7-data-flow-diagram)
8. [Quick Reference](#8-quick-reference)
9. [Common Pitfalls](#9-common-pitfalls)

---

## 1. Why a Data Pipeline?

LLMs have a fixed context window (4K–128K tokens). You cannot stuff an entire
knowledge base into every prompt. The data pipeline solves this by:

1. **Loading** — reading raw files from any source into a canonical format
2. **Chunking** — splitting documents into small, semantically coherent pieces
3. **Preparing** — attaching metadata so every chunk is traceable to its source

```
Raw files (PDF, HTML, DOCX, JSON …)
        │
        ▼
  DocumentLoader          ← "read any format"
        │
        ▼
    [Document]             ← canonical model: content + metadata
        │
        ▼
  ChunkingEngine          ← "split into pieces the LLM can use"
        │
        ▼
     [Chunk]              ← text + metadata + chunk_id + doc_id
```

---

## 2. Document Model

```python
# data_pipeline/document_loader.py
@dataclass
class Document:
    content:  str               # full text of the document
    metadata: Dict[str, Any]    # source, file_type, author, ingested_at, …
    doc_id:   str = ""          # SHA-256[:16] of content+metadata (auto)

    def __len__(self) -> int:           # len(doc) → character count
    def word_count(self) -> int:        # doc.word_count → token proxy
```

**ID generation** — deterministic hash so re-indexing the same file produces the
same `doc_id`. Enables de-duplication before embedding.

```python
doc = Document(content="Hello world", metadata={"source": "readme.txt"})
print(doc.doc_id)       # → "a3f1b2c9d4e57812"  (16-char hex)
print(len(doc))         # → 11
print(doc.word_count)   # → 2
```

---

## 3. Document Loaders

### 3.1 Class Hierarchy

```
BaseLoader (ABC)
├── TextLoader          ← .txt, .md, .rst
├── HTMLLoader          ← .html files OR any http/https URL
├── PDFLoader           ← .pdf via pypdf
├── DocxLoader          ← .docx via python-docx
├── JSONLoader          ← .json / .jsonl (pick content_key at runtime)
├── SQLLoader           ← SQL query → rows as documents
└── DirectoryLoader     ← recurse a directory; auto-dispatches by extension
```

### 3.2 TextLoader

```python
from data_pipeline import TextLoader

loader = TextLoader("./docs/policy.txt")
docs = loader.load()           # synchronous
docs = await loader.aload()    # async

print(docs[0].content)         # full file text (stripped)
print(docs[0].metadata)
# {'source': './docs/policy.txt', 'file_type': 'txt',
#  'file_size': 2048, 'ingested_at': 1700000000.0}
```

### 3.3 HTMLLoader — File or URL

```python
# From file
loader = HTMLLoader("./faq.html")

# From URL (uses requests.get internally)
loader = HTMLLoader("https://docs.example.com/api")
docs = loader.load()
# nav/header/footer/script/style tags are stripped automatically
```

### 3.4 JSONLoader — Flexible Schema

```python
loader = JSONLoader(
    "products.jsonl",
    content_key="description",        # which field becomes Document.content
    metadata_keys=["id", "category"],  # which fields go into metadata
)
docs = loader.load()   # one Document per JSON record
```

**Supports**: `.json` (object or array) + `.jsonl` (newline-delimited).
Records missing `content_key` are skipped with a warning.

### 3.5 DirectoryLoader — Auto-dispatch

```python
from data_pipeline import DirectoryLoader

loader = DirectoryLoader(
    "./knowledge_base/",
    recursive=True,           # descend into subdirectories
    max_files=500,            # safety cap
    exclude_patterns=["draft_*", "*.tmp"],
)
docs = loader.load()
# Returns all .txt, .md, .html, .pdf, .docx, .json, .jsonl files
# Each file is dispatched to the correct loader automatically
```

**Extension → Loader map**:
| Extension | Loader |
|---|---|
| `.txt`, `.md`, `.rst` | TextLoader |
| `.html`, `.htm` | HTMLLoader |
| `.pdf` | PDFLoader |
| `.docx` | DocxLoader |
| `.json`, `.jsonl` | JSONLoader |
| `http://`, `https://` | HTMLLoader |

### 3.6 DocumentLoader Façade (Recommended)

```python
from data_pipeline import DocumentLoader

loader = DocumentLoader()

# Auto-detect by extension / URL prefix
docs = loader.load("./policy.txt")
docs = loader.load("https://example.com/faq")
docs = loader.load("./knowledge_base/")   # → DirectoryLoader

# Override detection
docs = loader.load("./data.bin", loader_type="txt")

# Async
docs = await loader.aload("./reports/")
```

---

## 4. Chunking Strategies

### 4.1 Why Chunk?

| Problem | Solution |
|---|---|
| Document > context window | Split into smaller pieces |
| Irrelevant text dilutes the answer | Target only the relevant chunk |
| Embedding quality degrades on long text | Short chunks embed more accurately |

### 4.2 Strategy Comparison

| Strategy | Algorithm | Best for | Weakness |
|---|---|---|---|
| **Recursive** | Split on `\n\n` → `\n` → ` ` → char | General purpose (default) | May cut mid-sentence |
| **Token** | tiktoken BPE tokenizer | Precise token budgets for GPT | Slower; requires tiktoken |
| **Sentence** | NLTK sentence tokenizer | Conversational / dialogue text | Sentences can vary hugely in length |
| **Semantic** | Cosine similarity breakpoints | Scientific / technical documents | Requires embedding; expensive |

### 4.3 RecursiveChunker

The **default** strategy. Mimics how a human would split text: try paragraph
breaks first, then line breaks, then spaces, then characters as a last resort.

```python
from data_pipeline import RecursiveChunker

chunker = RecursiveChunker(
    chunk_size=512,          # target size in characters
    chunk_overlap=64,        # overlap to preserve context across boundaries
    separators=["\n\n", "\n", ". ", " ", ""],   # priority order
)
chunks = chunker.split(docs)
```

**How overlap works**:
```
Chunk 1: [═══════════════════════▓▓▓▓]
Chunk 2:                    [▓▓▓▓═══════════════════════]
                             ^^^^
                           64-char overlap
```
Overlap ensures a sentence split across boundaries still has both halves
findable via retrieval.

### 4.4 TokenChunker

```python
from data_pipeline import TokenChunker

chunker = TokenChunker(
    chunk_size=256,       # tokens (not characters)
    chunk_overlap=32,     # tokens
    encoding="cl100k_base",   # GPT-4 / GPT-3.5 encoding
)
```

Use when you need **exact token control** — e.g., fitting chunks into a 4K
context window with a fixed budget for system prompt + question + answer.

### 4.5 SentenceChunker

```python
from data_pipeline import SentenceChunker

chunker = SentenceChunker(
    chunk_size=1000,              # max characters per chunk
    chunk_overlap_sentences=2,   # sentences shared between chunks
)
```

Best for **FAQ documents** and **customer service transcripts** where each
sentence is a complete thought.

### 4.6 SemanticChunker (advanced)

Embeds each sentence and splits wherever cosine similarity between adjacent
sentences drops below a threshold — **natural topic boundaries**.

```python
chunker = ChunkingEngine(
    strategy="semantic",
    embedding_fn=embedder.encode_query,   # any callable: str → np.ndarray
    breakpoint_threshold=0.85,            # split if similarity < this
)
```

> ⚠️ **Cost**: requires one embedding call per sentence. Only use on documents
> where topic structure matters more than speed.

---

## 5. Chunk Model

```python
@dataclass
class Chunk:
    content:     str                # the text slice
    metadata:    Dict[str, Any]     # inherited from parent Document + chunk-level keys
    chunk_id:    str = ""           # SHA-256[:16] of content+metadata
    doc_id:      str = ""           # links back to parent Document.doc_id
    chunk_index: int = 0            # position within the document (0-based)
    token_count: int = 0            # approximate token count (chars / 4)
```

**Metadata inheritance**: every chunk automatically gets the parent document's
metadata (`source`, `file_type`, `author`, …) plus:

```python
chunk.metadata == {
    "source":      "policy.pdf",
    "file_type":   "pdf",
    "page":        3,
    "chunk_index": 7,
    "doc_id":      "a3f1b2c9d4e57812",
    "ingested_at": 1700000000.0,
}
```

---

## 6. ChunkingEngine Factory

```python
from data_pipeline import ChunkingEngine

# Simple usage: strategy name string
engine = ChunkingEngine(
    strategy="recursive",   # "recursive" | "token" | "sentence" | "semantic"
    chunk_size=512,
    chunk_overlap=64,
)

# Split a list of Documents
chunks: List[Chunk] = engine.split(docs)

# Split a single Document
chunks: List[Chunk] = engine.split_one(doc)

# Introspection
print(engine.strategy_name)   # "recursive"
```

---

## 7. Data Flow Diagram

```
                  ┌─────────────────────────┐
                  │     Raw Input Sources    │
                  │  PDF  HTML  DOCX  JSON   │
                  │  TXT   MD   URL   SQL    │
                  └───────────┬─────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │    DocumentLoader     │
                  │  (auto-detect type)   │
                  │                       │
                  │  TextLoader           │
                  │  HTMLLoader ──────────┤──▶ requests.get (URL)
                  │  PDFLoader ───────────┤──▶ pypdf
                  │  JSONLoader           │
                  │  DirectoryLoader      │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │     [Document]        │
                  │  content: str         │
                  │  metadata: dict       │
                  │  doc_id: str          │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │    ChunkingEngine     │
                  │                       │
                  │  RecursiveChunker     │◀── default
                  │  TokenChunker         │◀── GPT token budget
                  │  SentenceChunker      │◀── dialogue
                  │  SemanticChunker      │◀── scientific
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │       [Chunk]         │
                  │  content: str         │
                  │  chunk_id: str        │
                  │  doc_id: str          │
                  │  chunk_index: int     │
                  │  metadata: dict       │
                  └───────────────────────┘
                        │         │
                        ▼         ▼
                 EmbeddingEngine  BM25Index
```

---

## 8. Quick Reference

```python
# ── One-liner: load a directory and chunk everything ─────────────────────
from data_pipeline import DocumentLoader, ChunkingEngine

docs   = DocumentLoader().load("./knowledge_base/")
chunks = ChunkingEngine(strategy="recursive", chunk_size=512).split(docs)

# ── Access chunk data ────────────────────────────────────────────────────
for chunk in chunks:
    print(chunk.chunk_id)     # unique ID for this chunk
    print(chunk.doc_id)       # parent document ID
    print(chunk.chunk_index)  # position (0, 1, 2, …)
    print(chunk.content)      # text
    print(chunk.token_count)  # approximate token count
    print(chunk.metadata["source"])   # "policy.pdf"

# ── Choosing chunk_size ──────────────────────────────────────────────────
# Rule of thumb:
#   chunk_size  = ~20% of your LLM's context window
#   chunk_overlap = ~10-15% of chunk_size
#
# GPT-4o (128K context):   chunk_size=1024, overlap=128
# GPT-4o-mini (128K):      chunk_size=512,  overlap=64
# Llama 3.1 8B (8K):       chunk_size=256,  overlap=32
# Claude 3 (200K):         chunk_size=2048, overlap=256

# ── Async loading (for FastAPI / async services) ─────────────────────────
import asyncio
docs = asyncio.run(DocumentLoader().aload("./docs/"))
```

---

## 9. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| **chunk_size too small** | Answers lack context; citations useless | Increase to ≥ 300 chars |
| **chunk_size too large** | LLM ignores most of context; slow retrieval | Decrease; use token chunker |
| **No overlap** | Answer cut exactly at chunk boundary is missed | Set overlap ≥ 10% of size |
| **HTML not stripped** | Chunks full of `<div>`, `style=` garbage | HTMLLoader strips automatically |
| **PDF with scanned images** | Empty content after loading | Add OCR pre-processing step |
| **JSON wrong content_key** | All chunks empty → zero index | Check field name in your JSON |
| **Large directory, no max_files** | OOM or hours-long indexing | Set `max_files=1000` |
| **Re-indexing same file twice** | Duplicate chunks inflate results | Content-hash dedup via `doc_id` |
