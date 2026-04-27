# Document Processing — Loading & Chunking Strategies

## Conceptual Foundation

Document processing is the **entry point** of any RAG system. The quality of your document loading and chunking directly determines the ceiling of your RAG performance — even perfect retrieval and generation cannot recover from poorly processed documents.

**Key insight**: Documents contain **hierarchical structure** (sections, paragraphs, sentences) and **semantic boundaries** (topic transitions, logical breaks). Effective chunking preserves these boundaries while creating chunks that are:
1. **Semantically coherent** — each chunk covers one concept/topic
2. **Contextually sufficient** — contains enough context to be understood in isolation  
3. **Size-appropriate** — fits within embedding and LLM context windows

### The Document Processing Pipeline

```
Raw Sources → DocumentLoader → [Document] → ChunkingEngine → [Chunk] → Embeddings
```

**Critical decisions**:
- **Granularity**: Sentence-level vs paragraph-level vs section-level chunks
- **Overlap**: How much context to preserve across chunk boundaries
- **Metadata**: What document structure information to retain
- **Preprocessing**: How much cleaning and normalization to apply

---

## Mathematical Formulation

### Information Preservation

Document processing must balance **information density** vs **context completeness**:

```
Chunk Quality = λ × Coherence(chunk) + (1-λ) × Completeness(chunk)
```

Where:
- **Coherence**: semantic unity within chunk
- **Completeness**: sufficient context for understanding
- **λ**: balance parameter (domain-dependent)

### Overlap Optimization

**Sliding window chunking** with overlap `o`:
```
Chunk_i = Text[i×(s-o) : i×(s-o)+s]
```
Where `s` = chunk_size, `o` = overlap

**Information redundancy**:
```
Redundancy = o/s
```
- `o/s = 0`: No overlap (risk of boundary information loss)
- `o/s = 0.5`: 50% overlap (high redundancy, better recall)
- `o/s → 1`: Nearly complete overlap (inefficient storage)

### Semantic Boundary Detection

For **semantic chunking**, boundary score between sentences `i` and `i+1`:
```
Boundary_Score(i) = 1 - cosine_similarity(embed(sent_i), embed(sent_{i+1}))
```

Chunk boundaries occur where `Boundary_Score(i) > threshold`.

---

## Implementation Details

Your system (`data_pipeline/`) implements four complementary document loaders and chunking strategies.

### Document Loading

#### TextLoader
```python
class TextLoader:
    def load(self, file_path: str) -> Document:
        """
        Handles: .txt, .md, .py, .js, .json, .csv, .log
        Encoding: UTF-8 with fallback to latin1
        Preprocessing: Normalize line endings, strip BOM
        """
```

**Implementation highlights**:
```python
# From document_loader.py
def _safe_read_text(self, file_path: str) -> str:
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return self._normalize_text(content)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {file_path}")

def _normalize_text(self, text: str) -> str:
    """Remove BOM, normalize line endings, clean whitespace"""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Collapse multiple newlines
    return text.strip()
```

#### HTMLLoader  
```python
class HTMLLoader:
    def load(self, file_path: str) -> Document:
        """
        Uses BeautifulSoup for parsing
        Extracts: title, headings, body text, links
        Preserves: semantic structure (h1, h2, p tags)
        Filters: scripts, styles, navigation elements
        """
```

**Structure preservation**:
```python
def _extract_structured_text(self, soup) -> str:
    """
    Maintains document hierarchy:
    # Title (from <title> or <h1>)
    ## Section (from <h2>)  
    ### Subsection (from <h3>)
    
    Body paragraphs with preserved spacing
    """
    sections = []
    current_section = []
    
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li']):
        if element.name in ['h1', 'h2', 'h3', 'h4']:
            # Start new section
            if current_section:
                sections.append('\n'.join(current_section))
            level = int(element.name[1])
            prefix = '#' * level
            current_section = [f"{prefix} {element.get_text().strip()}"]
        else:
            current_section.append(element.get_text().strip())
    
    return '\n\n'.join(sections)
```

#### JSONLoader
```python
class JSONLoader:
    def load(self, file_path: str) -> Document:
        """
        Handles: .json, .jsonl files
        Strategy: Flatten nested objects to key-value text
        Preservation: Maintains data relationships via path notation
        """
```

**Flattening algorithm**:
```python
def _flatten_json(self, obj: Any, parent_key: str = '') -> str:
    """
    Convert nested JSON to readable text format:
    
    {"user": {"name": "John", "age": 30}} 
    →
    "user.name: John\nuser.age: 30"
    """
    items = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                items.append(self._flatten_json(value, new_key))
            else:
                items.append(f"{new_key}: {value}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_key = f"{parent_key}[{i}]"
            items.append(self._flatten_json(item, new_key))
    
    return '\n'.join(items)
```

#### DirectoryLoader
```python
class DirectoryLoader:
    def load_directory(self, dir_path: str, recursive: bool = True, 
                      max_files: int = 1000) -> List[Document]:
        """
        Traversal: BFS with configurable depth limit
        Filtering: Skip binary files, respect .gitignore patterns
        Batching: Process files in chunks to manage memory
        """
```

### Chunking Strategies

#### Recursive Character Splitting
```python
class RecursiveChunker(ChunkingEngine):
    """
    Strategy: Try splitting on major delimiters first, fallback to minor ones
    Priority: ['\n\n', '\n', '. ', ' ', '']
    Goal: Preserve natural text boundaries
    """
```

**Algorithm**:
```python
def split(self, documents: List[Document]) -> List[Chunk]:
    separators = ['\n\n', '\n', '. ', ' ', '']  # Priority order
    
    for doc in documents:
        chunks = self._recursive_split(doc.content, separators, 0)
        # Post-process: merge small chunks, split oversized chunks
        chunks = self._merge_small_chunks(chunks, min_size=50)
        chunks = self._handle_oversized(chunks, max_size=self.chunk_size * 1.5)
```

**Why recursive?** Preserves document structure:
1. Split on double newlines (paragraph boundaries) 
2. If chunk still too large → split on single newlines (sentence boundaries)
3. If still too large → split on periods (clause boundaries)
4. Final fallback → character splitting

#### Token-Based Splitting
```python
class TokenChunker(ChunkingEngine):
    """
    Strategy: Split based on tokenizer boundaries (subword level)
    Advantage: Precise control over LLM input size
    Use case: When exact token count matters (context window limits)
    """
```

**Implementation**:
```python
def _count_tokens(self, text: str) -> int:
    """Use tiktoken for GPT models, approximate for others"""
    if self.model_name.startswith('gpt'):
        import tiktoken
        enc = tiktoken.encoding_for_model(self.model_name)
        return len(enc.encode(text))
    else:
        # Approximate: 1 token ≈ 4 characters for most languages
        return len(text) // 4

def split_by_tokens(self, text: str) -> List[str]:
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = self._count_tokens(sentence)
        if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    return chunks
```

#### Sentence-Based Splitting
```python
class SentenceChunker(ChunkingEngine):
    """
    Strategy: Respect sentence boundaries using NLP parsing
    Library: spaCy or NLTK for robust sentence segmentation
    Advantage: Linguistically motivated, preserves complete thoughts
    """
```

**Sentence segmentation**:
```python
def _segment_sentences(self, text: str) -> List[str]:
    """Use spaCy for accurate sentence boundaries"""
    import spacy
    
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    nlp.add_pipe('sentencizer')  # Lightweight sentence segmentation
    
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def _group_sentences(self, sentences: List[str]) -> List[str]:
    """Group sentences into chunks of appropriate size"""
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Add overlap: include last sentence from previous chunk
        if chunks and len(current_chunk) == 0:
            overlap_sentence = self._get_last_sentence(chunks[-1])
            current_chunk.append(overlap_sentence)
            current_length = len(overlap_sentence)
        
        if current_length + len(sentence) > self.chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    return chunks
```

#### Semantic Chunking (Advanced)
```python
class SemanticChunker(ChunkingEngine):  
    """
    Strategy: Split based on semantic coherence using embeddings
    Algorithm: Compute sentence embeddings, detect topic shifts
    Advantage: Contextually meaningful chunks
    Cost: Requires embedding computation during preprocessing
    """
```

**Semantic boundary detection**:
```python
def _detect_semantic_boundaries(self, sentences: List[str]) -> List[int]:
    """Find topic shift points using embedding similarity"""
    
    # Embed all sentences
    embeddings = self.embedder.encode(sentences)
    
    # Compute pairwise similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)
    
    # Find local minima (topic boundaries)
    boundaries = []
    threshold = np.percentile(similarities, 25)  # Bottom quartile
    
    for i, sim in enumerate(similarities):
        if sim < threshold:
            # Check if it's a local minimum
            left_ok = (i == 0) or (similarities[i-1] > sim)
            right_ok = (i == len(similarities)-1) or (similarities[i+1] > sim)
            if left_ok and right_ok:
                boundaries.append(i + 1)  # Boundary after sentence i
    
    return boundaries
```

---

## Comparative Analysis

### Chunking Strategy Selection

| Strategy | Best For | Pros | Cons |
|---|---|---|---|
| **Recursive** | General text, markdown | Preserves structure | Not token-aware |
| **Token** | Strict context limits | Precise size control | May break mid-sentence |
| **Sentence** | Narrative text | Linguistic boundaries | Variable chunk sizes |
| **Semantic** | Technical docs | Topically coherent | Computationally expensive |

### Chunk Size Guidelines

**By content type**:
```python
# Technical documentation
chunk_size = 256     # Dense information, precise retrieval needed
overlap = 32         # 12.5% overlap

# Narrative text (books, articles)  
chunk_size = 1024    # More context needed for understanding
overlap = 128        # 12.5% overlap

# Code documentation
chunk_size = 512     # Balance between function-level and file-level context
overlap = 64         # Preserve cross-function relationships

# FAQ/Q&A content
chunk_size = 256     # Each Q&A pair should be self-contained
overlap = 0          # No overlap needed for discrete Q&A
```

**By LLM context window**:
```python
# GPT-4 (128K tokens) - can handle larger chunks
max_chunk_tokens = 2048

# Llama 3.1 8B (8K tokens) - need smaller chunks  
max_chunk_tokens = 512

# Claude 3 (200K tokens) - very large chunks possible
max_chunk_tokens = 4096
```

### Processing Trade-offs

**Preprocessing intensity**:

**Minimal** (fast indexing, some noise):
```python
# Just basic normalization
text = text.strip().replace('\r\n', '\n')
```

**Moderate** (balanced):
```python  
# Clean whitespace, normalize unicode
text = unicodedata.normalize('NFKC', text)
text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean paragraph breaks
```

**Intensive** (clean text, slow):
```python
# Full text cleaning pipeline
text = self._remove_headers_footers(text)
text = self._fix_encoding_issues(text)  
text = self._normalize_punctuation(text)
text = self._remove_excessive_whitespace(text)
text = self._fix_hyphenation(text)  # "word-\nbreak" → "wordbreak"
```

---

## Practical Guidelines

### Chunk Size Tuning

**Start with these defaults**:
```python
DOMAIN_DEFAULTS = {
    'technical_docs': {'chunk_size': 512, 'overlap': 64, 'strategy': 'recursive'},
    'narrative_text': {'chunk_size': 1024, 'overlap': 128, 'strategy': 'sentence'}, 
    'structured_data': {'chunk_size': 256, 'overlap': 32, 'strategy': 'token'},
    'code_repos': {'chunk_size': 512, 'overlap': 64, 'strategy': 'recursive'},
}
```

**Optimization process**:
1. **Start with domain defaults**
2. **Measure retrieval@10 performance** on eval set
3. **Binary search on chunk_size**: [256, 512, 1024, 2048]
4. **Tune overlap**: [0, 0.125, 0.25] × chunk_size  
5. **Try alternative strategies** if performance plateau

### Quality Assessment

**Chunk quality metrics**:
```python
def assess_chunk_quality(chunks: List[Chunk]) -> Dict[str, float]:
    return {
        'avg_length': np.mean([len(c.content) for c in chunks]),
        'length_std': np.std([len(c.content) for c in chunks]), 
        'completeness': measure_sentence_completeness(chunks),
        'redundancy': measure_overlap_redundancy(chunks),
        'boundary_quality': measure_semantic_boundaries(chunks)
    }
```

**Boundary completeness**:
```python
def measure_sentence_completeness(chunks: List[Chunk]) -> float:
    """Fraction of chunks that end with complete sentences"""
    complete = 0
    for chunk in chunks:
        text = chunk.content.strip()
        if text.endswith(('.', '!', '?', ':', ';')):
            complete += 1
    return complete / len(chunks)
```

### Memory and Performance

**Streaming processing** for large documents:
```python
def process_large_file(file_path: str, chunk_size_mb: int = 10) -> Iterator[Document]:
    """Process files larger than memory in streaming fashion"""
    with open(file_path, 'r') as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size_mb * 1024 * 1024)
            if not chunk:
                break
                
            buffer += chunk
            # Split on paragraph boundaries
            paragraphs = buffer.split('\n\n')
            
            # Yield complete paragraphs, keep incomplete in buffer
            for paragraph in paragraphs[:-1]:
                yield Document(content=paragraph, metadata={...})
            buffer = paragraphs[-1]
```

**Parallel processing**:
```python
from concurrent.futures import ProcessPoolExecutor

def parallel_chunk_processing(documents: List[Document], n_workers: int = 4):
    """Process chunking across multiple cores"""
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        chunk_groups = executor.map(chunker.split, 
                                  [documents[i::n_workers] for i in range(n_workers)])
    
    # Flatten results
    all_chunks = []
    for group in chunk_groups:
        all_chunks.extend(group)
    return all_chunks
```

### Common Issues & Solutions

**Issue**: Chunks cut off mid-sentence
```python
# Solution: Sentence-aware splitting
def ensure_sentence_boundary(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    
    # Find last sentence boundary before limit
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    last_question = truncated.rfind('?')
    last_exclamation = truncated.rfind('!')
    
    boundary = max(last_period, last_question, last_exclamation)
    if boundary > max_length * 0.5:  # Don't truncate too much
        return text[:boundary + 1]
    else:
        return truncated  # Accept mid-sentence if necessary
```

**Issue**: Important context split across chunks
```python
# Solution: Sliding window with strategic overlap
def create_overlapping_chunks(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end]
        
        # Extend to sentence boundary if we're not at end
        if end < len(text):
            next_period = text.find('.', end)
            if next_period != -1 and next_period - end < 50:  # Reasonable extension
                end = next_period + 1
                chunk_text = text[start:end]
        
        chunks.append(chunk_text)
        start = end - overlap  # Overlap with next chunk
        
        if start >= len(text):
            break
    
    return chunks
```

**Issue**: Metadata loss during chunking
```python
# Solution: Propagate and enrich metadata
def create_chunk_with_metadata(parent_doc: Document, chunk_text: str, 
                              chunk_index: int, start_pos: int) -> Chunk:
    metadata = parent_doc.metadata.copy()
    metadata.update({
        'chunk_index': chunk_index,
        'start_position': start_pos,
        'chunk_size': len(chunk_text),
        'parent_doc_id': parent_doc.doc_id,
        'estimated_reading_time': len(chunk_text.split()) / 200  # words per minute
    })
    
    return Chunk(
        chunk_id=generate_chunk_id(parent_doc.doc_id, chunk_index),
        doc_id=parent_doc.doc_id,
        content=chunk_text,
        metadata=metadata,
        chunk_index=chunk_index
    )
```

**Next concept**: Embedding Systems — how text becomes vectors and similarity computation
