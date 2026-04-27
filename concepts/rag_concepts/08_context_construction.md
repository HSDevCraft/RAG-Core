# Context Construction — Prompt Building & Citation Systems

## Conceptual Foundation

Context construction is the **critical bridge** between retrieval and generation in RAG systems. It transforms retrieved chunks into a coherent, token-aware context that maximizes the LLM's ability to generate accurate, grounded responses.

**Key insight**: Raw concatenation of retrieved chunks often fails due to information overload, redundancy, and token budget violations. Effective context construction requires **intelligent assembly**, **citation management**, and **token optimization**.

### The Context Assembly Challenge

```
Raw approach: chunk1 + chunk2 + chunk3 + ... → Token overflow, poor structure
Smart approach: 
  1. Deduplicate overlapping information
  2. Organize by relevance hierarchy  
  3. Add citation markers
  4. Respect token budget
  5. Preserve logical flow
```

**Critical trade-offs**:
- **Completeness vs Conciseness**: Include all relevant info vs stay within token limits
- **Accuracy vs Readability**: Exact quotes vs paraphrased summaries  
- **Citation Granularity**: Sentence-level vs chunk-level attribution

---

## Mathematical Formulation

### Token Budget Optimization

Given:
- Retrieved chunks: `C = {c₁, c₂, ..., cₙ}`
- Token limits: `T_max` (total), `T_query`, `T_template`
- Available for context: `T_context = T_max - T_query - T_template`

**Optimization problem**:
```
maximize: Σᵢ relevance(cᵢ) × xᵢ
subject to: Σᵢ tokens(cᵢ) × xᵢ ≤ T_context
           xᵢ ∈ {0, 1}  (binary: include chunk or not)
```

**Greedy approximation**:
```
efficiency(cᵢ) = relevance(cᵢ) / tokens(cᵢ)
Sort by efficiency, include while budget allows
```

### Citation Density

**Citation coverage ratio**:
```
citation_coverage = cited_tokens / total_context_tokens
```

Optimal range: 0.15-0.30 (15-30% of content should have citations)

**Citation granularity score**:
```
granularity = sentence_citations / total_sentences
```
Higher granularity = more precise attribution, but more tokens used.

---

## Implementation Details

Your system (`generation/prompt_builder.py`) implements sophisticated context construction with multiple citation styles and token management.

### ContextBuilder Core

```python
class ContextBuilder:
    def __init__(self, max_tokens: int = 3000, citation_style: str = "inline",
                 order: str = "relevance", include_metadata: bool = True,
                 metadata_fields: List[str] = None):
        """
        Intelligent context assembly with citation management
        
        Args:
            max_tokens: Hard limit on context length
            citation_style: "inline"|"footnote"|"none"
            order: "relevance"|"reverse_rank"|"source"|"chronological"
            include_metadata: Whether to include source information
            metadata_fields: Specific metadata to include
        """
        self.max_tokens = max_tokens
        self.citation_style = citation_style
        self.order = order
        self.include_metadata = include_metadata
        self.metadata_fields = metadata_fields or ["source", "page"]
        
    def build(self, search_results: List[SearchResult]) -> Tuple[str, List[Dict]]:
        """
        Build context string with citations from search results
        
        Returns:
            context_str: Formatted context with citations
            citations: List of citation metadata
        """
        
        if not search_results:
            return "", []
        
        # Step 1: Order chunks according to strategy
        ordered_chunks = self._order_chunks(search_results)
        
        # Step 2: Deduplicate similar content
        deduped_chunks = self._deduplicate_chunks(ordered_chunks)
        
        # Step 3: Apply token budget with intelligent truncation
        budget_chunks = self._apply_token_budget(deduped_chunks)
        
        # Step 4: Build context with citations
        context_str, citations = self._assemble_context(budget_chunks)
        
        return context_str, citations
    
    def _order_chunks(self, results: List[SearchResult]) -> List[SearchResult]:
        """Order chunks according to specified strategy"""
        
        if self.order == "relevance":
            # Sort by relevance score (highest first)
            return sorted(results, key=lambda x: x.score, reverse=True)
            
        elif self.order == "reverse_rank":
            # Sort by retrieval rank (lowest rank = highest relevance first)
            return sorted(results, key=lambda x: x.rank)
            
        elif self.order == "source":
            # Group by source, then by relevance within source
            grouped = defaultdict(list)
            for result in results:
                source = result.metadata.get("source", "unknown")
                grouped[source].append(result)
            
            ordered = []
            for source_results in grouped.values():
                source_results.sort(key=lambda x: x.score, reverse=True)
                ordered.extend(source_results)
            return ordered
            
        elif self.order == "chronological":
            # Sort by timestamp if available, fallback to relevance
            return sorted(results, key=lambda x: (
                x.metadata.get("timestamp", "1970-01-01"),
                -x.score  # Secondary sort by relevance (descending)
            ))
        
        return results  # No ordering
    
    def _deduplicate_chunks(self, chunks: List[SearchResult]) -> List[SearchResult]:
        """Remove chunks with excessive content overlap"""
        
        if len(chunks) <= 1:
            return chunks
        
        # Compute text similarity between all pairs
        texts = [chunk.content for chunk in chunks]
        
        # Simple token-based similarity (more sophisticated: use embeddings)
        def jaccard_similarity(text1: str, text2: str) -> float:
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            return intersection / union if union > 0 else 0
        
        # Keep track of chunks to remove
        to_remove = set()
        similarity_threshold = 0.7  # 70% token overlap = too similar
        
        for i in range(len(chunks)):
            if i in to_remove:
                continue
                
            for j in range(i + 1, len(chunks)):
                if j in to_remove:
                    continue
                    
                similarity = jaccard_similarity(texts[i], texts[j])
                
                if similarity > similarity_threshold:
                    # Remove the one with lower relevance score
                    if chunks[i].score >= chunks[j].score:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break  # Skip checking further for chunk i
        
        # Return chunks not marked for removal
        return [chunk for i, chunk in enumerate(chunks) if i not in to_remove]
    
    def _apply_token_budget(self, chunks: List[SearchResult]) -> List[SearchResult]:
        """Select chunks within token budget using greedy algorithm"""
        
        # Estimate tokens for each chunk (rough approximation: chars/4)
        chunk_tokens = [len(chunk.content) // 4 for chunk in chunks]
        
        # Greedy selection by relevance/token efficiency
        selected_chunks = []
        remaining_budget = self.max_tokens
        
        # Calculate efficiency scores
        chunk_efficiency = []
        for i, chunk in enumerate(chunks):
            tokens = chunk_tokens[i]
            if tokens > 0:
                efficiency = chunk.score / tokens
                chunk_efficiency.append((i, efficiency, tokens))
        
        # Sort by efficiency (relevance per token)
        chunk_efficiency.sort(key=lambda x: x[1], reverse=True)
        
        # Greedily select chunks
        for idx, efficiency, tokens in chunk_efficiency:
            if tokens <= remaining_budget:
                selected_chunks.append(chunks[idx])
                remaining_budget -= tokens
            elif remaining_budget > 100:  # If significant budget left, try truncation
                truncated_chunk = self._truncate_chunk(chunks[idx], remaining_budget)
                if truncated_chunk:
                    selected_chunks.append(truncated_chunk)
                break  # Budget exhausted
        
        return selected_chunks
    
    def _truncate_chunk(self, chunk: SearchResult, max_tokens: int) -> Optional[SearchResult]:
        """Intelligently truncate chunk to fit token budget"""
        
        max_chars = max_tokens * 4  # Rough token-to-char conversion
        
        if len(chunk.content) <= max_chars:
            return chunk
        
        # Try to truncate at sentence boundaries
        sentences = chunk.content.split('. ')
        truncated_content = ""
        
        for sentence in sentences:
            potential_content = truncated_content + sentence + ". "
            if len(potential_content) <= max_chars:
                truncated_content = potential_content
            else:
                break
        
        if len(truncated_content.strip()) < 50:  # Too short after truncation
            return None
        
        # Create truncated chunk
        truncated_chunk = SearchResult(
            chunk_id=chunk.chunk_id,
            content=truncated_content.strip() + " [...]",  # Indicate truncation
            score=chunk.score * 0.9,  # Slightly penalize truncated content
            metadata=chunk.metadata,
            rank=chunk.rank
        )
        
        return truncated_chunk
```

### Citation System Implementation

```python
def _assemble_context(self, chunks: List[SearchResult]) -> Tuple[str, List[Dict]]:
    """Assemble final context with citation system"""
    
    if self.citation_style == "none":
        return self._assemble_without_citations(chunks)
    elif self.citation_style == "inline":
        return self._assemble_with_inline_citations(chunks)
    elif self.citation_style == "footnote":
        return self._assemble_with_footnote_citations(chunks)
    
def _assemble_with_inline_citations(self, chunks: List[SearchResult]) -> Tuple[str, List[Dict]]:
    """Assemble context with inline [1], [2], [3] style citations"""
    
    context_parts = []
    citations = []
    
    for i, chunk in enumerate(chunks):
        citation_ref = i + 1
        
        # Create citation metadata
        citation = {
            "ref": citation_ref,
            "chunk_id": chunk.chunk_id,
            "score": chunk.score,
            "rank": chunk.rank
        }
        
        # Add metadata fields
        for field in self.metadata_fields:
            if field in chunk.metadata:
                citation[field] = chunk.metadata[field]
        
        citations.append(citation)
        
        # Add content with citation marker
        if self.include_metadata and "source" in chunk.metadata:
            source_info = f" (Source: {chunk.metadata['source']}"
            if "page" in chunk.metadata:
                source_info += f", Page {chunk.metadata['page']}"
            source_info += ")"
        else:
            source_info = ""
        
        # Add citation reference to the beginning of chunk
        cited_content = f"[{citation_ref}] {chunk.content}{source_info}"
        context_parts.append(cited_content)
    
    # Join all parts with double newlines for readability
    context_str = "\n\n".join(context_parts)
    
    return context_str, citations

def _assemble_with_footnote_citations(self, chunks: List[SearchResult]) -> Tuple[str, List[Dict]]:
    """Assemble context with footnote-style citations"""
    
    context_parts = []
    footnotes = []
    citations = []
    
    for i, chunk in enumerate(chunks):
        citation_ref = i + 1
        
        # Create citation metadata
        citation = {
            "ref": citation_ref,
            "chunk_id": chunk.chunk_id,
            "score": chunk.score,
            "rank": chunk.rank
        }
        
        for field in self.metadata_fields:
            if field in chunk.metadata:
                citation[field] = chunk.metadata[field]
        
        citations.append(citation)
        
        # Add superscript reference to content
        referenced_content = f"{chunk.content}^{citation_ref}"
        context_parts.append(referenced_content)
        
        # Create footnote
        footnote = f"{citation_ref}. "
        if "source" in chunk.metadata:
            footnote += f"Source: {chunk.metadata['source']}"
            if "page" in chunk.metadata:
                footnote += f", Page {chunk.metadata['page']}"
        footnote += f" (Relevance: {chunk.score:.3f})"
        
        footnotes.append(footnote)
    
    # Assemble: content + footnote section
    content_section = "\n\n".join(context_parts)
    footnote_section = "\n".join(footnotes)
    
    context_str = f"{content_section}\n\n---\nSources:\n{footnote_section}"
    
    return context_str, citations
```

### Advanced Context Optimization

```python
class AdaptiveContextBuilder(ContextBuilder):
    """Context builder that adapts strategy based on content characteristics"""
    
    def build(self, search_results: List[SearchResult], 
              query: str = None) -> Tuple[str, List[Dict]]:
        """Build context with adaptive strategies"""
        
        # Analyze content characteristics
        content_analysis = self._analyze_content(search_results, query)
        
        # Adapt parameters based on analysis
        self._adapt_parameters(content_analysis)
        
        # Use parent build method with adapted parameters
        return super().build(search_results)
    
    def _analyze_content(self, results: List[SearchResult], query: str) -> Dict:
        """Analyze retrieved content to guide context construction"""
        
        analysis = {
            "content_diversity": 0.0,    # How varied are the chunks?
            "factual_density": 0.0,      # How fact-heavy vs narrative?
            "query_complexity": 0.0,     # How complex is the query?
            "chunk_coherence": 0.0       # How well do chunks fit together?
        }
        
        # Content diversity (lexical diversity)
        all_tokens = set()
        chunk_tokens = []
        
        for result in results:
            tokens = set(result.content.lower().split())
            chunk_tokens.append(tokens)
            all_tokens.update(tokens)
        
        if len(results) > 1:
            # Calculate average pairwise Jaccard similarity
            similarities = []
            for i in range(len(chunk_tokens)):
                for j in range(i + 1, len(chunk_tokens)):
                    intersection = len(chunk_tokens[i] & chunk_tokens[j])
                    union = len(chunk_tokens[i] | chunk_tokens[j])
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
            
            # Diversity = 1 - average_similarity
            analysis["content_diversity"] = 1 - (sum(similarities) / len(similarities))
        
        # Factual density (heuristic: numbers, dates, proper nouns)
        total_tokens = 0
        factual_tokens = 0
        
        for result in results:
            tokens = result.content.split()
            total_tokens += len(tokens)
            
            for token in tokens:
                # Count numbers, dates, capitalized words as "factual"
                if (re.match(r'\d+', token) or 
                    re.match(r'\d{4}-\d{2}-\d{2}', token) or
                    (token[0].isupper() and len(token) > 2)):
                    factual_tokens += 1
        
        analysis["factual_density"] = factual_tokens / total_tokens if total_tokens > 0 else 0
        
        # Query complexity (length, question words, technical terms)
        if query:
            query_tokens = query.split()
            analysis["query_complexity"] = min(1.0, len(query_tokens) / 20)  # Normalized by 20 words
            
            question_words = ["what", "how", "why", "when", "where", "which", "who"]
            if any(word in query.lower() for word in question_words):
                analysis["query_complexity"] += 0.2
        
        return analysis
    
    def _adapt_parameters(self, analysis: Dict):
        """Adapt context building parameters based on content analysis"""
        
        # High diversity content → prioritize top results, shorter context
        if analysis["content_diversity"] > 0.7:
            self.max_tokens = int(self.max_tokens * 0.8)  # Reduce to focus on top results
            
        # High factual density → use more precise citations, preserve metadata
        if analysis["factual_density"] > 0.3:
            self.citation_style = "footnote"  # More detailed citations for facts
            self.include_metadata = True
            
        # Complex query → allow more context, use chronological ordering if available
        if analysis["query_complexity"] > 0.6:
            self.max_tokens = int(self.max_tokens * 1.2)  # Allow more context
            if self.order == "relevance":
                self.order = "source"  # Group by source for complex queries
```

---

## Comparative Analysis

### Citation Style Selection

| Style | Token Overhead | Precision | User Experience | Best For |
|---|---|---|---|---|
| **None** | 0% | Low | Clean | Internal systems, summaries |
| **Inline** | 5-15% | Medium | Good | General Q&A, documentation |
| **Footnote** | 10-25% | High | Academic | Research, legal, compliance |

### Context Ordering Strategies

```python
def compare_ordering_strategies(results: List[SearchResult], query: str) -> Dict[str, float]:
    """Compare different ordering strategies for given query/results"""
    
    strategies = ["relevance", "reverse_rank", "source", "chronological"]
    strategy_scores = {}
    
    for strategy in strategies:
        builder = ContextBuilder(order=strategy)
        context, citations = builder.build(results)
        
        # Score based on multiple factors
        score = 0.0
        
        # 1. Coherence (semantic flow between adjacent chunks)
        coherence = measure_context_coherence(context)
        score += coherence * 0.4
        
        # 2. Information density (unique facts per token)
        density = measure_information_density(context)
        score += density * 0.3
        
        # 3. Relevance preservation (top chunks near beginning)
        relevance_order = measure_relevance_preservation(results, citations)
        score += relevance_order * 0.3
        
        strategy_scores[strategy] = score
    
    return strategy_scores

def measure_context_coherence(context: str) -> float:
    """Measure semantic flow between context sections"""
    
    sections = context.split('\n\n')
    if len(sections) < 2:
        return 1.0  # Single section = perfect coherence
    
    # Compute sentence embeddings for each section
    embedder = SentenceTransformerEmbedder()
    section_embeddings = embedder.encode([s[:200] for s in sections])  # First 200 chars
    
    # Measure average cosine similarity between adjacent sections
    adjacency_similarities = []
    for i in range(len(section_embeddings) - 1):
        sim = cosine_similarity([section_embeddings[i]], [section_embeddings[i+1]])[0][0]
        adjacency_similarities.append(sim)
    
    return np.mean(adjacency_similarities)
```

---

## Practical Guidelines

### Token Budget Management

**Dynamic budget allocation**:
```python
class DynamicContextBuilder(ContextBuilder):
    def __init__(self, llm_context_window: int, query_length: int, 
                 template_length: int, response_budget: int = 1024):
        """
        Dynamically allocate context budget based on available tokens
        
        Args:
            llm_context_window: Model's maximum context (e.g., 8192 for GPT-4)
            query_length: Actual query token count
            template_length: Prompt template token overhead  
            response_budget: Tokens reserved for LLM response
        """
        
        # Calculate available context tokens
        used_tokens = query_length + template_length + response_budget
        safety_margin = int(llm_context_window * 0.1)  # 10% safety buffer
        
        max_context_tokens = llm_context_window - used_tokens - safety_margin
        max_context_tokens = max(500, max_context_tokens)  # Minimum viable context
        
        super().__init__(max_tokens=max_context_tokens)
        
        # Log allocation for debugging
        logger.info(f"Context budget: {max_context_tokens} tokens "
                   f"(Window: {llm_context_window}, Used: {used_tokens}, "
                   f"Safety: {safety_margin})")
```

**Multi-turn context management**:
```python
def manage_conversation_context(conversation_history: List[Dict], 
                               new_context: str, max_total_tokens: int) -> str:
    """Manage context across conversation turns"""
    
    # Prioritize: Current context > Recent history > Older history
    current_tokens = len(new_context) // 4
    remaining_budget = max_total_tokens - current_tokens
    
    if remaining_budget <= 0:
        return new_context  # No room for history
    
    # Add recent conversation turns until budget exhausted
    history_parts = []
    for turn in reversed(conversation_history):  # Most recent first
        turn_text = f"Q: {turn['query']}\nA: {turn['answer']}\n"
        turn_tokens = len(turn_text) // 4
        
        if turn_tokens <= remaining_budget:
            history_parts.insert(0, turn_text)  # Add at beginning
            remaining_budget -= turn_tokens
        else:
            break
    
    # Assemble: history + current context
    if history_parts:
        full_context = "Previous conversation:\n" + "".join(history_parts) + "\nCurrent context:\n" + new_context
    else:
        full_context = new_context
    
    return full_context
```

### Quality Optimization

**Context redundancy detection**:
```python
def detect_context_redundancy(context: str, threshold: float = 0.8) -> List[str]:
    """Identify redundant sentences in context"""
    
    sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 10]
    
    if len(sentences) < 2:
        return []
    
    # Compute sentence embeddings
    embedder = SentenceTransformerEmbedder()
    embeddings = embedder.encode(sentences)
    
    # Find highly similar sentence pairs
    redundant_sentences = []
    similarity_matrix = cosine_similarity(embeddings)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > threshold:
                # Keep the longer sentence (usually more informative)
                if len(sentences[i]) >= len(sentences[j]):
                    redundant_sentences.append(sentences[j])
                else:
                    redundant_sentences.append(sentences[i])
    
    return list(set(redundant_sentences))  # Remove duplicates

def remove_redundant_content(context: str) -> str:
    """Remove redundant sentences from context"""
    
    redundant = detect_context_redundancy(context)
    
    if not redundant:
        return context
    
    # Remove redundant sentences
    cleaned_context = context
    for sentence in redundant:
        cleaned_context = cleaned_context.replace(sentence + '.', '')
    
    # Clean up extra whitespace
    cleaned_context = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_context)
    
    return cleaned_context.strip()
```

### Performance Monitoring

**Context quality metrics**:
```python
def compute_context_quality_metrics(context: str, citations: List[Dict], 
                                  query: str) -> Dict[str, float]:
    """Comprehensive context quality assessment"""
    
    metrics = {}
    
    # 1. Citation coverage
    total_chars = len(context)
    citation_chars = sum(len(f"[{c['ref']}]") for c in citations)
    metrics["citation_coverage"] = citation_chars / total_chars if total_chars > 0 else 0
    
    # 2. Information density (unique facts per 100 tokens)
    sentences = context.split('.')
    unique_facts = len(set(s.strip().lower() for s in sentences if len(s.strip()) > 20))
    tokens = len(context) // 4
    metrics["information_density"] = (unique_facts / tokens * 100) if tokens > 0 else 0
    
    # 3. Query relevance (context similarity to query)
    embedder = SentenceTransformerEmbedder()
    query_embedding = embedder.encode([query])
    context_embedding = embedder.encode([context[:1000]])  # First 1000 chars
    metrics["query_relevance"] = cosine_similarity(query_embedding, context_embedding)[0][0]
    
    # 4. Readability (average sentence length)
    sentence_lengths = [len(s.split()) for s in sentences if len(s.strip()) > 5]
    metrics["readability"] = 1 / (1 + np.std(sentence_lengths)) if sentence_lengths else 0
    
    # 5. Source diversity
    sources = set(citation.get("source", "unknown") for citation in citations)
    metrics["source_diversity"] = len(sources) / len(citations) if citations else 0
    
    return metrics
```

### Common Issues & Solutions

**Issue**: Context exceeds token limit despite budget management
```python
# Problem: Token estimation (chars/4) is inaccurate for some text types
# Solution: Use proper tokenizer for accurate counting

def accurate_token_count(text: str, model_name: str) -> int:
    """Get accurate token count using model's tokenizer"""
    
    try:
        if "gpt" in model_name.lower():
            import tiktoken
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        elif "claude" in model_name.lower():
            # Anthropic's tokenizer (approximate)
            return len(text) // 3.5  # Claude tokens are slightly larger
        else:
            # Generic approximation
            return len(text.split())  # Word-based counting for other models
    except:
        return len(text) // 4  # Fallback to character-based estimation

class AccurateContextBuilder(ContextBuilder):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
    
    def _apply_token_budget(self, chunks: List[SearchResult]) -> List[SearchResult]:
        """Use accurate token counting"""
        
        selected_chunks = []
        remaining_budget = self.max_tokens
        
        for chunk in chunks:
            chunk_tokens = accurate_token_count(chunk.content, self.model_name)
            
            if chunk_tokens <= remaining_budget:
                selected_chunks.append(chunk)
                remaining_budget -= chunk_tokens
            else:
                # Try intelligent truncation with accurate counting
                truncated = self._accurate_truncate_chunk(chunk, remaining_budget)
                if truncated:
                    selected_chunks.append(truncated)
                break
        
        return selected_chunks
```

**Issue**: Citations become misaligned after context processing
```python
# Problem: Text processing changes citation references
# Solution: Maintain citation mapping throughout pipeline

class CitationTrackingBuilder(ContextBuilder):
    def _assemble_with_inline_citations(self, chunks: List[SearchResult]) -> Tuple[str, List[Dict]]:
        """Maintain citation integrity during assembly"""
        
        context_parts = []
        citations = []
        citation_mapping = {}  # chunk_id -> citation_ref
        
        for i, chunk in enumerate(chunks):
            citation_ref = i + 1
            citation_mapping[chunk.chunk_id] = citation_ref
            
            # Embed citation ID in content for tracking
            tracked_content = f"<CITE_START:{chunk.chunk_id}>[{citation_ref}] {chunk.content}<CITE_END:{chunk.chunk_id}>"
            context_parts.append(tracked_content)
            
            citations.append({
                "ref": citation_ref,
                "chunk_id": chunk.chunk_id,
                "source": chunk.metadata.get("source", "unknown"),
                "score": chunk.score
            })
        
        # Join and then clean up tracking markers
        raw_context = "\n\n".join(context_parts)
        
        # Remove tracking markers but preserve citation structure
        clean_context = re.sub(r'<CITE_START:[^>]+>', '', raw_context)
        clean_context = re.sub(r'<CITE_END:[^>]+>', '', clean_context)
        
        return clean_context, citations
```

**Issue**: Poor context flow between chunks from different sources
```python
# Problem: Abrupt transitions between unrelated content
# Solution: Add transitional text and logical connectors

def improve_context_flow(context: str, citations: List[Dict]) -> str:
    """Add transitions between context sections"""
    
    sections = context.split('\n\n')
    if len(sections) < 2:
        return context
    
    # Group sections by source
    source_groups = []
    current_source = None
    current_group = []
    
    section_citations = {}  # Map section to citation info
    for section in sections:
        citation_match = re.search(r'\[(\d+)\]', section)
        if citation_match:
            citation_ref = int(citation_match.group(1))
            citation = next((c for c in citations if c["ref"] == citation_ref), None)
            if citation:
                source = citation.get("source", "unknown")
                
                if source != current_source:
                    if current_group:
                        source_groups.append((current_source, current_group))
                    current_source = source
                    current_group = [section]
                else:
                    current_group.append(section)
    
    if current_group:
        source_groups.append((current_source, current_group))
    
    # Rebuild context with source transitions
    improved_sections = []
    
    for i, (source, group_sections) in enumerate(source_groups):
        if i > 0:  # Add transition between sources
            improved_sections.append(f"From {source}:")
        
        improved_sections.extend(group_sections)
    
    return '\n\n'.join(improved_sections)
```

**Next concept**: LLM Integration — provider abstractions, streaming, and cost optimization strategies
