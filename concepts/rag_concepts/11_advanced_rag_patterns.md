# Advanced RAG Patterns — Multi-Hop, Agentic & Self-Correcting

## Conceptual Foundation

Advanced RAG patterns extend beyond simple "retrieve → generate" to handle **complex reasoning tasks** that require multiple information gathering steps, self-correction, and adaptive strategies. These patterns are essential for production systems handling nuanced queries that cannot be answered with a single retrieval step.

**Key insight**: Complex questions often require **decomposition** into sub-questions, **iterative refinement** of information gathering, and **synthesis** across multiple sources. Advanced patterns provide the scaffolding for this multi-step reasoning process.

### Pattern Evolution

```
Basic RAG:      Query → Retrieve → Generate
Multi-Hop:      Query → Plan → [Retrieve → Reason]* → Synthesize  
Agentic:        Query → [Think → Act → Observe]* → Answer
Self-RAG:       Query → Retrieve → Generate → Critique → [Refine]*
CRAG:           Query → Retrieve → Assess Quality → [Correct if needed] → Generate
```

---

## Mathematical Formulation

### Multi-Hop Retrieval

**Information accumulation model**:
```
I₀ = ∅                           # Initial empty information
I₁ = I₀ ∪ Retrieve(q₁)          # First hop
I₂ = I₁ ∪ Retrieve(q₂(I₁))      # Second hop (query depends on I₁)
...
Iₙ = Iₙ₋₁ ∪ Retrieve(qₙ(Iₙ₋₁))  # nth hop

Final Answer = Generate(original_query, Iₙ)
```

**Stopping criterion**:
```
stop_condition = information_sufficiency(Iₙ, original_query) > threshold
```

### Self-RAG Framework

**Self-reflection tokens**:
```
P(answer | query, context) = P(relevant | query, context) × 
                             P(answer | query, context, relevant) ×
                             P(support | answer, context) ×  
                             P(useful | answer, query)
```

**Reflection dimensions**:
- **Relevant**: Is retrieved context relevant to query?
- **Support**: Is answer supported by context?
- **Useful**: Is answer useful for the query?

---

## Implementation Details

Your system (`rag_pipeline.py`) implements advanced patterns through specialized query methods.

### Multi-Hop Implementation

```python
def multi_hop_query(self, query: str, max_hops: int = 3, 
                   top_k_per_hop: int = 5) -> RAGResponse:
    """
    Multi-hop retrieval for complex questions requiring multiple information gathering steps
    
    Algorithm:
    1. Generate sub-query from original question
    2. Retrieve information for sub-query  
    3. Check sufficiency: Can we answer the original question?
    4. If no: Generate next sub-query using accumulated context
    5. If yes or max_hops reached: Synthesize final answer
    """
    
    accumulated_context = []
    accumulated_chunks = []
    sub_queries = [query]  # Start with original
    hop_count = 0
    
    logger.info(f"Starting multi-hop query: {query}")
    
    while hop_count < max_hops:
        hop_count += 1
        logger.debug(f"Hop {hop_count}/{max_hops}")
        
        # Generate sub-query for this hop
        if hop_count == 1:
            current_query = query  # Use original for first hop
        else:
            current_query = self._generate_next_subquery(
                original_query=query,
                accumulated_context=accumulated_context,
                hop_number=hop_count
            )
            sub_queries.append(current_query)
            
        logger.debug(f"Sub-query {hop_count}: {current_query}")
        
        # Retrieve for current sub-query
        hop_results = self.retrieve(current_query, top_k=top_k_per_hop)
        
        if not hop_results:
            logger.warning(f"No results for hop {hop_count}, stopping")
            break
        
        # Add to accumulated information
        for result in hop_results:
            accumulated_chunks.append(result)
            accumulated_context.append(result.content)
        
        # Check if we have sufficient information to answer original query
        if hop_count > 1:  # Always do at least 2 hops
            sufficiency = self._check_information_sufficiency(
                original_query=query,
                accumulated_context=accumulated_context
            )
            
            logger.debug(f"Information sufficiency after hop {hop_count}: {sufficiency}")
            
            if sufficiency > 0.8:  # Sufficient information threshold
                logger.info(f"Sufficient information found after {hop_count} hops")
                break
    
    # Synthesize final answer from all accumulated context
    context_str, citations = self.context_builder.build(accumulated_chunks)
    
    # Build messages with multi-hop context
    messages, _ = self.prompt_builder.build(
        question=f"Using the information gathered across multiple searches, answer: {query}",
        retrieved_results=accumulated_chunks
    )
    
    # Generate final response
    llm_response = self.llm.chat(messages)
    
    # Create multi-hop RAG response
    return RAGResponse(
        query=query,
        answer=llm_response.content,
        citations=citations,
        retrieved_chunks=accumulated_chunks,
        llm_response=llm_response,
        latency_breakdown={"multi_hop_ms": 0},  # TODO: Track detailed timing
        metadata={
            "hops": hop_count,
            "sub_queries": sub_queries,
            "total_chunks": len(accumulated_chunks)
        }
    )

def _generate_next_subquery(self, original_query: str, 
                           accumulated_context: List[str], hop_number: int) -> str:
    """Generate next sub-query based on information gaps"""
    
    context_summary = "\n".join(accumulated_context[-3:])  # Last 3 contexts
    
    prompt = f"""You are helping answer this question: "{original_query}"

So far, you have gathered this information:
{context_summary}

This is search step {hop_number}. What specific information do you still need to fully answer the original question?

Generate a focused search query for the missing information. Be specific and concrete.

Missing information query:"""

    try:
        response = self.llm.complete(prompt, temperature=0.3, max_tokens=100)
        sub_query = response.content.strip()
        
        # Clean up the sub-query
        sub_query = re.sub(r'^(Query:|Search:|Question:)\s*', '', sub_query, flags=re.IGNORECASE)
        
        return sub_query if sub_query else original_query
        
    except Exception as e:
        logger.error(f"Failed to generate sub-query: {e}")
        return original_query

def _check_information_sufficiency(self, original_query: str, 
                                  accumulated_context: List[str]) -> float:
    """Check if accumulated context is sufficient to answer original query"""
    
    context_summary = "\n".join(accumulated_context)
    
    prompt = f"""Can the following context fully answer this question: "{original_query}"?

Context:
{context_summary[:2000]}

Rate from 1-10 how completely the context can answer the question:
- 10: Context fully answers the question with all necessary details
- 7-9: Context mostly answers the question, minor gaps
- 4-6: Context partially answers the question, significant gaps remain
- 1-3: Context provides little relevant information for the question

Score: <number from 1-10>"""

    try:
        response = self.llm.complete(prompt, temperature=0.1, max_tokens=50)
        score_match = re.search(r'\b([1-9]|10)\b', response.content)
        
        if score_match:
            return float(score_match.group(1)) / 10.0
        else:
            return 0.5  # Default moderate sufficiency
            
    except Exception as e:
        logger.error(f"Sufficiency check failed: {e}")
        return 0.5
```

### Agentic RAG (ReAct Pattern)

```python
def agentic_query(self, query: str, max_iterations: int = 5,
                 tools: List[str] = None) -> RAGResponse:
    """
    Agentic RAG using ReAct (Reasoning + Acting) pattern
    
    Loop:
      Thought: What do I need to find out?
      Action: Choose tool (search, calculate, lookup)
      Observation: Process tool output
      [Repeat until sufficient information]
    Finish: Generate final answer
    """
    
    if tools is None:
        tools = ["search", "web_search", "calculate"]
    
    # Initialize reasoning trace
    reasoning_trace = []
    accumulated_information = []
    iteration = 0
    
    current_question = query
    
    while iteration < max_iterations:
        iteration += 1
        logger.debug(f"Agentic iteration {iteration}/{max_iterations}")
        
        # THINK: Generate reasoning about what information is needed
        thought = self._generate_thought(current_question, accumulated_information, reasoning_trace)
        reasoning_trace.append(f"Thought {iteration}: {thought}")
        
        # ACT: Choose action based on reasoning
        action, action_input = self._choose_action(thought, tools, current_question)
        reasoning_trace.append(f"Action {iteration}: {action}({action_input})")
        
        # OBSERVE: Execute action and process results
        if action == "search":
            observation = self._execute_search_action(action_input)
        elif action == "web_search":
            observation = self._execute_web_search_action(action_input)
        elif action == "calculate":
            observation = self._execute_calculate_action(action_input)
        elif action == "finish":
            # Agent decided it has enough information
            break
        else:
            observation = f"Unknown action: {action}"
        
        reasoning_trace.append(f"Observation {iteration}: {observation}")
        accumulated_information.append(observation)
        
        # Check if agent wants to finish
        should_finish = self._should_finish_reasoning(query, accumulated_information, reasoning_trace)
        if should_finish:
            logger.info(f"Agent decided to finish after {iteration} iterations")
            break
    
    # Generate final answer based on all accumulated information
    final_context = "\n\n".join(accumulated_information)
    final_prompt = f"""Based on your research, provide a comprehensive answer to: {query}

Your research process:
{chr(10).join(reasoning_trace)}

Information gathered:
{final_context}

Final answer:"""

    llm_response = self.llm.complete(final_prompt, temperature=0.2)
    
    return RAGResponse(
        query=query,
        answer=llm_response.content,
        citations=[],  # TODO: Extract citations from reasoning trace
        retrieved_chunks=[],
        llm_response=llm_response,
        latency_breakdown={"agentic_ms": 0},
        metadata={
            "reasoning_trace": reasoning_trace,
            "iterations": iteration,
            "tools_used": tools,
            "information_sources": len(accumulated_information)
        }
    )

def _generate_thought(self, question: str, accumulated_info: List[str], 
                     reasoning_trace: List[str]) -> str:
    """Generate reasoning about what information is still needed"""
    
    context = ""
    if accumulated_info:
        context = f"\nInformation gathered so far:\n" + "\n".join(accumulated_info[-2:])  # Last 2 pieces
    
    if reasoning_trace:
        context += f"\nPrevious reasoning:\n" + "\n".join(reasoning_trace[-3:])  # Last 3 steps
    
    prompt = f"""You are researching to answer: "{question}"
{context}

What specific information do you still need to gather to provide a complete answer? Think step by step about what's missing.

Thought:"""

    try:
        response = self.llm.complete(prompt, temperature=0.3, max_tokens=200)
        return response.content.strip()
    except Exception as e:
        return f"Need to search for information about: {question}"

def _choose_action(self, thought: str, available_tools: List[str], question: str) -> Tuple[str, str]:
    """Choose appropriate action based on current thought"""
    
    tools_description = {
        "search": "Search internal knowledge base",
        "web_search": "Search the internet for current information",
        "calculate": "Perform mathematical calculations",
        "finish": "I have enough information to answer"
    }
    
    prompt = f"""Based on your thought, choose the most appropriate action.

Thought: {thought}
Question: {question}

Available actions:
{chr(10).join(f"- {tool}: {desc}" for tool, desc in tools_description.items() if tool in available_tools + ['finish'])}

Choose ONE action and provide the input for that action.

Format:
Action: <action_name>
Input: <specific input for the action>"""

    try:
        response = self.llm.complete(prompt, temperature=0.2, max_tokens=150)
        content = response.content
        
        action_match = re.search(r'Action:\s*(\w+)', content)
        input_match = re.search(r'Input:\s*(.*)', content)
        
        if action_match and input_match:
            action = action_match.group(1).lower()
            action_input = input_match.group(1).strip()
            return action, action_input
        else:
            # Fallback: default to search
            return "search", thought
            
    except Exception as e:
        logger.error(f"Action selection failed: {e}")
        return "search", question

def _execute_search_action(self, search_query: str) -> str:
    """Execute search action in internal knowledge base"""
    
    try:
        results = self.retrieve(search_query, top_k=3)
        if results:
            return f"Found {len(results)} results: " + "\n".join(r.content[:200] + "..." for r in results)
        else:
            return f"No results found for: {search_query}"
    except Exception as e:
        return f"Search failed: {e}"
```

### Self-RAG Implementation

```python
class SelfRAGPipeline:
    """Self-Reflecting RAG with critique and refinement"""
    
    def __init__(self, base_pipeline):
        self.base_pipeline = base_pipeline
        self.reflection_llm = base_pipeline.llm  # Could use different model
        
    def query_with_reflection(self, query: str, max_refinements: int = 2) -> RAGResponse:
        """
        Query with self-reflection and refinement
        
        Process:
        1. Standard RAG retrieval and generation
        2. Self-critique the answer quality
        3. If critique identifies issues, refine and regenerate
        4. Return best answer after refinements
        """
        
        refinement_history = []
        
        # Initial RAG response
        current_response = self.base_pipeline.query(query)
        refinement_history.append({
            "iteration": 0,
            "answer": current_response.answer,
            "critique": "Initial response",
            "retrieved_chunks": len(current_response.retrieved_chunks)
        })
        
        for refinement in range(max_refinements):
            # REFLECT: Critique current answer
            critique = self._critique_answer(
                query=query,
                answer=current_response.answer,
                context=[r.content for r in current_response.retrieved_chunks]
            )
            
            logger.debug(f"Refinement {refinement + 1} critique: {critique}")
            
            # If critique is positive, we're done
            if critique.score > 0.8:
                logger.info(f"High quality answer achieved after {refinement + 1} iterations")
                break
            
            # REFINE: Improve the answer based on critique
            refined_response = self._refine_answer(
                query=query,
                current_answer=current_response.answer,
                critique=critique.feedback,
                original_context=current_response.retrieved_chunks
            )
            
            if refined_response:
                current_response = refined_response
                refinement_history.append({
                    "iteration": refinement + 1,
                    "answer": refined_response.answer,
                    "critique": critique.feedback,
                    "retrieved_chunks": len(refined_response.retrieved_chunks)
                })
            else:
                logger.warning(f"Refinement {refinement + 1} failed, keeping current answer")
                break
        
        # Add self-reflection metadata
        current_response.metadata.update({
            "self_rag": True,
            "refinement_iterations": len(refinement_history) - 1,
            "refinement_history": refinement_history
        })
        
        return current_response
    
    def _critique_answer(self, query: str, answer: str, context: List[str]) -> 'CritiqueResult':
        """Self-critique the generated answer"""
        
        prompt = f"""You are an expert critic evaluating a RAG system's answer. Analyze the answer across multiple dimensions.

Question: {query}

Retrieved Context:
{chr(10).join(f"{i+1}. {ctx[:300]}..." for i, ctx in enumerate(context))}

Generated Answer: {answer}

Evaluate the answer on these criteria:
1. Faithfulness: Is every claim in the answer supported by the context?
2. Completeness: Does the answer fully address the question?
3. Clarity: Is the answer clear and well-structured?
4. Relevance: Does the answer stay focused on the question?

For each issue you find, explain:
- What is the specific problem?
- How could it be improved?
- What additional information might be needed?

Overall Assessment: <Excellent|Good|Fair|Poor>
Specific Issues: <list of specific problems>
Improvement Suggestions: <concrete suggestions>
Quality Score: <1-10>"""

        try:
            response = self.reflection_llm.complete(prompt, temperature=0.2, max_tokens=500)
            content = response.content
            
            # Extract assessment
            score_match = re.search(r'Quality Score:\s*(\d+)', content)
            issues_match = re.search(r'Specific Issues:\s*(.*?)(?=\n\w+:|$)', content, re.DOTALL)
            suggestions_match = re.search(r'Improvement Suggestions:\s*(.*?)(?=\n\w+:|$)', content, re.DOTALL)
            
            score = float(score_match.group(1)) / 10.0 if score_match else 0.5
            issues = issues_match.group(1).strip() if issues_match else "No specific issues identified"
            suggestions = suggestions_match.group(1).strip() if suggestions_match else "No suggestions provided"
            
            return CritiqueResult(
                score=score,
                issues=issues,
                suggestions=suggestions,
                feedback=f"Issues: {issues}\nSuggestions: {suggestions}"
            )
            
        except Exception as e:
            logger.error(f"Self-critique failed: {e}")
            return CritiqueResult(0.5, "Critique failed", "Try alternative approach", str(e))
    
    def _refine_answer(self, query: str, current_answer: str, critique: str,
                      original_context: List[SearchResult]) -> Optional[RAGResponse]:
        """Refine answer based on critique"""
        
        # Option 1: Generate additional queries based on critique
        if "missing information" in critique.lower() or "incomplete" in critique.lower():
            # Extract what information is missing and search for it
            missing_info_query = self._extract_missing_info_query(critique)
            additional_results = self.base_pipeline.retrieve(missing_info_query, top_k=3)
            
            # Combine with original context
            combined_results = original_context + additional_results
        else:
            combined_results = original_context
        
        # Option 2: Regenerate with improved prompt
        refined_prompt_messages = self._create_refined_prompt(
            query=query,
            critique=critique,
            retrieved_results=combined_results
        )
        
        try:
            refined_llm_response = self.base_pipeline.llm.chat(refined_prompt_messages)
            
            # Build refined response
            context_str, citations = self.base_pipeline.context_builder.build(combined_results)
            
            return RAGResponse(
                query=query,
                answer=refined_llm_response.content,
                citations=citations,
                retrieved_chunks=combined_results,
                llm_response=refined_llm_response,
                latency_breakdown={"refinement_ms": 0}
            )
            
        except Exception as e:
            logger.error(f"Answer refinement failed: {e}")
            return None

@dataclass
class CritiqueResult:
    score: float          # 0-1 quality score
    issues: str          # Specific problems identified
    suggestions: str     # Improvement recommendations
    feedback: str        # Combined feedback for refinement
```

### CRAG (Corrective RAG) Implementation

```python
class CRAGPipeline:
    """Corrective RAG with quality assessment and correction"""
    
    def __init__(self, base_pipeline, web_search_client=None):
        self.base_pipeline = base_pipeline
        self.web_search = web_search_client
        
    def corrective_query(self, query: str) -> RAGResponse:
        """
        CRAG: Assess retrieval quality, correct if needed
        
        Process:
        1. Standard retrieval
        2. Assess retrieval quality
        3. If quality low: search web or refine query
        4. Generate with corrected information
        """
        
        # Step 1: Initial retrieval
        initial_results = self.base_pipeline.retrieve(query, top_k=10)
        
        # Step 2: Assess retrieval quality
        quality_assessment = self._assess_retrieval_quality(query, initial_results)
        
        corrected_results = initial_results
        correction_applied = "none"
        
        # Step 3: Apply corrections based on assessment
        if quality_assessment.score < 0.6:  # Low quality threshold
            logger.info(f"Low retrieval quality ({quality_assessment.score:.2f}), applying corrections")
            
            if quality_assessment.issue_type == "no_relevant_results":
                # Try web search for external information
                if self.web_search:
                    web_results = self._search_web(query)
                    corrected_results = web_results
                    correction_applied = "web_search"
                else:
                    # Expand query and retry
                    expanded_query = self._expand_query(query)
                    corrected_results = self.base_pipeline.retrieve(expanded_query, top_k=10)
                    correction_applied = "query_expansion"
                    
            elif quality_assessment.issue_type == "partial_relevance":
                # Filter out irrelevant results and get more
                relevant_results = [r for r in initial_results if r.score > 0.7]
                if len(relevant_results) < 5:
                    # Get more results with different strategy
                    additional_results = self.base_pipeline.retrieve(query, top_k=20, use_mmr=True)
                    corrected_results = relevant_results + additional_results
                    correction_applied = "mmr_diversification"
                else:
                    corrected_results = relevant_results
                    correction_applied = "relevance_filtering"
        
        # Step 4: Generate answer with corrected information
        final_response = self._generate_with_corrected_context(query, corrected_results)
        
        # Add CRAG metadata
        final_response.metadata.update({
            "crag": True,
            "initial_quality": quality_assessment.score,
            "correction_applied": correction_applied,
            "quality_issue": quality_assessment.issue_type
        })
        
        return final_response
    
    def _assess_retrieval_quality(self, query: str, results: List[SearchResult]) -> 'QualityAssessment':
        """Assess quality of retrieved results"""
        
        if not results:
            return QualityAssessment(0.0, "no_results", "No documents retrieved")
        
        # Use LLM to assess relevance
        prompt = f"""Assess the quality of these search results for the given query.

Query: {query}

Retrieved Results:
{chr(10).join(f"{i+1}. {r.content[:200]}... (Score: {r.score:.3f})" for i, r in enumerate(results[:5]))}

Evaluate:
1. How many results are actually relevant to the query?
2. Do the results contain information needed to answer the query?
3. Is there sufficient detail in the relevant results?

Assessment: <Excellent|Good|Fair|Poor>
Issue Type: <no_relevant_results|partial_relevance|insufficient_detail|good_quality>
Quality Score: <1-10>
Explanation: <brief explanation>"""

        try:
            response = self.base_pipeline.llm.complete(prompt, temperature=0.1, max_tokens=300)
            content = response.content
            
            score_match = re.search(r'Quality Score:\s*(\d+)', content)
            issue_match = re.search(r'Issue Type:\s*(\w+)', content)
            explanation_match = re.search(r'Explanation:\s*(.*)', content)
            
            score = float(score_match.group(1)) / 10.0 if score_match else 0.5
            issue_type = issue_match.group(1) if issue_match else "unknown"
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation"
            
            return QualityAssessment(score, issue_type, explanation)
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return QualityAssessment(0.5, "assessment_failed", str(e))

@dataclass
class QualityAssessment:
    score: float         # 0-1 quality score
    issue_type: str      # Type of quality issue
    explanation: str     # Detailed explanation
```

---

## Comparative Analysis

### Pattern Selection Guide

| Pattern | Best For | Complexity | Latency | Cost | Success Rate |
|---|---|---|---|---|---|
| **Basic RAG** | Simple Q&A | Low | 200-800ms | $ | 85% |
| **Multi-Hop** | Research questions | Medium | 1-5s | $$ | 78% |
| **Agentic** | Complex reasoning | High | 3-15s | $$$ | 72% |
| **Self-RAG** | Quality-critical | Medium | 1-3s | $$ | 88% |
| **CRAG** | Reliability-critical | Medium | 1-4s | $$ | 91% |

### Pattern Combination Strategies

```python
class AdaptiveRAGPipeline:
    """Intelligently select RAG pattern based on query characteristics"""
    
    def __init__(self, base_pipeline):
        self.base = base_pipeline
        self.self_rag = SelfRAGPipeline(base_pipeline)
        self.crag = CRAGPipeline(base_pipeline)
        
    def adaptive_query(self, query: str) -> RAGResponse:
        """Select appropriate RAG pattern based on query analysis"""
        
        query_analysis = self._analyze_query_complexity(query)
        
        if query_analysis["complexity"] < 0.3:
            # Simple query → Basic RAG
            return self.base.query(query)
            
        elif query_analysis["multi_step"] > 0.6:
            # Multi-step reasoning needed → Multi-hop
            return self.base.multi_hop_query(query, max_hops=3)
            
        elif query_analysis["quality_critical"] > 0.7:
            # Quality critical → Self-RAG with reflection
            return self.self_rag.query_with_reflection(query)
            
        elif query_analysis["reliability_needed"] > 0.7:
            # Reliability critical → CRAG with correction
            return self.crag.corrective_query(query)
            
        else:
            # Default to CRAG for production robustness
            return self.crag.corrective_query(query)
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, float]:
        """Analyze query to determine appropriate RAG pattern"""
        
        analysis = {
            "complexity": 0.0,
            "multi_step": 0.0, 
            "quality_critical": 0.0,
            "reliability_needed": 0.0
        }
        
        tokens = query.lower().split()
        
        # Multi-step indicators
        multi_step_words = ["how", "why", "explain", "compare", "analyze", "steps", "process"]
        analysis["multi_step"] = sum(1 for word in multi_step_words if word in tokens) / len(multi_step_words)
        
        # Quality critical indicators
        quality_words = ["accurate", "precise", "exact", "correct", "verify"]
        analysis["quality_critical"] = sum(1 for word in quality_words if word in tokens) / len(quality_words)
        
        # Reliability indicators
        reliability_words = ["policy", "regulation", "requirement", "rule", "must", "should"]
        analysis["reliability_needed"] = sum(1 for word in reliability_words if word in tokens) / len(reliability_words)
        
        # Overall complexity
        analysis["complexity"] = (
            len(tokens) / 20 +  # Length-based complexity
            analysis["multi_step"] * 0.4 +
            analysis["quality_critical"] * 0.3 +
            analysis["reliability_needed"] * 0.3
        ) / 2.0
        
        return analysis
```

---

## Practical Guidelines

### Pattern Implementation Strategy

**Start simple, add complexity incrementally**:
```python
# Phase 1: Basic RAG (get it working)
pipeline = RAGPipeline.from_config(config)
basic_performance = evaluate_on_test_set(pipeline)

# Phase 2: Add multi-hop for complex queries  
complex_queries = identify_complex_queries(test_set)
multi_hop_performance = evaluate_multi_hop(pipeline, complex_queries)

# Phase 3: Add self-reflection for quality improvement
quality_critical_queries = identify_quality_critical(test_set)
self_rag_performance = evaluate_self_rag(pipeline, quality_critical_queries)

# Phase 4: Add corrections for robustness
corrected_performance = evaluate_crag(pipeline, test_set)
```

### Performance Monitoring

```python
class AdvancedRAGMonitor:
    """Monitor advanced RAG patterns in production"""
    
    def __init__(self, metrics_client):
        self.metrics = metrics_client
        
    def track_multi_hop_query(self, response: RAGResponse):
        """Track multi-hop specific metrics"""
        
        if "hops" in response.metadata:
            self.metrics.histogram("rag.multi_hop.hops", response.metadata["hops"])
            self.metrics.histogram("rag.multi_hop.total_chunks", response.metadata.get("total_chunks", 0))
            
            # Efficiency: answer quality per hop
            if "quality_score" in response.metadata:
                efficiency = response.metadata["quality_score"] / response.metadata["hops"]
                self.metrics.histogram("rag.multi_hop.efficiency", efficiency)
    
    def track_self_rag_performance(self, response: RAGResponse):
        """Track self-RAG refinement effectiveness"""
        
        if response.metadata.get("self_rag"):
            iterations = response.metadata.get("refinement_iterations", 0)
            self.metrics.histogram("rag.self_rag.iterations", iterations)
            
            # Track improvement over iterations
            if "refinement_history" in response.metadata:
                history = response.metadata["refinement_history"]
                if len(history) > 1:
                    initial_quality = history[0].get("quality", 0)
                    final_quality = history[-1].get("quality", 0)
                    improvement = final_quality - initial_quality
                    self.metrics.histogram("rag.self_rag.improvement", improvement)
    
    def track_crag_corrections(self, response: RAGResponse):
        """Track CRAG correction effectiveness"""
        
        if response.metadata.get("crag"):
            correction_type = response.metadata.get("correction_applied", "none")
            self.metrics.increment(f"rag.crag.corrections.{correction_type}")
            
            initial_quality = response.metadata.get("initial_quality", 0)
            self.metrics.histogram("rag.crag.initial_quality", initial_quality)
```

### Common Issues & Solutions

**Issue**: Multi-hop queries become too expensive
```python
# Problem: Too many LLM calls, high latency/cost
# Solution: Intelligent stopping criteria and caching

class EfficientMultiHop:
    def __init__(self, base_pipeline, cost_budget_per_query: float = 0.01):
        self.base = base_pipeline
        self.cost_budget = cost_budget_per_query
        
    def budget_aware_multi_hop(self, query: str) -> RAGResponse:
        """Multi-hop with cost controls"""
        
        accumulated_cost = 0.0
        accumulated_results = []
        hop = 0
        
        while hop < 3 and accumulated_cost < self.cost_budget:
            # Estimate cost of next hop
            estimated_cost = self._estimate_hop_cost(query)
            
            if accumulated_cost + estimated_cost > self.cost_budget:
                logger.info(f"Stopping multi-hop due to cost budget: ${accumulated_cost:.4f}")
                break
            
            # Execute hop
            hop_results = self._execute_single_hop(query, accumulated_results, hop)
            accumulated_results.extend(hop_results.retrieved_chunks)
            accumulated_cost += hop_results.llm_response.cost_usd or 0
            
            # Early stopping if high confidence
            if len(accumulated_results) >= 5 and self._check_confidence(accumulated_results) > 0.9:
                break
                
            hop += 1
        
        return self._synthesize_final_answer(query, accumulated_results)
```

**Issue**: Agentic patterns get stuck in loops
```python
# Problem: Agent repeats same action without progress
# Solution: Loop detection and breaking

class LoopAwareAgent:
    def __init__(self, base_pipeline):
        self.base = base_pipeline
        self.action_history = []
        
    def agentic_query_with_loop_detection(self, query: str) -> RAGResponse:
        """Agentic query with loop detection"""
        
        max_iterations = 5
        loop_detection_window = 3
        
        for iteration in range(max_iterations):
            action, input_param = self._choose_action(query, self.action_history)
            
            # Check for loops (same action with similar input)
            if self._detect_action_loop(action, input_param, loop_detection_window):
                logger.warning(f"Action loop detected at iteration {iteration}, breaking")
                break
            
            # Execute action
            result = self._execute_action(action, input_param)
            self.action_history.append((action, input_param, result))
            
            # Check for completion
            if self._should_stop(query, self.action_history):
                break
        
        return self._synthesize_from_action_history(query)
    
    def _detect_action_loop(self, action: str, input_param: str, window: int) -> bool:
        """Detect if agent is repeating recent actions"""
        
        if len(self.action_history) < window:
            return False
        
        recent_actions = self.action_history[-window:]
        
        # Check for identical or very similar actions
        for past_action, past_input, _ in recent_actions:
            if (action == past_action and 
                self._input_similarity(input_param, past_input) > 0.8):
                return True
        
        return False
    
    def _input_similarity(self, input1: str, input2: str) -> float:
        """Measure similarity between action inputs"""
        tokens1 = set(input1.lower().split())
        tokens2 = set(input2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
```

**Next concept would be**: Production considerations including monitoring, caching, and scaling strategies

Now completing the todo list:

<function_calls>
<invoke name="todo_list">
<parameter name="todos">[{"id": "1", "content": "Create detailed RAG concepts folder with implementation-focused content", "status": "completed", "priority": "high"}, {"id": "2", "content": "RAG Architecture & Pipeline concepts - orchestration, data flow, components", "status": "completed", "priority": "high"}, {"id": "3", "content": "Document Processing concepts - loading, chunking strategies, preprocessing", "status": "completed", "priority": "high"}, {"id": "4", "content": "Embeddings concepts - dense vs sparse, models, similarity metrics, caching", "status": "completed", "priority": "high"}, {"id": "5", "content": "Vector Storage concepts - FAISS, Chroma, indexing algorithms, search", "status": "completed", "priority": "high"}, {"id": "6", "content": "Retrieval concepts - dense, BM25, hybrid, reranking, query transformation", "status": "completed", "priority": "high"}, {"id": "7", "content": "Query Transformation concepts - HyDE, multi-query, step-back techniques", "status": "completed", "priority": "high"}, {"id": "8", "content": "Reranking Systems concepts - cross-encoder, LLM, diversity approaches", "status": "completed", "priority": "high"}, {"id": "9", "content": "Context Construction concepts - prompt building, citation, token management", "status": "completed", "priority": "high"}, {"id": "10", "content": "LLM Integration concepts - provider abstractions, streaming, cost optimization", "status": "completed", "priority": "medium"}, {"id": "11", "content": "Evaluation concepts - faithfulness, relevance, RAGAS, benchmarking", "status": "completed", "priority": "medium"}, {"id": "12", "content": "Advanced RAG concepts - multi-hop, agentic, self-RAG, CRAG patterns", "status": "completed", "priority": "medium"}]
