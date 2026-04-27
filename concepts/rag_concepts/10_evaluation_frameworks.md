# Evaluation Frameworks — Faithfulness, Relevance & RAGAS

## Conceptual Foundation

Evaluation is the **quality assurance system** for RAG applications. Without systematic evaluation, you cannot reliably improve performance, deploy with confidence, or debug failures. RAG evaluation is uniquely challenging because it spans **multiple dimensions** and **multiple stages** of the pipeline.

**Key insight**: RAG evaluation requires both **component-level metrics** (retrieval quality, generation quality) and **end-to-end metrics** (user satisfaction, task completion). A system can have perfect retrieval but poor generation, or vice versa.

### The RAG Evaluation Challenge

```
Traditional ML:    Input → Model → Output → Compare with Ground Truth → Metric
RAG Evaluation:    Query → Retrieve → Context → Generate → Answer
                     ↓         ↓         ↓        ↓
                  Retrieval  Context   Generation End-to-End
                  Metrics    Quality   Metrics    Metrics
```

**Evaluation dimensions**:
- **Retrieval**: Did we find the right information?
- **Faithfulness**: Is the answer grounded in retrieved context?  
- **Relevance**: Does the answer address the question?
- **Completeness**: Is all necessary information included?
- **Conciseness**: Is the answer appropriately brief?
- **Factual Accuracy**: Are the claims correct?

---

## Mathematical Formulation

### Retrieval Evaluation Metrics

**Precision@K**:
```
P@K = |{relevant documents} ∩ {retrieved documents}| / K
```

**Recall@K**:
```
R@K = |{relevant documents} ∩ {retrieved documents}| / |{relevant documents}|
```

**Mean Reciprocal Rank (MRR)**:
```
MRR = (1/|Q|) × Σᵢ (1/rankᵢ)
```
where rankᵢ is the rank of the first relevant document for query i.

**Normalized Discounted Cumulative Gain (NDCG)**:
```
DCG@K = Σᵢ₌₁ᴷ (2^(relᵢ) - 1) / log₂(i + 1)
NDCG@K = DCG@K / IDCG@K
```

### Generation Evaluation Metrics

**ROUGE-L** (Longest Common Subsequence):
```
LCS(X,Y) = longest common subsequence of X and Y
ROUGE-L = F-measure based on LCS
```

**BLEU Score** (N-gram overlap):
```
BLEU = BP × exp(Σₙ₌₁ᴺ wₙ log pₙ)
```
where pₙ is n-gram precision, BP is brevity penalty.

**BERTScore** (Semantic similarity):
```
BERTScore = F₁(cos_sim(BERT(candidate), BERT(reference)))
```

---

## Implementation Details

Your system (`evaluation/evaluator.py`) implements comprehensive evaluation with multiple metrics and frameworks.

### RAGEvaluator Core

```python
class RAGEvaluator:
    def __init__(self, llm: LLMInterface, embedder: EmbeddingEngine,
                 metrics: List[str] = None, thresholds: Dict[str, float] = None):
        """
        Comprehensive RAG evaluation framework
        
        Args:
            llm: LLM for judge-based metrics
            embedder: For semantic similarity metrics
            metrics: List of metrics to compute
            thresholds: Pass/fail thresholds for each metric
        """
        self.llm = llm
        self.embedder = embedder
        
        self.available_metrics = {
            # Reference-free (no ground truth needed)
            "faithfulness": self._evaluate_faithfulness,
            "answer_relevancy": self._evaluate_answer_relevancy,
            "context_precision": self._evaluate_context_precision,
            
            # Reference-based (need ground truth)
            "answer_correctness": self._evaluate_answer_correctness,
            "context_recall": self._evaluate_context_recall,
            
            # String-based metrics
            "rouge_l": self._evaluate_rouge_l,
            "token_f1": self._evaluate_token_f1,
            "bleu": self._evaluate_bleu,
            
            # Semantic metrics
            "bert_score": self._evaluate_bert_score,
            "semantic_similarity": self._evaluate_semantic_similarity
        }
        
        self.metrics = metrics or ["faithfulness", "answer_relevancy", "rouge_l"]
        self.thresholds = thresholds or {
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_precision": 0.6,
            "answer_correctness": 0.7,
            "rouge_l": 0.5,
            "token_f1": 0.6
        }
        
    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a single QA sample"""
        
        start_time = time.perf_counter()
        metrics_results = {}
        
        for metric_name in self.metrics:
            if metric_name in self.available_metrics:
                try:
                    score = self.available_metrics[metric_name](sample)
                    threshold = self.thresholds.get(metric_name, 0.5)
                    
                    metrics_results[metric_name] = MetricScore(
                        name=metric_name,
                        score=score.score if hasattr(score, 'score') else score,
                        reason=score.reason if hasattr(score, 'reason') else f"Score: {score:.3f}",
                        passed=score.score >= threshold if hasattr(score, 'score') else score >= threshold
                    )
                except Exception as e:
                    logger.error(f"Failed to compute {metric_name}: {e}")
                    metrics_results[metric_name] = MetricScore(
                        name=metric_name,
                        score=0.0,
                        reason=f"Evaluation failed: {e}",
                        passed=False
                    )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return EvaluationResult(
            sample=sample,
            metrics=metrics_results,
            latency_ms=latency_ms
        )
```

### Faithfulness Evaluation

Faithfulness measures whether the generated answer is grounded in the provided context.

```python
def _evaluate_faithfulness(self, sample: EvaluationSample) -> MetricScore:
    """
    Evaluate if answer is faithful to the retrieved context
    Uses LLM as judge to verify grounding
    """
    
    prompt = f"""Please analyze if the given answer is faithful to the provided context.

Context: {' '.join(sample.contexts)}

Answer: {sample.answer}

A faithful answer should:
1. Only make claims that can be directly supported by the context
2. Not contradict any information in the context  
3. Not add information not present in the context
4. Clearly indicate when information is uncertain or missing

Rate the faithfulness on a scale from 1-10 where:
- 10: Completely faithful - every claim is directly supported by context
- 7-9: Mostly faithful - minor unsupported details or slight inference
- 4-6: Partially faithful - some claims supported, others not
- 1-3: Largely unfaithful - many unsupported or contradictory claims

Think step by step:
1. Identify key claims in the answer
2. Check if each claim is supported by the context
3. Look for any contradictions or fabricated information
4. Consider overall faithfulness

Reasoning: <your step-by-step analysis>
Score: <score from 1-10>"""

    try:
        response = self.llm.complete(prompt, temperature=0.1, max_tokens=500)
        content = response.content
        
        # Extract score and reasoning
        score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', content)
        reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=\nScore:|$)', content, re.DOTALL)
        
        if score_match:
            score = float(score_match.group(1)) / 10.0  # Normalize to 0-1
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        else:
            # Fallback: look for any number
            numbers = re.findall(r'\b([1-9]|10)\b', content)
            score = float(numbers[-1]) / 10.0 if numbers else 0.5
            reasoning = "Score extracted from LLM response"
        
        return MetricScore(
            name="faithfulness",
            score=score,
            reason=reasoning,
            passed=score >= self.thresholds.get("faithfulness", 0.8)
        )
        
    except Exception as e:
        logger.error(f"Faithfulness evaluation failed: {e}")
        return MetricScore("faithfulness", 0.0, f"Evaluation failed: {e}", False)
```

### Answer Relevancy Evaluation

```python
def _evaluate_answer_relevancy(self, sample: EvaluationSample) -> MetricScore:
    """
    Evaluate how well the answer addresses the original question
    """
    
    prompt = f"""Evaluate how well the answer addresses the given question.

Question: {sample.question}

Answer: {sample.answer}

A relevant answer should:
1. Directly address what the question is asking
2. Provide information that helps answer the question
3. Stay focused on the topic of the question
4. Be appropriately detailed for the question type

Rate the relevancy on a scale from 1-10 where:
- 10: Perfectly relevant - directly and completely answers the question
- 7-9: Highly relevant - addresses the main question with good detail
- 4-6: Moderately relevant - partially addresses the question
- 1-3: Low relevance - barely related to the question

Consider:
- Does the answer address the specific information requested?
- Is the answer focused on the question topic?
- Does it provide useful information for the questioner?

Reasoning: <your analysis>
Score: <score from 1-10>"""

    try:
        response = self.llm.complete(prompt, temperature=0.1, max_tokens=400)
        content = response.content
        
        score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', content)
        reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=\nScore:|$)', content, re.DOTALL)
        
        if score_match:
            score = float(score_match.group(1)) / 10.0
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        else:
            score = 0.5
            reasoning = "Could not extract score from LLM response"
        
        return MetricScore(
            name="answer_relevancy", 
            score=score,
            reason=reasoning,
            passed=score >= self.thresholds.get("answer_relevancy", 0.7)
        )
        
    except Exception as e:
        return MetricScore("answer_relevancy", 0.0, f"Evaluation failed: {e}", False)
```

### Context Evaluation Metrics

```python
def _evaluate_context_precision(self, sample: EvaluationSample) -> MetricScore:
    """
    Evaluate what fraction of retrieved contexts are relevant to the question
    """
    
    if not sample.contexts:
        return MetricScore("context_precision", 0.0, "No contexts provided", False)
    
    relevant_count = 0
    evaluations = []
    
    for i, context in enumerate(sample.contexts):
        prompt = f"""Is this context relevant for answering the given question?

Question: {sample.question}

Context: {context}

A context is relevant if it contains information that could help answer the question.

Answer with "Yes" or "No" and briefly explain why.

Relevant: <Yes/No>
Reason: <brief explanation>"""

        try:
            response = self.llm.complete(prompt, temperature=0.0, max_tokens=100)
            content = response.content.lower()
            
            is_relevant = "yes" in content.split("\n")[0] or content.startswith("yes")
            if is_relevant:
                relevant_count += 1
                
            evaluations.append(f"Context {i+1}: {'Relevant' if is_relevant else 'Not relevant'}")
            
        except Exception as e:
            logger.warning(f"Failed to evaluate context {i}: {e}")
            evaluations.append(f"Context {i+1}: Evaluation failed")
    
    precision = relevant_count / len(sample.contexts)
    reasoning = f"Found {relevant_count}/{len(sample.contexts)} relevant contexts. " + "; ".join(evaluations)
    
    return MetricScore(
        name="context_precision",
        score=precision,
        reason=reasoning,
        passed=precision >= self.thresholds.get("context_precision", 0.6)
    )

def _evaluate_context_recall(self, sample: EvaluationSample) -> MetricScore:
    """
    Evaluate what fraction of relevant information was retrieved
    Requires ground truth context or manual annotation
    """
    
    if not sample.ground_truth:
        return MetricScore("context_recall", 0.0, "Ground truth required for context recall", False)
    
    prompt = f"""Given the question and correct answer, evaluate if the provided contexts contain enough information to generate this answer.

Question: {sample.question}

Correct Answer: {sample.ground_truth}

Retrieved Contexts:
{chr(10).join(f"{i+1}. {ctx}" for i, ctx in enumerate(sample.contexts))}

Rate from 1-10 how well the contexts support generating the correct answer:
- 10: All necessary information is present
- 7-9: Most necessary information is present  
- 4-6: Some key information is missing
- 1-3: Most necessary information is missing

Score: <score from 1-10>
Missing Information: <what key information is missing, if any>"""

    try:
        response = self.llm.complete(prompt, temperature=0.1, max_tokens=200)
        content = response.content
        
        score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', content)
        missing_match = re.search(r'Missing Information:\s*(.*)', content, re.DOTALL)
        
        if score_match:
            score = float(score_match.group(1)) / 10.0
            missing_info = missing_match.group(1).strip() if missing_match else "None identified"
            reasoning = f"Context coverage score: {score:.2f}. Missing: {missing_info}"
        else:
            score = 0.5
            reasoning = "Could not extract score from evaluation"
        
        return MetricScore(
            name="context_recall",
            score=score,
            reason=reasoning,
            passed=score >= self.thresholds.get("context_recall", 0.7)
        )
        
    except Exception as e:
        return MetricScore("context_recall", 0.0, f"Evaluation failed: {e}", False)
```

### String-Based Metrics

```python
def _evaluate_rouge_l(self, sample: EvaluationSample) -> float:
    """ROUGE-L: Longest Common Subsequence based metric"""
    
    if not sample.ground_truth:
        return 0.0
    
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(sample.ground_truth, sample.answer)
    
    return scores['rougeL'].fmeasure

def _evaluate_token_f1(self, sample: EvaluationSample) -> float:
    """Token-level F1 score"""
    
    if not sample.ground_truth:
        return 0.0
    
    # Tokenize both texts
    def tokenize(text: str) -> set:
        return set(text.lower().split())
    
    pred_tokens = tokenize(sample.answer)
    true_tokens = tokenize(sample.ground_truth)
    
    if not pred_tokens and not true_tokens:
        return 1.0
    if not pred_tokens or not true_tokens:
        return 0.0
    
    # Calculate F1
    intersection = pred_tokens & true_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(true_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def _evaluate_semantic_similarity(self, sample: EvaluationSample) -> float:
    """Semantic similarity using embeddings"""
    
    if not sample.ground_truth:
        return 0.0
    
    # Get embeddings
    answer_embedding = self.embedder.encode_query(sample.answer)
    truth_embedding = self.embedder.encode_query(sample.ground_truth)
    
    # Compute cosine similarity
    similarity = np.dot(answer_embedding, truth_embedding) / (
        np.linalg.norm(answer_embedding) * np.linalg.norm(truth_embedding)
    )
    
    return float(similarity)
```

---

## RAGAS Integration

RAGAS (RAG Assessment) is the industry standard framework for RAG evaluation.

```python
class RagasEvaluator:
    def __init__(self, llm: LLMInterface, embedder: EmbeddingEngine):
        """
        RAGAS evaluation framework integration
        
        Requires: pip install ragas
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                faithfulness, 
                context_recall,
                context_precision,
                answer_correctness,
                answer_similarity
            )
            
            self.evaluate_func = evaluate
            self.metrics = {
                "answer_relevancy": answer_relevancy,
                "faithfulness": faithfulness,
                "context_recall": context_recall, 
                "context_precision": context_precision,
                "answer_correctness": answer_correctness,
                "answer_similarity": answer_similarity
            }
            
        except ImportError:
            raise ImportError("RAGAS not installed. Run: pip install ragas")
        
        self.llm = llm
        self.embedder = embedder
        
    def evaluate(self, dataset: List[Dict], 
                metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate dataset using RAGAS metrics
        
        Args:
            dataset: List of dicts with keys: question, answer, contexts, ground_truths
            metrics: Metrics to compute (default: all available)
        """
        
        from datasets import Dataset
        
        # Convert to RAGAS format
        ragas_dataset = Dataset.from_list(dataset)
        
        # Select metrics
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        selected_metrics = [self.metrics[m] for m in metrics if m in self.metrics]
        
        if not selected_metrics:
            raise ValueError(f"No valid metrics specified. Available: {list(self.metrics.keys())}")
        
        # Configure metrics with LLM and embeddings
        for metric in selected_metrics:
            if hasattr(metric, 'llm'):
                metric.llm = self._adapt_llm_for_ragas(self.llm)
            if hasattr(metric, 'embeddings'):
                metric.embeddings = self._adapt_embedder_for_ragas(self.embedder)
        
        # Evaluate
        try:
            result = self.evaluate_func(
                dataset=ragas_dataset,
                metrics=selected_metrics,
                llm=self._adapt_llm_for_ragas(self.llm),
                embeddings=self._adapt_embedder_for_ragas(self.embedder)
            )
            
            return dict(result)
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {metric: 0.0 for metric in metrics}
    
    def _adapt_llm_for_ragas(self, llm: LLMInterface):
        """Adapt our LLM interface to RAGAS format"""
        # RAGAS expects specific LLM interface - this would need implementation
        # based on RAGAS version and requirements
        pass
    
    def _adapt_embedder_for_ragas(self, embedder: EmbeddingEngine):
        """Adapt our embedder to RAGAS format"""  
        pass
```

---

## Comparative Analysis

### Metric Selection Guide

| Metric | Use Case | Reference Needed | LLM Judge | Computational Cost |
|---|---|---|---|---|
| **Faithfulness** | Hallucination detection | No | Yes | High |
| **Answer Relevancy** | Off-topic detection | No | Yes | High |
| **Context Precision** | Retrieval noise | No | Yes | High |
| **Context Recall** | Missing information | Yes | Yes | High |
| **Answer Correctness** | Factual accuracy | Yes | Yes | High |
| **ROUGE-L** | Surface similarity | Yes | No | Low |
| **Token F1** | Keyword overlap | Yes | No | Low |
| **Semantic Similarity** | Meaning preservation | Yes | No | Medium |

### Evaluation Strategy by Use Case

**Development/Debugging**:
```python
dev_metrics = ["faithfulness", "answer_relevancy", "token_f1"]
# Fast feedback loop, catches major issues
```

**Pre-production Validation**:
```python
validation_metrics = ["faithfulness", "answer_relevancy", "context_precision", 
                     "answer_correctness", "rouge_l"]
# Comprehensive quality check
```

**Production Monitoring**:
```python
prod_metrics = ["faithfulness", "answer_relevancy"]
# Automated quality monitoring without ground truth
```

**Research/Optimization**:
```python
research_metrics = ["all_available"]
# Full evaluation for systematic improvement
```

---

## Practical Guidelines

### Evaluation Dataset Construction

```python
class EvaluationDatasetBuilder:
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        
    def build_from_queries(self, queries: List[str], 
                          domain_expert_answers: Dict[str, str] = None) -> List[EvaluationSample]:
        """Build evaluation dataset from list of queries"""
        
        samples = []
        
        for query in queries:
            # Generate answer using current pipeline
            response = self.pipeline.query(query)
            
            # Get ground truth if available
            ground_truth = domain_expert_answers.get(query) if domain_expert_answers else None
            
            sample = EvaluationSample(
                question=query,
                ground_truth=ground_truth,
                answer=response.answer,
                contexts=[r.content for r in response.retrieved_chunks]
            )
            samples.append(sample)
        
        return samples
    
    def build_from_documents(self, documents: List[str], 
                           questions_per_doc: int = 3) -> List[EvaluationSample]:
        """Generate evaluation samples from documents using LLM"""
        
        samples = []
        
        for doc in documents:
            # Generate questions from document
            questions = self._generate_questions_from_document(doc, questions_per_doc)
            
            for question, expected_answer in questions:
                # Get pipeline answer
                response = self.pipeline.query(question)
                
                sample = EvaluationSample(
                    question=question,
                    ground_truth=expected_answer,
                    answer=response.answer,
                    contexts=[r.content for r in response.retrieved_chunks]
                )
                samples.append(sample)
        
        return samples
    
    def _generate_questions_from_document(self, document: str, 
                                        num_questions: int) -> List[Tuple[str, str]]:
        """Generate Q&A pairs from document content"""
        
        prompt = f"""Based on the following document, generate {num_questions} question-answer pairs that would be good for testing a RAG system.

Document: {document[:2000]}...

Generate questions that:
1. Can be answered from the document content
2. Vary in complexity (simple facts, reasoning, comparisons)
3. Test different aspects of the document

Format as:
Q1: <question>
A1: <answer>
Q2: <question>  
A2: <answer>
..."""

        try:
            response = self.pipeline.llm.complete(prompt, max_tokens=1000)
            return self._parse_qa_pairs(response.content)
        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            return []
    
    def _parse_qa_pairs(self, content: str) -> List[Tuple[str, str]]:
        """Parse Q&A pairs from LLM response"""
        
        pairs = []
        lines = content.strip().split('\n')
        
        current_q = None
        current_a = None
        
        for line in lines:
            line = line.strip()
            if re.match(r'^Q\d+:', line):
                current_q = re.sub(r'^Q\d+:\s*', '', line)
            elif re.match(r'^A\d+:', line) and current_q:
                current_a = re.sub(r'^A\d+:\s*', '', line)
                pairs.append((current_q, current_a))
                current_q = None
                current_a = None
        
        return pairs
```

### Automated Quality Monitoring

```python
class ProductionQualityMonitor:
    def __init__(self, evaluator: RAGEvaluator, 
                 alert_thresholds: Dict[str, float],
                 sample_rate: float = 0.1):
        """
        Monitor RAG quality in production
        
        Args:
            evaluator: Configured evaluator
            alert_thresholds: Metric thresholds for alerts
            sample_rate: Fraction of queries to evaluate (0.1 = 10%)
        """
        self.evaluator = evaluator
        self.alert_thresholds = alert_thresholds
        self.sample_rate = sample_rate
        
        # Sliding window for quality tracking
        self.quality_history = []
        self.window_size = 100
        
    def monitor_query(self, query: str, answer: str, 
                     retrieved_contexts: List[str]) -> Optional[Dict]:
        """Monitor a single query (called on sample_rate fraction)"""
        
        if random.random() > self.sample_rate:
            return None  # Skip this query
        
        # Evaluate (reference-free metrics only)
        sample = EvaluationSample(
            question=query,
            ground_truth=None,  # Not available in production
            answer=answer,
            contexts=retrieved_contexts
        )
        
        result = self.evaluator.evaluate_sample(sample)
        
        # Track quality over time
        self.quality_history.append(result)
        if len(self.quality_history) > self.window_size:
            self.quality_history.pop(0)
        
        # Check for quality degradation
        alerts = self._check_for_alerts(result)
        
        return {
            "metrics": {name: score.score for name, score in result.metrics.items()},
            "alerts": alerts,
            "query_id": hash(query)  # For debugging specific failures
        }
    
    def _check_for_alerts(self, result: EvaluationResult) -> List[str]:
        """Check if metrics fall below alert thresholds"""
        
        alerts = []
        
        for metric_name, score in result.metrics.items():
            threshold = self.alert_thresholds.get(metric_name)
            if threshold and score.score < threshold:
                alerts.append(f"{metric_name} below threshold: {score.score:.3f} < {threshold}")
        
        # Check for trend degradation
        if len(self.quality_history) >= 20:  # Need sufficient history
            recent_avg = self._get_recent_average("faithfulness", window=10)
            older_avg = self._get_recent_average("faithfulness", window=20, offset=10)
            
            if recent_avg and older_avg and recent_avg < older_avg * 0.9:  # 10% degradation
                alerts.append(f"Quality trend degradation detected: {recent_avg:.3f} vs {older_avg:.3f}")
        
        return alerts
    
    def _get_recent_average(self, metric_name: str, window: int, offset: int = 0) -> Optional[float]:
        """Get average metric score for recent window"""
        
        if len(self.quality_history) < window + offset:
            return None
        
        start_idx = len(self.quality_history) - window - offset
        end_idx = len(self.quality_history) - offset
        
        scores = []
        for result in self.quality_history[start_idx:end_idx]:
            if metric_name in result.metrics:
                scores.append(result.metrics[metric_name].score)
        
        return sum(scores) / len(scores) if scores else None
```

### Benchmarking Against Standards

```python
class RAGBenchmark:
    """Compare against standard RAG benchmarks"""
    
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        
    def run_ms_marco_benchmark(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate on MS MARCO dataset"""
        
        # Load MS MARCO dataset (simplified)
        marco_samples = self._load_ms_marco_samples(num_samples)
        
        results = []
        for sample in marco_samples:
            # Run pipeline
            response = self.pipeline.query(sample["query"])
            
            # Create evaluation sample
            eval_sample = EvaluationSample(
                question=sample["query"],
                ground_truth=sample["answers"][0] if sample["answers"] else None,
                answer=response.answer,
                contexts=[r.content for r in response.retrieved_chunks]
            )
            
            results.append(eval_sample)
        
        # Evaluate
        evaluator = RAGEvaluator(
            llm=self.pipeline.llm,
            embedder=self.pipeline.embedder,
            metrics=["answer_relevancy", "rouge_l", "semantic_similarity"]
        )
        
        scores = {}
        for result in results:
            eval_result = evaluator.evaluate_sample(result)
            for metric_name, score in eval_result.metrics.items():
                if metric_name not in scores:
                    scores[metric_name] = []
                scores[metric_name].append(score.score)
        
        # Return averages
        return {metric: sum(values) / len(values) for metric, values in scores.items()}
    
    def run_natural_questions_benchmark(self) -> Dict[str, float]:
        """Evaluate on Natural Questions dataset"""
        # Implementation similar to MS MARCO
        pass
        
    def run_hotpot_qa_benchmark(self) -> Dict[str, float]:
        """Evaluate on HotpotQA multi-hop reasoning"""
        pass
```

### Common Issues & Solutions

**Issue**: LLM judge metrics are inconsistent across runs
```python
# Problem: Temperature too high in judge LLM
# Solution: Use deterministic settings and structured prompts

def create_consistent_judge_config() -> Dict:
    """LLM configuration for consistent evaluation"""
    return {
        "temperature": 0.0,        # Fully deterministic
        "top_p": 1.0,             # No nucleus sampling
        "frequency_penalty": 0.0,  # No repetition penalty
        "max_tokens": 500,        # Sufficient for reasoning
        "seed": 42                # Fixed seed if supported
    }

def create_structured_evaluation_prompt(question: str, answer: str, 
                                      context: str, metric: str) -> str:
    """Create highly structured prompt for consistent evaluation"""
    
    prompt = f"""You are evaluating a RAG system response for {metric}.

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}

EVALUATION CRITERIA FOR {metric.upper()}:
"""
    
    criteria = {
        "faithfulness": """
        Score 1-10 based on:
        - 10: Every claim in answer is directly stated in context
        - 8-9: Minor inferences that are clearly supported
        - 6-7: Some claims go slightly beyond context
        - 4-5: Several unsupported claims
        - 1-3: Many fabricated or contradictory claims
        """,
        "relevancy": """
        Score 1-10 based on:
        - 10: Answer directly and completely addresses the question
        - 8-9: Answer addresses main aspects of question
        - 6-7: Answer partially relevant to question
        - 4-5: Answer tangentially related to question  
        - 1-3: Answer largely irrelevant to question
        """
    }
    
    prompt += criteria.get(metric, "Score 1-10 based on quality.")
    prompt += "\n\nProvide your evaluation as:\nReasoning: [step-by-step analysis]\nScore: [single number 1-10]"
    
    return prompt
```

**Issue**: Ground truth answers are too different from generated answers
```python
# Problem: Ground truth style doesn't match generated style
# Solution: Normalize both for comparison

def normalize_for_comparison(text: str) -> str:
    """Normalize text for fair comparison"""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common prefixes/suffixes
    prefixes = ["the answer is", "based on the context", "according to"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Remove punctuation for token-based metrics
    text = re.sub(r'[^\w\s]', ' ', text)
    
    return text

def compute_normalized_token_f1(predicted: str, ground_truth: str) -> float:
    """Token F1 with normalization"""
    
    pred_normalized = normalize_for_comparison(predicted)
    truth_normalized = normalize_for_comparison(ground_truth)
    
    pred_tokens = set(pred_normalized.split())
    truth_tokens = set(truth_normalized.split())
    
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    intersection = pred_tokens & truth_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(truth_tokens)
    
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
```

**Next concept**: Advanced RAG Patterns — multi-hop reasoning, agentic RAG, self-RAG, and CRAG architectures
