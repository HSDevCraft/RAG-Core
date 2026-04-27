# Module 07 — Evaluation
### `evaluation/evaluator.py`

---

## Table of Contents
1. [Why Evaluate RAG?](#1-why-evaluate-rag)
2. [Evaluation Taxonomy](#2-evaluation-taxonomy)
3. [Core Data Models](#3-core-data-models)
4. [String-Based Metrics](#4-string-based-metrics)
5. [LLM-Judge Metrics](#5-llm-judge-metrics)
6. [RAGAS Integration](#6-ragas-integration)
7. [RAGEvaluator Façade](#7-ragevaluator-façade)
8. [Benchmark Datasets](#8-benchmark-datasets)
9. [EvaluationReport](#9-evaluationreport)
10. [Quick Reference](#10-quick-reference)
11. [Common Pitfalls](#11-common-pitfalls)

---

## 1. Why Evaluate RAG?

Without evaluation, you're flying blind. Evaluation answers:

| Question | Metric |
|---|---|
| Is the answer grounded in context? | **Faithfulness** |
| Does the answer address the question? | **Answer Relevancy** |
| Did we retrieve the right chunks? | **Context Precision / Recall** |
| Is the answer factually correct? | **Answer Correctness** |
| How fast is the pipeline? | **Latency (p50/p95/p99)** |
| How much does it cost per query? | **Cost per query ($)** |

```
Development cycle with evaluation:

Change (chunking/retrieval/prompt)
          │
          ▼
    Run evaluator on test set (50–200 samples)
          │
          ▼
    Did metrics improve?
    YES → keep change, update baseline
    NO  → revert, try something else
```

---

## 2. Evaluation Taxonomy

```
Metrics
├── Reference-Free (no ground truth needed)
│   ├── Faithfulness      ← answer grounded in retrieved context?
│   └── Answer Relevancy  ← answer relevant to the question?
│
├── Reference-Based (need ground truth answer)
│   ├── Answer Correctness  ← matches expected answer?
│   ├── ROUGE-L             ← longest common subsequence overlap
│   └── Token F1            ← token-level precision + recall
│
└── Retrieval-Only
    ├── Context Precision   ← fraction of retrieved chunks that are relevant
    └── Context Recall      ← fraction of relevant chunks actually retrieved
```

---

## 3. Core Data Models

### EvaluationSample

One row in your evaluation dataset:

```python
@dataclass
class EvaluationSample:
    question:          str               # the input query
    ground_truth:      str               # expected correct answer
    answer:            str               # RAG-generated answer (to evaluate)
    contexts:          List[str]         # retrieved chunk texts
    context_relevance: Optional[List[bool]] = None  # manual relevance labels
```

### MetricScore

```python
@dataclass
class MetricScore:
    name:    str     # e.g., "faithfulness"
    score:   float   # 0.0–1.0
    reason:  str     # explanation from LLM judge
    passed:  bool    # score >= threshold
```

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    sample:   EvaluationSample
    metrics:  Dict[str, MetricScore]   # metric_name → MetricScore
    latency_ms: float
```

### EvaluationReport

```python
@dataclass
class EvaluationReport:
    results:     List[EvaluationResult]
    summary:     Dict[str, float]     # metric_name → mean score across samples
    failures:    List[EvaluationResult]  # results below threshold
    config:      Dict
    created_at:  float

    def to_dict(self) -> dict: ...
    def save(self, path: str) -> None: ...
```

---

## 4. String-Based Metrics

Fast, free, no LLM needed. Use for quick automated regression testing.

### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)

Measures the **longest common subsequence** between generated and reference answer.

```
Reference: "refund within 30 days"
Generated: "customers get a refund within 30 days of purchase"

LCS: "refund within 30 days"  (length 4 tokens)
Precision: 4/8 = 0.50
Recall:    4/4 = 1.00
F1:        2×0.5×1.0/(0.5+1.0) = 0.667
```

### Token F1

Token-level overlap regardless of order:

```
Reference tokens: {refund, within, 30, days}
Generated tokens: {customers, get, refund, within, 30, days, purchase}

Precision: |intersection| / |generated| = 4/7 = 0.571
Recall:    |intersection| / |reference| = 4/4 = 1.000
F1 = 0.727
```

```python
from evaluation.evaluator import StringMetrics

metrics = StringMetrics()

rouge_score = metrics.rouge_l(
    prediction="customers get a refund within 30 days",
    reference="refund within 30 days",
)   # → 0.667

f1_score = metrics.token_f1(
    prediction="customers get a refund within 30 days",
    reference="refund within 30 days",
)   # → 0.727
```

**Thresholds** (guideline):
| Score | Interpretation |
|---|---|
| ≥ 0.8 | Excellent — near-verbatim match |
| 0.6–0.8 | Good — correct answer, different wording |
| 0.4–0.6 | Partial — some relevant content |
| < 0.4 | Poor — likely incorrect or incomplete |

---

## 5. LLM-Judge Metrics

The LLM evaluates quality aspects that string matching can't capture.
Each metric gets a 0–10 score from the judge, normalised to 0–1.

### 5.1 Faithfulness

**"Does every claim in the answer appear in the retrieved context?"**

```
Context: "Customers can return items within 30 days for a full refund."
Answer:  "You can get a full refund within 30 days."
→ Faithful: YES (0.95)

Answer:  "You can get a full refund within 60 days and free return shipping."
→ Faithful: NO  (0.30) — "60 days" and "free return shipping" not in context
```

```python
from evaluation.evaluator import LLMJudge

judge = LLMJudge(llm=llm_interface)

score = judge.faithfulness(
    answer="You can get a full refund within 30 days.",
    contexts=["Customers can return items within 30 days for a full refund."],
)
# MetricScore(name="faithfulness", score=0.95, reason="All claims supported…", passed=True)
```

**Judge prompt**:
```
Rate (0-10) how well the answer is supported by the context.
10 = every claim in the answer is directly stated in the context
0  = the answer contradicts or ignores the context entirely

Context: {context}
Answer: {answer}

Score:
```

### 5.2 Answer Relevancy

**"Does the answer actually address what was asked?"**

```
Question: "What is the refund policy?"
Answer: "Our products are high quality." → Score: 0.05 (irrelevant!)
Answer: "Refunds are available within 30 days." → Score: 0.92
```

### 5.3 Context Precision

**"What fraction of retrieved chunks are actually relevant?"**

```
Retrieved: [relevant, irrelevant, relevant, relevant, irrelevant]
Precision: 3/5 = 0.60
```

High precision → retriever is not polluting context with noise.

### 5.4 Context Recall

**"Were all relevant chunks actually retrieved?"**

```
Ground truth needed chunks: [chunk-A, chunk-B, chunk-C]
Retrieved:                  [chunk-A, chunk-B, chunk-D]
Recall: 2/3 = 0.67  (chunk-C was missed)
```

High recall → retriever doesn't miss important information.

### 5.5 Answer Correctness

```python
score = judge.answer_correctness(
    question="What is the refund window?",
    answer="30 days from purchase date.",
    ground_truth="Customers have 30 days to return products.",
)
# 0.90 — semantically correct, slightly different wording
```

---

## 6. RAGAS Integration

[RAGAS](https://docs.ragas.io) is the industry-standard RAG evaluation framework.

```python
from evaluation.evaluator import RagasEvaluator

evaluator = RagasEvaluator(llm=llm_interface, embedder=embedder)

# RAGAS dataset format
dataset = [
    {
        "question": "What is the refund policy?",
        "answer": "30 days full refund.",
        "contexts": ["policy.pdf chunk 1", "policy.pdf chunk 2"],
        "ground_truth": "Customers can return items within 30 days.",
    },
    ...
]

report = evaluator.evaluate(dataset, metrics=["faithfulness", "answer_relevancy",
                                               "context_precision", "context_recall"])
print(report["faithfulness"])          # 0.87
print(report["context_precision"])     # 0.72
```

**RAGAS metrics use the LLM internally** — each metric makes 1–3 LLM calls
per sample. Budget accordingly:
```
100 samples × 4 metrics × 2 LLM calls × $0.00015 = ~$0.12 (gpt-4o-mini)
```

---

## 7. RAGEvaluator Façade

```python
from evaluation import RAGEvaluator

evaluator = RAGEvaluator(
    llm=llm_interface,
    embedder=embedder,
    metrics=["faithfulness", "answer_relevancy", "rouge_l", "token_f1"],
    thresholds={
        "faithfulness":     0.8,
        "answer_relevancy": 0.7,
        "rouge_l":          0.5,
        "token_f1":         0.6,
    },
)

# Evaluate a single sample
sample = EvaluationSample(
    question="What is the refund policy?",
    ground_truth="30-day full refund.",
    answer=pipeline.query("What is the refund policy?").answer,
    contexts=[r.content for r in pipeline.retrieve("What is the refund policy?")],
)
result = evaluator.evaluate_sample(sample)
print(result.metrics["faithfulness"].score)   # 0.91

# Evaluate a full test set
report = evaluator.evaluate_dataset(samples)
report.save("./evaluation_report.json")
print(report.summary)
# {'faithfulness': 0.87, 'answer_relevancy': 0.82, 'rouge_l': 0.71, 'token_f1': 0.74}
```

### Failure Analysis

```python
# Samples that fell below threshold
for failure in report.failures:
    print(f"Q: {failure.sample.question}")
    print(f"A: {failure.sample.answer[:100]}")
    for name, score in failure.metrics.items():
        if not score.passed:
            print(f"  FAIL {name}: {score.score:.2f} — {score.reason}")
    print()
```

---

## 8. Benchmark Datasets

```python
# Load a pre-built benchmark (HuggingFace datasets integration)
samples = evaluator.load_benchmark(
    name="squad",        # "squad" | "natural_questions" | "hotpot_qa" | custom
    split="validation",
    max_samples=100,
)

# Or build your own from the RAG pipeline
samples = []
for q, gt in your_qna_pairs:
    response = pipeline.query(q)
    samples.append(EvaluationSample(
        question=q,
        ground_truth=gt,
        answer=response.answer,
        contexts=[r.content for r in response.retrieved_chunks],
    ))
```

---

## 9. EvaluationReport

```python
report.save("./evaluation_report.json")
# Saves:
# {
#   "summary": {"faithfulness": 0.87, "answer_relevancy": 0.82, …},
#   "num_samples": 100,
#   "num_failures": 13,
#   "failure_rate": 0.13,
#   "results": [...per-sample details...],
#   "config": {"metrics": […], "thresholds": {…}},
#   "created_at": "2024-01-15T10:30:00Z"
# }
```

---

## 10. Quick Reference

```python
from evaluation import RAGEvaluator, EvaluationSample

# ── Create evaluator ──────────────────────────────────────────────────────
ev = RAGEvaluator(llm=llm, embedder=emb, metrics=["faithfulness", "rouge_l"])

# ── Single sample ─────────────────────────────────────────────────────────
sample = EvaluationSample(
    question="Q", ground_truth="GT", answer="A", contexts=["ctx1", "ctx2"]
)
result = ev.evaluate_sample(sample)
print(result.metrics["faithfulness"].score)  # 0.0 – 1.0
print(result.metrics["faithfulness"].passed) # True / False

# ── Full dataset ─────────────────────────────────────────────────────────
report = ev.evaluate_dataset(samples)
print(report.summary)          # mean scores per metric
print(len(report.failures))    # count below threshold
report.save("report.json")

# ── RAGAS ────────────────────────────────────────────────────────────────
from evaluation.evaluator import RagasEvaluator
ragas_ev = RagasEvaluator(llm=llm, embedder=emb)
scores = ragas_ev.evaluate(ragas_dataset)

# ── String metrics only (no LLM) ─────────────────────────────────────────
from evaluation.evaluator import StringMetrics
sm = StringMetrics()
print(sm.rouge_l(prediction="answer", reference="expected"))
print(sm.token_f1(prediction="answer", reference="expected"))
```

---

## 11. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| **LLM judge is lenient** | Faithfulness always 0.9+ | Use stricter judge prompt or GPT-4 instead of mini |
| **Too few eval samples** | Metrics vary widely run-to-run | Use ≥ 50 samples for stable averages |
| **Ground truth too vague** | Token F1 always low | Write specific, precise ground-truth answers |
| **Evaluating with same LLM as generation** | Judge agrees with generator | Use a different (stronger) model for judging |
| **RAGAS API costs** | $5+ per eval run | Use gpt-4o-mini; batch to 100 samples max |
| **No reference → can't use correctness** | KeyError | Use reference-free metrics (faithfulness, relevancy) |
| **Failure threshold too strict** | Everything fails | Start with 0.6–0.7 threshold, raise over time |
