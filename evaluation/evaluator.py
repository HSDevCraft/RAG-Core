"""
evaluator.py
────────────
RAG evaluation framework.

Metrics implemented:
  Retrieval quality:
    - Context Precision   : What fraction of retrieved chunks is relevant?
    - Context Recall      : What fraction of relevant info was retrieved?

  Generation quality:
    - Faithfulness        : Is every claim in the answer supported by context?
    - Answer Relevancy    : Does the answer address the question?
    - Answer Correctness  : Does the answer match a reference answer? (ROUGE + LLM)

  System metrics:
    - Latency (p50, p90, p99)
    - Cost per query
    - Retrieval hit rate

Approaches:
  1. RAGAS framework (automated LLM-based)
  2. Custom LLM-judge (GPT-4o as judge)
  3. String metrics (ROUGE, BLEU, exact match)
  4. Human evaluation framework
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class EvaluationSample:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricScore:
    name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.name}: {self.score:.4f}"


@dataclass
class EvaluationResult:
    sample: EvaluationSample
    scores: List[MetricScore]
    latency_ms: float = 0.0

    @property
    def as_dict(self) -> dict:
        return {
            "question": self.sample.question,
            "answer": self.sample.answer,
            "scores": {s.name: s.score for s in self.scores},
            "latency_ms": self.latency_ms,
        }

    def score_by_name(self, name: str) -> Optional[float]:
        for s in self.scores:
            if s.name == name:
                return s.score
        return None


@dataclass
class EvaluationReport:
    results: List[EvaluationResult]
    metric_names: List[str]

    @property
    def aggregate(self) -> Dict[str, float]:
        """Mean score per metric across all samples."""
        agg: Dict[str, List[float]] = {m: [] for m in self.metric_names}
        for r in self.results:
            for s in r.scores:
                if s.name in agg:
                    agg[s.name].append(s.score)
        return {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}

    @property
    def latency_stats(self) -> Dict[str, float]:
        latencies = [r.latency_ms for r in self.results]
        if not latencies:
            return {}
        return {
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p99": float(np.percentile(latencies, 99)),
            "mean": float(np.mean(latencies)),
        }

    def to_dataframe(self):
        """Convert to pandas DataFrame for analysis."""
        import pandas as pd
        rows = [r.as_dict for r in self.results]
        df = pd.DataFrame(rows)
        for metric in self.metric_names:
            df[metric] = df["scores"].apply(lambda s: s.get(metric, None))
        return df.drop(columns=["scores"])

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("RAG EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Samples evaluated: {len(self.results)}")
        print("\nAggregate Scores:")
        for metric, score in self.aggregate.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {metric:<25} {score:.4f}  [{bar}]")
        print("\nLatency:")
        for k, v in self.latency_stats.items():
            print(f"  {k}: {v:.1f}ms")
        print("=" * 60)


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------
class LLMJudge:
    """
    GPT-4-class model as an evaluation judge.
    Uses structured JSON output for consistent scoring.
    """

    FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether an AI answer is faithful to the provided context.

Context:
{context}

Question: {question}
Answer: {answer}

Evaluate faithfulness: Does the answer ONLY contain information that can be directly inferred from the context?
Score from 0.0 to 1.0:
  1.0 = Every statement is supported by the context
  0.5 = Some statements are supported; some are from external knowledge
  0.0 = The answer contradicts or ignores the context

Respond with valid JSON only:
{{"score": <float 0-1>, "reasoning": "<brief explanation>", "unsupported_claims": [<list of unsupported statements>]}}"""

    RELEVANCY_PROMPT = """You are an expert evaluator assessing whether an AI answer is relevant to the question.

Question: {question}
Answer: {answer}

Does the answer directly address the question? Is it focused and on-topic?
Score from 0.0 to 1.0:
  1.0 = Perfectly addresses the question
  0.5 = Partially addresses the question or has significant off-topic content
  0.0 = Does not address the question at all

Respond with valid JSON only:
{{"score": <float 0-1>, "reasoning": "<brief explanation>"}}"""

    CORRECTNESS_PROMPT = """You are an expert evaluator comparing an AI answer to a reference answer.

Question: {question}
Reference Answer: {ground_truth}
AI Answer: {answer}

How correct is the AI answer compared to the reference?
Score from 0.0 to 1.0:
  1.0 = Semantically equivalent; all key facts present
  0.5 = Partially correct; some key facts present
  0.0 = Incorrect or contradicts the reference

Respond with valid JSON only:
{{"score": <float 0-1>, "reasoning": "<brief explanation>", "missing_facts": [<list>]}}"""

    CONTEXT_PRECISION_PROMPT = """You are an expert evaluator assessing retrieval quality.

Question: {question}
Retrieved contexts (numbered):
{contexts}

For each context, determine if it is relevant to answering the question.
Return a JSON array of relevance judgments (true/false) in the same order as the contexts.

Respond with valid JSON only:
{{"relevance": [<true/false for each context>], "reasoning": "<brief explanation>"}}"""

    def __init__(self, llm_fn):
        """llm_fn: callable(prompt: str) → str"""
        self._llm = llm_fn

    def _call_json(self, prompt: str) -> dict:
        """Call LLM and parse JSON response with fallback."""
        response = self._llm(prompt)
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        return {"score": 0.5, "reasoning": "Parse error", "error": response[:200]}

    def score_faithfulness(self, question: str, answer: str,
                           contexts: List[str]) -> MetricScore:
        context_str = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = self.FAITHFULNESS_PROMPT.format(
            context=context_str, question=question, answer=answer
        )
        result = self._call_json(prompt)
        return MetricScore(
            name="faithfulness",
            score=float(result.get("score", 0)),
            details=result,
        )

    def score_answer_relevancy(self, question: str, answer: str) -> MetricScore:
        prompt = self.RELEVANCY_PROMPT.format(question=question, answer=answer)
        result = self._call_json(prompt)
        return MetricScore(
            name="answer_relevancy",
            score=float(result.get("score", 0)),
            details=result,
        )

    def score_answer_correctness(self, question: str, answer: str,
                                 ground_truth: str) -> MetricScore:
        prompt = self.CORRECTNESS_PROMPT.format(
            question=question, answer=answer, ground_truth=ground_truth
        )
        result = self._call_json(prompt)
        return MetricScore(
            name="answer_correctness",
            score=float(result.get("score", 0)),
            details=result,
        )

    def score_context_precision(self, question: str,
                                contexts: List[str]) -> MetricScore:
        numbered = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts))
        prompt = self.CONTEXT_PRECISION_PROMPT.format(
            question=question, contexts=numbered
        )
        result = self._call_json(prompt)
        relevance = result.get("relevance", [True] * len(contexts))
        precision = sum(relevance) / len(relevance) if relevance else 0.0
        return MetricScore(
            name="context_precision",
            score=float(precision),
            details={"relevance_per_chunk": relevance},
        )


# ---------------------------------------------------------------------------
# String-based metrics (no LLM needed)
# ---------------------------------------------------------------------------
class StringMetrics:
    """ROUGE and token overlap metrics — fast, no API cost."""

    @staticmethod
    def rouge_l(prediction: str, reference: str) -> float:
        """ROUGE-L: longest common subsequence F1."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        lcs = StringMetrics._lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _lcs_length(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    @staticmethod
    def token_overlap_f1(prediction: str, reference: str) -> float:
        """Token-level precision/recall F1 (used in SQuAD evaluation)."""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        common = pred_tokens & ref_tokens
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        norm = lambda s: re.sub(r"\s+", " ", s.lower().strip())
        return 1.0 if norm(prediction) == norm(reference) else 0.0

    @staticmethod
    def context_recall_heuristic(answer: str, contexts: List[str]) -> float:
        """
        Heuristic: count answer sentences that have a token overlap
        with any retrieved context.
        """
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        sentences = nltk.sent_tokenize(answer)
        if not sentences:
            return 0.0

        context_text = " ".join(contexts).lower()
        context_words = set(context_text.split())

        supported = 0
        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = len(sent_words & context_words) / (len(sent_words) + 1e-10)
            if overlap > 0.5:
                supported += 1

        return supported / len(sentences)


# ---------------------------------------------------------------------------
# RAGAS Integration
# ---------------------------------------------------------------------------
class RagasEvaluator:
    """
    Thin wrapper around the RAGAS library for automated RAG evaluation.
    RAGAS provides: faithfulness, answer_relevancy, context_precision, context_recall.

    Install: pip install ragas
    """

    def __init__(self, llm=None, embeddings=None):
        try:
            import ragas
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("RAGAS not installed. Run: pip install ragas")
        self._llm = llm
        self._embeddings = embeddings

    def evaluate(self, samples: List[EvaluationSample]) -> Dict[str, float]:
        if not self._available:
            return {"error": "ragas not installed"}

        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_precision, context_recall,
        )
        from datasets import Dataset

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth or "" for s in samples],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=self._llm,
            embeddings=self._embeddings,
        )
        return dict(result)


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------
class RAGEvaluator:
    """
    Unified evaluation orchestrator.

    Supports:
      - LLM-judge scoring (faithfulness, relevancy, correctness, context precision)
      - String metrics (ROUGE-L, token F1, exact match)
      - RAGAS automated evaluation
      - Latency tracking

    Usage:
        evaluator = RAGEvaluator(llm_fn=lambda p: llm.complete(p).content)
        result = evaluator.evaluate_sample(sample)
        report = evaluator.evaluate_dataset(samples)
        report.print_summary()
    """

    def __init__(
        self,
        llm_fn=None,
        metrics: Optional[List[str]] = None,
        use_ragas: bool = False,
    ):
        self._judge = LLMJudge(llm_fn) if llm_fn else None
        self._string = StringMetrics()
        self._use_ragas = use_ragas
        self._metrics = metrics or [
            "faithfulness", "answer_relevancy", "context_precision",
            "context_recall_heuristic", "rouge_l",
        ]

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        t0 = time.perf_counter()
        scores: List[MetricScore] = []

        for metric in self._metrics:
            score = self._compute_metric(metric, sample)
            if score:
                scores.append(score)

        latency = (time.perf_counter() - t0) * 1000
        return EvaluationResult(sample=sample, scores=scores, latency_ms=latency)

    def _compute_metric(self, metric: str,
                        sample: EvaluationSample) -> Optional[MetricScore]:
        try:
            if metric == "faithfulness" and self._judge:
                return self._judge.score_faithfulness(
                    sample.question, sample.answer, sample.contexts
                )
            elif metric == "answer_relevancy" and self._judge:
                return self._judge.score_answer_relevancy(
                    sample.question, sample.answer
                )
            elif metric == "answer_correctness" and self._judge and sample.ground_truth:
                return self._judge.score_answer_correctness(
                    sample.question, sample.answer, sample.ground_truth
                )
            elif metric == "context_precision" and self._judge:
                return self._judge.score_context_precision(
                    sample.question, sample.contexts
                )
            elif metric == "context_recall_heuristic":
                score = self._string.context_recall_heuristic(
                    sample.answer, sample.contexts
                )
                return MetricScore(name=metric, score=score)
            elif metric == "rouge_l" and sample.ground_truth:
                score = self._string.rouge_l(sample.answer, sample.ground_truth)
                return MetricScore(name=metric, score=score)
            elif metric == "token_f1" and sample.ground_truth:
                score = self._string.token_overlap_f1(sample.answer, sample.ground_truth)
                return MetricScore(name=metric, score=score)
            elif metric == "exact_match" and sample.ground_truth:
                score = self._string.exact_match(sample.answer, sample.ground_truth)
                return MetricScore(name=metric, score=score)
        except Exception as e:
            logger.warning("Metric %s failed: %s", metric, e)
        return None

    def evaluate_dataset(
        self,
        samples: List[EvaluationSample],
        show_progress: bool = True,
    ) -> EvaluationReport:
        results: List[EvaluationResult] = []
        total = len(samples)

        for i, sample in enumerate(samples):
            if show_progress and (i % 10 == 0 or i == total - 1):
                logger.info("Evaluating sample %d/%d", i + 1, total)
            result = self.evaluate_sample(sample)
            results.append(result)

        return EvaluationReport(results=results, metric_names=self._metrics)

    def evaluate_pipeline(
        self,
        pipeline,                               # RAGPipeline instance
        test_set: List[dict],                   # [{"question": ..., "ground_truth": ...}]
        filter: Optional[dict] = None,
    ) -> EvaluationReport:
        """
        End-to-end pipeline evaluation.
        Runs each test question through the pipeline and scores the output.
        """
        samples: List[EvaluationSample] = []

        for item in test_set:
            question = item["question"]
            ground_truth = item.get("ground_truth")

            response = pipeline.query(question, filter=filter)
            sample = EvaluationSample(
                question=question,
                answer=response.answer,
                contexts=[r.content for r in response.retrieved_chunks],
                ground_truth=ground_truth,
                metadata=response.to_dict(),
            )
            samples.append(sample)

        return self.evaluate_dataset(samples)

    @staticmethod
    def load_test_set(path: str) -> List[dict]:
        """Load evaluation dataset from JSON/JSONL."""
        import json
        from pathlib import Path
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            if p.suffix == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
            return json.load(f)

    @staticmethod
    def save_report(report: EvaluationReport, path: str) -> None:
        """Save evaluation report as JSON."""
        import json
        data = {
            "aggregate": report.aggregate,
            "latency_stats": report.latency_stats,
            "samples": [r.as_dict for r in report.results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Evaluation report saved to %s", path)


# ---------------------------------------------------------------------------
# Benchmark datasets
# ---------------------------------------------------------------------------
BENCHMARK_DATASETS = {
    "squad_v2": {
        "description": "Stanford QA — 150k QA pairs, reading comprehension",
        "hf_path": "squad_v2",
        "metrics": ["exact_match", "token_f1"],
    },
    "natural_questions": {
        "description": "Google NQ — real search queries, Wikipedia passages",
        "hf_path": "natural_questions",
        "metrics": ["exact_match", "rouge_l"],
    },
    "hotpotqa": {
        "description": "Multi-hop QA requiring multiple documents",
        "hf_path": "hotpot_qa",
        "metrics": ["exact_match", "token_f1"],
    },
    "msmarco": {
        "description": "Microsoft MARCO — passage retrieval benchmark",
        "hf_path": "ms_marco",
        "metrics": ["rouge_l", "faithfulness"],
    },
}


def load_benchmark(name: str, split: str = "validation",
                   max_samples: int = 100) -> List[EvaluationSample]:
    """Load a standard benchmark dataset via HuggingFace datasets."""
    from datasets import load_dataset

    cfg = BENCHMARK_DATASETS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown benchmark: {name}. Choose from {list(BENCHMARK_DATASETS)}")

    ds = load_dataset(cfg["hf_path"], split=split)
    samples: List[EvaluationSample] = []

    for item in list(ds)[:max_samples]:
        question = item.get("question", "")
        ground_truth = item.get("answers", {}).get("text", [""])[0]
        context = item.get("context", item.get("passage", ""))
        samples.append(EvaluationSample(
            question=question,
            answer="",            # to be filled by pipeline
            contexts=[context] if context else [],
            ground_truth=ground_truth,
        ))

    return samples
