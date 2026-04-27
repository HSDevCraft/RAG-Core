"""
rag_pipeline.py
───────────────
Master orchestrator for the end-to-end RAG pipeline.

Architecture flow:
  INDEXING  : Documents → Chunks → Embeddings → VectorStore + BM25
  RETRIEVAL : Query → Embed → [HyDE/MultiQuery] → Hybrid Search → Rerank → [MMR]
  GENERATION: Context → PromptBuilder → LLM → Response + Citations

Advanced capabilities:
  - Multi-hop retrieval (iterative evidence gathering)
  - Streaming generation
  - Caching (Redis / disk)
  - Observability hooks (Langfuse)
  - Memory-augmented (conversation history)
  - Tool-augmented (pluggable tool registry)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Callable

from config import RAGConfig, DEFAULT_CONFIG
from data_pipeline.document_loader import DocumentLoader, Document
from data_pipeline.chunking import ChunkingEngine, Chunk
from embeddings.embedding import EmbeddingEngine
from vector_store.vector_store import VectorStoreFactory, BaseVectorStore, SearchResult
from retrieval.retriever import DenseRetriever, BM25Retriever, HybridRetriever, QueryTransformer, maximal_marginal_relevance
from retrieval.reranker import CrossEncoderReranker, RerankerFactory
from generation.prompt_builder import PromptBuilder, ConversationHistory
from generation.llm_interface import LLMInterface, LLMResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------
@dataclass
class RAGResponse:
    query: str
    answer: str
    citations: List[dict]
    retrieved_chunks: List[SearchResult]
    llm_response: Optional[LLMResponse] = None
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return sum(self.latency_breakdown.values())

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "num_chunks_retrieved": len(self.retrieved_chunks),
            "latency_ms": self.latency_breakdown,
            "total_latency_ms": self.total_latency_ms,
            "llm_tokens": self.llm_response.to_dict().get("tokens") if self.llm_response else None,
        }


# ---------------------------------------------------------------------------
# Indexing result
# ---------------------------------------------------------------------------
@dataclass
class IndexingResult:
    num_documents: int
    num_chunks: int
    embedding_latency_ms: float
    total_latency_ms: float
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------
class RAGPipeline:
    """
    Full RAG pipeline — indexing + retrieval + generation.

    Usage (quick start):
        pipeline = RAGPipeline.from_config()
        pipeline.index(["./docs/"])
        response = pipeline.query("What is the refund policy?")
        print(response.answer)

    Usage (custom):
        pipeline = RAGPipeline(
            embedder=EmbeddingEngine(provider="openai"),
            vector_store=VectorStoreFactory.create("chroma"),
            llm=LLMInterface(provider="openai", model="gpt-4o"),
            config=RAGConfig(),
        )
    """

    def __init__(
        self,
        embedder: EmbeddingEngine,
        vector_store: BaseVectorStore,
        llm: LLMInterface,
        config: RAGConfig = DEFAULT_CONFIG,
        reranker=None,
        cache=None,
        observability=None,
    ):
        self.config = config
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm
        self._reranker = reranker
        self._cache = cache
        self._obs = observability

        # Build internal components
        self._doc_loader = DocumentLoader()
        self._chunker = ChunkingEngine(
            strategy=config.chunking.strategy,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        self._bm25 = BM25Retriever()
        self._dense_retriever = DenseRetriever(vector_store, embedder)
        self._hybrid_retriever = HybridRetriever(
            self._dense_retriever,
            self._bm25,
            use_rrf=True,
        )
        self._prompt_builder = PromptBuilder(
            template="with_citations" if config.generation.citation_style != "none" else "default",
            max_context_tokens=config.generation.max_context_tokens,
            citation_style=config.generation.citation_style,
        )

        # Chunk registry (for MMR embedding lookup)
        self._chunk_embeddings: Dict[str, Any] = {}

        # Conversation histories (session_id → ConversationHistory)
        self._histories: Dict[str, ConversationHistory] = {}

        logger.info("RAGPipeline initialized: embedder=%s, store=%s, llm=%s",
                    embedder.provider, type(vector_store).__name__, llm.provider)

    # -----------------------------------------------------------------------
    # Factory method
    # -----------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: RAGConfig = DEFAULT_CONFIG) -> "RAGPipeline":
        """Build pipeline from a RAGConfig object."""
        embedder = EmbeddingEngine(
            provider=config.embedding.provider,
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            use_cache=True,
        )
        vector_store = VectorStoreFactory.create(
            backend=config.vector_store.backend,
            dimension=config.embedding.dimension,
            index_type=config.vector_store.index_type,
            metric=config.vector_store.metric,
            persist_dir=config.vector_store.persist_dir,
        )
        llm = LLMInterface(
            provider=config.llm.provider,
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        reranker = (
            CrossEncoderReranker(
                model_name=config.retrieval.reranker_model,
                batch_size=config.retrieval.reranker_batch_size,
            )
            if config.retrieval.use_reranker else None
        )
        return cls(
            embedder=embedder,
            vector_store=vector_store,
            llm=llm,
            config=config,
            reranker=reranker,
        )

    # -----------------------------------------------------------------------
    # INDEXING
    # -----------------------------------------------------------------------
    def index(
        self,
        sources: List[str],
        loader_kwargs: Optional[Dict] = None,
        batch_size: int = 256,
    ) -> IndexingResult:
        """
        Index one or more sources (files, directories, URLs).

        Args:
            sources: List of file paths, directory paths, or URLs.
            loader_kwargs: Extra kwargs passed to DocumentLoader.
            batch_size: Embedding batch size.

        Returns:
            IndexingResult with stats.
        """
        t_start = time.perf_counter()
        errors: List[str] = []
        all_docs: List[Document] = []

        for src in sources:
            try:
                docs = self._doc_loader.load(src, **(loader_kwargs or {}))
                all_docs.extend(docs)
                logger.info("Loaded %d docs from %s", len(docs), src)
            except Exception as e:
                errors.append(f"{src}: {e}")
                logger.error("Failed to load %s: %s", src, e)

        if not all_docs:
            return IndexingResult(0, 0, 0, 0, errors)

        # Chunk
        chunks = self._chunker.split(all_docs)
        logger.info("Chunking: %d docs → %d chunks", len(all_docs), len(chunks))

        # Embed
        t_embed = time.perf_counter()
        texts = [c.content for c in chunks]
        embedding_result = self._embedder.encode(texts, batch_size=batch_size)
        vectors = embedding_result.embeddings
        embed_latency = (time.perf_counter() - t_embed) * 1000

        # Store in vector DB
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self._vector_store.add(ids, vectors, texts, metadatas)

        # Store for MMR lookup
        for i, chunk in enumerate(chunks):
            self._chunk_embeddings[chunk.chunk_id] = vectors[i]

        # Build BM25 index
        self._bm25.build(texts, ids, metadatas)

        total_latency = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Indexing complete: %d docs, %d chunks, embed_latency=%.0fms, total=%.0fms",
            len(all_docs), len(chunks), embed_latency, total_latency,
        )

        return IndexingResult(
            num_documents=len(all_docs),
            num_chunks=len(chunks),
            embedding_latency_ms=embed_latency,
            total_latency_ms=total_latency,
            errors=errors,
        )

    def index_documents(self, documents: List[Document], batch_size: int = 256) -> IndexingResult:
        """
        Index pre-loaded Document objects directly (skip the loading step).

        Use this when you have already loaded and pre-processed your documents
        outside the pipeline (e.g., from a database, API, or custom loader).

        Args:
            documents: List of Document objects to chunk, embed, and index.
            batch_size: Number of texts to embed per batch.

        Returns:
            IndexingResult with chunk count, latency, and any errors.
        """
        if not documents:
            logger.warning("index_documents called with empty document list")
            return IndexingResult(0, 0, 0.0, 0.0, ["No documents provided"])

        t_start = time.perf_counter()
        chunks = self._chunker.split(documents)
        logger.info("index_documents: %d docs → %d chunks", len(documents), len(chunks))

        texts = [c.content for c in chunks]
        t_embed = time.perf_counter()
        embedding_result = self._embedder.encode(texts, batch_size=batch_size)
        vectors = embedding_result.embeddings
        embed_latency = (time.perf_counter() - t_embed) * 1000

        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self._vector_store.add(ids, vectors, texts, metadatas)

        for i, chunk in enumerate(chunks):
            self._chunk_embeddings[chunk.chunk_id] = vectors[i]

        self._bm25.build(texts, ids, metadatas)
        total_latency = (time.perf_counter() - t_start) * 1000
        logger.info(
            "index_documents complete: %d chunks, embed=%.0fms, total=%.0fms",
            len(chunks), embed_latency, total_latency,
        )
        return IndexingResult(len(documents), len(chunks), embed_latency, total_latency)

    # -----------------------------------------------------------------------
    # RETRIEVAL
    # -----------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
        use_hyde: bool = False,
        use_multi_query: bool = False,
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks for a query.

        Optional query transformations:
          use_hyde       : embed a hypothetical answer instead of the raw query
          use_multi_query: generate N paraphrases and union results
        """
        cfg = self.config.retrieval
        n_candidates = top_k or cfg.top_k
        final_k = cfg.final_top_k

        t0 = time.perf_counter()

        # Query transformation
        effective_query = query
        if use_hyde or use_multi_query:
            transformer = QueryTransformer(
                llm_fn=lambda p: self._llm.complete(p).content
            )
            if use_hyde:
                effective_query = transformer.hyde(query)
                logger.debug("HyDE query: %s", effective_query[:100])
            elif use_multi_query:
                queries = transformer.multi_query(query, n=3)
                # Union results from all query variants
                seen: Dict[str, SearchResult] = {}
                for q in queries:
                    results = self._get_candidates(q, n_candidates, filter)
                    for r in results:
                        if r.chunk_id not in seen:
                            seen[r.chunk_id] = r
                candidates = list(seen.values())
                logger.debug("MultiQuery union: %d unique chunks", len(candidates))
                return self._finalize(candidates, query, final_k)

        candidates = self._get_candidates(effective_query, n_candidates, filter)
        return self._finalize(candidates, query, final_k)

    def _get_candidates(self, query: str, top_k: int,
                        filter: Optional[dict]) -> List[SearchResult]:
        """Run hybrid or dense retrieval."""
        if self.config.retrieval.use_hybrid and self._bm25._bm25 is not None:
            return self._hybrid_retriever.retrieve(query, top_k=top_k, filter=filter)
        return self._dense_retriever.retrieve(query, top_k=top_k, filter=filter)

    def _finalize(self, candidates: List[SearchResult], query: str,
                  top_k: int) -> List[SearchResult]:
        """Rerank → MMR → trim to top_k."""
        if self._reranker and candidates:
            candidates = self._reranker.rerank(query, candidates)

        if self.config.retrieval.use_mmr and self._chunk_embeddings:
            import numpy as np
            query_vec = self._embedder.encode_query(query)
            cand_vecs = []
            valid_candidates = []
            for c in candidates:
                if c.chunk_id in self._chunk_embeddings:
                    cand_vecs.append(self._chunk_embeddings[c.chunk_id])
                    valid_candidates.append(c)
            if cand_vecs:
                cand_arr = np.vstack(cand_vecs)
                candidates = maximal_marginal_relevance(
                    query_vec, cand_arr, valid_candidates,
                    top_k=top_k,
                    lambda_param=self.config.retrieval.mmr_lambda,
                )

        return candidates[:top_k]

    # -----------------------------------------------------------------------
    # GENERATION
    # -----------------------------------------------------------------------
    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        filter: Optional[dict] = None,
        use_hyde: bool = False,
        use_multi_query: bool = False,
        template: Optional[str] = None,
    ) -> RAGResponse:
        """
        End-to-end RAG: retrieve + generate.

        Args:
            question    : User question string.
            session_id  : Optional session ID for multi-turn conversation.
            filter      : Metadata filter for retrieval (e.g. {"source": "policy.pdf"}).
            use_hyde    : Use HyDE query transformation.
            use_multi_query: Use multi-query transformation.
            template    : Override the default prompt template.

        Returns:
            RAGResponse with answer, citations, and latency breakdown.
        """
        timings: Dict[str, float] = {}

        # Check cache
        cache_key = self._cache_key(question, filter)
        if self._cache:
            cached = self._get_cache(cache_key)
            if cached:
                logger.debug("Cache HIT for query: %s", question[:60])
                return cached

        # --- Retrieval ---
        t0 = time.perf_counter()
        results = self.retrieve(
            question, filter=filter,
            use_hyde=use_hyde, use_multi_query=use_multi_query,
        )
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        if not results:
            return RAGResponse(
                query=question,
                answer="I could not find relevant information to answer your question.",
                citations=[],
                retrieved_chunks=[],
                latency_breakdown=timings,
            )

        # --- Prompt Building ---
        t1 = time.perf_counter()
        history = self._histories.get(session_id) if session_id else None

        # Override template if requested
        if template:
            builder = PromptBuilder(
                template=template,
                max_context_tokens=self.config.generation.max_context_tokens,
            )
        else:
            builder = self._prompt_builder

        messages, citations = builder.build(question, results, history=history)
        timings["prompt_build_ms"] = (time.perf_counter() - t1) * 1000

        # --- Generation ---
        t2 = time.perf_counter()
        llm_response = self._llm.chat(messages)
        timings["generation_ms"] = (time.perf_counter() - t2) * 1000

        # --- Update conversation history ---
        if session_id:
            if session_id not in self._histories:
                self._histories[session_id] = ConversationHistory(max_tokens=2000)
            self._histories[session_id].add("user", question)
            self._histories[session_id].add("assistant", llm_response.content)

        response = RAGResponse(
            query=question,
            answer=llm_response.content,
            citations=citations,
            retrieved_chunks=results,
            llm_response=llm_response,
            latency_breakdown=timings,
        )

        # Store in cache
        if self._cache:
            self._set_cache(cache_key, response)

        return response

    def stream_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        filter: Optional[dict] = None,
    ) -> Iterator[str]:
        """Stream tokens from the LLM as they are generated."""
        results = self.retrieve(question, filter=filter)
        messages, _ = self._prompt_builder.build(
            question, results,
            history=self._histories.get(session_id) if session_id else None,
        )
        return self._llm.stream(messages)

    # -----------------------------------------------------------------------
    # MULTI-HOP RETRIEVAL
    # -----------------------------------------------------------------------
    def multi_hop_query(
        self,
        question: str,
        max_hops: int = 3,
        filter: Optional[dict] = None,
    ) -> RAGResponse:
        """
        Iterative multi-hop retrieval.

        At each hop:
          1. Retrieve evidence for current query.
          2. Ask LLM: "Do you have enough information? If not, what else is needed?"
          3. If more info needed, generate a follow-up query and repeat.

        Use case: complex multi-part questions requiring multiple evidence passages.
        Example: "Who is the CEO of the company that acquired Slack?"
        """
        accumulated_results: List[SearchResult] = []
        current_query = question
        all_content: List[str] = []

        for hop in range(max_hops):
            logger.info("Multi-hop %d/%d: '%s'", hop + 1, max_hops, current_query[:80])
            results = self._get_candidates(current_query, self.config.retrieval.top_k, filter)
            new_results = [r for r in results
                           if r.chunk_id not in {e.chunk_id for e in accumulated_results}]
            accumulated_results.extend(new_results)
            all_content.extend(r.content for r in new_results)

            # Ask LLM if enough info
            gathered = "\n\n".join(all_content[:5000])     # token-limit guard
            check_prompt = (
                f"Original question: {question}\n\n"
                f"Evidence gathered so far:\n{gathered}\n\n"
                "Can you answer the original question with the information above? "
                "Reply 'YES' if yes. If not, reply 'NEED: <follow-up question to search>'."
            )
            check_response = self._llm.complete(check_prompt)
            check_text = check_response.content.strip()

            if check_text.upper().startswith("YES"):
                break

            if check_text.upper().startswith("NEED:"):
                current_query = check_text[5:].strip()
            else:
                break

        # Final generation over all accumulated evidence
        final_context = "\n\n".join(
            f"[{i+1}] {r.content}" for i, r in enumerate(accumulated_results)
        )
        final_response = self._llm.answer(question, final_context)

        return RAGResponse(
            query=question,
            answer=final_response.content,
            citations=[{"ref": i+1, "chunk_id": r.chunk_id, "source": r.metadata.get("source")}
                       for i, r in enumerate(accumulated_results)],
            retrieved_chunks=accumulated_results,
            llm_response=final_response,
            metadata={"hops": max_hops},
        )

    # -----------------------------------------------------------------------
    # AGENTIC RAG
    # -----------------------------------------------------------------------
    def agentic_query(
        self,
        question: str,
        tools: Optional[Dict[str, Callable]] = None,
        max_iterations: int = 5,
    ) -> RAGResponse:
        """
        Agentic RAG with ReAct (Reason + Act) loop.

        The agent can:
          - retrieve(query)       → search the knowledge base
          - calculate(expr)       → run math expressions
          - Custom tools via `tools` dict

        Loop: Thought → Action → Observation → Thought → ... → Final Answer

        Use when: Complex queries requiring reasoning + multiple retrieval steps
                  + external tool use (calculators, APIs, DBs).
        """
        base_tools = {
            "retrieve": lambda q: "\n".join(
                r.content for r in self._get_candidates(q, 3, None)
            ),
        }
        if tools:
            base_tools.update(tools)

        tool_descriptions = "\n".join(
            f"  - {name}(input): {fn.__doc__ or 'Tool'}"
            for name, fn in base_tools.items()
        )

        system_prompt = f"""You are an AI assistant using the ReAct framework.
Available tools:
{tool_descriptions}
  - finish(answer): Return the final answer.

Format each step as:
Thought: <your reasoning>
Action: <tool_name>(<input>)
Observation: <result>

Continue until you call finish().
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Question: {question}"},
        ]

        trajectory: List[str] = []
        all_results: List[SearchResult] = []

        for i in range(max_iterations):
            response = self._llm.chat(messages)
            content = response.content
            messages.append({"role": "assistant", "content": content})
            trajectory.append(content)

            # Parse action
            if "finish(" in content.lower():
                import re
                match = re.search(r"finish\((.+?)\)", content, re.IGNORECASE | re.DOTALL)
                answer = match.group(1).strip().strip("\"'") if match else content
                return RAGResponse(
                    query=question,
                    answer=answer,
                    citations=[],
                    retrieved_chunks=all_results,
                    llm_response=response,
                    metadata={"trajectory": trajectory, "iterations": i + 1},
                )

            import re
            action_match = re.search(r"Action:\s*(\w+)\((.+?)\)", content, re.DOTALL)
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_match.group(2).strip().strip("\"'")

                if tool_name in base_tools:
                    try:
                        observation = base_tools[tool_name](tool_input)
                        if tool_name == "retrieve":
                            partial = self._get_candidates(tool_input, 3, None)
                            all_results.extend(partial)
                    except Exception as e:
                        observation = f"Error: {e}"
                else:
                    observation = f"Unknown tool: {tool_name}"

                messages.append({"role": "user", "content": f"Observation: {observation}"})

        # Fallback if max iterations reached
        final = self._llm.answer(question, "\n".join(trajectory[-3:]))
        return RAGResponse(
            query=question,
            answer=final.content,
            citations=[],
            retrieved_chunks=all_results,
            llm_response=final,
            metadata={"trajectory": trajectory, "max_iterations_reached": True},
        )

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist vector store and BM25 index."""
        from pathlib import Path
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._vector_store.persist(str(p / "vector_store"))
        self._bm25.save(str(p / "bm25"))
        logger.info("RAGPipeline saved to %s", path)

    def load(self, path: str) -> None:
        """Load persisted indexes."""
        from pathlib import Path
        p = Path(path)
        self._vector_store.load(str(p / "vector_store"))
        self._bm25.load(str(p / "bm25"))
        # Re-wire retrievers after loading
        self._dense_retriever = DenseRetriever(self._vector_store, self._embedder)
        self._hybrid_retriever = HybridRetriever(
            self._dense_retriever, self._bm25, use_rrf=True
        )
        logger.info("RAGPipeline loaded from %s", path)

    # -----------------------------------------------------------------------
    # Cache helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _cache_key(query: str, filter: Optional[dict]) -> str:
        payload = query + json.dumps(filter or {}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _get_cache(self, key: str) -> Optional[RAGResponse]:
        if self._cache is None:
            return None
        try:
            return self._cache.get(key)
        except Exception:
            return None

    def _set_cache(self, key: str, value: RAGResponse, ttl: int = 3600) -> None:
        if self._cache is None:
            return
        try:
            self._cache.set(key, value, expire=ttl)
        except Exception as e:
            logger.warning("Cache set failed: %s", e)

    # -----------------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------------
    def clear_session(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    def get_session_history(self, session_id: str) -> Optional[ConversationHistory]:
        return self._histories.get(session_id)

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------
    @property
    def index_size(self) -> int:
        return self._vector_store.count
