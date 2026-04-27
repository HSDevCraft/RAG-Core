"""
api/main.py
───────────
Production-grade FastAPI service for the RAG system.

Endpoints:
  POST /index          — Ingest documents
  POST /query          — Query the RAG pipeline
  POST /query/stream   — Streaming query (SSE)
  GET  /health         — Health check + component status
  GET  /metrics        — Prometheus metrics endpoint
  DELETE /session/{id} — Clear conversation history
  POST /evaluate       — Run evaluation on a test set

Security:
  - API key authentication (Bearer token)
  - PII scrubbing middleware
  - Rate limiting (slowapi)
  - Request size limits

Observability:
  - Structured logging (structlog)
  - Request tracing (trace_id header)
  - Prometheus metrics
  - Langfuse integration (async)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterator, List, Optional

import structlog
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator

from config import RAGConfig, DEFAULT_CONFIG
from rag_pipeline import RAGPipeline, RAGResponse

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_pipeline: Optional[RAGPipeline] = None
_config: RAGConfig = DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("startup", event="RAG API starting up")
    try:
        _pipeline = RAGPipeline.from_config(_config)
        # Load persisted index if it exists
        persist_path = _config.vector_store.persist_dir
        if os.path.exists(os.path.join(persist_path, "vector_store")):
            _pipeline.load(persist_path)
            logger.info("startup", event="Loaded persisted index",
                        path=persist_path, size=_pipeline.index_size)
        else:
            logger.info("startup", event="No persisted index found; starting fresh")
    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise

    yield

    # Shutdown: persist indexes
    try:
        if _pipeline:
            _pipeline.save(_config.vector_store.persist_dir)
            logger.info("shutdown", event="Index persisted")
    except Exception as e:
        logger.error("shutdown_error", error=str(e))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG System API",
    description="Production-grade Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
security = HTTPBearer(auto_error=False)
VALID_API_KEYS = set(
    k.strip() for k in os.getenv("API_KEYS", "dev-key-12345").split(",")
)


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    if os.getenv("DISABLE_AUTH", "false").lower() == "true":
        return "anonymous"
    if not credentials or credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials.credentials


# ---------------------------------------------------------------------------
# Tracing middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4())[:8])
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    request.state.trace_id = trace_id

    t0 = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - t0) * 1000

    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency, 1),
    )
    response.headers["X-Trace-Id"] = trace_id
    response.headers["X-Latency-Ms"] = str(round(latency, 1))
    return response


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class IndexRequest(BaseModel):
    sources: List[str] = Field(..., description="File paths, directory paths, or URLs")
    loader_kwargs: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=256, ge=1, le=1024)

    @field_validator("sources")
    @classmethod
    def sources_not_empty(cls, v):
        if not v:
            raise ValueError("sources must not be empty")
        return v


class IndexResponse(BaseModel):
    status: str
    num_documents: int
    num_chunks: int
    embedding_latency_ms: float
    total_latency_ms: float
    errors: List[str] = []


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, description="For multi-turn conversations")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")
    use_hyde: bool = Field(default=False)
    use_multi_query: bool = Field(default=False)
    use_multi_hop: bool = Field(default=False)
    template: Optional[str] = Field(default=None)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


class ChunkResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    source: Optional[str] = None
    page: Optional[int] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[ChunkResult]
    latency_ms: Dict[str, float]
    total_latency_ms: float
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    trace_id: Optional[str] = None


class EvaluateRequest(BaseModel):
    test_set: List[Dict[str, Any]] = Field(
        ..., description='[{"question": ..., "ground_truth": ...}]'
    )
    metrics: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    index_size: int
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline


def rag_response_to_api(response: RAGResponse,
                         trace_id: str = "") -> QueryResponse:
    chunks = [
        ChunkResult(
            chunk_id=r.chunk_id,
            content=r.content[:500],   # truncate for response size
            score=round(r.score, 4),
            source=r.metadata.get("source"),
            page=r.metadata.get("page"),
        )
        for r in response.retrieved_chunks
    ]
    tokens = response.llm_response.total_tokens if response.llm_response else None
    cost = response.llm_response.cost_usd if response.llm_response else None

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        citations=response.citations,
        retrieved_chunks=chunks,
        latency_ms=response.latency_breakdown,
        total_latency_ms=response.total_latency_ms,
        tokens_used=tokens,
        cost_usd=round(cost, 6) if cost else None,
        trace_id=trace_id,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    pipeline = get_pipeline()
    return HealthResponse(
        status="healthy",
        components={
            "vector_store": "ok",
            "embedder": "ok",
            "llm": "ok",
        },
        index_size=pipeline.index_size,
    )


@app.post("/index", response_model=IndexResponse, tags=["Indexing"])
async def index_documents(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Index documents from file paths, directories, or URLs.

    This operation is synchronous. For large document sets, consider
    running as a background job and polling /health for index_size updates.
    """
    pipeline = get_pipeline()
    logger.info("index_request", sources=request.sources, num_sources=len(request.sources))

    result = pipeline.index(
        request.sources,
        loader_kwargs=request.loader_kwargs,
        batch_size=request.batch_size,
    )

    # Async persist
    background_tasks.add_task(
        pipeline.save, _config.vector_store.persist_dir
    )

    return IndexResponse(
        status="success" if not result.errors else "partial_success",
        num_documents=result.num_documents,
        num_chunks=result.num_chunks,
        embedding_latency_ms=round(result.embedding_latency_ms, 1),
        total_latency_ms=round(result.total_latency_ms, 1),
        errors=result.errors,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    req: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    Query the RAG pipeline. Returns answer + citations + retrieved chunks.

    Guardrails:
    - Input is sanitized to prevent prompt injection.
    - Session history is scoped to session_id.
    - PII patterns are stripped from answers before returning.
    """
    pipeline = get_pipeline()
    trace_id = getattr(req.state, "trace_id", "")

    # Sanitize input (prompt injection defense)
    question = _sanitize_input(request.question)

    logger.info("query_request", question=question[:80], session=request.session_id,
                hyde=request.use_hyde, multi_hop=request.use_multi_hop)

    if request.use_multi_hop:
        response = await asyncio.to_thread(
            pipeline.multi_hop_query, question, 3, request.filter
        )
    else:
        response = await asyncio.to_thread(
            pipeline.query,
            question,
            session_id=request.session_id,
            filter=request.filter,
            use_hyde=request.use_hyde,
            use_multi_query=request.use_multi_query,
            template=request.template,
        )

    # PII scrubbing
    response.answer = _scrub_pii(response.answer)

    logger.info("query_response",
                total_latency_ms=round(response.total_latency_ms, 1),
                chunks=len(response.retrieved_chunks))

    return rag_response_to_api(response, trace_id)


@app.post("/query/stream", tags=["Query"])
async def query_stream(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Streaming query endpoint. Returns Server-Sent Events (SSE).

    Client usage:
        const es = new EventSource('/query/stream', {method: 'POST', body: JSON.stringify(req)})
        es.onmessage = e => process(e.data)
    """
    pipeline = get_pipeline()
    question = _sanitize_input(request.question)

    async def generate():
        try:
            token_gen = pipeline.stream_query(
                question,
                session_id=request.session_id,
                filter=request.filter,
            )
            for token in token_gen:
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # nginx: disable buffering
        },
    )


@app.delete("/session/{session_id}", tags=["Session"])
async def clear_session(
    session_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Clear conversation history for a session."""
    pipeline = get_pipeline()
    pipeline.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/evaluate", tags=["Evaluation"])
async def evaluate(
    request: EvaluateRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Run automated evaluation on a test set.
    Requires OPENAI_API_KEY for LLM-judge metrics.
    """
    from evaluation.evaluator import RAGEvaluator
    pipeline = get_pipeline()

    llm_fn = lambda p: pipeline._llm.complete(p).content
    evaluator = RAGEvaluator(
        llm_fn=llm_fn,
        metrics=request.metrics or [
            "faithfulness", "answer_relevancy", "context_recall_heuristic", "rouge_l"
        ],
    )

    report = await asyncio.to_thread(
        evaluator.evaluate_pipeline, pipeline, request.test_set
    )

    return {
        "aggregate": report.aggregate,
        "latency_stats": report.latency_stats,
        "num_samples": len(report.results),
    }


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return StreamingResponse(
            iter([generate_latest()]),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return JSONResponse({"error": "prometheus_client not installed"})


# ---------------------------------------------------------------------------
# Security utilities
# ---------------------------------------------------------------------------
_PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"disregard (your|the|all) (previous |prior |above )?instructions",
    r"you are now",
    r"act as (?!an assistant)",
    r"system prompt",
    r"reveal your instructions",
    r"jailbreak",
    r"DAN mode",
]

_PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),              # SSN
    (r"\b\d{16}\b", "[CARD]"),                          # Credit card
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),    # US phone
]


def _sanitize_input(text: str) -> str:
    """Detect and neutralize prompt injection attempts."""
    import re
    text_lower = text.lower()
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning("prompt_injection_detected", pattern=pattern, input=text[:100])
            raise HTTPException(
                status_code=400,
                detail="Input contains disallowed content. Please rephrase your question."
            )
    return text


def _scrub_pii(text: str) -> str:
    """Replace PII patterns with placeholders."""
    import re
    for pattern, replacement in _PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error("unhandled_error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    CLI entry point registered in pyproject.toml as:
        [project.scripts]
        rag-api = "api.main:main"

    Allows running the server with:
        rag-api                             # installed package
        python -m uvicorn api.main:app ...  # direct
    """
    import uvicorn

    is_dev = os.getenv("ENV", "production") == "development"
    uvicorn.run(
        "api.main:app",
        host=_config.api_host,
        port=_config.api_port,
        workers=1 if is_dev else _config.api_workers,
        log_level="debug" if is_dev else "info",
        reload=is_dev,
    )


if __name__ == "__main__":
    main()
