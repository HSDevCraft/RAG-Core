# Module 08 — FastAPI Service
### `api/main.py`

---

## Table of Contents
1. [API Overview](#1-api-overview)
2. [Endpoint Reference](#2-endpoint-reference)
3. [Authentication](#3-authentication)
4. [Request & Response Models](#4-request--response-models)
5. [Streaming (SSE)](#5-streaming-sse)
6. [Security Middleware](#6-security-middleware)
7. [Prometheus Metrics](#7-prometheus-metrics)
8. [Running the Server](#8-running-the-server)
9. [Docker Deployment](#9-docker-deployment)
10. [Quick Reference](#10-quick-reference)
11. [Common Pitfalls](#11-common-pitfalls)

---

## 1. API Overview

```
                     ┌─────────────────────────────────────┐
                     │           FastAPI App               │
                     │                                     │
  Client ──HTTPS──▶  │  Auth middleware (X-API-Key)        │
                     │  PII scrubbing middleware            │
                     │  Prompt injection defense           │
                     │  Rate limiting (per API key)         │
                     │                                     │
                     │  POST /index          ──▶ pipeline.index()
                     │  POST /query          ──▶ pipeline.query()
                     │  POST /query/stream   ──▶ pipeline.stream_query()
                     │  DELETE /sessions/:id ──▶ pipeline.clear_session()
                     │  GET  /health         ──▶ system status
                     │  GET  /metrics        ──▶ Prometheus metrics
                     │  GET  /docs           ──▶ Swagger UI
                     └─────────────────────────────────────┘
```

**Base URL**: `http://localhost:8080`  
**Auth header**: `X-API-Key: <your-api-key>`  
**Content-Type**: `application/json`

---

## 2. Endpoint Reference

### `GET /health`

Health check — no auth required (load balancer probes use this).

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "healthy",
  "index_size": 312,
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_provider": "openai"
}
```

---

### `POST /index`

Index documents from file paths, directories, or URLs.

```bash
curl -X POST http://localhost:8080/index \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "sources": ["./docs/", "https://example.com/policy"],
    "batch_size": 128
  }'
```

```json
{
  "status": "success",
  "num_documents": 47,
  "num_chunks": 312,
  "embedding_latency_ms": 3420.5,
  "total_latency_ms": 4100.2,
  "errors": []
}
```

**Status values**:
- `"success"` — all sources indexed
- `"partial_success"` — some sources failed (check `errors`)

---

### `POST /query`

Synchronous query — waits for full answer before returning.

```bash
curl -X POST http://localhost:8080/query \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the refund policy?",
    "top_k": 10,
    "template": "with_citations",
    "session_id": "user-abc-session-001",
    "filter": {"file_type": "pdf"}
  }'
```

```json
{
  "query": "What is the refund policy?",
  "answer": "Based on the documentation, customers are eligible for a full refund within 30 days of purchase [1]. After 30 days, only store credit is available [1]. Digital downloads are non-refundable once accessed [2].",
  "citations": [
    {"ref": 1, "chunk_id": "a3f1b2c9", "source": "policy.pdf", "page": 3, "score": 0.923},
    {"ref": 2, "chunk_id": "d4e57812", "source": "policy.pdf", "page": 4, "score": 0.887}
  ],
  "total_latency_ms": 1850.3,
  "token_usage": {"prompt": 450, "completion": 120, "total": 570},
  "cost_usd": 0.000086
}
```

---

### `POST /query/stream`

Streaming query — returns tokens as Server-Sent Events (SSE).

```bash
curl -X POST http://localhost:8080/query/stream \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "Explain the refund policy", "session_id": "user-abc"}'
```

```
data: {"token": "Based"}
data: {"token": " on"}
data: {"token": " the"}
data: {"token": " documentation"}
data: {"token": ","}
...
data: [DONE]
```

**JavaScript EventSource**:
```javascript
const source = new EventSource('/query/stream');
const resp = await fetch('/query/stream', {
  method: 'POST',
  headers: {'X-API-Key': 'dev-key-12345', 'Content-Type': 'application/json'},
  body: JSON.stringify({query: 'What is the refund policy?'})
});
const reader = resp.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const {value, done} = await reader.read();
  if (done) break;
  const text = decoder.decode(value);
  // parse SSE lines, extract token
  console.log(text);
}
```

---

### `DELETE /sessions/{session_id}`

Clear conversation history for a session.

```bash
curl -X DELETE http://localhost:8080/sessions/user-abc-session-001 \
  -H "X-API-Key: dev-key-12345"
```

```json
{"status": "cleared", "session_id": "user-abc-session-001"}
```

---

### `POST /evaluate`

Run evaluation on a list of QA samples.

```bash
curl -X POST http://localhost:8080/evaluate \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "question": "What is the refund policy?",
        "ground_truth": "30-day full refund."
      }
    ],
    "metrics": ["faithfulness", "rouge_l"]
  }'
```

---

## 3. Authentication

The API uses **static API key authentication** via the `X-API-Key` header.

```python
# api/main.py — auth implementation
async def verify_api_key(x_api_key: str = Header(...)) -> str:
    valid_keys = os.getenv("API_KEYS", "dev-key-12345").split(",")
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key
```

**Configuration**:
```bash
# .env
API_KEYS=prod-key-abc123,service-key-xyz789,dev-key-12345
DISABLE_AUTH=false   # set true only in dev/testing
```

**Multiple keys** — rotate keys without downtime by adding the new key
before removing the old one:
```bash
API_KEYS=old-key-123,new-key-456   # both active during rotation
# → after clients migrate →
API_KEYS=new-key-456
```

---

## 4. Request & Response Models

### QueryRequest

```python
class QueryRequest(BaseModel):
    query:      str   = Field(..., min_length=1, max_length=4096)
    top_k:      int   = Field(10, ge=1, le=50)
    template:   str   = Field("default")
    session_id: Optional[str] = None
    filter:     Optional[Dict[str, Any]] = None
    use_hyde:   bool  = False
    use_mmr:    bool  = False
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)
```

### IndexRequest

```python
class IndexRequest(BaseModel):
    sources:       List[str] = Field(..., min_items=1)
    loader_kwargs: Optional[Dict] = None
    batch_size:    int = Field(128, ge=1, le=512)
```

### QueryResponse

```python
class QueryResponse(BaseModel):
    query:           str
    answer:          str
    citations:       List[Dict]
    total_latency_ms: float
    token_usage:     Dict[str, int]
    cost_usd:        float
    session_id:      Optional[str]
```

### Error Responses

| Code | Meaning | Example |
|---|---|---|
| 400 | Blocked (injection/PII) | `{"detail": "Query contains prohibited patterns"}` |
| 403 | Invalid API key | `{"detail": "Invalid API key"}` |
| 422 | Validation error | `{"detail": [{"loc": ["body","query"], "msg": "field required"}]}` |
| 429 | Rate limit exceeded | `{"detail": "Rate limit: 100 req/min exceeded"}` |
| 500 | Internal error | `{"detail": "Internal server error", "type": "RuntimeError"}` |

---

## 5. Streaming (SSE)

The `/query/stream` endpoint uses **Server-Sent Events** (SSE), the standard
for server-push text streams over HTTP.

```python
# api/main.py — SSE implementation
@app.post("/query/stream")
async def stream_query(request: QueryRequest, api_key=Depends(verify_api_key)):
    async def generate():
        pipeline = get_pipeline()
        # Run retrieval synchronously, then stream generation
        for token in pipeline.stream_query(
            request.query,
            session_id=request.session_id,
            top_k=request.top_k,
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

**SSE format rules**:
- Each event: `data: {json}\n\n`
- Terminator: `data: [DONE]\n\n`
- Client must handle partial chunks (network fragmentation)

---

## 6. Security Middleware

### PII Scrubbing

Strips personal data from queries before processing:

```python
PII_PATTERNS = [
    (r'\b\d{16}\b',                    '[CREDIT_CARD]'),     # credit card
    (r'\b\d{3}-\d{2}-\d{4}\b',        '[SSN]'),              # social security
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
    (r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]'),
]

def scrub_pii(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text
```

Input query `"My SSN is 123-45-6789"` becomes `"My SSN is [SSN]"`.

### Prompt Injection Defense

Rejects or sanitises queries containing patterns that attempt to override
the system prompt:

```python
INJECTION_PATTERNS = [
    r"ignore (?:all )?(?:previous|above) instructions",
    r"disregard (?:your |the )?(?:system |previous )?(?:prompt|instructions)",
    r"you are now (?:a|an|my)",
    r"jailbreak",
    r"act as (?:an? )?(?:evil|unrestricted|DAN)",
]
```

Returns HTTP 400 if a pattern matches.

---

## 7. Prometheus Metrics

Available at `GET /metrics` (port 8001 in Docker).

```
# HELP rag_requests_total Total RAG requests
# TYPE rag_requests_total counter
rag_requests_total{endpoint="/query",status="200"} 1523

# HELP rag_request_duration_seconds Query duration histogram
# TYPE rag_request_duration_seconds histogram
rag_request_duration_seconds_bucket{le="0.5"}  892
rag_request_duration_seconds_bucket{le="1.0"}  1234
rag_request_duration_seconds_bucket{le="2.0"}  1501

# HELP rag_index_size_total Number of indexed chunks
# TYPE rag_index_size_total gauge
rag_index_size_total 312

# HELP rag_token_cost_usd_total Cumulative LLM cost in USD
# TYPE rag_token_cost_usd_total counter
rag_token_cost_usd_total 0.23481
```

**Grafana dashboard queries** (example PromQL):
```promql
# Request rate
rate(rag_requests_total[5m])

# p95 latency
histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m]))

# Error rate
rate(rag_requests_total{status=~"5.."}[5m]) / rate(rag_requests_total[5m])
```

---

## 8. Running the Server

### Development

```bash
# With Makefile
make api                       # hot-reload, port 8080

# Direct
uvicorn api.main:app --reload --port 8080

# With custom settings
ENV=development PORT=9000 make api
```

### Production

```bash
# Makefile (gunicorn-backed uvicorn workers)
make api-prod WORKERS=4

# Direct
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4 \
  --log-level warning

# As installed CLI entry-point (after pip install -e .)
rag-api
```

### Configuration via Environment

```bash
# .env
OPENAI_API_KEY=sk-...
API_KEYS=prod-key-abc,dev-key-xyz
DISABLE_AUTH=false
ENV=production

# Vector store
VECTOR_STORE_BACKEND=faiss
VECTOR_STORE_DIR=./vector_store_data

# LLM
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Caching
REDIS_URL=redis://localhost:6379/0
```

---

## 9. Docker Deployment

### Single Service

```bash
docker build -t rag-system:latest .
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -e API_KEYS=my-prod-key \
  -v $(pwd)/vector_store_data:/app/vector_store_data \
  rag-system:latest
```

### Full Stack (docker-compose)

```bash
# API + Redis only (minimal)
make docker-up

# API + Redis + Prometheus + Grafana
make docker-up-full
```

```yaml
# docker-compose.yml services:
#   rag-api    → http://localhost:8080
#   redis      → localhost:6379
#   chroma     → http://localhost:8000
#   prometheus → http://localhost:9090
#   grafana    → http://localhost:3001  (admin/admin)
```

---

## 10. Quick Reference

```bash
# ── Health check ──────────────────────────────────────────────────────────
curl http://localhost:8080/health

# ── Index documents ───────────────────────────────────────────────────────
curl -X POST http://localhost:8080/index \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"sources": ["./docs/"]}'

# ── Query ─────────────────────────────────────────────────────────────────
curl -X POST http://localhost:8080/query \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "top_k": 10}'

# ── Stream query ──────────────────────────────────────────────────────────
curl -X POST http://localhost:8080/query/stream \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the refund policy"}'

# ── Clear session ─────────────────────────────────────────────────────────
curl -X DELETE http://localhost:8080/sessions/user-abc \
  -H "X-API-Key: dev-key-12345"

# ── Swagger UI ────────────────────────────────────────────────────────────
open http://localhost:8080/docs

# ── Prometheus metrics ────────────────────────────────────────────────────
curl http://localhost:8001/metrics
```

---

## 11. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| **Forgetting auth header** | 403 on every request | Include `X-API-Key` header |
| **top_k > 50** | 422 Validation error | Max is 50 (configurable in model) |
| **Streaming with wrong Content-Type** | Gets buffered, not streamed | Client must accept `text/event-stream` |
| **Multi-worker session state** | Session history lost between requests | Use Redis for session storage |
| **Prometheus on wrong port** | `curl /metrics` 404 | Metrics are on port 8001, not 8080 |
| **Workers=1 in production** | CPU bottleneck on single core | Use `workers = 2 × CPU_cores + 1` |
| **No graceful shutdown** | Requests dropped on deploy | Use `--timeout-graceful-shutdown 30` |
| **Docker volume not mounted** | Index lost on container restart | Mount `./vector_store_data:/app/...` |
