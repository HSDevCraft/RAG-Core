# LLM Integration — Provider Abstractions & Optimization

## Conceptual Foundation

LLM integration is the **final stage** in RAG systems where carefully constructed context and queries are transformed into natural language responses. The challenge lies in **provider diversity** — different APIs, pricing models, capabilities, and reliability characteristics require a unified abstraction.

**Key insight**: Production RAG systems need **provider-agnostic interfaces** with **graceful degradation**, **cost optimization**, and **streaming capabilities**. A single provider failure shouldn't bring down the entire system.

### The Multi-Provider Challenge

```
Provider Landscape:
├── OpenAI: High quality, expensive, rate limits
├── Anthropic: Long context, expensive, good reasoning  
├── Cohere: Specialized for RAG, moderate cost
├── Azure: Enterprise features, complex pricing
├── Ollama: Local deployment, free, variable quality
└── HuggingFace: Open models, self-hosted, complex setup
```

**Core requirements**:
- **Unified interface** across all providers
- **Automatic failover** when primary provider fails
- **Cost optimization** through model selection and caching
- **Streaming support** for responsive UIs
- **Token and cost tracking** for budgeting

---

## Mathematical Formulation

### Cost Optimization Model

**Total cost per query**:
```
C_total = C_input + C_output + C_overhead
```

Where:
- `C_input = tokens_input × price_per_input_token`
- `C_output = tokens_output × price_per_output_token`  
- `C_overhead = API_call_latency × compute_cost_per_ms`

**Provider selection optimization**:
```
minimize: α × Cost + β × Latency + γ × (1 - Quality)
subject to: Availability(provider) = True
           Budget_remaining > Cost
```

### Streaming Mathematics

**Token arrival rate** (for streaming):
```
λ(t) = tokens_per_second at time t
```

**User perceived latency**:
```
L_perceived = T_first_token + Σ(1/λ(t)) for incomplete thoughts
```

**Optimization**: Minimize time-to-first-token while maintaining consistent streaming rate.

---

## Implementation Details

Your system (`generation/llm_interface.py`) provides a unified LLM interface with multiple provider implementations.

### LLMInterface Core

```python
@dataclass
class LLMResponse:
    content: str                    # Generated response text
    model: str                     # Model that generated it  
    prompt_tokens: int             # Input tokens consumed
    completion_tokens: int         # Output tokens generated
    total_tokens: int             # prompt + completion
    latency_ms: float             # Wall-clock generation time
    finish_reason: str            # "stop"|"length"|"content_filter"
    cost_usd: Optional[float] = None  # Calculated cost
    
    @property
    def tokens_per_second(self) -> float:
        """Generation speed in tokens/second"""
        if self.latency_ms > 0:
            return (self.completion_tokens / self.latency_ms) * 1000
        return 0.0
    
    def to_dict(self) -> dict:
        """Serializable representation for logging/caching"""
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "cost_usd": self.cost_usd,
            "tokens_per_second": self.tokens_per_second
        }

class LLMInterface(ABC):
    """Unified interface for all LLM providers"""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Chat completion with message history"""
        pass
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Single prompt completion"""
        pass
    
    @abstractmethod
    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """Streaming chat completion"""
        pass
    
    def answer(self, question: str, context: str, **kwargs) -> LLMResponse:
        """Convenience method for RAG-style Q&A"""
        messages = [
            {"role": "system", "content": "Answer the question based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
        return self.chat(messages, **kwargs)
```

### OpenAI Implementation

```python
class OpenAILLM(LLMInterface):
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None,
                 temperature: float = 0.1, max_tokens: int = 1024,
                 timeout: float = 30.0):
        """
        OpenAI LLM implementation
        
        Models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
        """
        import openai
        
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=timeout
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Pricing table ($/1M tokens) - updated periodically
        self.pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
        }
    
    def chat(self, messages: List[Dict[str, str]], 
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             json_mode: bool = False,
             **kwargs) -> LLMResponse:
        """OpenAI chat completion"""
        
        start_time = time.perf_counter()
        
        try:
            # Prepare request parameters
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                **kwargs
            }
            
            # Enable JSON mode if requested
            if json_mode and "gpt-4" in self.model_name:
                request_params["response_format"] = {"type": "json_object"}
                # Ensure system message mentions JSON
                if messages and messages[0]["role"] == "system":
                    if "json" not in messages[0]["content"].lower():
                        messages[0]["content"] += " Respond in valid JSON format."
            
            # Make API call
            response = self.client.chat.completions.create(**request_params)
            
            # Calculate timing and costs
            latency_ms = (time.perf_counter() - start_time) * 1000
            cost_usd = self._calculate_cost(response.usage)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason,
                cost_usd=cost_usd
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMError(f"OpenAI chat completion failed: {e}")
    
    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """OpenAI streaming completion"""
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise LLMError(f"OpenAI streaming failed: {e}")
    
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on token usage and current pricing"""
        
        pricing = self.pricing.get(self.model_name, {"input": 1.0, "output": 3.0})
        
        input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
```

### Azure OpenAI Implementation

```python
class AzureOpenAILLM(LLMInterface):
    def __init__(self, deployment_name: str, api_key: str = None,
                 azure_endpoint: str = None, api_version: str = "2024-02-01", 
                 **kwargs):
        """
        Azure OpenAI implementation with enterprise features
        
        Args:
            deployment_name: Azure deployment name (not model name)
            azure_endpoint: https://yourservice.openai.azure.com/
        """
        import openai
        
        self.client = openai.AzureOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=api_version
        )
        self.deployment_name = deployment_name
        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 1024)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Azure OpenAI chat completion with content filtering"""
        
        start_time = time.perf_counter()
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,  # Azure uses deployment name
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Check for content filtering
            choice = response.choices[0]
            if hasattr(choice, 'content_filter_results'):
                filter_results = choice.content_filter_results
                if any(result.filtered for result in filter_results.values()):
                    raise LLMError("Content filtered by Azure safety systems")
            
            return LLMResponse(
                content=choice.message.content,
                model=response.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=latency_ms,
                finish_reason=choice.finish_reason,
                cost_usd=None  # Azure pricing varies by contract
            )
            
        except Exception as e:
            logger.error(f"Azure OpenAI error: {e}")
            raise LLMError(f"Azure OpenAI failed: {e}")
```

### Ollama Implementation (Local)

```python
class OllamaLLM(LLMInterface):
    def __init__(self, model_name: str = "llama3.1:8b", 
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.1):
        """
        Ollama local LLM implementation
        
        Models: llama3.1:8b, llama3.1:70b, codellama, mistral, etc.
        Requires: ollama serve running locally
        """
        import requests
        
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.session = requests.Session()
        
        # Test connection
        try:
            self._test_connection()
        except Exception as e:
            logger.warning(f"Ollama connection test failed: {e}")
    
    def _test_connection(self):
        """Test Ollama server connectivity"""
        response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Ollama chat completion"""
        
        start_time = time.perf_counter()
        
        # Convert messages to Ollama format
        prompt = self._messages_to_prompt(messages)
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get('temperature', self.temperature),
                        "num_predict": kwargs.get('max_tokens', 1024)
                    }
                },
                timeout=kwargs.get('timeout', 60)
            )
            response.raise_for_status()
            
            result = response.json()
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return LLMResponse(
                content=result["response"],
                model=self.model_name,
                prompt_tokens=result.get("prompt_eval_count", 0),
                completion_tokens=result.get("eval_count", 0),
                total_tokens=result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
                latency_ms=latency_ms,
                finish_reason="stop" if result.get("done", False) else "length",
                cost_usd=0.0  # Local model = no cost
            )
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise LLMError(f"Ollama generation failed: {e}")
    
    def stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """Ollama streaming generation"""
        
        prompt = self._messages_to_prompt(messages)
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": kwargs.get('temperature', self.temperature)
                    }
                },
                stream=True,
                timeout=kwargs.get('timeout', 120)
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
                        
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise LLMError(f"Ollama streaming failed: {e}")
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-format messages to plain prompt"""
        
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")  # Prompt for response
        return "\n\n".join(prompt_parts)
```

### Fallback & Load Balancing System

```python
class ResilientLLMInterface:
    def __init__(self, primary_llm: LLMInterface, 
                 fallback_llm: Optional[LLMInterface] = None,
                 circuit_breaker_threshold: int = 3,
                 circuit_breaker_timeout: int = 60):
        """
        LLM interface with automatic fallback and circuit breaker
        
        Args:
            primary_llm: Main LLM provider
            fallback_llm: Backup provider (e.g., Ollama)
            circuit_breaker_threshold: Failures before switching
            circuit_breaker_timeout: Seconds before retry
        """
        self.primary = primary_llm
        self.fallback = fallback_llm
        self.circuit_breaker = CircuitBreaker(circuit_breaker_threshold, circuit_breaker_timeout)
        
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Chat with automatic failover"""
        
        # Try primary provider
        if self.circuit_breaker.can_execute():
            try:
                response = self.primary.chat(messages, **kwargs)
                self.circuit_breaker.on_success()
                return response
            except LLMError as e:
                self.circuit_breaker.on_failure()
                logger.warning(f"Primary LLM failed: {e}")
        
        # Fall back to secondary provider
        if self.fallback:
            try:
                logger.info("Using fallback LLM provider")
                return self.fallback.chat(messages, **kwargs)
            except LLMError as e:
                logger.error(f"Fallback LLM also failed: {e}")
        
        raise LLMError("All LLM providers failed")

class CircuitBreaker:
    def __init__(self, threshold: int, timeout: int):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True
    
    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.threshold:
            self.state = "OPEN"
```

---

## Comparative Analysis

### Provider Selection Matrix

| Factor | OpenAI | Azure | Anthropic | Cohere | Ollama |
|---|---|---|---|---|---|
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Cost** | High | Variable | High | Medium | Free |
| **Latency** | 200-800ms | 300-1200ms | 300-1000ms | 200-600ms | 50-500ms |
| **Context Length** | 128K | 128K | 200K | 128K | 32K |
| **Reliability** | 99.9% | 99.95% | 99.5% | 99.5% | Self-hosted |
| **Enterprise** | Basic | Advanced | Basic | Medium | Full control |

### Cost Optimization Strategies

```python
def select_optimal_provider(query_complexity: float, budget_per_query: float, 
                          latency_requirement_ms: int) -> str:
    """Select provider based on requirements and constraints"""
    
    provider_specs = {
        "gpt-4o-mini": {
            "cost_per_1k_tokens": 0.0002,  # $0.0002 per 1K tokens
            "quality_score": 0.85,
            "avg_latency_ms": 400,
            "max_context": 128000
        },
        "gpt-4o": {
            "cost_per_1k_tokens": 0.0050,
            "quality_score": 0.95,
            "avg_latency_ms": 800,
            "max_context": 128000  
        },
        "claude-3-haiku": {
            "cost_per_1k_tokens": 0.0008,
            "quality_score": 0.80,
            "avg_latency_ms": 600,
            "max_context": 200000
        },
        "ollama-llama3.1": {
            "cost_per_1k_tokens": 0.0000,
            "quality_score": 0.75,
            "avg_latency_ms": 200,
            "max_context": 32000
        }
    }
    
    # Estimate tokens for query (rough approximation)
    estimated_tokens = len(query_complexity) * 50  # Assume complex = longer
    
    viable_providers = []
    
    for provider, specs in provider_specs.items():
        estimated_cost = (estimated_tokens / 1000) * specs["cost_per_1k_tokens"]
        
        if (estimated_cost <= budget_per_query and 
            specs["avg_latency_ms"] <= latency_requirement_ms):
            
            # Calculate value score (quality per dollar per ms)
            if estimated_cost > 0:
                value_score = specs["quality_score"] / (estimated_cost * specs["avg_latency_ms"])
            else:
                value_score = specs["quality_score"] * 1000  # Free gets bonus
            
            viable_providers.append((provider, value_score, specs))
    
    if not viable_providers:
        return "ollama-llama3.1"  # Fallback to free option
    
    # Return highest value provider
    return max(viable_providers, key=lambda x: x[1])[0]
```

---

## Practical Guidelines

### Cost Management

**Token budget tracking**:
```python
class TokenBudgetManager:
    def __init__(self, daily_budget_usd: float = 10.0):
        self.daily_budget = daily_budget_usd
        self.daily_spend = 0.0
        self.last_reset = datetime.now().date()
        
    def can_afford(self, estimated_cost: float) -> bool:
        """Check if request fits within budget"""
        self._reset_if_new_day()
        return (self.daily_spend + estimated_cost) <= self.daily_budget
    
    def record_spend(self, cost: float):
        """Record actual spending"""
        self._reset_if_new_day()
        self.daily_spend += cost
        
        if self.daily_spend >= self.daily_budget * 0.9:  # 90% threshold
            logger.warning(f"Approaching daily budget limit: ${self.daily_spend:.4f} / ${self.daily_budget}")
    
    def _reset_if_new_day(self):
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_spend = 0.0
            self.last_reset = today

# Usage in LLM interface
budget_manager = TokenBudgetManager(daily_budget_usd=50.0)

def budget_aware_chat(llm: LLMInterface, messages: List[Dict], **kwargs) -> LLMResponse:
    """Chat completion with budget management"""
    
    # Estimate cost before calling
    estimated_tokens = sum(len(m["content"]) // 4 for m in messages)
    estimated_cost = (estimated_tokens / 1000) * 0.0002  # Rough estimate
    
    if not budget_manager.can_afford(estimated_cost):
        # Switch to cheaper model or fail gracefully
        logger.warning("Budget limit reached, switching to Ollama")
        ollama_llm = OllamaLLM()
        return ollama_llm.chat(messages, **kwargs)
    
    # Proceed with expensive model
    response = llm.chat(messages, **kwargs)
    budget_manager.record_spend(response.cost_usd or estimated_cost)
    
    return response
```

### Streaming Implementation

**Server-Sent Events (SSE) for web UIs**:
```python
from fastapi.responses import StreamingResponse

@app.post("/query/stream")
async def stream_query_endpoint(request: QueryRequest):
    """FastAPI endpoint for streaming responses"""
    
    async def generate():
        try:
            # Retrieve context (non-streaming part)
            retrieved_results = retriever.retrieve(request.query)
            context, citations = context_builder.build(retrieved_results)
            
            # Build messages
            messages = [
                {"role": "system", "content": "Answer based on context provided."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {request.query}"}
            ]
            
            # Stream response
            for token in llm.stream(messages):
                yield f"data: {json.dumps({'token': token})}\n\n"
                
            # Send completion signal
            yield f"data: {json.dumps({'citations': citations})}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**WebSocket for real-time chat**:
```python
@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    try:
        while True:
            # Receive query
            data = await websocket.receive_json()
            query = data.get("query")
            
            if not query:
                continue
            
            # Retrieve and build context
            retrieved_results = retriever.retrieve(query, session_id=session_id)
            context, citations = context_builder.build(retrieved_results)
            
            # Get conversation history
            history = conversation_manager.get_history(session_id)
            messages = build_messages_with_history(query, context, history)
            
            # Stream response
            response_tokens = []
            async for token in llm.stream_async(messages):  # Async streaming
                await websocket.send_json({"type": "token", "content": token})
                response_tokens.append(token)
            
            # Send completion
            full_response = "".join(response_tokens)
            await websocket.send_json({
                "type": "complete",
                "response": full_response,
                "citations": citations
            })
            
            # Update conversation history
            conversation_manager.add_exchange(session_id, query, full_response)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
```

### Performance Optimization

**Request batching for efficiency**:
```python
class BatchedLLMInterface:
    def __init__(self, base_llm: LLMInterface, max_batch_size: int = 5, 
                 batch_timeout_ms: int = 100):
        self.base_llm = base_llm
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.pending_requests = []
        self.batch_lock = asyncio.Lock()
        
    async def chat_async(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Async chat with automatic batching"""
        
        # Create request future
        request_future = asyncio.Future()
        request_item = (messages, kwargs, request_future)
        
        async with self.batch_lock:
            self.pending_requests.append(request_item)
            
            # Trigger batch processing if conditions met
            if (len(self.pending_requests) >= self.max_batch_size):
                await self._process_batch()
            else:
                # Set timer for batch timeout
                asyncio.create_task(self._batch_timeout())
        
        # Wait for result
        return await request_future
    
    async def _process_batch(self):
        """Process accumulated requests as a batch"""
        
        if not self.pending_requests:
            return
        
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        
        # Process batch (simplified - could use provider's batch API)
        tasks = []
        for messages, kwargs, future in batch:
            task = asyncio.create_task(self._single_request(messages, kwargs, future))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def _single_request(self, messages, kwargs, future):
        """Process single request and set result"""
        try:
            result = self.base_llm.chat(messages, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
```

### Error Handling & Monitoring

```python
class MonitoredLLMInterface:
    def __init__(self, base_llm: LLMInterface, metrics_client):
        self.base_llm = base_llm
        self.metrics = metrics_client
        
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Chat with comprehensive monitoring"""
        
        start_time = time.perf_counter()
        
        # Estimate request size for monitoring
        request_size = sum(len(m["content"]) for m in messages)
        
        try:
            response = self.base_llm.chat(messages, **kwargs)
            
            # Record success metrics
            self.metrics.increment("llm.requests.success", tags={
                "model": response.model,
                "provider": self.base_llm.__class__.__name__
            })
            
            self.metrics.histogram("llm.latency", response.latency_ms, tags={
                "model": response.model
            })
            
            self.metrics.histogram("llm.tokens.input", response.prompt_tokens)
            self.metrics.histogram("llm.tokens.output", response.completion_tokens)
            
            if response.cost_usd:
                self.metrics.histogram("llm.cost_usd", response.cost_usd)
            
            return response
            
        except LLMError as e:
            # Record failure metrics
            self.metrics.increment("llm.requests.error", tags={
                "error_type": e.__class__.__name__,
                "provider": self.base_llm.__class__.__name__
            })
            
            logger.error(f"LLM request failed: {e}", extra={
                "model": getattr(self.base_llm, 'model_name', 'unknown'),
                "request_size": request_size,
                "error": str(e)
            })
            
            raise
```

### Common Issues & Solutions

**Issue**: Token limit exceeded errors
```python
# Problem: Context + query + response > model's context window
# Solution: Intelligent context truncation

def truncate_for_model(messages: List[Dict], model_name: str, 
                      response_budget: int = 1024) -> List[Dict]:
    """Truncate messages to fit model's context window"""
    
    context_limits = {
        "gpt-4o-mini": 128000,
        "gpt-4o": 128000,
        "claude-3-haiku": 200000,
        "llama3.1:8b": 32000
    }
    
    max_context = context_limits.get(model_name, 4096)
    available_tokens = max_context - response_budget
    
    # Accurate token counting
    def count_tokens(text: str) -> int:
        return len(text) // 4  # Simplified approximation
    
    # Truncate from the middle, preserve system message and latest user message
    total_tokens = sum(count_tokens(m["content"]) for m in messages)
    
    if total_tokens <= available_tokens:
        return messages
    
    # Keep system message (first) and user message (last)
    if len(messages) >= 2:
        result = [messages[0]]  # System message
        remaining_budget = available_tokens - count_tokens(messages[0]["content"])
        
        # Add latest user message
        user_msg = messages[-1]
        user_tokens = count_tokens(user_msg["content"])
        
        if user_tokens <= remaining_budget:
            result.append(user_msg)
        else:
            # Truncate user message
            max_chars = remaining_budget * 4
            truncated_content = user_msg["content"][:max_chars] + "..."
            result.append({"role": user_msg["role"], "content": truncated_content})
        
        return result
    
    return messages  # Fallback: return as-is
```

**Issue**: Inconsistent response quality across providers
```python
# Problem: Different models need different prompting strategies
# Solution: Provider-specific prompt templates

class AdaptivePromptBuilder:
    def __init__(self):
        self.provider_templates = {
            "openai": {
                "system": "You are a helpful assistant. Answer questions based on the provided context.",
                "style": "direct"
            },
            "anthropic": {
                "system": "I'm Claude, an AI assistant. I'll answer your question based on the context provided, being careful to only use information from the context.",
                "style": "conversational"
            },
            "ollama": {
                "system": "Answer the question using only the information in the context. Be concise and factual.",
                "style": "minimal"
            }
        }
    
    def build_for_provider(self, question: str, context: str, provider: str) -> List[Dict]:
        """Build provider-optimized messages"""
        
        template = self.provider_templates.get(provider, self.provider_templates["openai"])
        
        if template["style"] == "minimal":
            # Ollama prefers simpler prompts
            messages = [
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": f"Context:\n{context}\n\nQ: {question}\nA:"}
            ]
        else:
            # Standard format for API providers
            messages = [
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]
        
        return messages
```

**Next concept**: Evaluation Frameworks — faithfulness, relevance metrics, RAGAS integration, and benchmarking strategies
