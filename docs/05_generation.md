# Module 05 — Generation
### `generation/prompt_builder.py` · `generation/llm_interface.py`

---

## Table of Contents
1. [Generation Overview](#1-generation-overview)
2. [PromptTemplate](#2-prompttemplate)
3. [ContextBuilder](#3-contextbuilder)
4. [ConversationHistory](#4-conversationhistory)
5. [PromptBuilder Façade](#5-promptbuilder-façade)
6. [LLMInterface](#6-llminterface)
7. [LLMResponse Model](#7-llmresponse-model)
8. [Provider Deep-Dive](#8-provider-deep-dive)
9. [Streaming](#9-streaming)
10. [Quick Reference](#10-quick-reference)
11. [Common Pitfalls](#11-common-pitfalls)

---

## 1. Generation Overview

The generation layer converts retrieved chunks into a final natural-language
answer. It has three responsibilities:

```
Retrieved Chunks  +  User Query
        │
        ▼
  ContextBuilder      ← token-aware context assembly + citation numbering
        │
        ▼
  PromptBuilder       ← pick a template, inject context + query + history
        │
        ▼
   [messages list]    ← OpenAI-format: [{role, content}, …]
        │
        ▼
  LLMInterface        ← call the actual LLM (OpenAI / Ollama / Anthropic / …)
        │
        ▼
   LLMResponse        ← answer text + token counts + latency + cost
```

---

## 2. PromptTemplate

A `PromptTemplate` defines the *shape* of the prompt — the system persona
and the user message format. Seven built-in templates ship out-of-the-box:

```python
@dataclass
class PromptTemplate:
    name:           str   # unique identifier
    description:    str
    system_message: str   # persona and instructions for the LLM
    user_template:  str   # f-string with {context} and {question} slots
```

### Built-in Templates

| Name | Best for | Key characteristic |
|---|---|---|
| `default` | General QA | Balanced; cites sources |
| `with_citations` | Research, legal | Explicit `[1]` style citations |
| `conversational` | Chatbots | Friendly; uses conversation history |
| `summarization` | Long documents | Structured bullet-point output |
| `structured_output` | JSON APIs | Forces JSON-formatted answers |
| `chain_of_thought` | Complex reasoning | Step-by-step reasoning |
| `safety_strict` | Regulated domains | Refuses uncertain answers |

### System Message Examples

**`default`**:
```
You are a helpful, accurate assistant. Answer questions based ONLY on
the provided context. If the context doesn't contain the answer, say
"I don't have enough information to answer that." Do not fabricate.
```

**`safety_strict`**:
```
You are a compliance assistant. You MUST:
1. Only state facts explicitly present in the context.
2. Use exact quotes when possible.
3. If uncertain, respond: "I cannot confirm this from available documents."
4. Never speculate, infer, or extrapolate.
```

**`chain_of_thought`**:
```
You are an analytical assistant. For every question:
1. First, identify key facts from the context.
2. Reason step-by-step.
3. Only then provide your final answer.
Format: "Reasoning: ... | Answer: ..."
```

### Registering a Custom Template

```python
from generation import PromptBuilder, PromptTemplate, TEMPLATES

my_template = PromptTemplate(
    name="legal_review",
    description="Legal document reviewer",
    system_message="You are a legal analyst. Only cite primary sources.",
    user_template=(
        "Legal Context:\n{context}\n\n"
        "Legal Question: {question}\n\n"
        "Legal Analysis:"
    ),
)

builder = PromptBuilder(template="default")
builder.register_template(my_template)    # now available to all builders

# Use it
builder2 = PromptBuilder(template="legal_review")
```

---

## 3. ContextBuilder

Assembles retrieved chunks into a single context string while staying within
a token budget and attaching citation metadata.

```python
from generation.prompt_builder import ContextBuilder

builder = ContextBuilder(
    max_tokens=3000,           # hard limit; chunks are dropped if budget exceeded
    citation_style="inline",   # "inline" | "footnote" | "none"
    order="relevance",         # "relevance" | "reverse_rank" | "source"
    include_metadata=True,     # include source/page in context
    metadata_fields=["source", "page"],
)

context_str, citations = builder.build(search_results)
```

### Citation Styles

**`inline`** (default):
```
[1] Customers are eligible for a full refund within 30 days of purchase.
[2] Digital downloads are non-refundable once accessed.
```

**`footnote`**:
```
Customers are eligible for a full refund within 30 days¹.

---
¹ Source: policy.pdf, page 3
```

**`none`**:
```
Customers are eligible for a full refund within 30 days of purchase.
Digital downloads are non-refundable once accessed.
```

### Token Budget Algorithm

```
remaining = max_tokens

for chunk in sorted_by_relevance:
    tokens = len(chunk.content) / 4    # approximate (chars / 4 ≈ tokens)
    if tokens <= remaining:
        include chunk
        remaining -= tokens
    else:
        stop (budget exhausted)
```

This ensures the context never exceeds the LLM's window even with many chunks.

### Citation Object Structure

```python
citations = [
    {"ref": 1, "chunk_id": "abc123", "source": "policy.pdf",
     "page": 3, "score": 0.923},
    {"ref": 2, "chunk_id": "def456", "source": "faq.md",
     "page": None, "score": 0.887},
]
```

---

## 4. ConversationHistory

Maintains multi-turn conversation state for chatbot use cases.

```python
from generation.prompt_builder import ConversationHistory

history = ConversationHistory(max_tokens=2000)  # oldest messages dropped when exceeded

# Add turns
history.add("user", "What is the refund policy?")
history.add("assistant", "You can return items within 30 days for a full refund.")
history.add("user", "What about digital downloads?")

# Get OpenAI-format message list
messages = history.get_messages()
# [
#   {"role": "user",      "content": "What is the refund policy?"},
#   {"role": "assistant", "content": "You can return items within 30 days…"},
#   {"role": "user",      "content": "What about digital downloads?"},
# ]

print(len(history))     # 3 messages
history.clear()         # reset
```

### How `max_tokens` Works

When history exceeds the token budget, the **oldest** messages are dropped
first (FIFO eviction), keeping the most recent context:

```
[msg1] [msg2] [msg3] [msg4] [msg5]    → total 2500 tokens, max=2000
 ↑ dropped                            → keep [msg2] [msg3] [msg4] [msg5]
```

---

## 5. PromptBuilder Façade

Combines template + context builder + history into a single call:

```python
from generation import PromptBuilder

builder = PromptBuilder(
    template="with_citations",
    max_context_tokens=3000,
    citation_style="inline",
    custom_system_prompt=None,   # override template's system_message
)

# Basic: query + retrieved results
messages, citations = builder.build(
    question="What is the refund policy?",
    retrieved_results=search_results,
)
# messages = [{"role": "system", "content": "…"}, {"role": "user", "content": "…"}]

# With conversation history (multi-turn)
messages, citations = builder.build(
    question="What about digital downloads?",
    retrieved_results=search_results,
    history=conversation_history,
)
# messages = [system, user_turn1, assistant_turn1, user_turn2]

# Standalone (no retrieved chunks — manual context)
messages = builder.build_standalone(
    question="Summarize the following",
    context="Some manually provided context text here",
)

# Introspection
print(builder.available_templates)   # list of all template names
print(builder.template_name)         # currently active template
```

---

## 6. LLMInterface

The unified façade over all LLM providers.

```python
from generation import LLMInterface

llm = LLMInterface(
    provider="openai",              # "openai"|"azure"|"ollama"|"anthropic"|"huggingface"
    model="gpt-4o-mini",
    temperature=0.1,                # low for factual QA
    max_tokens=1024,
    api_key=os.getenv("OPENAI_API_KEY"),
    fallback_provider="ollama",     # if primary fails, fall back to Ollama
    fallback_model="llama3.1:8b",
)

# Chat (OpenAI-format messages)
response = llm.chat(messages)

# Convenience: answer(question, context)
response = llm.answer("What is the refund policy?", context_str)

# Complete (single prompt, no messages)
response = llm.complete("Answer this: What is 2+2?")

# Stream
for token in llm.stream(messages):
    print(token, end="", flush=True)
```

---

## 7. LLMResponse Model

```python
@dataclass
class LLMResponse:
    content:           str     # the generated answer text
    model:             str     # model that generated it
    prompt_tokens:     int     # input tokens used
    completion_tokens: int     # output tokens generated
    total_tokens:      int     # prompt + completion
    latency_ms:        float   # wall-clock time for the API call
    finish_reason:     str     # "stop" | "length" | "content_filter"

    @property
    def cost_usd(self) -> float:
        # Computed from token counts using per-model pricing table
        ...

    def to_dict(self) -> dict:
        # Serializable summary: content, model, tokens, latency_ms, cost_usd
        ...
```

```python
response = llm.chat(messages)
print(response.content)           # "Customers can return items within 30 days…"
print(response.total_tokens)      # 312
print(f"${response.cost_usd:.5f}")  # $0.00005 (gpt-4o-mini)
print(response.latency_ms)        # 1234.5 ms
```

### Cost Tracking (built-in pricing table)

| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|---|---|---|
| `gpt-4o` | $2.50 | $10.00 |
| `gpt-4o-mini` | $0.15 | $0.60 |
| `gpt-4-turbo` | $10.00 | $30.00 |
| `claude-3-5-sonnet` | $3.00 | $15.00 |
| `llama3.1:8b` (Ollama) | $0.00 | $0.00 |

---

## 8. Provider Deep-Dive

### OpenAILLM

```python
llm = LLMInterface(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1024,
    top_p=0.95,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

# JSON mode (guaranteed JSON output)
response = llm.chat(messages, json_mode=True)
data = json.loads(response.content)
```

### Azure OpenAI

```python
llm = LLMInterface(
    provider="azure",
    model="gpt-4o",                           # deployment name
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_api_version="2024-02-01",
)
```

### Ollama (Local, Free)

```python
llm = LLMInterface(
    provider="ollama",
    model="llama3.1:8b",
    base_url="http://localhost:11434",   # or Docker service name
)
# Requires: ollama serve + ollama pull llama3.1:8b
```

### Anthropic Claude

```python
llm = LLMInterface(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=2048,
)
```

### Fallback Pattern

```python
# If OpenAI is down or rate-limited, automatically fall back to local Ollama
llm = LLMInterface(
    provider="openai",
    model="gpt-4o-mini",
    fallback_provider="ollama",
    fallback_model="llama3.1:8b",
)
# If openai.chat() raises → automatically retries with ollama
```

---

## 9. Streaming

Streaming returns tokens as they're generated — essential for responsive UIs.

```python
# Server-Sent Events in FastAPI
async def stream_endpoint(query: str):
    messages, _ = builder.build(query, retriever.retrieve(query))
    for token in llm.stream(messages):
        yield f"data: {token}\n\n"

# CLI streaming
for token in llm.stream(messages):
    print(token, end="", flush=True)
print()  # newline at end
```

**Under the hood (OpenAI)**:
```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True,        # ← key flag
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        yield delta.content
```

---

## 10. Quick Reference

```python
from generation import PromptBuilder, LLMInterface, ConversationHistory

# ── Setup ────────────────────────────────────────────────────────────────
builder = PromptBuilder(template="with_citations", max_context_tokens=3000)
llm     = LLMInterface(provider="openai", model="gpt-4o-mini", temperature=0.1)
history = ConversationHistory(max_tokens=2000)

# ── Single turn ──────────────────────────────────────────────────────────
messages, citations = builder.build("What is the refund policy?", results)
response = llm.chat(messages)
print(response.content)

# ── Multi-turn ───────────────────────────────────────────────────────────
history.add("user", "What is the refund policy?")
history.add("assistant", response.content)
messages, _ = builder.build("What about digital downloads?", results, history)
response2 = llm.chat(messages)

# ── Streaming ────────────────────────────────────────────────────────────
for token in llm.stream(messages):
    print(token, end="", flush=True)

# ── JSON structured output ───────────────────────────────────────────────
builder_json = PromptBuilder(template="structured_output")
messages, _  = builder_json.build("Extract key facts about refunds", results)
response     = llm.chat(messages, json_mode=True)
data         = json.loads(response.content)

# ── Cost tracking ─────────────────────────────────────────────────────────
print(f"Tokens: {response.total_tokens}, Cost: ${response.cost_usd:.5f}")

# ── Available templates ──────────────────────────────────────────────────
print(builder.available_templates)
# ['default', 'with_citations', 'conversational', 'summarization',
#  'structured_output', 'chain_of_thought', 'safety_strict']
```

---

## 11. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| **Temperature too high** | Hallucinations in QA | Set `temperature=0.0–0.2` for factual tasks |
| **max_context_tokens too large** | `context_length_exceeded` error | Keep ≤ 80% of model's context window |
| **No system message** | LLM ignores instructions | Always provide a system message via template |
| **Forgetting `[DONE]` in stream** | Infinite loop in SSE | Stream implementation handles this internally |
| **JSON mode without JSON instruction** | Non-JSON response | Use `structured_output` template with JSON mode |
| **History not cleared between users** | Cross-user data leakage | Use `session_id` scoped histories |
| **Fallback not configured** | 500 errors on API outage | Always set `fallback_provider="ollama"` |
| **Azure wrong deployment name** | `404 NotFound` | `model` = deployment name, not model family |
