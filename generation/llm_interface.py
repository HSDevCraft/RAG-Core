"""
llm_interface.py
────────────────
Unified LLM abstraction supporting multiple providers.

Providers:
  - openai       (GPT-4o, GPT-4-turbo, GPT-3.5-turbo)
  - azure_openai (enterprise Azure OpenAI deployments)
  - ollama       (local open-source models: Llama3, Mistral, Phi-3)
  - huggingface  (direct HuggingFace transformers, GPU required)
  - anthropic    (Claude-3.5-Sonnet, Claude-3-Haiku)

Features:
  - Streaming support (SSE token-by-token)
  - Retry with exponential backoff (tenacity)
  - Token usage tracking
  - Structured output (JSON mode)
  - Cost estimation
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw: Any = None

    @property
    def cost_usd(self) -> float:
        """Approximate cost based on public pricing."""
        PRICING = {
            "gpt-4o":           (5.0,  15.0),    # (input, output) per 1M tokens
            "gpt-4o-mini":      (0.15,  0.6),
            "gpt-4-turbo":      (10.0, 30.0),
            "gpt-3.5-turbo":    (0.5,   1.5),
            "claude-3-5-sonnet-20241022": (3.0, 15.0),
            "claude-3-haiku-20240307":    (0.25, 1.25),
        }
        inp_price, out_price = PRICING.get(self.model, (1.0, 3.0))
        return (self.prompt_tokens * inp_price + self.completion_tokens * out_price) / 1_000_000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "tokens": {"prompt": self.prompt_tokens,
                       "completion": self.completion_tokens,
                       "total": self.total_tokens},
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "finish_reason": self.finish_reason,
        }


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class BaseLLM:
    def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        raise NotImplementedError

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        raise NotImplementedError

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Simple text-completion wrapper."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
class OpenAILLM(BaseLLM):
    """
    OpenAI chat completions.

    Model selection guide:
      gpt-4o          → best accuracy, $5/1M in + $15/1M out
      gpt-4o-mini     → 95% quality, 10× cheaper → use for high-volume inference
      gpt-3.5-turbo   → fastest, cheapest, good for simple tasks

    JSON mode: set response_format={"type": "json_object"} + mention JSON in prompt.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 60,
        seed: Optional[int] = 42,
    ):
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._seed = seed
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=self._api_key, timeout=self._timeout
            )
        return self._client

    def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        from tenacity import retry, stop_after_attempt, wait_exponential

        client = self._get_client()
        params = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }
        if self._seed is not None:
            params["seed"] = self._seed
        if kwargs.get("json_mode"):
            params["response_format"] = {"type": "json_object"}

        @retry(stop=stop_after_attempt(3),
               wait=wait_exponential(multiplier=1, min=2, max=30))
        def _call():
            return client.chat.completions.create(**params)

        t0 = time.perf_counter()
        response = _call()
        latency = (time.perf_counter() - t0) * 1000

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency,
            finish_reason=choice.finish_reason,
            raw=response,
        )

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        client = self._get_client()
        params = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "stream": True,
        }
        response = client.chat.completions.create(**params)
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------
class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI deployment — same API as OpenAI but enterprise-grade."""

    def __init__(
        self,
        deployment_name: str,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self._deployment = deployment_name
        self._endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self._api_version = api_version
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.AzureOpenAI(
                azure_endpoint=self._endpoint,
                api_key=self._api_key,
                api_version=self._api_version,
            )
        return self._client

    def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        client = self._get_client()
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        latency = (time.perf_counter() - t0) * 1000
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=self._deployment,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency_ms=latency,
            finish_reason=choice.finish_reason,
        )

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._deployment, messages=messages, stream=True,
            temperature=kwargs.get("temperature", self._temperature),
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


# ---------------------------------------------------------------------------
# Ollama (local open-source)
# ---------------------------------------------------------------------------
class OllamaLLM(BaseLLM):
    """
    Ollama local inference server.

    Popular models:
      llama3.1:8b    → best open-source at 8B scale
      mistral:7b     → fast, strong instruction following
      phi3:mini      → 3.8B, extremely fast, good for simple RAG
      gemma2:9b      → excellent multilingual
      qwen2:7b       → strong coding + multilingual

    Start server: `ollama serve`
    Pull model:   `ollama pull llama3.1:8b`
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens

    def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        import requests as req
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self._temperature),
                "num_predict": kwargs.get("max_tokens", self._max_tokens),
            },
        }
        t0 = time.perf_counter()
        response = req.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        latency = (time.perf_counter() - t0) * 1000
        data = response.json()
        content = data.get("message", {}).get("content", "")
        return LLMResponse(
            content=content,
            model=self._model,
            latency_ms=latency,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        )

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        import requests as req
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": kwargs.get("temperature", self._temperature)},
        }
        with req.post(f"{self._base_url}/api/chat", json=payload,
                      stream=True, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------
class AnthropicLLM(BaseLLM):
    """Anthropic Claude models — excellent instruction following and safety."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        client = self._get_client()
        # Anthropic separates system prompt from user messages
        system_msg = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append(m)

        t0 = time.perf_counter()
        response = client.messages.create(
            model=self._model,
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            system=system_msg,
            messages=user_messages,
        )
        latency = (time.perf_counter() - t0) * 1000
        content = response.content[0].text if response.content else ""
        return LLMResponse(
            content=content,
            model=self._model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=latency,
            finish_reason=response.stop_reason or "stop",
        )

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        client = self._get_client()
        system_msg = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append(m)

        with client.messages.stream(
            model=self._model,
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            system=system_msg,
            messages=user_messages,
        ) as stream_resp:
            for text in stream_resp.text_stream:
                yield text


# ---------------------------------------------------------------------------
# HuggingFace Transformers (local GPU)
# ---------------------------------------------------------------------------
class HuggingFaceLLM(BaseLLM):
    """
    Direct HuggingFace transformers inference.
    Requires GPU for reasonable speed (8B+ models).
    Use with 4-bit quantization (bitsandbytes) for 24GB VRAM.
    """

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "cuda",
        load_in_4bit: bool = True,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ):
        self._model_id = model_id
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline, BitsAndBytesConfig

            bnb_config = None
            if self._load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            self._pipeline = pipeline(
                "text-generation",
                model=self._model_id,
                device_map="auto",
                model_kwargs={"quantization_config": bnb_config} if bnb_config else {},
                torch_dtype=torch.float16,
            )
            logger.info("HuggingFace model loaded: %s", self._model_id)

    def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        self._load()
        t0 = time.perf_counter()
        outputs = self._pipeline(
            messages,
            max_new_tokens=kwargs.get("max_tokens", self._max_new_tokens),
            do_sample=self._temperature > 0,
            temperature=self._temperature if self._temperature > 0 else None,
            return_full_text=False,
        )
        latency = (time.perf_counter() - t0) * 1000
        content = outputs[0]["generated_text"]
        if isinstance(content, list):
            content = content[-1].get("content", "")
        return LLMResponse(
            content=str(content),
            model=self._model_id,
            latency_ms=latency,
        )

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        from transformers import TextIteratorStreamer
        import threading
        self._load()
        streamer = TextIteratorStreamer(self._pipeline.tokenizer, skip_special_tokens=True)
        thread = threading.Thread(
            target=self._pipeline,
            kwargs={"text_inputs": messages, "streamer": streamer,
                    "max_new_tokens": kwargs.get("max_tokens", self._max_new_tokens)},
        )
        thread.start()
        for token in streamer:
            yield token


# ---------------------------------------------------------------------------
# LLM Interface — unified façade
# ---------------------------------------------------------------------------
class LLMInterface:
    """
    Main entry-point. Wraps provider selection, fallback, and logging.

    Usage:
        llm = LLMInterface(provider="openai", model="gpt-4o-mini")
        response = llm.chat(messages)
        answer = llm.answer(query, context_str)
        for token in llm.stream_answer(query, context_str): print(token, end="")
    """

    _PROVIDER_MAP = {
        "openai":       OpenAILLM,
        "azure_openai": AzureOpenAILLM,
        "ollama":       OllamaLLM,
        "anthropic":    AnthropicLLM,
        "huggingface":  HuggingFaceLLM,
    }

    def __init__(
        self,
        provider: str = "openai",
        fallback_provider: Optional[str] = None,
        **kwargs,
    ):
        if provider not in self._PROVIDER_MAP:
            raise ValueError(f"Unknown provider: {provider}. "
                             f"Choose from {list(self._PROVIDER_MAP)}")
        self._llm: BaseLLM = self._PROVIDER_MAP[provider](**kwargs)
        self._fallback: Optional[BaseLLM] = None
        if fallback_provider:
            self._fallback = self._PROVIDER_MAP[fallback_provider]()
        self._provider = provider

    def chat(self, messages: List[dict], **kwargs) -> LLMResponse:
        try:
            return self._llm.chat(messages, **kwargs)
        except Exception as e:
            if self._fallback:
                logger.warning("Primary LLM failed (%s), using fallback.", e)
                return self._fallback.chat(messages, **kwargs)
            raise

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        return self._llm.stream(messages, **kwargs)

    def answer(self, query: str, context: str,
               system_prompt: Optional[str] = None) -> LLMResponse:
        """Convenience: build a simple RAG prompt and call."""
        sys_msg = system_prompt or (
            "Answer the question using ONLY the provided context. "
            "If the answer is not there, say so."
        )
        messages = [
            {"role": "system",  "content": sys_msg},
            {"role": "user",    "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        return self.chat(messages)

    def stream_answer(self, query: str, context: str,
                      system_prompt: Optional[str] = None) -> Iterator[str]:
        sys_msg = system_prompt or (
            "Answer the question using ONLY the provided context."
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        return self.stream(messages)

    @property
    def provider(self) -> str:
        return self._provider
