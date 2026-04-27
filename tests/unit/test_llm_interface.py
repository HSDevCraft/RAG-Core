"""
tests/unit/test_llm_interface.py
──────────────────────────────────
Unit tests for the LLM abstraction layer.

All external API calls are mocked — these tests run with no API key.
Coverage:
  - LLMResponse model (cost_usd, to_dict)
  - OpenAILLM: chat(), stream(), complete()
  - OllamaLLM: chat(), stream()
  - LLMInterface: provider routing, fallback, answer(), stream_answer()
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pytest

from generation.llm_interface import LLMInterface, LLMResponse, OpenAILLM, OllamaLLM


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------
class TestLLMResponse:
    def test_basic_fields(self):
        r = LLMResponse(content="hello", model="gpt-4o-mini",
                        prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert r.content == "hello"
        assert r.model == "gpt-4o-mini"
        assert r.total_tokens == 150

    def test_cost_usd_gpt4o_mini(self):
        r = LLMResponse(
            content="x", model="gpt-4o-mini",
            prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000,
        )
        # $0.15/1M in + $0.6/1M out = $0.75 total
        assert abs(r.cost_usd - 0.75) < 0.01

    def test_cost_usd_unknown_model_uses_fallback(self):
        r = LLMResponse(content="x", model="unknown-model",
                        prompt_tokens=1_000_000, completion_tokens=1_000_000,
                        total_tokens=2_000_000)
        # Fallback: $1/1M + $3/1M = $4
        assert abs(r.cost_usd - 4.0) < 0.01

    def test_to_dict_has_required_keys(self):
        r = LLMResponse(content="answer", model="gpt-4o-mini",
                        prompt_tokens=10, completion_tokens=20, total_tokens=30,
                        latency_ms=500.0)
        d = r.to_dict()
        assert "content" in d
        assert "model" in d
        assert "tokens" in d
        assert "latency_ms" in d
        assert "cost_usd" in d
        assert d["tokens"]["total"] == 30

    def test_finish_reason_default(self):
        r = LLMResponse(content="x", model="gpt-4o")
        assert r.finish_reason == "stop"


# ---------------------------------------------------------------------------
# OpenAILLM (mocked)
# ---------------------------------------------------------------------------
class TestOpenAILLM:
    def _make_mock_response(self, content: str = "test answer",
                             model: str = "gpt-4o-mini",
                             finish: str = "stop"):
        """Build a mock OpenAI SDK response object."""
        choice = MagicMock()
        choice.message.content = content
        choice.finish_reason = finish

        usage = MagicMock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 25
        usage.total_tokens = 75

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        response.model = model
        return response

    def test_chat_returns_llm_response(self):
        llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test")
        mock_resp = self._make_mock_response("Hello there!")

        with patch("openai.OpenAI") as MockOpenAI:
            client = MockOpenAI.return_value
            client.chat.completions.create.return_value = mock_resp
            llm._client = client

            result = llm.chat([{"role": "user", "content": "Hi"}])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello there!"
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 25

    def test_chat_passes_temperature(self):
        llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test", temperature=0.5)
        mock_resp = self._make_mock_response()

        with patch("openai.OpenAI") as MockOpenAI:
            client = MockOpenAI.return_value
            client.chat.completions.create.return_value = mock_resp
            llm._client = client
            llm.chat([{"role": "user", "content": "test"}])
            call_kwargs = client.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == 0.5

    def test_chat_json_mode(self):
        llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test")
        mock_resp = self._make_mock_response('{"answer": "yes"}')

        with patch("openai.OpenAI") as MockOpenAI:
            client = MockOpenAI.return_value
            client.chat.completions.create.return_value = mock_resp
            llm._client = client
            result = llm.chat([{"role": "user", "content": "Q"}], json_mode=True)
            call_kwargs = client.chat.completions.create.call_args[1]
            assert call_kwargs.get("response_format") == {"type": "json_object"}

    def test_complete_wraps_chat(self):
        llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test")
        mock_resp = self._make_mock_response("completion result")

        with patch("openai.OpenAI") as MockOpenAI:
            client = MockOpenAI.return_value
            client.chat.completions.create.return_value = mock_resp
            llm._client = client
            result = llm.complete("Just a prompt")

        assert result.content == "completion result"

    def test_stream_yields_tokens(self):
        llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test")

        def _make_chunk(token):
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = token
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = delta
            return chunk

        chunks = [_make_chunk(t) for t in ["Hello", " world", "!"]]

        with patch("openai.OpenAI") as MockOpenAI:
            client = MockOpenAI.return_value
            client.chat.completions.create.return_value = iter(chunks)
            llm._client = client
            tokens = list(llm.stream([{"role": "user", "content": "test"}]))

        assert tokens == ["Hello", " world", "!"]


# ---------------------------------------------------------------------------
# OllamaLLM (mocked)
# ---------------------------------------------------------------------------
class TestOllamaLLM:
    def test_chat_parses_response(self):
        llm = OllamaLLM(model="llama3.1:8b", base_url="http://localhost:11434")
        mock_json = {
            "message": {"content": "Ollama answer"},
            "prompt_eval_count": 30,
            "eval_count": 20,
        }

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_json
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = llm.chat([{"role": "user", "content": "test"}])

        assert result.content == "Ollama answer"
        assert result.prompt_tokens == 30
        assert result.completion_tokens == 20

    def test_chat_posts_to_correct_url(self):
        llm = OllamaLLM(model="llama3.1:8b", base_url="http://ollama:11434")
        mock_json = {"message": {"content": "ok"}, "prompt_eval_count": 0, "eval_count": 0}

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_json
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response
            llm.chat([{"role": "user", "content": "x"}])
            call_url = mock_post.call_args[0][0]
            assert call_url == "http://ollama:11434/api/chat"

    def test_stream_yields_tokens(self):
        llm = OllamaLLM(model="llama3.1:8b")
        import json as _json

        stream_lines = [
            _json.dumps({"message": {"content": "tok1"}, "done": False}),
            _json.dumps({"message": {"content": " tok2"}, "done": False}),
            _json.dumps({"message": {"content": ""}, "done": True}),
        ]

        with patch("requests.post") as mock_post:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_cm)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_cm.iter_lines.return_value = [l.encode() for l in stream_lines]
            mock_cm.raise_for_status = MagicMock()
            mock_post.return_value = mock_cm

            tokens = list(llm.stream([{"role": "user", "content": "test"}]))

        assert "tok1" in tokens
        assert " tok2" in tokens


# ---------------------------------------------------------------------------
# LLMInterface — façade + fallback
# ---------------------------------------------------------------------------
class TestLLMInterface:
    def test_valid_provider_creates_interface(self):
        llm = LLMInterface(provider="openai", api_key="sk-test")
        assert llm.provider == "openai"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMInterface(provider="nonexistent_provider")

    def test_chat_delegates_to_provider(self, mock_llm):
        """The mock_llm fixture returns a deterministic LLMResponse."""
        result = mock_llm.chat([{"role": "user", "content": "Hello!"}])
        assert isinstance(result, LLMResponse)
        assert "Mock answer" in result.content

    def test_complete_convenience_method(self, mock_llm):
        result = mock_llm.complete("What is 2+2?")
        assert isinstance(result, LLMResponse)

    def test_stream_yields_strings(self, mock_llm):
        tokens = list(mock_llm.stream([{"role": "user", "content": "test"}]))
        assert all(isinstance(t, str) for t in tokens)
        assert len(tokens) > 0

    def test_answer_convenience_method(self, mock_llm):
        result = mock_llm.answer("question?", "some context")
        assert isinstance(result, LLMResponse)

    def test_fallback_called_on_primary_failure(self):
        """If primary LLM raises, fallback should be called."""
        primary = MagicMock()
        primary.chat.side_effect = Exception("Primary failed")

        fallback = MagicMock()
        fallback.chat.return_value = LLMResponse(
            content="fallback answer", model="fallback-model"
        )

        llm = LLMInterface.__new__(LLMInterface)
        llm._llm = primary
        llm._fallback = fallback
        llm._provider = "mock"

        result = llm.chat([{"role": "user", "content": "q"}])
        assert result.content == "fallback answer"
        primary.chat.assert_called_once()
        fallback.chat.assert_called_once()

    def test_no_fallback_reraises_on_failure(self):
        llm = LLMInterface.__new__(LLMInterface)
        llm._provider = "mock"
        llm._fallback = None
        failing_llm = MagicMock()
        failing_llm.chat.side_effect = RuntimeError("Service unavailable")
        llm._llm = failing_llm

        with pytest.raises(RuntimeError, match="Service unavailable"):
            llm.chat([{"role": "user", "content": "q"}])
