"""
tests/unit/test_prompt_builder.py
──────────────────────────────────
Unit tests for PromptBuilder, ContextBuilder, and ConversationHistory.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from generation.prompt_builder import (
    TEMPLATES,
    ContextBuilder,
    ConversationHistory,
    PromptBuilder,
    PromptTemplate,
)
from vector_store.vector_store import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_results(n: int = 3, base_score: float = 0.9) -> list[SearchResult]:
    return [
        SearchResult(
            chunk_id=f"chunk-{i}",
            content=f"This is chunk {i} with relevant content about refund policies.",
            score=base_score - i * 0.05,
            metadata={"source": f"doc_{i}.pdf", "page": i + 1},
            rank=i,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------
class TestPromptTemplate:
    def test_format_user_substitutes_context_and_question(self):
        template = TEMPLATES["default"]
        result = template.format_user(context="ctx here", question="What?")
        assert "ctx here" in result
        assert "What?" in result

    def test_all_builtin_templates_have_required_fields(self):
        required = {"name", "system_message", "user_template"}
        for name, tmpl in TEMPLATES.items():
            assert tmpl.name == name, f"Template '{name}' has wrong name field"
            assert tmpl.system_message, f"Template '{name}' has empty system_message"
            assert "{context}" in tmpl.user_template, \
                f"Template '{name}' missing {{context}} in user_template"
            assert "{question}" in tmpl.user_template, \
                f"Template '{name}' missing {{question}} in user_template"

    def test_builtin_templates_list(self):
        expected = {
            "default", "with_citations", "conversational",
            "summarization", "structured_output", "chain_of_thought", "safety_strict"
        }
        assert expected.issubset(set(TEMPLATES.keys()))


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------
class TestContextBuilder:
    def test_build_returns_non_empty_context(self):
        builder = ContextBuilder(max_tokens=1000, citation_style="inline")
        results = _make_results(3)
        context, citations = builder.build(results)
        assert context.strip() != ""

    def test_build_returns_correct_citation_count(self):
        builder = ContextBuilder(max_tokens=1000)
        results = _make_results(3)
        _, citations = builder.build(results)
        assert len(citations) == 3

    def test_citation_ref_numbers_sequential(self):
        builder = ContextBuilder(max_tokens=1000)
        results = _make_results(4)
        _, citations = builder.build(results)
        refs = [c["ref"] for c in citations]
        assert refs == list(range(1, len(refs) + 1))

    def test_citation_has_required_keys(self):
        builder = ContextBuilder(max_tokens=1000)
        results = _make_results(2)
        _, citations = builder.build(results)
        for c in citations:
            assert "ref" in c
            assert "chunk_id" in c
            assert "source" in c
            assert "score" in c

    def test_token_budget_limits_chunks(self):
        """Very small token budget should include fewer chunks."""
        builder_small = ContextBuilder(max_tokens=30)
        builder_large = ContextBuilder(max_tokens=10000)
        results = _make_results(10)
        _, cit_small = builder_small.build(results)
        _, cit_large = builder_large.build(results)
        assert len(cit_small) <= len(cit_large)

    def test_empty_results_returns_fallback(self):
        builder = ContextBuilder(max_tokens=1000)
        context, citations = builder.build([])
        assert "No relevant context" in context
        assert citations == []

    def test_inline_citation_markers_in_context(self):
        builder = ContextBuilder(max_tokens=2000, citation_style="inline")
        results = _make_results(2)
        context, _ = builder.build(results)
        assert "[1]" in context
        assert "[2]" in context

    def test_metadata_included_when_flag_true(self):
        builder = ContextBuilder(max_tokens=2000, include_metadata=True,
                                 metadata_fields=["source"])
        results = _make_results(1)
        context, _ = builder.build(results)
        assert "source=" in context or "doc_0.pdf" in context

    def test_reverse_rank_order(self):
        """With order='reverse_rank', last result should appear first in context."""
        builder = ContextBuilder(max_tokens=5000, order="reverse_rank",
                                 citation_style="none", include_metadata=False)
        results = _make_results(3)
        context, _ = builder.build(results)
        # chunk-2 should appear before chunk-0 in context string
        pos_0 = context.find("chunk 0")
        pos_2 = context.find("chunk 2")
        if pos_0 != -1 and pos_2 != -1:
            assert pos_2 < pos_0


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------
class TestConversationHistory:
    def test_add_and_format(self):
        h = ConversationHistory()
        h.add("user", "Hello")
        h.add("assistant", "Hi there!")
        formatted = h.get_formatted()
        assert "Hello" in formatted
        assert "Hi there!" in formatted

    def test_get_messages_openai_format(self):
        h = ConversationHistory()
        h.add("user", "What is RAG?")
        h.add("assistant", "RAG is Retrieval-Augmented Generation.")
        msgs = h.get_messages()
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_len_returns_message_count(self):
        h = ConversationHistory()
        assert len(h) == 0
        h.add("user", "msg")
        assert len(h) == 1
        h.add("assistant", "resp")
        assert len(h) == 2

    def test_clear_empties_history(self):
        h = ConversationHistory()
        h.add("user", "msg")
        h.add("assistant", "resp")
        h.clear()
        assert len(h) == 0

    def test_token_budget_truncates_old_messages(self):
        """Very small budget should drop older messages."""
        h = ConversationHistory(max_tokens=5)
        for i in range(20):
            h.add("user", f"message number {i} with extra padding")
        msgs = h.get_messages()
        # Should have fewer than 20 messages after truncation
        assert len(msgs) < 20

    def test_get_formatted_role_casing(self):
        h = ConversationHistory()
        h.add("user", "hi")
        formatted = h.get_formatted()
        assert "User:" in formatted


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------
class TestPromptBuilder:
    def test_default_template_builds_messages(self):
        builder = PromptBuilder(template="default")
        results = _make_results(2)
        messages, citations = builder.build("What is the refund policy?", results)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_present(self):
        builder = PromptBuilder(template="default")
        results = _make_results(1)
        messages, _ = builder.build("Question?", results)
        assert messages[0]["role"] == "system"
        assert len(messages[0]["content"]) > 0

    def test_user_message_contains_question(self):
        builder = PromptBuilder(template="default")
        results = _make_results(1)
        messages, _ = builder.build("specific question here", results)
        assert "specific question here" in messages[1]["content"]

    def test_user_message_contains_context(self):
        builder = PromptBuilder(template="default")
        results = _make_results(2)
        messages, _ = builder.build("question", results)
        assert "chunk 0" in messages[1]["content"] or "chunk 1" in messages[1]["content"]

    def test_citations_returned(self):
        builder = PromptBuilder(template="with_citations")
        results = _make_results(3)
        _, citations = builder.build("question", results)
        assert len(citations) == 3

    def test_invalid_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            PromptBuilder(template="nonexistent_template_xyz")

    def test_custom_system_prompt_overrides(self):
        custom = "You are a custom assistant."
        builder = PromptBuilder(template="default", custom_system_prompt=custom)
        results = _make_results(1)
        messages, _ = builder.build("q", results)
        assert messages[0]["content"] == custom

    def test_with_conversation_history(self):
        builder = PromptBuilder(template="default")
        history = ConversationHistory()
        history.add("user", "Previous question")
        history.add("assistant", "Previous answer")
        results = _make_results(1)
        messages, _ = builder.build("New question", results, history=history)
        # System + history(2) + new user = 4 messages
        assert len(messages) == 4
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]

    def test_available_templates_list(self):
        builder = PromptBuilder()
        templates = builder.available_templates
        assert "default" in templates
        assert "with_citations" in templates
        assert len(templates) >= 7

    def test_template_name_property(self):
        builder = PromptBuilder(template="summarization")
        assert builder.template_name == "summarization"

    def test_build_standalone(self):
        builder = PromptBuilder(template="default")
        messages = builder.build_standalone("my question", "my context")
        assert len(messages) == 2
        assert "my question" in messages[1]["content"]
        assert "my context" in messages[1]["content"]

    def test_empty_results_still_returns_messages(self):
        builder = PromptBuilder(template="default")
        messages, citations = builder.build("question with no context", [])
        assert len(messages) == 2
        assert citations == []

    def test_register_custom_template(self):
        builder = PromptBuilder(template="default")
        custom = PromptTemplate(
            name="my_template",
            system_message="You are a test assistant.",
            user_template="Context:\n{context}\n\nQ: {question}\nA:",
        )
        builder.register_template(custom)
        assert "my_template" in TEMPLATES

    def test_all_builtin_templates_buildable(self):
        """Every built-in template should produce valid messages without error."""
        results = _make_results(2)
        history = ConversationHistory()
        history.add("user", "prev question")
        history.add("assistant", "prev answer")

        for tmpl_name in TEMPLATES:
            builder = PromptBuilder(template=tmpl_name)
            try:
                if tmpl_name == "conversational":
                    messages, _ = builder.build("question", results, history=history)
                else:
                    messages, _ = builder.build("question", results)
                assert len(messages) >= 2, f"Template '{tmpl_name}' returned < 2 messages"
            except Exception as e:
                pytest.fail(f"Template '{tmpl_name}' raised {type(e).__name__}: {e}")
