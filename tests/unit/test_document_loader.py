"""
tests/unit/test_document_loader.py
───────────────────────────────────
Unit tests for the data ingestion layer.

Coverage targets:
  - TextLoader / PDFLoader / HTMLLoader / JSONLoader
  - DirectoryLoader auto-dispatch
  - DocumentLoader façade (detect_type + load)
  - Document model (hash, word_count, len)
  - Edge cases: empty files, missing keys, unsupported extensions
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from data_pipeline.document_loader import (
    Document,
    DocumentLoader,
    DirectoryLoader,
    HTMLLoader,
    JSONLoader,
    TextLoader,
)


# ── Document model ────────────────────────────────────────────────────────────
class TestDocument:
    def test_auto_id_generated(self):
        doc = Document(content="Hello world", metadata={})
        assert doc.doc_id != ""
        assert len(doc.doc_id) == 16

    def test_same_content_same_id(self):
        d1 = Document(content="Hello", metadata={"source": "a"})
        d2 = Document(content="Hello", metadata={"source": "a"})
        assert d1.doc_id == d2.doc_id

    def test_different_metadata_different_id(self):
        d1 = Document(content="Hello", metadata={"source": "a"})
        d2 = Document(content="Hello", metadata={"source": "b"})
        assert d1.doc_id != d2.doc_id

    def test_len_returns_char_count(self):
        doc = Document(content="Hello world")
        assert len(doc) == 11

    def test_word_count(self):
        doc = Document(content="The quick brown fox")
        assert doc.word_count == 4

    def test_explicit_doc_id_preserved(self):
        doc = Document(content="Test", doc_id="custom-id-123")
        assert doc.doc_id == "custom-id-123"


# ── TextLoader ────────────────────────────────────────────────────────────────
class TestTextLoader:
    def test_loads_txt_file(self, sample_text_file, sample_text):
        loader = TextLoader(sample_text_file)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].content == sample_text.strip()

    def test_metadata_contains_source(self, sample_text_file):
        loader = TextLoader(sample_text_file)
        docs = loader.load()
        assert docs[0].metadata["source"] == sample_text_file

    def test_metadata_file_type_is_txt(self, sample_text_file):
        loader = TextLoader(sample_text_file)
        docs = loader.load()
        assert docs[0].metadata["file_type"] == "txt"

    def test_metadata_has_ingested_at(self, sample_text_file):
        loader = TextLoader(sample_text_file)
        docs = loader.load()
        assert "ingested_at" in docs[0].metadata
        assert docs[0].metadata["ingested_at"] > 0

    def test_async_load(self, sample_text_file):
        import asyncio
        loader = TextLoader(sample_text_file)
        docs = asyncio.run(loader.aload())
        assert len(docs) == 1


# ── HTMLLoader ────────────────────────────────────────────────────────────────
class TestHTMLLoader:
    def test_loads_html_file(self, sample_html_file):
        loader = HTMLLoader(sample_html_file)
        docs = loader.load()
        assert len(docs) == 1
        assert len(docs[0].content) > 0

    def test_strips_navigation(self, sample_html_file):
        loader = HTMLLoader(sample_html_file)
        docs = loader.load()
        # nav content should be stripped
        assert "Navigation menu" not in docs[0].content

    def test_strips_footer(self, sample_html_file):
        loader = HTMLLoader(sample_html_file)
        docs = loader.load()
        assert "Footer content" not in docs[0].content

    def test_preserves_main_content(self, sample_html_file):
        loader = HTMLLoader(sample_html_file)
        docs = loader.load()
        assert "return a product" in docs[0].content.lower()

    def test_file_type_metadata(self, sample_html_file):
        loader = HTMLLoader(sample_html_file)
        docs = loader.load()
        assert docs[0].metadata["file_type"] == "html"

    def test_url_detection(self):
        loader = HTMLLoader("https://example.com/page")
        assert loader._is_url is True

    def test_file_path_not_url(self, sample_html_file):
        loader = HTMLLoader(sample_html_file)
        assert loader._is_url is False

    def test_url_load_mocked(self):
        """Verify URL loading uses requests and parses correctly."""
        html_content = "<html><body><p>Test content here</p></body></html>"
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.raise_for_status = MagicMock()

        with patch("data_pipeline.document_loader.requests.get",
                   return_value=mock_response) as mock_get:
            loader = HTMLLoader("https://example.com")
            docs = loader.load()

        mock_get.assert_called_once_with("https://example.com", timeout=30)
        assert len(docs) == 1
        assert "Test content here" in docs[0].content


# ── JSONLoader ────────────────────────────────────────────────────────────────
class TestJSONLoader:
    def test_loads_jsonl_file(self, sample_json_file):
        loader = JSONLoader(sample_json_file, content_key="text", metadata_keys=["id"])
        docs = loader.load()
        assert len(docs) == 3

    def test_content_key_extracted(self, sample_json_file):
        loader = JSONLoader(sample_json_file, content_key="text")
        docs = loader.load()
        assert "Laptop" in docs[0].content or "SmartWatch" in docs[0].content or "AC" in docs[0].content

    def test_metadata_keys_included(self, sample_json_file):
        loader = JSONLoader(sample_json_file, content_key="text", metadata_keys=["id"])
        docs = loader.load()
        for doc in docs:
            assert "id" in doc.metadata

    def test_missing_content_key_skips_record(self, tmp_path):
        """Records without the content key should be silently skipped."""
        data = [{"name": "no-text-field"}, {"text": "valid record"}]
        f = tmp_path / "data.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in data))
        loader = JSONLoader(str(f), content_key="text")
        docs = loader.load()
        assert len(docs) == 1

    def test_loads_json_array(self, tmp_path):
        data = [{"text": "item one"}, {"text": "item two"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        loader = JSONLoader(str(f), content_key="text")
        docs = loader.load()
        assert len(docs) == 2

    def test_loads_json_object(self, tmp_path):
        data = {"text": "single document"}
        f = tmp_path / "doc.json"
        f.write_text(json.dumps(data))
        loader = JSONLoader(str(f), content_key="text")
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].content == "single document"


# ── DirectoryLoader ───────────────────────────────────────────────────────────
class TestDirectoryLoader:
    def test_loads_all_supported_formats(self, tmp_path, sample_text):
        """DirectoryLoader should detect and load .txt, .md, .json files."""
        (tmp_path / "policy.txt").write_text(sample_text)
        (tmp_path / "faq.md").write_text("# FAQ\nSome content here.")
        products = [{"text": "Product A details"}]
        (tmp_path / "products.json").write_text(json.dumps(products))

        loader = DirectoryLoader(str(tmp_path))
        docs = loader.load()
        assert len(docs) >= 3    # at least one per file

    def test_skips_unsupported_extensions(self, tmp_path):
        (tmp_path / "data.csv").write_text("col1,col2\nval1,val2")
        (tmp_path / "readme.txt").write_text("Some text content")
        loader = DirectoryLoader(str(tmp_path))
        docs = loader.load()
        # CSV is unsupported; only txt should be loaded
        sources = [d.metadata.get("file_type") for d in docs]
        assert "txt" in sources
        assert "csv" not in sources

    def test_exclude_pattern_respected(self, tmp_path):
        (tmp_path / "include.txt").write_text("included content")
        (tmp_path / "exclude_this.txt").write_text("excluded content")
        loader = DirectoryLoader(str(tmp_path), exclude_patterns=["exclude_this"])
        docs = loader.load()
        sources = [d.metadata.get("source", "") for d in docs]
        assert not any("exclude_this" in s for s in sources)

    def test_max_files_limit(self, tmp_path):
        for i in range(5):
            (tmp_path / f"file_{i}.txt").write_text(f"Content of file {i}")
        loader = DirectoryLoader(str(tmp_path), max_files=2)
        docs = loader.load()
        assert len(docs) <= 2

    def test_empty_directory_returns_empty_list(self, tmp_path):
        loader = DirectoryLoader(str(tmp_path))
        docs = loader.load()
        assert docs == []

    def test_failed_file_does_not_crash(self, tmp_path):
        """A corrupt/unreadable file should be logged and skipped, not crash."""
        good = tmp_path / "good.txt"
        good.write_text("Good content")
        bad = tmp_path / "bad.txt"
        bad.write_text("\x00\x01\x02")   # binary garbage

        loader = DirectoryLoader(str(tmp_path))
        # Should not raise; bad file is logged and skipped
        try:
            docs = loader.load()
            assert any("Good content" in d.content for d in docs)
        except Exception:
            pytest.fail("DirectoryLoader should not raise on corrupt files")


# ── DocumentLoader façade ─────────────────────────────────────────────────────
class TestDocumentLoaderFacade:
    def test_auto_detects_txt(self, sample_text_file):
        loader = DocumentLoader()
        docs = loader.load(sample_text_file)
        assert len(docs) == 1

    def test_auto_detects_html(self, sample_html_file):
        loader = DocumentLoader()
        docs = loader.load(sample_html_file)
        assert len(docs) == 1

    def test_auto_detects_directory(self, tmp_path, sample_text):
        (tmp_path / "doc.txt").write_text(sample_text)
        loader = DocumentLoader()
        docs = loader.load(str(tmp_path))
        assert len(docs) >= 1

    def test_explicit_loader_type_overrides_detection(self, sample_text_file):
        loader = DocumentLoader()
        docs = loader.load(sample_text_file, loader_type="txt")
        assert len(docs) == 1

    def test_unsupported_type_raises(self):
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported loader type"):
            loader.load("some_file.xyz", loader_type="xyz")

    def test_url_auto_detected(self):
        loader = DocumentLoader()
        assert loader._detect_type("https://example.com") == "url"
        assert loader._detect_type("http://example.com/page") == "url"

    def test_async_load(self, sample_text_file):
        import asyncio
        loader = DocumentLoader()
        docs = asyncio.run(loader.aload(sample_text_file))
        assert len(docs) == 1
