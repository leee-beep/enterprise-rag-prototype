"""Tests for recursive local document loading."""

import re
from pathlib import Path

import pytest

from enterprise_rag.config import load_settings
from enterprise_rag.documents import (
    DocumentDecodeError,
    DocumentsDirectoryNotFoundError,
    NoSupportedDocumentsError,
    load_documents,
)

FIXTURES = Path(__file__).parent / "fixtures" / "documents"


def test_loads_markdown_txt_and_nested_files() -> None:
    result = load_documents(FIXTURES)
    by_source = {document.source: document for document in result.documents}

    assert "rag_basics.md" in by_source
    assert "retrieval_notes.txt" in by_source
    assert "advanced/hybrid_search.md" in by_source
    assert by_source["rag_basics.md"].content.startswith("# RAG Basics")
    assert by_source["retrieval_notes.txt"].content.startswith("Retrieval notes")


def test_ignores_unsupported_extensions_and_reports_blank_files() -> None:
    result = load_documents(FIXTURES)
    sources = {document.source for document in result.documents}

    assert "ignored.pdf" not in sources
    assert result.skipped_empty == ("empty.txt",)


def test_document_metadata_is_relative_and_complete() -> None:
    result = load_documents(FIXTURES)
    document = next(doc for doc in result.documents if doc.source == "rag_basics.md")

    assert document.file_name == "rag_basics.md"
    assert document.file_type == ".md"
    assert document.source == "rag_basics.md"
    assert document.document_id.startswith("doc-")
    assert len(document.document_id) == 68


def test_document_id_is_stable_across_loads() -> None:
    first = {doc.source: doc.document_id for doc in load_documents(FIXTURES).documents}
    second = {doc.source: doc.document_id for doc in load_documents(FIXTURES).documents}

    assert first == second


def test_empty_supported_file_is_visible_in_result(tmp_path: Path) -> None:
    (tmp_path / "blank.md").write_text("\n  \t", encoding="utf-8")

    result = load_documents(tmp_path)

    assert result.documents == ()
    assert result.skipped_empty == ("blank.md",)


def test_empty_directory_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(NoSupportedDocumentsError, match="找不到支援的檔案"):
        load_documents(tmp_path)


def test_missing_directory_has_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(DocumentsDirectoryNotFoundError, match=re.escape(str(missing))):
        load_documents(missing)


def test_invalid_utf8_error_contains_file_path(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(DocumentDecodeError, match="invalid.txt"):
        load_documents(tmp_path)


def test_loader_does_not_require_gemini_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = load_settings(load_env_file=False)

    result = load_documents(FIXTURES)

    assert settings.gemini_api_key is None
    assert len(result.documents) == 3