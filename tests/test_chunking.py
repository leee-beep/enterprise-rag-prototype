"""Tests for deterministic character-based document splitting."""

from pathlib import Path

import pytest

from enterprise_rag.chunking import split_documents
from enterprise_rag.config import Settings, load_settings
from enterprise_rag.documents import load_documents
from enterprise_rag.models import LoadedDocument

FIXTURES = Path(__file__).parent / "fixtures" / "documents"


def make_settings(*, chunk_size: int, chunk_overlap: int) -> Settings:
    return Settings(
        gemini_api_key=None,
        generation_model="unused-generation-model",
        embedding_model="unused-embedding-model",
        documents_dir=FIXTURES,
        vector_store_dir=Path("unused-vector-store"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=4,
    )


def make_document(content: str) -> LoadedDocument:
    return LoadedDocument(
        content=content,
        source="sample.txt",
        file_name="sample.txt",
        file_type=".txt",
        document_id="doc-stable-test-id",
    )


def test_long_documents_are_split_with_size_limit() -> None:
    documents = load_documents(FIXTURES).documents
    settings = make_settings(chunk_size=80, chunk_overlap=10)

    chunks = split_documents(documents, settings)

    assert len(chunks) > len(documents)
    assert all(0 < len(chunk.content) <= 80 for chunk in chunks)


def test_chunk_metadata_preserves_document_metadata() -> None:
    document = make_document("A" * 25)
    chunks = split_documents([document], make_settings(chunk_size=10, chunk_overlap=2))

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    for chunk in chunks:
        assert chunk.source == document.source
        assert chunk.file_name == document.file_name
        assert chunk.file_type == document.file_type
        assert chunk.document_id == document.document_id
        assert chunk.chunk_id == (
            f"{document.document_id}:chunk-{chunk.chunk_index:06d}"
        )


def test_chunk_ids_are_stable() -> None:
    document = make_document("0123456789" * 4)
    settings = make_settings(chunk_size=12, chunk_overlap=3)

    first = split_documents([document], settings)
    second = split_documents([document], settings)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.content for chunk in first] == [chunk.content for chunk in second]


def test_chunk_overlap_is_exact() -> None:
    document = make_document("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    chunks = split_documents([document], make_settings(chunk_size=10, chunk_overlap=3))

    assert [chunk.content for chunk in chunks] == [
        "ABCDEFGHIJ",
        "HIJKLMNOPQ",
        "OPQRSTUVWX",
        "VWXYZ",
    ]
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.content[-3:] == current.content[:3]


def test_prefers_paragraph_boundary_when_available() -> None:
    document = make_document(
        "First paragraph has text.\n\nSecond paragraph has more text for testing."
    )

    chunks = split_documents([document], make_settings(chunk_size=35, chunk_overlap=5))

    assert chunks[0].content.endswith("\n\n")
    assert len(chunks[0].content) <= 35


def test_high_overlap_still_makes_progress() -> None:
    document = make_document("line one\nline two\nline three\nline four")

    chunks = split_documents([document], make_settings(chunk_size=12, chunk_overlap=10))

    assert chunks
    assert len(chunks) < len(document.content)
    assert all(len(chunk.content) <= 12 for chunk in chunks)


def test_splitter_does_not_require_gemini_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("DOCUMENTS_DIR", str(FIXTURES))
    monkeypatch.setenv("CHUNK_SIZE", "90")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")
    settings = load_settings(load_env_file=False)
    documents = load_documents(settings.documents_dir).documents

    chunks = split_documents(documents, settings)

    assert settings.gemini_api_key is None
    assert chunks