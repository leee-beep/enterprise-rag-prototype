"""Tests for the minimal batch embedding module."""

from collections.abc import Sequence
from pathlib import Path

import pytest

import enterprise_rag.embeddings as embedding_module
from enterprise_rag.config import ConfigurationError, Settings
from enterprise_rag.embeddings import (
    EmbeddingValidationError,
    create_embedding_client,
    embed_chunks,
)
from enterprise_rag.models import DocumentChunk


class FakeEmbeddingClient:
    """In-memory client that records calls and never accesses the network."""

    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def embed(
        self,
        *,
        model: str,
        contents: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        self.calls.append((model, tuple(contents)))
        return self.vectors


def make_settings(*, api_key: str | None = "test-only-fake-key") -> Settings:
    return Settings(
        gemini_api_key=api_key,
        generation_model="unused-generation-model",
        embedding_model="test-embedding-model",
        documents_dir=Path("unused-documents"),
        vector_store_dir=Path("unused-vectors"),
        chunk_size=100,
        chunk_overlap=10,
        top_k=4,
    )


def make_chunk(index: int, content: str) -> DocumentChunk:
    return DocumentChunk(
        content=content,
        source="guide.md",
        file_name="guide.md",
        file_type=".md",
        document_id="doc-stable",
        chunk_index=index,
        chunk_id=f"doc-stable:chunk-{index:06d}",
    )


def test_missing_api_key_fails_only_when_embedding_runs() -> None:
    settings = make_settings(api_key=None)
    client = FakeEmbeddingClient([[0.1, 0.2]])

    assert embed_chunks([], settings, client=client) == ()
    assert client.calls == []

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        embed_chunks([make_chunk(0, "content")], settings, client=client)

    assert client.calls == []


def test_real_client_factory_validates_key_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = False

    def fail_if_constructed(*, api_key: str) -> None:
        nonlocal constructed
        constructed = True

    monkeypatch.setattr(
        embedding_module,
        "GoogleGenAIEmbeddingClient",
        fail_if_constructed,
    )

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        create_embedding_client(make_settings(api_key=None))

    assert constructed is False


def test_multiple_chunks_are_embedded_in_one_batch() -> None:
    chunks = [make_chunk(0, "first"), make_chunk(1, "second")]
    client = FakeEmbeddingClient([[0.1, 0.2], [0.3, 0.4]])

    results = embed_chunks(chunks, make_settings(), client=client)

    assert client.calls == [("test-embedding-model", ("first", "second"))]
    assert [result.vector for result in results] == [(0.1, 0.2), (0.3, 0.4)]


def test_embedding_result_preserves_original_chunk_and_metadata() -> None:
    chunk = make_chunk(3, "metadata test")
    client = FakeEmbeddingClient([[1, 2, 3]])

    result = embed_chunks([chunk], make_settings(), client=client)[0]

    assert result.chunk is chunk
    assert result.chunk.source == "guide.md"
    assert result.chunk.document_id == "doc-stable"
    assert result.chunk.chunk_index == 3
    assert result.chunk.chunk_id == "doc-stable:chunk-000003"
    assert result.embedding_model == "test-embedding-model"
    assert result.vector == (1.0, 2.0, 3.0)


def test_empty_vector_is_rejected() -> None:
    client = FakeEmbeddingClient([[]])

    with pytest.raises(EmbeddingValidationError, match="is empty"):
        embed_chunks([make_chunk(0, "content")], make_settings(), client=client)


def test_inconsistent_vector_dimensions_are_rejected() -> None:
    client = FakeEmbeddingClient([[0.1, 0.2], [0.3, 0.4, 0.5]])
    chunks = [make_chunk(0, "first"), make_chunk(1, "second")]

    with pytest.raises(EmbeddingValidationError, match="dimensions are inconsistent"):
        embed_chunks(chunks, make_settings(), client=client)


def test_vector_count_must_match_chunk_count() -> None:
    client = FakeEmbeddingClient([[0.1, 0.2]])
    chunks = [make_chunk(0, "first"), make_chunk(1, "second")]

    with pytest.raises(EmbeddingValidationError, match="count does not match"):
        embed_chunks(chunks, make_settings(), client=client)


def test_fake_client_path_does_not_construct_real_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_real_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("Real Gemini client must not be constructed in tests")

    monkeypatch.setattr(
        embedding_module,
        "GoogleGenAIEmbeddingClient",
        forbidden_real_client,
    )
    client = FakeEmbeddingClient([[0.5, 0.6]])

    results = embed_chunks(
        [make_chunk(0, "offline test")],
        make_settings(),
        client=client,
    )

    assert results[0].vector == (0.5, 0.6)