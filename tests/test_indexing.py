"""Offline tests for the runnable document-to-index application service."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest

from enterprise_rag.cli import main, run_cli
from enterprise_rag.config import Settings
from enterprise_rag.indexing import (
    INDEX_MANIFEST_FILE_NAME,
    IndexingError,
    IndexingService,
    load_index_manifest,
    validate_index_compatibility,
)
from enterprise_rag.retrieval import Retriever
from enterprise_rag.vector_store import (
    INDEX_FILE_NAME,
    METADATA_FILE_NAME,
    FaissVectorStore,
)


class FakeEmbeddingClient:
    """Deterministic batch/query client that never performs network I/O."""

    def __init__(
        self,
        *,
        vectors: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.queries: list[str] = []

    def embed(self, *, model: str, contents: Sequence[str]):
        self.calls.append((model, tuple(contents)))
        if self.vectors is not None:
            return self.vectors
        return tuple(
            (float(index), float(len(content)))
            for index, content in enumerate(contents)
        )

    def embed_query(self, question: str):
        self.queries.append(question)
        return (0.0, 1.0)


class FailingEmbeddingClient(FakeEmbeddingClient):
    def embed(self, *, model: str, contents: Sequence[str]):
        raise RuntimeError("configured embedding provider is unavailable")


def settings(
    documents: Path,
    output: Path,
    *,
    provider: str = "ollama",
    model: str = "fake-embedding",
    chunk_size: int = 40,
    chunk_overlap: int = 5,
) -> Settings:
    return Settings(
        gemini_api_key=None,
        generation_model="unused-generation",
        embedding_model=model if provider == "gemini" else "unused-gemini",
        documents_dir=documents,
        vector_store_dir=output,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=2,
        embedding_provider=provider,
        ollama_embedding_model=model,
    )


def write_documents(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "guide.md").write_text(
        "# RAG\n\nRetrieval augments generation with selected context.",
        encoding="utf-8",
    )
    nested = directory / "advanced"
    nested.mkdir()
    (nested / "search.txt").write_text(
        "Vector search finds nearby chunks.",
        encoding="utf-8",
    )


def service(
    documents: Path,
    output: Path,
    *,
    client: FakeEmbeddingClient | None = None,
    **setting_overrides,
) -> IndexingService:
    return IndexingService(
        settings(documents, output, **setting_overrides),
        embedding_client=client or FakeEmbeddingClient(),
        clock=lambda: datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc),
        monotonic=iter((10.0, 10.25)).__next__,
    )


def test_full_indexing_workflow_persists_loadable_index(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "index"
    write_documents(documents)
    client = FakeEmbeddingClient()

    result = service(documents, output, client=client).build()
    loaded = FaissVectorStore.load(output)

    assert result.loaded_document_count == 2
    assert result.chunk_count == result.embedded_vector_count == loaded.size
    assert result.embedding_dimension == loaded.dimension == 2
    assert result.output_path == output.resolve()
    assert result.elapsed_seconds == pytest.approx(0.25)
    assert (output / INDEX_FILE_NAME).is_file()
    assert (output / METADATA_FILE_NAME).is_file()
    assert (output / INDEX_MANIFEST_FILE_NAME).is_file()
    assert client.calls[0][0] == "fake-embedding"


def test_missing_input_directory_has_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(IndexingError, match="does not exist"):
        service(missing, tmp_path / "index").build()


def test_input_must_be_a_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.md"
    file_path.write_text("content", encoding="utf-8")
    with pytest.raises(IndexingError, match="not a directory"):
        service(file_path, tmp_path / "index").build()


def test_no_supported_documents_has_clear_error(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "ignored.pdf").write_bytes(b"fake")
    with pytest.raises(IndexingError, match="Document loading failed"):
        service(documents, tmp_path / "index").build()


def test_empty_input_directory_has_clear_error(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    with pytest.raises(IndexingError, match="Document loading failed"):
        service(documents, tmp_path / "index").build()


def test_all_empty_documents_cannot_create_chunks(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "empty.md").write_text(" \n", encoding="utf-8")
    with pytest.raises(IndexingError, match="No non-empty documents"):
        service(documents, tmp_path / "index").build()


def test_loader_output_without_chunks_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = tmp_path / "documents"
    write_documents(documents)
    monkeypatch.setattr(
        "enterprise_rag.indexing.split_documents",
        lambda documents, settings: (),
    )

    with pytest.raises(IndexingError, match="produced no chunks"):
        service(documents, tmp_path / "index").build()


def test_input_and_output_cannot_overlap(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    write_documents(documents)
    with pytest.raises(IndexingError, match="must not contain"):
        service(documents, documents / "index").build()


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (((), (0.0, 1.0)), "is empty"),
        (((0.0, 1.0),), "count does not match"),
        (((0.0, 1.0), (0.0, 1.0, 2.0)), "dimensions are inconsistent"),
    ],
)
def test_invalid_embedding_batches_fail_without_publishing(
    tmp_path: Path,
    vectors,
    message: str,
) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "index"
    write_documents(documents)

    with pytest.raises(IndexingError, match=message):
        service(
            documents,
            output,
            client=FakeEmbeddingClient(vectors=vectors),
            chunk_size=1000,
        ).build()
    assert not output.exists()


def test_existing_output_requires_overwrite_and_preserves_contents(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "index"
    write_documents(documents)
    output.mkdir()
    marker = output / "old.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(IndexingError, match="--overwrite"):
        service(documents, output).build()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_overwrite_replaces_complete_directory(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "index"
    write_documents(documents)
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")

    result = service(documents, output).build(overwrite=True)

    assert result.output_path == output.resolve()
    assert not (output / "old.txt").exists()
    assert FaissVectorStore.load(output).size == result.chunk_count


def test_persistence_failure_keeps_existing_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "index"
    write_documents(documents)
    output.mkdir()
    marker = output / "old.txt"
    marker.write_text("stable", encoding="utf-8")

    def fail_save(self, directory):
        raise OSError("simulated write failure")

    monkeypatch.setattr(FaissVectorStore, "save", fail_save)
    with pytest.raises(IndexingError, match="safely save"):
        service(documents, output).build(overwrite=True)
    assert marker.read_text(encoding="utf-8") == "stable"


def test_embedding_provider_failure_is_wrapped_without_publishing(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "index"
    write_documents(documents)

    with pytest.raises(IndexingError, match="provider is unavailable"):
        service(
            documents,
            output,
            client=FailingEmbeddingClient(),
        ).build()
    assert not output.exists()


def test_manifest_records_vector_space_and_build_settings(tmp_path: Path) -> None:
    documents = tmp_path / "knowledge"
    output = tmp_path / "index"
    write_documents(documents)
    result = service(documents, output).build()

    manifest = load_index_manifest(output)

    assert manifest == {
        "schema_version": 1,
        "built_at": "2026-07-30T08:00:00Z",
        "source_identifier": "knowledge",
        "embedding_provider": "ollama",
        "embedding_model": "fake-embedding",
        "embedding_dimension": 2,
        "chunk_size": 40,
        "chunk_overlap": 5,
        "document_count": 2,
        "chunk_count": result.chunk_count,
    }
    assert str(tmp_path) not in json.dumps(manifest)


def test_manifest_compatibility_rejects_provider_or_model_mismatch(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "index"
    write_documents(documents)
    service(documents, output).build()

    validate_index_compatibility(output, settings(documents, output))
    with pytest.raises(IndexingError, match="incompatible"):
        validate_index_compatibility(
            output,
            settings(documents, output, model="different-model"),
        )


def test_ask_checks_new_manifest_before_querying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "index"
    write_documents(documents)
    service(documents, output).build()
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "different-model")
    query_client = FakeEmbeddingClient()

    with pytest.raises(IndexingError, match="incompatible"):
        main(
            ["ask", "Question", "--index-path", str(output)],
            query_embedding_client=query_client,
            generation_client=object(),
        )
    assert query_client.queries == []


def test_cli_success_and_verbose_output_are_offline(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "index"
    write_documents(documents)
    stdout = StringIO()
    stderr = StringIO()

    assert run_cli(
        [
            "build-index",
            "--input",
            str(documents),
            "--output",
            str(output),
            "--verbose",
        ],
        indexing_service=service(documents, output),
        output=stdout,
        error_output=stderr,
    ) == 0
    assert stderr.getvalue() == ""
    text = stdout.getvalue()
    assert "Loading documents from" in text
    assert "Documents loaded: 2" in text
    assert "Chunks created:" in text
    assert "Vectors embedded:" in text
    assert "Embedding dimension: 2" in text
    assert f"Index saved to: {output.resolve()}" in text
    assert FaissVectorStore.load(output).size > 0


def test_cli_failure_returns_nonzero_without_traceback(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    stderr = StringIO()

    assert run_cli(
        ["build-index", "--input", str(missing), "--output", str(tmp_path / "index")],
        indexing_service=service(missing, tmp_path / "index"),
        error_output=stderr,
    ) == 2
    assert "does not exist" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def test_cli_missing_gemini_key_fails_before_network(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "index"
    write_documents(documents)
    gemini_settings = settings(
        documents,
        output,
        provider="gemini",
        model="fake-gemini-model",
    )
    stderr = StringIO()

    assert run_cli(
        ["build-index"],
        indexing_service=IndexingService(gemini_settings),
        error_output=stderr,
    ) == 2
    assert "GEMINI_API_KEY" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()
    assert not output.exists()


def test_offline_e2e_load_chunk_embed_save_load_retrieve(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    output = tmp_path / "index"
    write_documents(documents)
    client = FakeEmbeddingClient()
    service(documents, output, client=client, chunk_size=1000).build()

    loaded = FaissVectorStore.load(output)
    results = Retriever(client, loaded).retrieve("What is RAG?", top_k=1)

    assert len(results) == 1
    assert results[0].embedded_chunk.chunk.source in {
        "guide.md",
        "advanced/search.txt",
    }
    assert client.queries == ["What is RAG?"]
