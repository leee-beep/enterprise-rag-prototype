"""Offline tests for FAISS persistence and persistence-aware CLI commands."""
from __future__ import annotations

import json
from io import StringIO

import pytest

from enterprise_rag.cli import (
    CLIConfigurationError,
    main,
    run_build_index_command,
)
from enterprise_rag.models import DocumentChunk, EmbeddedChunk
from enterprise_rag.vector_store import (
    INDEX_FILE_NAME,
    METADATA_FILE_NAME,
    FaissVectorStore,
    VectorStoreValidationError,
    build_vector_store,
)


def embedded(index: int, vector, *, source="guide.md"):
    chunk = DocumentChunk(
        content=f"chunk {index}",
        source=source,
        file_name=source.split("/")[-1],
        file_type=".md",
        document_id="doc-1",
        chunk_index=index,
        chunk_id=f"doc-1:chunk-{index:06d}",
        metadata={"category": "offline", "order": index},
    )
    return EmbeddedChunk(
        chunk=chunk,
        vector=tuple(vector),
        embedding_model="fake-embedding",
    )


class FakeQueryEmbeddingClient:
    def __init__(self, vector):
        self.vector = vector
        self.questions = []

    def embed_query(self, question):
        self.questions.append(question)
        return self.vector


class FakeGenerationClient:
    provider = "fake"
    model = "fake-generation"

    def __init__(self, answer="persisted answer"):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.answer


def saved_store(tmp_path):
    store = build_vector_store(
        [embedded(0, (0.0, 0.0)), embedded(1, (2.0, 2.0))]
    )
    store.save(tmp_path)
    return store


def test_save_load_round_trip_preserves_index_chunks_and_metadata(tmp_path):
    original = saved_store(tmp_path)
    loaded = FaissVectorStore.load(tmp_path)

    assert loaded.size == original.size == 2
    assert loaded.dimension == original.dimension == 2
    hits = loaded.search((0.0, 0.0), 2)
    assert [item.chunk.content for item in hits] == ["chunk 0", "chunk 1"]
    assert hits[0].chunk.metadata == {"category": "offline", "order": 0}
    assert hits[0].embedding_model == "fake-embedding"
    assert hits[0].vector == pytest.approx((0.0, 0.0))


@pytest.mark.parametrize("missing_name", [INDEX_FILE_NAME, METADATA_FILE_NAME])
def test_load_rejects_missing_files(tmp_path, missing_name):
    saved_store(tmp_path)
    (tmp_path / missing_name).unlink()
    with pytest.raises(VectorStoreValidationError, match="missing"):
        FaissVectorStore.load(tmp_path)


def test_load_rejects_corrupted_metadata_json(tmp_path):
    saved_store(tmp_path)
    (tmp_path / METADATA_FILE_NAME).write_text("{broken", encoding="utf-8")
    with pytest.raises(VectorStoreValidationError, match="invalid JSON"):
        FaissVectorStore.load(tmp_path)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda data: data.update(version=999), "version is incompatible"),
        (lambda data: data.update(dimension=3), "dimension does not match"),
        (lambda data: data.update(item_count=3), "matching item_count"),
        (lambda data: data["items"][0].pop("embedding_model"), "embedding_model"),
        (lambda data: data["items"][0]["chunk"].update(metadata=[]), "metadata"),
    ],
)
def test_load_validates_metadata_schema_and_index_consistency(
    tmp_path, mutate, match
):
    saved_store(tmp_path)
    metadata_path = tmp_path / METADATA_FILE_NAME
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(data)
    metadata_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(VectorStoreValidationError, match=match):
        FaissVectorStore.load(tmp_path)


def test_build_index_cli_saves_injected_fake_data(tmp_path):
    store = build_vector_store([embedded(0, (0.0, 0.0))])
    calls = []
    output = StringIO()
    assert run_build_index_command(
        lambda: calls.append("build") or store,
        index_path=tmp_path,
        output=output,
    ) == 0
    assert calls == ["build"]
    assert (tmp_path / INDEX_FILE_NAME).is_file()
    assert (tmp_path / METADATA_FILE_NAME).is_file()
    assert "Saved 1 chunks" in output.getvalue()


def test_build_index_cli_requires_injected_builder(tmp_path):
    with pytest.raises(CLIConfigurationError, match="index builder"):
        main(["build-index", "--index-path", str(tmp_path)])


def test_ask_cli_loads_existing_index_without_rebuilding(tmp_path, capfd):
    saved_store(tmp_path)
    embedding = FakeQueryEmbeddingClient((0.0, 0.0))
    generation = FakeGenerationClient()

    assert main(
        ["ask", "What is stored?", "--index-path", str(tmp_path), "--top-k", "1"],
        query_embedding_client=embedding,
        generation_client=generation,
    ) == 0

    assert capfd.readouterr().out == "persisted answer\n"
    assert embedding.questions == ["What is stored?"]
    assert "chunk 0" in generation.prompts[0]


def test_ask_without_pipeline_or_index_has_clear_error():
    with pytest.raises(CLIConfigurationError, match="does not automatically rebuild"):
        main(["ask", "Question"])
