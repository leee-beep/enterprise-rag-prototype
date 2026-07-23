"""Offline tests for provider-neutral retrieval and CLI presentation."""
from __future__ import annotations

from io import StringIO
from collections.abc import Sequence
import pytest

from enterprise_rag.cli import CLIConfigurationError, main, run_cli, run_retrieve_command
from enterprise_rag.models import DocumentChunk, EmbeddedChunk
from enterprise_rag.retrieval import RetrievalError, Retriever
from enterprise_rag.vector_store import build_vector_store

class FakeQueryEmbeddingClient:
    def __init__(self, vector: Sequence[float]) -> None:
        self.vector = vector
        self.queries: list[str] = []
    def embed_query(self, text: str) -> Sequence[float]:
        self.queries.append(text)
        return self.vector
    def embed(self, *, model: str, contents: Sequence[str]) -> Sequence[Sequence[float]]:
        raise AssertionError("Retriever must use embed_query, not document embedding.")

class FakeVectorStore:
    def __init__(self, hits=(), *, size: int | None = None) -> None:
        self.hits = tuple(hits)
        self._size = len(self.hits) if size is None else size
        self.calls: list[tuple[tuple[float, ...], int]] = []
    @property
    def size(self) -> int:
        return self._size
    def search_with_scores(self, query_vector, top_k):
        self.calls.append((tuple(query_vector), top_k))
        return self.hits[:top_k]

def embedded(index: int, vector=(0.0, 0.0), *, source="guide.md", metadata=None):
    chunk = DocumentChunk(
        content=f"chunk {index}", source=source, file_name=source.split("/")[-1],
        file_type=".md", document_id="doc-1", chunk_index=index,
        chunk_id=f"doc-1:chunk-{index:06d}", metadata=metadata or {},
    )
    return EmbeddedChunk(chunk=chunk, vector=tuple(vector), embedding_model="fake-model")

def test_empty_index_returns_no_results_without_embedding():
    client = FakeQueryEmbeddingClient((1.0, 2.0))
    store = FakeVectorStore(size=0)
    assert Retriever(client, store).retrieve("What is RAG?", 4) == ()
    assert client.queries == [] and store.calls == []

def test_top_k_is_forwarded_to_vector_store():
    items = [(embedded(0), 0.0), (embedded(1), 1.0), (embedded(2), 2.0)]
    client = FakeQueryEmbeddingClient((0.1, 0.2))
    store = FakeVectorStore(items)
    results = Retriever(client, store).retrieve(" question ", 2)
    assert len(results) == 2
    assert client.queries == ["question"]
    assert store.calls == [((0.1, 0.2), 2)]

def test_result_preserves_chunk_and_normalizes_metadata():
    item = embedded(7, source="advanced/search.md", metadata={"category":"retrieval"})
    result = Retriever(FakeQueryEmbeddingClient((0, 0)), FakeVectorStore([(item, 0.0)])).retrieve("q", 1)[0]
    assert result.embedded_chunk is item
    assert result.metadata["category"] == "retrieval"
    assert result.metadata["source"] == "advanced/search.md"
    assert result.metadata["chunk_id"] == item.chunk.chunk_id
    assert result.metadata["embedding_model"] == "fake-model"

def test_score_is_normalized_from_squared_l2_distance():
    item = embedded(0)
    result = Retriever(FakeQueryEmbeddingClient((0, 0)), FakeVectorStore([(item, 3.0)])).retrieve("q", 1)[0]
    assert result.score == pytest.approx(0.25)

def test_results_are_sorted_by_relevance_score():
    farther, nearer = embedded(0), embedded(1)
    store = FakeVectorStore([(farther, 4.0), (nearer, 0.0)])
    results = Retriever(FakeQueryEmbeddingClient((0, 0)), store).retrieve("q", 2)
    assert [result.embedded_chunk for result in results] == [nearer, farther]
    assert results[0].score > results[1].score

def test_provider_and_vector_store_are_injected_mocks():
    client = FakeQueryEmbeddingClient((9.0, 8.0))
    store = FakeVectorStore([(embedded(0), 1.0)])
    Retriever(client, store).retrieve("offline", 1)
    assert client.queries == ["offline"]
    assert store.calls == [((9.0, 8.0), 1)]

def test_real_faiss_store_returns_scored_hits_in_order():
    first = embedded(0, (0.0, 0.0)); second = embedded(1, (2.0, 2.0))
    store = build_vector_store([second, first])
    results = Retriever(FakeQueryEmbeddingClient((0.0, 0.0)), store).retrieve("q", 2)
    assert [result.embedded_chunk for result in results] == [first, second]
    assert results[0].score == pytest.approx(1.0)

def test_invalid_question_and_top_k_are_rejected():
    retriever = Retriever(FakeQueryEmbeddingClient((0, 0)), FakeVectorStore([(embedded(0), 0)]))
    with pytest.raises(RetrievalError, match="question"): retriever.retrieve("  ", 1)
    with pytest.raises(RetrievalError, match="top_k"): retriever.retrieve("q", 0)
    with pytest.raises(RetrievalError, match="positive integer"): retriever.retrieve("q", 1.5)
    with pytest.raises(RetrievalError, match="positive integer"): retriever.retrieve("q", True)

def test_cli_retrieve_prints_top_k_score_source_and_chunk():
    retriever = Retriever(FakeQueryEmbeddingClient((0, 0)), FakeVectorStore([(embedded(0), 0.0)]))
    output = StringIO()
    assert run_retrieve_command(retriever, question="What is RAG?", top_k=1, output=output) == 0
    rendered = output.getvalue()
    assert "Top 1" in rendered and "Score:" in rendered
    assert "Source: guide.md" in rendered and "Chunk: chunk 0" in rendered

def test_cli_prompts_for_question_and_requires_in_memory_retriever():
    retriever = Retriever(FakeQueryEmbeddingClient((0, 0)), FakeVectorStore(size=0))
    output = StringIO(); prompts=[]
    run_retrieve_command(retriever, question=None, top_k=2, input_fn=lambda p: prompts.append(p) or "q", output=output)
    assert prompts == ["Question: "] and output.getvalue() == "No retrieval results.\n"
    with pytest.raises(CLIConfigurationError, match="in-memory Retriever"):
        main(["retrieve", "q"])
def test_retrieval_result_metadata_is_copied_canonical_and_read_only():
    original = {"category":"retrieval", "source":"stale.md"}
    item = embedded(1, source="canonical.md", metadata=original)
    result = Retriever(FakeQueryEmbeddingClient((0, 0)), FakeVectorStore([(item, 0.0)])).retrieve("q", 1)[0]
    assert original == {"category":"retrieval", "source":"stale.md"}
    assert result.metadata["source"] == "canonical.md"
    with pytest.raises(TypeError):
        result.metadata["source"] = "mutated.md"

def test_retriever_caps_misbehaving_store_results_at_top_k():
    class OverReturningStore(FakeVectorStore):
        def search_with_scores(self, query_vector, top_k):
            return self.hits
    hits = [(embedded(0),0.0),(embedded(1),1.0),(embedded(2),2.0)]
    results = Retriever(FakeQueryEmbeddingClient((0,0)), OverReturningStore(hits)).retrieve("q", 2)
    assert len(results) == 2

def test_cli_boundary_reports_configuration_error_without_traceback():
    error_output = StringIO()
    exit_code = run_cli(["retrieve", "q"], error_output=error_output)
    assert exit_code == 2
    assert error_output.getvalue().startswith("Error: ")
    assert "Traceback" not in error_output.getvalue()