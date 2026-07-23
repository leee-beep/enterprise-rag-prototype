"""Offline tests for the minimal in-memory FAISS vector store."""

import faiss
import pytest

from enterprise_rag.models import DocumentChunk, EmbeddedChunk
from enterprise_rag.vector_store import (
    FaissVectorStore,
    VectorStoreValidationError,
    build_vector_store,
)


def make_embedded_chunk(
    index: int,
    vector: tuple[float, ...],
    *,
    source: str = "guide.md",
) -> EmbeddedChunk:
    chunk = DocumentChunk(
        content=f"chunk content {index}",
        source=source,
        file_name=source.split("/")[-1],
        file_type=".md",
        document_id=f"doc-{source}",
        chunk_index=index,
        chunk_id=f"doc-{source}:chunk-{index:06d}",
    )
    return EmbeddedChunk(
        chunk=chunk,
        vector=vector,
        embedding_model="fake-embedding-model",
    )


def test_empty_batch_cannot_build_index() -> None:
    with pytest.raises(VectorStoreValidationError, match="empty"):
        build_vector_store([])


def test_stored_vector_dimensions_must_match() -> None:
    items = [
        make_embedded_chunk(0, (0.0, 1.0)),
        make_embedded_chunk(1, (0.0, 1.0, 2.0)),
    ]

    with pytest.raises(VectorStoreValidationError, match="dimensions are inconsistent"):
        build_vector_store(items)


def test_builds_index_flat_l2_with_all_vectors() -> None:
    items = [
        make_embedded_chunk(0, (0.0, 0.0)),
        make_embedded_chunk(1, (1.0, 1.0)),
        make_embedded_chunk(2, (5.0, 5.0)),
    ]

    store = build_vector_store(items)

    assert isinstance(store, FaissVectorStore)
    assert isinstance(store.index, faiss.IndexFlatL2)
    assert store.index.ntotal == 3
    assert store.size == 3
    assert store.dimension == 2


def test_top_k_must_be_at_least_one() -> None:
    store = build_vector_store([make_embedded_chunk(0, (0.0, 0.0))])

    with pytest.raises(VectorStoreValidationError, match="top_k"):
        store.search((0.0, 0.0), top_k=0)


def test_search_returns_requested_number_in_distance_order() -> None:
    first = make_embedded_chunk(0, (0.0, 0.0))
    second = make_embedded_chunk(1, (1.0, 1.0))
    third = make_embedded_chunk(2, (5.0, 5.0))
    store = build_vector_store([first, second, third])

    results = store.search((0.1, 0.1), top_k=2)

    assert results == (first, second)


def test_search_caps_result_count_at_store_size() -> None:
    items = [
        make_embedded_chunk(0, (0.0, 0.0)),
        make_embedded_chunk(1, (1.0, 1.0)),
    ]
    store = build_vector_store(items)

    results = store.search((0.0, 0.0), top_k=10)

    assert len(results) == 2


def test_search_result_preserves_original_metadata() -> None:
    expected = make_embedded_chunk(
        7,
        (0.0, 0.0),
        source="advanced/hybrid_search.md",
    )
    store = build_vector_store(
        [expected, make_embedded_chunk(8, (10.0, 10.0))]
    )

    result = store.search((0.0, 0.0), top_k=1)[0]

    assert result is expected
    assert result.chunk.source == "advanced/hybrid_search.md"
    assert result.chunk.chunk_index == 7
    assert result.chunk.chunk_id.endswith("chunk-000007")
    assert result.embedding_model == "fake-embedding-model"


def test_query_vector_dimension_must_match_index() -> None:
    store = build_vector_store([make_embedded_chunk(0, (0.0, 0.0))])

    with pytest.raises(VectorStoreValidationError, match="dimension does not match"):
        store.search((0.0, 0.0, 0.0), top_k=1)