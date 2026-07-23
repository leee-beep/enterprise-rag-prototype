"""Provider-independent semantic retrieval over a scored vector store."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from enterprise_rag.embeddings import validate_embedding_vectors
from enterprise_rag.models import EmbeddedChunk, RetrievalResult

class RetrievalError(RuntimeError):
    """Raised when a retrieval request or vector-store response is invalid."""

class QueryEmbeddingClient(Protocol):
    """Minimal embedding contract needed by Retriever."""
    def embed_query(self, text: str) -> Sequence[float]: ...

class ScoredVectorStore(Protocol):
    """Minimal scored-search contract needed by Retriever."""
    @property
    def size(self) -> int: ...

    def search_with_scores(
        self, query_vector: Sequence[float], top_k: int
    ) -> Sequence[tuple[EmbeddedChunk, float]]: ...

class Retriever:
    """Embed one question and return provider-independent ranked chunks.

    Relevance is ``1 / (1 + squared_l2_distance)``. It is a monotonic display
    score: higher means closer within the same embedding model and FAISS index.
    It is not cosine similarity, a probability, or comparable across models or
    indexes.
    """
    def __init__(
        self, embedding_client: QueryEmbeddingClient, vector_store: ScoredVectorStore
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    def retrieve(self, question: str, top_k: int) -> tuple[RetrievalResult, ...]:
        if not isinstance(question, str) or not question.strip():
            raise RetrievalError("question must be a non-empty string.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise RetrievalError("top_k must be a positive integer.")
        if self._vector_store.size == 0:
            return ()

        query_vector = validate_embedding_vectors(
            [self._embedding_client.embed_query(question.strip())], expected_count=1
        )[0]
        hits = tuple(self._vector_store.search_with_scores(query_vector, top_k))[:top_k]
        results: list[RetrievalResult] = []
        for embedded_chunk, distance in hits:
            distance = float(distance)
            if not math.isfinite(distance) or distance < 0:
                raise RetrievalError("Vector store returned an invalid squared L2 distance.")
            results.append(
                RetrievalResult(
                    score=1.0 / (1.0 + distance),
                    embedded_chunk=embedded_chunk,
                    metadata=embedded_chunk.chunk.metadata,
                )
            )
        return tuple(sorted(results, key=lambda result: result.score, reverse=True))

def retrieve_context(*args: object, **kwargs: object) -> list[object]:
    """Backward-compatible placeholder retained until RAG orchestration is designed."""
    raise NotImplementedError("Use Retriever.retrieve() for retrieval results.")