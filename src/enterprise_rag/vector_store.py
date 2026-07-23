"""Minimal in-memory FAISS vector store for embedded chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import faiss
import numpy as np

from enterprise_rag.config import Settings
from enterprise_rag.models import EmbeddedChunk


class VectorStoreError(RuntimeError):
    """Base error for vector-store operations."""


class VectorStoreValidationError(VectorStoreError):
    """Raised when stored or query vectors are invalid."""


class FaissVectorStore:
    """An IndexFlatL2 plus an ordered mapping back to EmbeddedChunk objects."""

    def __init__(self, embedded_chunks: Sequence[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            raise VectorStoreValidationError(
                "Cannot build a vector store from an empty EmbeddedChunk batch."
            )

        dimensions = {len(item.vector) for item in embedded_chunks}
        if 0 in dimensions:
            raise VectorStoreValidationError("Stored embedding vectors cannot be empty.")
        if len(dimensions) != 1:
            raise VectorStoreValidationError(
                "Stored embedding vector dimensions are inconsistent: "
                f"received {sorted(dimensions)}."
            )

        self._items = tuple(embedded_chunks)
        self._dimension = dimensions.pop()
        self._index = faiss.IndexFlatL2(self._dimension)
        matrix = np.asarray(
            [item.vector for item in self._items],
            dtype=np.float32,
        )
        self._index.add(matrix)

    @property
    def index(self) -> faiss.IndexFlatL2:
        """Expose the underlying index for inspection, not metadata lookup."""
        return self._index

    @property
    def size(self) -> int:
        """Return the number of indexed embedded chunks."""
        return len(self._items)

    @property
    def dimension(self) -> int:
        """Return the common embedding-vector dimension."""
        return self._dimension

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> tuple[EmbeddedChunk, ...]:
        """Return the nearest embedded chunks ordered by squared L2 distance."""
        if top_k < 1:
            raise VectorStoreValidationError("top_k must be greater than or equal to 1.")
        if len(query_vector) == 0:
            raise VectorStoreValidationError("Query vector cannot be empty.")
        if len(query_vector) != self._dimension:
            raise VectorStoreValidationError(
                "Query vector dimension does not match the index: "
                f"expected {self._dimension}, received {len(query_vector)}."
            )

        query = np.asarray([query_vector], dtype=np.float32)
        result_count = min(top_k, self.size)
        _, indices = self._index.search(query, result_count)
        return tuple(self._items[index] for index in indices[0] if index >= 0)


def build_vector_store(
    embedded_chunks: Sequence[EmbeddedChunk],
) -> FaissVectorStore:
    """Build the minimal in-memory FAISS store."""
    return FaissVectorStore(embedded_chunks)


def save_vector_store(vector_store: Any, settings: Settings) -> None:
    """Persist a vector store under the configured local data directory."""
    # TODO: Save the FAISS index and EmbeddedChunk mapping locally.
    raise NotImplementedError("Vector-store persistence is not implemented yet.")


def load_vector_store(embeddings: Any, settings: Settings) -> Any:
    """Load a previously generated local vector store."""
    # TODO: Load and validate a trusted local FAISS index and metadata mapping.
    raise NotImplementedError("Vector-store loading is not implemented yet.")