"""Minimal in-memory FAISS vector store for embedded chunks."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from enterprise_rag.config import Settings
from enterprise_rag.models import DocumentChunk, EmbeddedChunk

INDEX_FILE_NAME = "index.faiss"
METADATA_FILE_NAME = "metadata.json"
PERSISTENCE_VERSION = 1


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

    def _validate_query(self, query_vector: Sequence[float], top_k: int) -> None:
        if top_k < 1:
            raise VectorStoreValidationError("top_k must be greater than or equal to 1.")
        if len(query_vector) == 0:
            raise VectorStoreValidationError("Query vector cannot be empty.")
        if len(query_vector) != self._dimension:
            raise VectorStoreValidationError(
                "Query vector dimension does not match the index: "
                f"expected {self._dimension}, received {len(query_vector)}."
            )

    def search_with_scores(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> tuple[tuple[EmbeddedChunk, float], ...]:
        """Return chunks paired with squared L2 distances, nearest first."""
        self._validate_query(query_vector, top_k)
        query = np.asarray([query_vector], dtype=np.float32)
        result_count = min(top_k, self.size)
        distances, indices = self._index.search(query, result_count)
        return tuple(
            (self._items[index], float(distance))
            for index, distance in zip(indices[0], distances[0])
            if index >= 0
        )

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> tuple[EmbeddedChunk, ...]:
        """Return nearest chunks while preserving the original public API."""
        return tuple(
            item for item, _distance in self.search_with_scores(query_vector, top_k)
        )

    def save(self, directory: str | Path) -> None:
        """Save the FAISS index and versioned chunk metadata."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": PERSISTENCE_VERSION,
            "index_type": "IndexFlatL2",
            "dimension": self.dimension,
            "item_count": self.size,
            "items": [
                {
                    "embedding_model": item.embedding_model,
                    "chunk": {
                        "content": item.chunk.content,
                        "source": item.chunk.source,
                        "file_name": item.chunk.file_name,
                        "file_type": item.chunk.file_type,
                        "document_id": item.chunk.document_id,
                        "chunk_index": item.chunk.chunk_index,
                        "chunk_id": item.chunk.chunk_id,
                        "metadata": item.chunk.metadata,
                    },
                }
                for item in self._items
            ],
        }
        try:
            serialized = json.dumps(
                metadata, ensure_ascii=False, indent=2, sort_keys=True
            )
        except (TypeError, ValueError) as exc:
            raise VectorStoreValidationError(
                "Vector-store metadata must contain only JSON-serializable values."
            ) from exc
        try:
            faiss.write_index(self._index, str(target / INDEX_FILE_NAME))
            (target / METADATA_FILE_NAME).write_text(
                serialized + "\n", encoding="utf-8"
            )
        except OSError as exc:
            raise VectorStoreError(
                f"Could not save vector store to '{target}'."
            ) from exc

    @classmethod
    def load(cls, directory: str | Path) -> FaissVectorStore:
        """Load and validate a trusted local FAISS index and metadata mapping."""
        target = Path(directory)
        index_path = target / INDEX_FILE_NAME
        metadata_path = target / METADATA_FILE_NAME
        if not index_path.is_file():
            raise VectorStoreValidationError(
                f"FAISS index file is missing: '{index_path}'."
            )
        if not metadata_path.is_file():
            raise VectorStoreValidationError(
                f"Vector-store metadata file is missing: '{metadata_path}'."
            )
        try:
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise VectorStoreValidationError(
                f"Vector-store metadata is not valid UTF-8: '{metadata_path}'."
            ) from exc
        except json.JSONDecodeError as exc:
            raise VectorStoreValidationError(
                f"Vector-store metadata is invalid JSON at line {exc.lineno}, "
                f"column {exc.colno}: '{metadata_path}'."
            ) from exc
        metadata = _validate_persistence_metadata(raw_metadata, metadata_path)
        try:
            index = faiss.read_index(str(index_path))
        except RuntimeError as exc:
            raise VectorStoreValidationError(
                f"FAISS index could not be loaded or is corrupted: '{index_path}'."
            ) from exc
        if not isinstance(index, faiss.IndexFlatL2):
            raise VectorStoreValidationError(
                "FAISS index type is incompatible; expected IndexFlatL2."
            )
        if index.d != metadata["dimension"]:
            raise VectorStoreValidationError(
                "FAISS index dimension does not match metadata: "
                f"index={index.d}, metadata={metadata['dimension']}."
            )
        if index.ntotal != metadata["item_count"]:
            raise VectorStoreValidationError(
                "FAISS index item count does not match metadata: "
                f"index={index.ntotal}, metadata={metadata['item_count']}."
            )
        items = tuple(
            _deserialize_embedded_chunk(item, index.reconstruct(position))
            for position, item in enumerate(metadata["items"])
        )
        store = cls.__new__(cls)
        store._items = items
        store._dimension = index.d
        store._index = index
        return store


def _validate_persistence_metadata(value: object, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VectorStoreValidationError(
            f"Vector-store metadata root must be a JSON object: '{path}'."
        )
    version = value.get("version")
    if version != PERSISTENCE_VERSION:
        raise VectorStoreValidationError(
            "Vector-store metadata version is incompatible: "
            f"expected {PERSISTENCE_VERSION}, received {version!r}."
        )
    if value.get("index_type") != "IndexFlatL2":
        raise VectorStoreValidationError(
            "Vector-store metadata index_type must be 'IndexFlatL2'."
        )
    dimension = value.get("dimension")
    item_count = value.get("item_count")
    items = value.get("items")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise VectorStoreValidationError(
            "Vector-store metadata dimension must be a positive integer."
        )
    if isinstance(item_count, bool) or not isinstance(item_count, int) or item_count < 1:
        raise VectorStoreValidationError(
            "Vector-store metadata item_count must be a positive integer."
        )
    if not isinstance(items, list) or len(items) != item_count:
        raise VectorStoreValidationError(
            "Vector-store metadata items must be an array matching item_count."
        )
    for position, item in enumerate(items):
        _validate_persisted_item(item, position)
    return value


def _validate_persisted_item(value: object, position: int) -> None:
    if not isinstance(value, dict):
        raise VectorStoreValidationError(
            f"Vector-store metadata item {position} must be an object."
        )
    if not isinstance(value.get("embedding_model"), str) or not value["embedding_model"].strip():
        raise VectorStoreValidationError(
            f"Vector-store metadata item {position} has an invalid embedding_model."
        )
    chunk = value.get("chunk")
    if not isinstance(chunk, dict):
        raise VectorStoreValidationError(
            f"Vector-store metadata item {position} must contain a chunk object."
        )
    for field_name in (
        "content", "source", "file_name", "file_type", "document_id", "chunk_id"
    ):
        if not isinstance(chunk.get(field_name), str):
            raise VectorStoreValidationError(
                f"Vector-store metadata item {position} chunk.{field_name} "
                "must be a string."
            )
    chunk_index = chunk.get("chunk_index")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise VectorStoreValidationError(
            f"Vector-store metadata item {position} chunk.chunk_index "
            "must be a non-negative integer."
        )
    if not isinstance(chunk.get("metadata"), dict):
        raise VectorStoreValidationError(
            f"Vector-store metadata item {position} chunk.metadata must be an object."
        )


def _deserialize_embedded_chunk(
    value: dict[str, Any], vector: Sequence[float]
) -> EmbeddedChunk:
    chunk_data = value["chunk"]
    chunk = DocumentChunk(
        content=chunk_data["content"],
        source=chunk_data["source"],
        file_name=chunk_data["file_name"],
        file_type=chunk_data["file_type"],
        document_id=chunk_data["document_id"],
        chunk_index=chunk_data["chunk_index"],
        chunk_id=chunk_data["chunk_id"],
        metadata=chunk_data["metadata"],
    )
    return EmbeddedChunk(
        chunk=chunk,
        vector=tuple(float(number) for number in vector),
        embedding_model=value["embedding_model"],
    )
def build_vector_store(
    embedded_chunks: Sequence[EmbeddedChunk],
) -> FaissVectorStore:
    """Build the minimal in-memory FAISS store."""
    return FaissVectorStore(embedded_chunks)


def save_vector_store(vector_store: Any, settings: Settings) -> None:
    """Persist a vector store under the configured local data directory."""
    if not isinstance(vector_store, FaissVectorStore):
        raise VectorStoreValidationError(
            "save_vector_store requires a FaissVectorStore."
        )
    vector_store.save(settings.vector_store_dir)


def load_vector_store(embeddings: Any, settings: Settings) -> Any:
    """Load a previously generated local vector store."""
    return FaissVectorStore.load(settings.vector_store_dir)
