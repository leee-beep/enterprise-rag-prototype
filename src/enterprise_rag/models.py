"""Data models shared by ingestion, chunking, embedding, and retrieval."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

@dataclass(frozen=True)
class LoadedDocument:
    """A local document and its stable source metadata."""
    content: str
    source: str
    file_name: str
    file_type: str
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DocumentChunk:
    """A deterministic text chunk retaining its source document metadata."""
    content: str
    source: str
    file_name: str
    file_type: str
    document_id: str
    chunk_index: int
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class EmbeddedChunk:
    """A source chunk paired with its validated embedding vector."""
    chunk: DocumentChunk
    vector: tuple[float, ...]
    embedding_model: str

@dataclass(frozen=True)
class DocumentLoadWarning:
    """A non-fatal ingestion problem that was explicitly reported."""
    source: str
    message: str
    json_path: str | None = None
    line_number: int | None = None

@dataclass(frozen=True)
class RetrievalResult:
    """A ranked chunk with finite relevance score and immutable metadata.

    ``score`` is a monotonic transformation of squared L2 distance, not a
    probability or cosine similarity. Canonical identity metadata is always
    derived from ``embedded_chunk`` so stale caller values cannot override it.
    """
    score: float
    embedded_chunk: EmbeddedChunk
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("RetrievalResult score must be finite.")
        chunk = self.embedded_chunk.chunk
        normalized = dict(self.metadata)
        normalized.update(
            {
                "source": chunk.source,
                "file_name": chunk.file_name,
                "file_type": chunk.file_type,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "embedding_model": self.embedded_chunk.embedding_model,
            }
        )
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "metadata", MappingProxyType(normalized))