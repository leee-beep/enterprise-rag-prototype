"""Data models shared by ingestion, chunking, embedding, and vector storage."""

from dataclasses import dataclass, field
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