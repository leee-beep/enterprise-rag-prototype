"""Deterministic character-based text splitting."""

from __future__ import annotations

from collections.abc import Iterable

from enterprise_rag.config import Settings
from enterprise_rag.models import DocumentChunk, LoadedDocument


def _preferred_end(
    text: str, start: int, maximum_end: int, chunk_overlap: int
) -> int:
    """Prefer paragraph or line boundaries without creating tiny chunks."""
    if maximum_end >= len(text):
        return len(text)

    minimum_end = start + max(chunk_overlap + 1, (maximum_end - start) // 2)

    paragraph = text.rfind("\n\n", minimum_end, maximum_end)
    if paragraph != -1:
        return paragraph + 2

    line = text.rfind("\n", minimum_end, maximum_end)
    if line != -1:
        return line + 1

    return maximum_end


def _chunk_id(document_id: str, chunk_index: int) -> str:
    """Build a stable chunk identifier from stable document information."""
    return f"{document_id}:chunk-{chunk_index:06d}"


def split_documents(
    documents: Iterable[LoadedDocument],
    settings: Settings,
) -> tuple[DocumentChunk, ...]:
    """Split documents with deterministic character windows and exact overlap.

    Paragraph and newline boundaries are preferred in the latter half of each
    candidate window. When no suitable boundary exists, the text is hard-cut at
    ``chunk_size`` so every chunk remains within the configured maximum.
    """
    if settings.chunk_size < 1:
        raise ValueError("chunk_size 必須大於或等於 1。")
    if settings.chunk_overlap < 0:
        raise ValueError("chunk_overlap 不可小於 0。")
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("chunk_overlap 必須小於 chunk_size。")

    chunks: list[DocumentChunk] = []
    for document in documents:
        if not document.content.strip():
            continue

        start = 0
        chunk_index = 0
        while start < len(document.content):
            maximum_end = min(start + settings.chunk_size, len(document.content))
            end = _preferred_end(
                document.content, start, maximum_end, settings.chunk_overlap
            )
            content = document.content[start:end]

            chunks.append(
                DocumentChunk(
                    content=content,
                    source=document.source,
                    file_name=document.file_name,
                    file_type=document.file_type,
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    chunk_id=_chunk_id(document.document_id, chunk_index),
                    metadata=dict(document.metadata),
                )
            )

            if end >= len(document.content):
                break
            start = end - settings.chunk_overlap
            chunk_index += 1

    return tuple(chunks)