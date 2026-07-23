"""Provider-neutral embedding interfaces and shared validation."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from enterprise_rag.config import Settings
from enterprise_rag.models import DocumentChunk, EmbeddedChunk

class EmbeddingError(RuntimeError):
    """Base error for embedding operations."""

class EmbeddingValidationError(EmbeddingError):
    """Raised when an embedding response is incomplete or inconsistent."""

class EmbeddingClient(Protocol):
    def embed(self, *, model: str, contents: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector for each input string."""
        ...

# Backward-compatible legacy Gemini adapter. New provider construction lives in factory.py.
class GoogleGenAIEmbeddingClient:
    def __init__(self, *, api_key: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)

    def embed(self, *, model: str, contents: Sequence[str]) -> Sequence[Sequence[float]]:
        response = self._client.models.embed_content(model=model, contents=list(contents))
        return [embedding.values or () for embedding in (response.embeddings or ())]

def validate_embedding_vectors(
    raw_vectors: Sequence[Sequence[float]], *, expected_count: int | None = None
) -> tuple[tuple[float, ...], ...]:
    vectors = tuple(tuple(float(value) for value in vector) for vector in raw_vectors)
    if expected_count is not None and len(vectors) != expected_count:
        raise EmbeddingValidationError(
            "Embedding response count does not match the input chunk count: "
            f"expected {expected_count}, received {len(vectors)}."
        )
    dimensions: set[int] = set()
    for index, vector in enumerate(vectors):
        if not vector:
            raise EmbeddingValidationError(f"Embedding response for chunk index {index} is empty.")
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingValidationError(f"Embedding response for chunk index {index} contains NaN or Infinity.")
        dimensions.add(len(vector))
    if len(dimensions) > 1:
        raise EmbeddingValidationError(
            "Embedding vector dimensions are inconsistent within the batch: "
            f"received {sorted(dimensions)}."
        )
    return vectors

def create_embedding_client(settings: Settings) -> EmbeddingClient:
    """Backward-compatible thin wrapper around the canonical provider factory."""
    from enterprise_rag.factory import create_embedding_client as provider_factory
    return provider_factory(settings)

def embed_chunks(
    chunks: Sequence[DocumentChunk], settings: Settings, *, client: EmbeddingClient | None = None
) -> tuple[EmbeddedChunk, ...]:
    """Embed chunks with the selected provider and shared vector validation."""
    if not chunks:
        return ()
    if client is None:
        embedding_client = create_embedding_client(settings)
    else:
        embedding_client = client
    model = settings.selected_embedding_model
    raw = embedding_client.embed(model=model, contents=[chunk.content for chunk in chunks])
    vectors = validate_embedding_vectors(raw, expected_count=len(chunks))
    return tuple(
        EmbeddedChunk(chunk=chunk, vector=vector, embedding_model=model)
        for chunk, vector in zip(chunks, vectors)
    )