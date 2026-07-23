"""Minimal, testable Gemini embedding support for document chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from enterprise_rag.config import Settings
from enterprise_rag.models import DocumentChunk, EmbeddedChunk


class EmbeddingError(RuntimeError):
    """Base error for embedding operations."""


class EmbeddingValidationError(EmbeddingError):
    """Raised when an embedding response is incomplete or inconsistent."""


class EmbeddingClient(Protocol):
    """Small injectable interface implemented by real and fake clients."""

    def embed(self, *, model: str, contents: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector for each input string."""
        ...


class GoogleGenAIEmbeddingClient:
    """Adapter around the Google Gen AI SDK synchronous embedding API."""

    def __init__(self, *, api_key: str) -> None:
        # Import lazily so fake-client tests never initialize the Google SDK.
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def embed(
        self,
        *,
        model: str,
        contents: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        response = self._client.models.embed_content(
            model=model,
            contents=list(contents),
        )
        return [embedding.values or () for embedding in (response.embeddings or ())]


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    """Create the real Gemini adapter after explicitly validating the API key."""
    api_key = settings.require_gemini_api_key()
    return GoogleGenAIEmbeddingClient(api_key=api_key)


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    settings: Settings,
    *,
    client: EmbeddingClient | None = None,
) -> tuple[EmbeddedChunk, ...]:
    """Embed a batch of chunks and validate the returned vectors.

    A client can be injected for tests. API-key validation still occurs at this
    embedding boundary, before either a fake or real client is invoked.
    """
    if not chunks:
        return ()

    if client is None:
        embedding_client = create_embedding_client(settings)
    else:
        settings.require_gemini_api_key()
        embedding_client = client

    raw_vectors = embedding_client.embed(
        model=settings.embedding_model,
        contents=[chunk.content for chunk in chunks],
    )
    vectors = tuple(tuple(float(value) for value in vector) for vector in raw_vectors)

    if len(vectors) != len(chunks):
        raise EmbeddingValidationError(
            "Embedding response count does not match the input chunk count: "
            f"expected {len(chunks)}, received {len(vectors)}."
        )

    dimensions: set[int] = set()
    for index, vector in enumerate(vectors):
        if not vector:
            raise EmbeddingValidationError(
                f"Embedding response for chunk index {index} is empty."
            )
        dimensions.add(len(vector))

    if len(dimensions) != 1:
        raise EmbeddingValidationError(
            "Embedding vector dimensions are inconsistent within the batch: "
            f"received {sorted(dimensions)}."
        )

    return tuple(
        EmbeddedChunk(
            chunk=chunk,
            vector=vector,
            embedding_model=settings.embedding_model,
        )
        for chunk, vector in zip(chunks, vectors)
    )