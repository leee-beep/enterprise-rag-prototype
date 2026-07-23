"""Google Gen AI provider adapters with delayed client construction."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from enterprise_rag.config import Settings
from enterprise_rag.embeddings import validate_embedding_vectors
from enterprise_rag.generation import validate_generated_text

class GeminiEmbeddingClient:
    def __init__(self, settings: Settings, *, sdk_client: Any | None = None) -> None:
        self._settings = settings
        self._client = sdk_client

    def _sdk_client(self) -> Any:
        api_key = self._settings.require_gemini_api_key()
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=api_key)
        return self._client

    def embed(self, *, model: str, contents: Sequence[str]) -> Sequence[Sequence[float]]:
        response = self._sdk_client().models.embed_content(model=model, contents=list(contents))
        raw = [embedding.values or () for embedding in (getattr(response, "embeddings", None) or ())]
        return validate_embedding_vectors(raw, expected_count=len(contents))

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return validate_embedding_vectors(
            self.embed(model=self._settings.embedding_model, contents=texts),
            expected_count=len(texts),
        )

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self.embed_documents([text])[0]

class GeminiGenerationClient:
    def __init__(self, settings: Settings, *, sdk_client: Any | None = None) -> None:
        self._settings = settings
        self._client = sdk_client

    def _sdk_client(self) -> Any:
        api_key = self._settings.require_gemini_api_key()
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        response = self._sdk_client().models.generate_content(
            model=self._settings.generation_model,
            contents=prompt,
        )
        return validate_generated_text(getattr(response, "text", None))