"""Provider factories that construct clients without performing API calls."""
from __future__ import annotations

from enterprise_rag.config import ConfigurationError, Settings
from enterprise_rag.embeddings import EmbeddingClient
from enterprise_rag.generation import GenerationClient
from enterprise_rag.providers.gemini import GeminiEmbeddingClient, GeminiGenerationClient
from enterprise_rag.providers.ollama import OllamaEmbeddingClient, OllamaGenerationClient

def create_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == "gemini":
        return GeminiEmbeddingClient(settings)
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
            timeout=settings.ollama_timeout_seconds,
        )
    raise ConfigurationError(f"Unsupported embedding provider: {settings.embedding_provider!r}.")

def create_generation_client(settings: Settings) -> GenerationClient:
    if settings.generation_provider == "gemini":
        return GeminiGenerationClient(settings)
    if settings.generation_provider == "ollama":
        return OllamaGenerationClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            timeout=settings.ollama_timeout_seconds,
        )
    raise ConfigurationError(f"Unsupported generation provider: {settings.generation_provider!r}.")