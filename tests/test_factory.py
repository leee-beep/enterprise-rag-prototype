from dataclasses import replace
from pathlib import Path
import pytest

from enterprise_rag.config import ConfigurationError, Settings
from enterprise_rag.factory import create_embedding_client, create_generation_client
from enterprise_rag.providers.gemini import GeminiEmbeddingClient, GeminiGenerationClient
from enterprise_rag.providers.ollama import OllamaEmbeddingClient, OllamaGenerationClient

def settings(**changes):
    base = Settings(
        gemini_api_key=None, generation_model="gemini-chat", embedding_model="gemini-embed",
        documents_dir=Path("docs"), vector_store_dir=Path("vectors"),
        chunk_size=100, chunk_overlap=10, top_k=4,
        ollama_embedding_model="ollama-embed", ollama_chat_model="ollama-chat",
    )
    return replace(base, **changes)

def test_gemini_embedding_factory_has_no_api_call():
    assert isinstance(create_embedding_client(settings()), GeminiEmbeddingClient)

def test_ollama_embedding_factory_has_no_api_call():
    assert isinstance(create_embedding_client(settings(embedding_provider="ollama")), OllamaEmbeddingClient)

def test_gemini_generation_factory_has_no_api_call():
    assert isinstance(create_generation_client(settings()), GeminiGenerationClient)

def test_ollama_generation_factory_has_no_api_call():
    assert isinstance(create_generation_client(settings(generation_provider="ollama")), OllamaGenerationClient)

@pytest.mark.parametrize("embedding,generation", [
    ("gemini", "gemini"), ("ollama", "ollama"),
    ("ollama", "gemini"), ("gemini", "ollama"),
])
def test_providers_can_be_mixed_independently(embedding, generation):
    value = settings(embedding_provider=embedding, generation_provider=generation)
    assert type(create_embedding_client(value)).__name__.lower().startswith(embedding)
    assert type(create_generation_client(value)).__name__.lower().startswith(generation)

def test_unsupported_embedding_provider_fails():
    with pytest.raises(ConfigurationError, match="Unsupported embedding"):
        create_embedding_client(settings(embedding_provider="bad"))

def test_unsupported_generation_provider_fails():
    with pytest.raises(ConfigurationError, match="Unsupported generation"):
        create_generation_client(settings(generation_provider="bad"))