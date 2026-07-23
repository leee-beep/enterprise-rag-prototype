from __future__ import annotations

import pytest
from enterprise_rag.config import ConfigurationError, load_settings
from enterprise_rag.factory import create_embedding_client
from enterprise_rag.providers.ollama import OllamaEmbeddingClient

PROVIDER_ENV = (
    "EMBEDDING_PROVIDER", "GENERATION_PROVIDER", "OLLAMA_BASE_URL",
    "OLLAMA_EMBEDDING_MODEL", "OLLAMA_CHAT_MODEL", "OLLAMA_TIMEOUT_SECONDS",
    "GEMINI_API_KEY", "GEMINI_CHAT_MODEL", "GEMINI_GENERATION_MODEL",
)

def clear(monkeypatch):
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)

def test_provider_defaults(monkeypatch):
    clear(monkeypatch)
    settings = load_settings(load_env_file=False)
    assert settings.embedding_provider == "gemini"
    assert settings.generation_provider == "gemini"
    assert settings.ollama_base_url == "http://localhost:11434"

def test_provider_case_and_whitespace_normalization(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "  OLLAMA ")
    monkeypatch.setenv("GENERATION_PROVIDER", " Gemini ")
    settings = load_settings(load_env_file=False)
    assert settings.embedding_provider == "ollama"
    assert settings.generation_provider == "gemini"

def test_unsupported_provider_is_rejected(monkeypatch):
    clear(monkeypatch); monkeypatch.setenv("EMBEDDING_PROVIDER", "unknown")
    with pytest.raises(ConfigurationError, match="EMBEDDING_PROVIDER"):
        load_settings(load_env_file=False)

def test_ollama_does_not_require_gemini_key(monkeypatch):
    clear(monkeypatch); monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    settings = load_settings(load_env_file=False)
    assert settings.gemini_api_key is None
    assert isinstance(create_embedding_client(settings), OllamaEmbeddingClient)

def test_timeout_must_be_positive(monkeypatch):
    clear(monkeypatch); monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "0")
    with pytest.raises(ConfigurationError, match="OLLAMA_TIMEOUT_SECONDS"):
        load_settings(load_env_file=False)

def test_base_url_trailing_slash_is_removed(monkeypatch):
    clear(monkeypatch); monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434///")
    assert load_settings(load_env_file=False).ollama_base_url == "http://localhost:11434"

def test_gemini_chat_model_supports_new_and_legacy_names(monkeypatch):
    clear(monkeypatch); monkeypatch.setenv("GEMINI_GENERATION_MODEL", "legacy")
    assert load_settings(load_env_file=False).generation_model == "legacy"
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "new-name")
    assert load_settings(load_env_file=False).gemini_chat_model == "new-name"