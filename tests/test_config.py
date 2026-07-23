"""Tests for environment-based configuration loading."""

import pytest

from enterprise_rag.config import ConfigurationError, PROJECT_ROOT, load_settings

SETTING_NAMES = (
    "GEMINI_API_KEY",
    "GEMINI_GENERATION_MODEL",
    "GEMINI_EMBEDDING_MODEL",
    "DOCUMENTS_DIR",
    "VECTOR_STORE_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "EMBEDDING_PROVIDER",
    "GENERATION_PROVIDER",
    "GEMINI_CHAT_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_EMBEDDING_MODEL",
    "OLLAMA_CHAT_MODEL",
    "OLLAMA_TIMEOUT_SECONDS",
)


def clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove relevant variables so tests never depend on a developer's .env."""
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_local_settings_load_without_gemini_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings(monkeypatch)

    settings = load_settings(load_env_file=False)

    assert settings.gemini_api_key is None
    assert settings.documents_dir == (PROJECT_ROOT / "data/documents").resolve()
    assert settings.chunk_size == 200
    assert settings.chunk_overlap == 50
    assert settings.top_k == 4


def test_gemini_feature_requires_api_key_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings(monkeypatch)
    settings = load_settings(load_env_file=False)

    with pytest.raises(ConfigurationError, match="這項功能需要 Gemini API") as error:
        settings.require_gemini_api_key()

    assert ".env" in str(error.value)
    assert "GitHub" in str(error.value)


def test_placeholder_api_key_is_rejected_at_gemini_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    settings = load_settings(load_env_file=False)

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        settings.require_gemini_api_key()


def test_loads_values_without_printing_or_revealing_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clear_settings(monkeypatch)
    fake_key = "test-only-fake-gemini-key"
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "test-generation-model")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("DOCUMENTS_DIR", "test-data/documents")
    monkeypatch.setenv("VECTOR_STORE_DIR", "test-data/vectors")
    monkeypatch.setenv("CHUNK_SIZE", "300")
    monkeypatch.setenv("CHUNK_OVERLAP", "30")
    monkeypatch.setenv("TOP_K", "5")

    settings = load_settings(load_env_file=False)

    assert settings.require_gemini_api_key() == fake_key
    assert settings.generation_model == "test-generation-model"
    assert settings.embedding_model == "test-embedding-model"
    assert settings.documents_dir == (PROJECT_ROOT / "test-data/documents").resolve()
    assert settings.vector_store_dir == (PROJECT_ROOT / "test-data/vectors").resolve()
    assert settings.chunk_size == 300
    assert settings.chunk_overlap == 30
    assert settings.top_k == 5
    assert fake_key not in repr(settings)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("CHUNK_SIZE", "not-a-number", "CHUNK_SIZE 必須是整數"),
        ("TOP_K", "0", "TOP_K 必須大於或等於 1"),
    ],
)
def test_rejects_invalid_integer_settings_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    clear_settings(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        load_settings(load_env_file=False)


def test_chunk_overlap_validation_does_not_require_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings(monkeypatch)
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    with pytest.raises(ConfigurationError, match="CHUNK_OVERLAP 必須小於 CHUNK_SIZE"):
        load_settings(load_env_file=False)