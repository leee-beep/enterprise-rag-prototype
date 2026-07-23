import pytest
from enterprise_rag.json_config import JsonConfigurationError
from enterprise_rag.config import load_settings

NAMES = ("JSON_EXTRACTION_MODE", "JSON_CONTENT_KEYS", "JSON_TITLE_KEYS", "JSON_METADATA_KEYS", "JSON_EXCLUDE_KEYS", "JSON_MAX_DEPTH", "JSON_MAX_RECORDS", "JSON_MAX_FILE_SIZE_MB", "JSON_STRICT_MODE")

def clear(monkeypatch):
    for name in NAMES: monkeypatch.delenv(name, raising=False)

def test_json_config_defaults(monkeypatch):
    clear(monkeypatch)
    config = load_settings(load_env_file=False).json_loader
    assert config.extraction_mode == "auto"
    assert "content" in config.content_keys and config.strict_mode is True

def test_json_config_custom_environment(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("JSON_EXTRACTION_MODE", "recursive")
    monkeypatch.setenv("JSON_CONTENT_KEYS", "article, copy")
    monkeypatch.setenv("JSON_STRICT_MODE", "false")
    config = load_settings(load_env_file=False).json_loader
    assert config.extraction_mode == "recursive"
    assert config.content_keys == ("article", "copy")
    assert config.strict_mode is False

def test_json_config_rejects_invalid_mode(monkeypatch):
    clear(monkeypatch); monkeypatch.setenv("JSON_EXTRACTION_MODE", "dump")
    with pytest.raises(JsonConfigurationError): load_settings(load_env_file=False)