"""Application settings loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from enterprise_rag.json_config import JsonLoaderSettings, load_json_loader_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
API_KEY_PLACEHOLDERS = {"your_gemini_api_key_here", "replace_me", "changeme"}
SUPPORTED_PROVIDERS = frozenset({"gemini", "ollama"})

class ConfigurationError(ValueError):
    """Raised when local environment settings are missing or invalid."""

@dataclass(frozen=True)
class Settings:
    """Validated local, Gemini, and Ollama settings."""
    gemini_api_key: str | None = field(repr=False)
    generation_model: str
    embedding_model: str
    documents_dir: Path
    vector_store_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    json_loader: JsonLoaderSettings = field(default_factory=JsonLoaderSettings)
    embedding_provider: str = "gemini"
    generation_provider: str = "gemini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_chat_model: str = "llama3.2"
    ollama_timeout_seconds: float = 30.0

    @property
    def gemini_chat_model(self) -> str:
        """Backward-compatible alias for the Gemini generation model."""
        return self.generation_model

    @property
    def selected_embedding_model(self) -> str:
        return self.embedding_model if self.embedding_provider == "gemini" else self.ollama_embedding_model

    @property
    def selected_generation_model(self) -> str:
        return self.generation_model if self.generation_provider == "gemini" else self.ollama_chat_model

    def require_gemini_api_key(self) -> str:
        value = (self.gemini_api_key or "").strip()
        if not value or value.lower() in API_KEY_PLACEHOLDERS:
            raise ConfigurationError(
                "這項功能需要 Gemini API，因此必須設定有效的 GEMINI_API_KEY。"
                "請在專案根目錄建立 .env，並加入 GEMINI_API_KEY=你的金鑰。"
                "不要把 .env 上傳到 GitHub。"
            )
        return value

def _optional_api_key() -> str | None:
    value = os.getenv("GEMINI_API_KEY", "").strip()
    return value or None

def _text_setting(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ConfigurationError(f"環境變數 {name} 不可為空白；請在 .env 中填入有效值。")
    return value

def _optional_text_setting(name: str, default: str) -> str:
    return os.getenv(name, default).strip()

def _provider_setting(name: str, default: str) -> str:
    value = os.getenv(name, default).strip().casefold()
    if value not in SUPPORTED_PROVIDERS:
        allowed = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ConfigurationError(f"環境變數 {name} 只支援 {allowed}，目前收到：{value!r}。")
    return value

def _integer_setting(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"環境變數 {name} 必須是整數，目前收到：{raw!r}。") from exc
    if value < minimum:
        raise ConfigurationError(f"環境變數 {name} 必須大於或等於 {minimum}，目前收到：{value}。")
    return value

def _positive_float_setting(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"環境變數 {name} 必須是數字，目前收到：{raw!r}。") from exc
    if value <= 0:
        raise ConfigurationError(f"環境變數 {name} 必須大於 0，目前收到：{value}。")
    return value

def _path_setting(name: str, default: str) -> Path:
    path = Path(_text_setting(name, default)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()

def load_settings(*, env_file: Path | None = None, load_env_file: bool = True) -> Settings:
    """Load settings without validating API keys or performing network calls."""
    if load_env_file:
        load_dotenv(dotenv_path=env_file or DEFAULT_ENV_FILE, override=False)
    chunk_size = _integer_setting("CHUNK_SIZE", 200, minimum=1)
    chunk_overlap = _integer_setting("CHUNK_OVERLAP", 50, minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigurationError("CHUNK_OVERLAP 必須小於 CHUNK_SIZE，否則文字切塊無法正常前進。")
    generation_model = os.getenv(
        "GEMINI_CHAT_MODEL",
        os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.1-flash-lite"),
    ).strip()
    if not generation_model:
        raise ConfigurationError("環境變數 GEMINI_CHAT_MODEL 不可為空白。")
    return Settings(
        gemini_api_key=_optional_api_key(),
        generation_model=generation_model,
        embedding_model=_text_setting("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        documents_dir=_path_setting("DOCUMENTS_DIR", "data/documents"),
        vector_store_dir=_path_setting("VECTOR_STORE_DIR", "data/vector_store"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=_integer_setting("TOP_K", 4, minimum=1),
        json_loader=load_json_loader_settings(),
        embedding_provider=_provider_setting("EMBEDDING_PROVIDER", "gemini"),
        generation_provider=_provider_setting("GENERATION_PROVIDER", "gemini"),
        ollama_base_url=_text_setting("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_embedding_model=_optional_text_setting("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        ollama_chat_model=_optional_text_setting("OLLAMA_CHAT_MODEL", "llama3.2"),
        ollama_timeout_seconds=_positive_float_setting("OLLAMA_TIMEOUT_SECONDS", 30.0),
    )