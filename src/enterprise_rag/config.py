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


class ConfigurationError(ValueError):
    """Raised when local environment settings are missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Validated settings shared by local and Gemini-backed features."""

    gemini_api_key: str | None = field(repr=False)
    generation_model: str
    embedding_model: str
    documents_dir: Path
    vector_store_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    json_loader: JsonLoaderSettings = field(default_factory=JsonLoaderSettings)

    def require_gemini_api_key(self) -> str:
        """Return a validated key immediately before a Gemini API operation.

        Local-only features may use ``Settings`` without a key. Embedding and LLM
        code must call this method before constructing a Gemini client or sending
        a request.
        """
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
        raise ConfigurationError(
            f"環境變數 {name} 不可為空白；請在 .env 中填入有效值。"
        )
    return value


def _integer_setting(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(
            f"環境變數 {name} 必須是整數，目前收到：{raw_value!r}。"
        ) from exc

    if value < minimum:
        raise ConfigurationError(
            f"環境變數 {name} 必須大於或等於 {minimum}，目前收到：{value}。"
        )
    return value


def _path_setting(name: str, default: str) -> Path:
    raw_value = _text_setting(name, default)
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_settings(
    *,
    env_file: Path | None = None,
    load_env_file: bool = True,
) -> Settings:
    """Load local settings without requiring access to the Gemini API.

    Args:
        env_file: Optional path to a dotenv file. Defaults to ``<project>/.env``.
        load_env_file: Set to ``False`` in tests or managed environments to avoid
            reading any local dotenv file.
    """
    if load_env_file:
        load_dotenv(dotenv_path=env_file or DEFAULT_ENV_FILE, override=False)

    chunk_size = _integer_setting("CHUNK_SIZE", 200, minimum=1)
    chunk_overlap = _integer_setting("CHUNK_OVERLAP", 50, minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigurationError(
            "CHUNK_OVERLAP 必須小於 CHUNK_SIZE，否則文字切塊無法正常前進。"
        )

    return Settings(
        gemini_api_key=_optional_api_key(),
        generation_model=_text_setting(
            "GEMINI_GENERATION_MODEL", "gemini-3.1-flash-lite"
        ),
        embedding_model=_text_setting(
            "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
        ),
        documents_dir=_path_setting("DOCUMENTS_DIR", "data/documents"),
        vector_store_dir=_path_setting("VECTOR_STORE_DIR", "data/vector_store"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=_integer_setting("TOP_K", 4, minimum=1),
        json_loader=load_json_loader_settings(),
    )