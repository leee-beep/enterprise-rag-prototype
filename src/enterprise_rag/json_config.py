"""Configuration for JSON and JSONL document extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_CONTENT_KEYS = (
    "content",
    "text",
    "body",
    "description",
    "summary",
    "answer",
    "paragraph",
    "paragraphs",
)
DEFAULT_TITLE_KEYS = ("title", "heading", "name", "question")
DEFAULT_METADATA_KEYS = (
    "id",
    "url",
    "author",
    "category",
    "tags",
    "language",
    "status",
    "version",
    "created_at",
    "updated_at",
)
DEFAULT_EXCLUDE_KEYS = (
    "embedding",
    "embeddings",
    "vector",
    "vectors",
    "raw_html",
    "base64",
    "image",
)
VALID_EXTRACTION_MODES = frozenset({"auto", "record", "recursive"})


class JsonConfigurationError(ValueError):
    """Raised when JSON ingestion configuration is invalid."""


@dataclass(frozen=True)
class JsonLoaderSettings:
    """Validated extraction strategy and safety limits."""

    extraction_mode: str = "auto"
    content_keys: tuple[str, ...] = DEFAULT_CONTENT_KEYS
    title_keys: tuple[str, ...] = DEFAULT_TITLE_KEYS
    metadata_keys: tuple[str, ...] = DEFAULT_METADATA_KEYS
    exclude_keys: tuple[str, ...] = DEFAULT_EXCLUDE_KEYS
    max_depth: int = 20
    max_records: int = 1000
    max_file_size_mb: float = 10.0
    strict_mode: bool = True

    def __post_init__(self) -> None:
        if self.extraction_mode not in VALID_EXTRACTION_MODES:
            raise JsonConfigurationError(
                "JSON_EXTRACTION_MODE must be auto, record, or recursive."
            )
        if self.max_depth < 0:
            raise JsonConfigurationError("JSON_MAX_DEPTH must be at least 0.")
        if self.max_records < 1:
            raise JsonConfigurationError("JSON_MAX_RECORDS must be at least 1.")
        if self.max_file_size_mb <= 0:
            raise JsonConfigurationError("JSON_MAX_FILE_SIZE_MB must be greater than 0.")
        for name, keys in (
            ("content_keys", self.content_keys),
            ("title_keys", self.title_keys),
            ("metadata_keys", self.metadata_keys),
            ("exclude_keys", self.exclude_keys),
        ):
            if not keys or any(not key.strip() for key in keys):
                raise JsonConfigurationError(f"{name} must contain non-empty keys.")


def _csv_setting(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values = tuple(part.strip().casefold() for part in raw.split(",") if part.strip())
    if not values:
        raise JsonConfigurationError(f"{name} must contain at least one key.")
    return values


def _int_setting(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise JsonConfigurationError(f"{name} must be an integer, received {raw!r}.") from exc
    if value < minimum:
        raise JsonConfigurationError(f"{name} must be at least {minimum}.")
    return value


def _float_setting(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise JsonConfigurationError(f"{name} must be numeric, received {raw!r}.") from exc
    if value <= 0:
        raise JsonConfigurationError(f"{name} must be greater than 0.")
    return value


def _bool_setting(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().casefold()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise JsonConfigurationError(
        f"{name} must be true/false, yes/no, on/off, or 1/0."
    )


def load_json_loader_settings() -> JsonLoaderSettings:
    """Load JSON ingestion settings from environment variables."""
    mode = os.getenv("JSON_EXTRACTION_MODE", "auto").strip().casefold()
    return JsonLoaderSettings(
        extraction_mode=mode,
        content_keys=_csv_setting("JSON_CONTENT_KEYS", DEFAULT_CONTENT_KEYS),
        title_keys=_csv_setting("JSON_TITLE_KEYS", DEFAULT_TITLE_KEYS),
        metadata_keys=_csv_setting("JSON_METADATA_KEYS", DEFAULT_METADATA_KEYS),
        exclude_keys=_csv_setting("JSON_EXCLUDE_KEYS", DEFAULT_EXCLUDE_KEYS),
        max_depth=_int_setting("JSON_MAX_DEPTH", 20, 0),
        max_records=_int_setting("JSON_MAX_RECORDS", 1000, 1),
        max_file_size_mb=_float_setting("JSON_MAX_FILE_SIZE_MB", 10.0),
        strict_mode=_bool_setting("JSON_STRICT_MODE", True),
    )