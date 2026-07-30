"""Application-level orchestration for building a persisted FAISS index."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from enterprise_rag.chunking import split_documents
from enterprise_rag.config import Settings
from enterprise_rag.documents import DocumentLoadingError, load_documents
from enterprise_rag.embeddings import EmbeddingClient, embed_chunks
from enterprise_rag.factory import create_embedding_client
from enterprise_rag.vector_store import (
    FaissVectorStore,
    VectorStoreError,
    build_vector_store,
)

INDEX_MANIFEST_FILE_NAME = "index_manifest.json"
INDEX_MANIFEST_SCHEMA_VERSION = 1


class IndexingError(RuntimeError):
    """Raised when the document-to-index workflow cannot complete safely."""


@dataclass(frozen=True)
class IndexingResult:
    """Summary of a completed and persisted indexing run."""

    loaded_document_count: int
    chunk_count: int
    embedded_vector_count: int
    embedding_dimension: int
    output_path: Path
    embedding_provider: str
    embedding_model: str
    elapsed_seconds: float


class IndexingService:
    """Compose existing ingestion, chunking, embedding, and FAISS components."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedding_client: EmbeddingClient | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_client = embedding_client
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        if monotonic is None:
            from time import monotonic as system_monotonic

            monotonic = system_monotonic
        self._monotonic = monotonic

    def build(
        self,
        *,
        input_directory: Path | None = None,
        output_directory: Path | None = None,
        overwrite: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> IndexingResult:
        """Build and safely publish a persisted index."""
        started = self._monotonic()
        input_path, output_path = _validate_paths(
            input_directory or self._settings.documents_dir,
            output_directory or self._settings.vector_store_dir,
            overwrite=overwrite,
        )
        _report(progress, f"Loading documents from {input_path}")
        try:
            loaded = load_documents(
                input_path,
                json_config=self._settings.json_loader,
            )
        except DocumentLoadingError as exc:
            raise IndexingError(f"Document loading failed: {exc}") from exc

        if not loaded.documents:
            skipped = len(loaded.skipped_empty)
            raise IndexingError(
                "No non-empty documents are available for indexing"
                f" ({skipped} empty supported file(s) were skipped)."
            )
        _report(progress, f"Loaded {len(loaded.documents)} document(s)")

        chunks = split_documents(loaded.documents, self._settings)
        if not chunks:
            raise IndexingError(
                "Document chunking produced no chunks; check the input contents "
                "and chunk settings."
            )
        _report(progress, f"Created {len(chunks)} chunk(s)")

        try:
            client = self._embedding_client or create_embedding_client(self._settings)
            embedded_chunks = embed_chunks(
                chunks,
                self._settings,
                client=client,
            )
        except Exception as exc:
            raise IndexingError(f"Embedding failed: {exc}") from exc
        if len(embedded_chunks) != len(chunks):
            raise IndexingError(
                "Embedding failed: vector count does not match the chunk count."
            )
        _report(progress, f"Embedded {len(embedded_chunks)} vector(s)")

        try:
            vector_store = build_vector_store(embedded_chunks)
        except VectorStoreError as exc:
            raise IndexingError(f"FAISS index construction failed: {exc}") from exc

        built_at = self._clock().astimezone(timezone.utc)
        manifest = _build_manifest(
            settings=self._settings,
            input_path=input_path,
            document_count=len(loaded.documents),
            chunk_count=len(chunks),
            vector_store=vector_store,
            built_at=built_at,
        )
        _publish_index(
            vector_store,
            manifest,
            output_path,
            overwrite=overwrite,
        )
        _report(progress, f"Saved index to {output_path}")

        return IndexingResult(
            loaded_document_count=len(loaded.documents),
            chunk_count=len(chunks),
            embedded_vector_count=len(embedded_chunks),
            embedding_dimension=vector_store.dimension,
            output_path=output_path,
            embedding_provider=self._settings.embedding_provider,
            embedding_model=self._settings.selected_embedding_model,
            elapsed_seconds=max(0.0, self._monotonic() - started),
        )


def load_index_manifest(directory: str | Path) -> dict[str, Any] | None:
    """Load a Milestone 8 manifest, or return None for a legacy index."""
    path = Path(directory) / INDEX_MANIFEST_FILE_NAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise IndexingError(f"Index manifest is not valid UTF-8: '{path}'.") from exc
    except json.JSONDecodeError as exc:
        raise IndexingError(
            "Index manifest is invalid JSON at "
            f"line {exc.lineno}, column {exc.colno}: '{path}'."
        ) from exc
    return _validate_manifest(raw, path)


def validate_index_compatibility(
    directory: str | Path,
    settings: Settings,
) -> None:
    """Reject a new-format index built in a different embedding vector space."""
    manifest = load_index_manifest(directory)
    if manifest is None:
        return
    expected = (
        settings.embedding_provider,
        settings.selected_embedding_model,
    )
    actual = (
        manifest["embedding_provider"],
        manifest["embedding_model"],
    )
    if actual != expected:
        raise IndexingError(
            "Saved index embedding configuration is incompatible with the "
            "current query configuration: "
            f"index uses provider={actual[0]!r}, model={actual[1]!r}; "
            f"current settings use provider={expected[0]!r}, model={expected[1]!r}."
        )


def _validate_paths(
    input_directory: Path,
    output_directory: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    input_path = Path(input_directory).expanduser().resolve()
    output_path = Path(output_directory).expanduser().resolve()
    if not input_path.exists():
        raise IndexingError(f"Input directory does not exist: '{input_path}'.")
    if not input_path.is_dir():
        raise IndexingError(f"Input path is not a directory: '{input_path}'.")
    if input_path == output_path:
        raise IndexingError("Input and output directories must be different.")
    if input_path in output_path.parents or output_path in input_path.parents:
        raise IndexingError(
            "Input and output directories must not contain one another."
        )
    if output_path.exists() and not output_path.is_dir():
        raise IndexingError(f"Output path exists and is not a directory: '{output_path}'.")
    if output_path.exists() and not overwrite:
        raise IndexingError(
            f"Output directory already exists: '{output_path}'. "
            "Use --overwrite to replace it."
        )
    return input_path, output_path


def _build_manifest(
    *,
    settings: Settings,
    input_path: Path,
    document_count: int,
    chunk_count: int,
    vector_store: FaissVectorStore,
    built_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
        "built_at": built_at.isoformat().replace("+00:00", "Z"),
        "source_identifier": input_path.name,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.selected_embedding_model,
        "embedding_dimension": vector_store.dimension,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "document_count": document_count,
        "chunk_count": chunk_count,
    }


def _validate_manifest(value: object, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IndexingError(f"Index manifest root must be a JSON object: '{path}'.")
    required_strings = (
        "built_at",
        "source_identifier",
        "embedding_provider",
        "embedding_model",
    )
    required_positive_ints = (
        "embedding_dimension",
        "chunk_size",
        "document_count",
        "chunk_count",
    )
    if value.get("schema_version") != INDEX_MANIFEST_SCHEMA_VERSION:
        raise IndexingError(
            "Index manifest schema is incompatible: "
            f"expected {INDEX_MANIFEST_SCHEMA_VERSION}, "
            f"received {value.get('schema_version')!r}."
        )
    for name in required_strings:
        if not isinstance(value.get(name), str) or not value[name].strip():
            raise IndexingError(f"Index manifest field {name!r} must be a non-empty string.")
    for name in required_positive_ints:
        field_value = value.get(name)
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value < 1
        ):
            raise IndexingError(
                f"Index manifest field {name!r} must be a positive integer."
            )
    overlap = value.get("chunk_overlap")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise IndexingError(
            "Index manifest field 'chunk_overlap' must be a non-negative integer."
        )
    if value["chunk_overlap"] >= value["chunk_size"]:
        raise IndexingError(
            "Index manifest chunk_overlap must be smaller than chunk_size."
        )
    return value


def _publish_index(
    vector_store: FaissVectorStore,
    manifest: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    parent = output_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=parent)
        ).resolve()
    except OSError as exc:
        raise IndexingError(
            f"Output directory is not writable: '{parent}'."
        ) from exc
    backup: Path | None = None
    try:
        vector_store.save(temporary)
        manifest_path = temporary / INDEX_MANIFEST_FILE_NAME
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        FaissVectorStore.load(temporary)
        _validate_manifest(manifest, manifest_path)

        if output_path.exists():
            if not overwrite:
                raise IndexingError(
                    f"Output directory already exists: '{output_path}'. "
                    "Use --overwrite to replace it."
                )
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output_path.name}.backup-", dir=parent)
            ).resolve()
            backup.rmdir()
            os.replace(output_path, backup)
        try:
            os.replace(temporary, output_path)
        except OSError:
            if backup is not None and backup.exists() and not output_path.exists():
                os.replace(backup, output_path)
                backup = None
            raise
        if backup is not None:
            _remove_safe_directory(backup, parent)
            backup = None
    except IndexingError:
        raise
    except (OSError, VectorStoreError, TypeError, ValueError) as exc:
        raise IndexingError(
            f"Could not safely save the index to '{output_path}'."
        ) from exc
    finally:
        if temporary.exists():
            _remove_safe_directory(temporary, parent)
        if backup is not None and backup.exists():
            if not output_path.exists():
                os.replace(backup, output_path)
            else:
                _remove_safe_directory(backup, parent)


def _remove_safe_directory(path: Path, expected_parent: Path) -> None:
    resolved = path.resolve()
    parent = expected_parent.resolve()
    if resolved.parent != parent or not (
        resolved.name.startswith(".") and
        (".tmp-" in resolved.name or ".backup-" in resolved.name)
    ):
        raise IndexingError(f"Refusing to remove unsafe temporary path: '{resolved}'.")
    shutil.rmtree(resolved)


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
