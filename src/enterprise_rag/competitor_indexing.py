"""Private competitor-PDF indexing orchestration built from existing components."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from enterprise_rag.chunking import split_documents
from enterprise_rag.competitor_metadata import CompetitorDocumentMetadata
from enterprise_rag.config import Settings
from enterprise_rag.embeddings import EmbeddingClient, embed_chunks
from enterprise_rag.factory import create_embedding_client
from enterprise_rag.indexing import INDEX_MANIFEST_SCHEMA_VERSION, _publish_index
from enterprise_rag.pdf_loader import PDFLoadingError, load_competitor_pdf
from enterprise_rag.vector_store import VectorStoreError, build_vector_store


SUPPORTED_COMPANIES = ("gigabyte", "asus", "msi")


class CompetitorIndexingError(RuntimeError):
    """Raised when a private competitor index cannot be built safely."""


@dataclass(frozen=True)
class CompetitorIndexingResult:
    company_id: str
    loaded_document_count: int
    chunk_count: int
    embedded_vector_count: int
    embedding_dimension: int
    embedding_request_count: int
    output_path: Path
    elapsed_seconds: float
    manifest: dict[str, Any]


class CompetitorIndexingService:
    """Build one persisted FAISS index from one selected competitor PDF."""

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
        source_root: str | Path,
        manifest_path: str | Path,
        company_id: str,
        output_directory: str | Path,
        fiscal_year: int = 2025,
        document_type: str = "annual_report",
        overwrite: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> CompetitorIndexingResult:
        started = self._monotonic()
        company = _validate_company(company_id)
        root = Path(source_root).expanduser().resolve()
        if not root.is_dir():
            raise CompetitorIndexingError("Competitor source root is unavailable.")
        output = Path(output_directory).expanduser().resolve()
        if output.exists() and not overwrite:
            raise CompetitorIndexingError(
                f"Company index already exists for {company}; use overwrite to replace it."
            )
        entry = _select_entry(
            manifest_path, company, fiscal_year=fiscal_year, document_type=document_type
        )
        relative_file = _safe_pdf_path(entry.get("file"))
        source = (root / PurePosixPath(relative_file)).resolve()
        if not source.is_relative_to(root):
            raise CompetitorIndexingError("Manifest PDF path escapes the private source root.")
        metadata = _metadata_from_entry(source, entry)
        _report(progress, f"Loading {company} {fiscal_year} {document_type}")
        try:
            loaded = load_competitor_pdf(source, metadata)
        except PDFLoadingError as exc:
            raise CompetitorIndexingError(f"Competitor PDF loading failed: {exc}") from exc
        chunks = split_documents(loaded.documents, self._settings)
        if not chunks:
            raise CompetitorIndexingError("Competitor PDF produced no indexable chunks.")
        _report(progress, f"Created {len(chunks)} chunks from {len(loaded.documents)} pages")
        try:
            client = self._embedding_client or create_embedding_client(self._settings)
            embedded = embed_chunks(chunks, self._settings, client=client)
            store = build_vector_store(embedded)
        except Exception as exc:
            raise CompetitorIndexingError(f"Competitor embedding/index build failed: {exc}") from exc
        if len(embedded) != len(chunks):
            raise CompetitorIndexingError("Embedded vector count does not match chunk count.")
        request_count = (
            (len(chunks) + self._settings.ollama_embedding_batch_size - 1)
            // self._settings.ollama_embedding_batch_size
            if self._settings.embedding_provider == "ollama" else 1
        )
        manifest = {
            "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
            "built_at": self._clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_identifier": metadata.source_document_id,
            "embedding_provider": self._settings.embedding_provider,
            "embedding_model": self._settings.selected_embedding_model,
            "embedding_dimension": store.dimension,
            "chunk_size": self._settings.chunk_size,
            "chunk_overlap": self._settings.chunk_overlap,
            "document_count": len(loaded.documents),
            "page_count": loaded.quality.page_count,
            "chunk_count": len(chunks),
            "company_id": metadata.company_id,
            "company_name": metadata.company_name,
            "ticker": metadata.ticker,
            "fiscal_year": metadata.fiscal_year,
            "document_type": metadata.document_type,
            "source_document_id": metadata.source_document_id,
            "source_sha256": metadata.source_sha256,
        }
        try:
            _publish_index(store, manifest, output, overwrite=overwrite)
        except (OSError, VectorStoreError, ValueError) as exc:
            raise CompetitorIndexingError(f"Could not persist index for {company}.") from exc
        return CompetitorIndexingResult(
            company, len(loaded.documents), len(chunks), len(embedded), store.dimension,
            request_count, output, max(0.0, self._monotonic() - started), manifest,
        )


def _select_entry(path: str | Path, company: str, *, fiscal_year: int, document_type: str) -> dict[str, Any]:
    safe_name = Path(path).name or "source manifest"
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompetitorIndexingError(f"Could not read competitor source manifest: {safe_name}.") from exc
    if not isinstance(value, list):
        raise CompetitorIndexingError("Competitor source manifest must be a JSON array.")
    matches = [item for item in value if isinstance(item, dict) and item.get("company_id") == company and item.get("fiscal_year") == fiscal_year and item.get("document_type") == document_type]
    if len(matches) != 1:
        raise CompetitorIndexingError(
            f"Expected exactly one {company} {fiscal_year} {document_type} manifest entry; found {len(matches)}."
        )
    return matches[0]


def _metadata_from_entry(path: Path, entry: dict[str, Any]) -> CompetitorDocumentMetadata:
    required = ("company_id", "company_name", "ticker", "fiscal_year", "period", "document_type", "title", "language", "source_url", "source_relative_path")
    missing = [name for name in required if name not in entry]
    if missing:
        raise CompetitorIndexingError("Competitor manifest entry is missing required metadata: " + ", ".join(missing) + ".")
    try:
        return CompetitorDocumentMetadata.from_pdf(path, **{name: entry[name] for name in required})
    except (TypeError, ValueError) as exc:
        raise CompetitorIndexingError(f"Competitor metadata is invalid for {path.name}.") from exc


def _safe_pdf_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompetitorIndexingError("Competitor manifest entry requires a PDF file.")
    path = PurePosixPath(value.strip().replace("\\", "/"))
    if path.is_absolute() or path.anchor or ".." in path.parts or ":" in path.parts[0] or path.suffix.casefold() != ".pdf":
        raise CompetitorIndexingError("Competitor manifest file must be a safe relative PDF path.")
    return path.as_posix()


def _validate_company(value: str) -> str:
    company = value.strip().casefold() if isinstance(value, str) else ""
    if company not in SUPPORTED_COMPANIES:
        raise CompetitorIndexingError(f"Unknown competitor company: {value!r}.")
    return company


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
