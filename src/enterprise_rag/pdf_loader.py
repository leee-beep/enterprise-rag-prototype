"""Local, page-aware PDF extraction for validated competitor documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError, WrongPasswordError

from enterprise_rag.competitor_metadata import (
    CompetitorDocumentMetadata,
    sha256_file,
)
from enterprise_rag.models import LoadedDocument


MIN_EXTRACTABLE_PAGE_RATIO = 0.20
MIN_TOTAL_CHARACTERS = 20
MAX_REPLACEMENT_CHARACTER_RATIO = 0.05


class PDFLoadingError(RuntimeError):
    """Base error for competitor PDF loading."""


class PDFPathError(PDFLoadingError):
    """Raised when a PDF path is missing, invalid, or unsupported."""


class PDFEncryptedError(PDFLoadingError):
    """Raised when a PDF requires decryption."""


class PDFMalformedError(PDFLoadingError):
    """Raised when pypdf cannot parse a PDF."""


class PDFTextExtractionError(PDFLoadingError):
    """Raised when text extraction fails for a specific page."""


class PDFMetadataMismatchError(PDFLoadingError):
    """Raised when validated metadata does not describe the current file bytes."""


class PDFQualityError(PDFLoadingError):
    """Raised when extracted text is clearly insufficient for RAG ingestion."""

    def __init__(self, message: str, report: "PDFExtractionQualityReport") -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class PDFExtractionQualityReport:
    page_count: int
    pages_with_text: int
    empty_page_count: int
    extractable_page_ratio: float
    total_characters: int
    average_characters_per_nonempty_page: float
    replacement_character_count: int
    replacement_character_ratio: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PDFLoadResult:
    documents: tuple[LoadedDocument, ...]
    skipped_empty_pages: tuple[int, ...]
    quality: PDFExtractionQualityReport


def load_competitor_pdf(
    pdf_path: str | Path,
    metadata: CompetitorDocumentMetadata,
) -> PDFLoadResult:
    """Extract meaningful pages while preserving safe competitor provenance."""
    path = _validate_pdf_path(pdf_path)
    current_digest = sha256_file(path)
    if current_digest != metadata.source_sha256:
        raise PDFMetadataMismatchError(
            f"PDF bytes no longer match validated metadata for {path.name}."
        )

    reader = _open_reader(path)
    try:
        pages = tuple(reader.pages)
    except (PdfReadError, OSError, ValueError) as exc:
        raise PDFMalformedError(f"PDF page tree is unreadable: {path.name}.") from exc
    page_count = len(pages)
    if page_count == 0:
        report = _quality_report((), page_count=0)
        raise PDFQualityError(f"PDF has no pages: {path.name}.", report)

    page_texts: list[str] = []
    for page_number, page in enumerate(pages, start=1):
        try:
            extracted = page.extract_text()
        except Exception as exc:
            raise PDFTextExtractionError(
                f"Could not extract text from page {page_number} of {path.name}."
            ) from exc
        page_texts.append(extracted or "")

    report = _quality_report(tuple(page_texts), page_count=page_count)
    _validate_quality(report, path.name)

    documents: list[LoadedDocument] = []
    skipped_empty: list[int] = []
    inherited = metadata.to_dict()
    file_name = PurePosixPath(metadata.source_relative_path).name
    for page_number, extracted in enumerate(page_texts, start=1):
        content = extracted.strip()
        if not content:
            skipped_empty.append(page_number)
            continue
        page_metadata = dict(inherited)
        page_metadata.update(
            {
                "page_number": page_number,
                "page_count": page_count,
                "source_document_id": metadata.source_document_id,
                "source_sha256": metadata.source_sha256,
            }
        )
        documents.append(
            LoadedDocument(
                content=content,
                source=metadata.source_relative_path,
                file_name=file_name,
                file_type=".pdf",
                document_id=f"{metadata.source_document_id}:page-{page_number:04d}",
                metadata=page_metadata,
            )
        )
    return PDFLoadResult(tuple(documents), tuple(skipped_empty), report)


def _validate_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)
    safe_name = path.name or "unnamed source"
    if not path.exists():
        raise PDFPathError(f"PDF file does not exist: {safe_name}.")
    if not path.is_file():
        raise PDFPathError(f"PDF path is not a file: {safe_name}.")
    if path.suffix.casefold() != ".pdf":
        raise PDFPathError(f"Unsupported file extension for {safe_name}; expected .pdf.")
    return path


def _open_reader(path: Path) -> PdfReader:
    try:
        reader = PdfReader(path, strict=False)
        if reader.is_encrypted:
            # The empty password is the only decryption attempt supported by
            # this MVP. No password guessing is attempted.
            if not reader.decrypt(""):
                raise PDFEncryptedError(
                    f"PDF requires a non-empty password and cannot be loaded: {path.name}."
                )
    except DependencyError as exc:
        raise PDFEncryptedError(
            f"PDF encryption support is unavailable for {path.name}; "
            "install the configured cryptography dependency."
        ) from exc
    except WrongPasswordError as exc:
        raise PDFEncryptedError(
            f"PDF requires a non-empty password and cannot be loaded: {path.name}."
        ) from exc
    except (PdfReadError, OSError, ValueError) as exc:
        raise PDFMalformedError(f"PDF is malformed or unreadable: {path.name}.") from exc
    return reader


def _quality_report(
    page_texts: tuple[str, ...],
    *,
    page_count: int,
) -> PDFExtractionQualityReport:
    normalized = tuple(text.strip() for text in page_texts)
    meaningful = tuple(text for text in normalized if text)
    pages_with_text = len(meaningful)
    empty_page_count = page_count - pages_with_text
    total_characters = sum(len(text) for text in normalized)
    replacement_count = sum(text.count("\ufffd") for text in normalized)
    page_ratio = pages_with_text / page_count if page_count else 0.0
    replacement_ratio = (
        replacement_count / total_characters if total_characters else 0.0
    )
    warnings: list[str] = []
    if empty_page_count:
        warnings.append(
            f"{empty_page_count} of {page_count} pages contained no extractable text."
        )
    if replacement_count:
        warnings.append(
            f"Extracted text contained {replacement_count} Unicode replacement characters."
        )
    return PDFExtractionQualityReport(
        page_count=page_count,
        pages_with_text=pages_with_text,
        empty_page_count=empty_page_count,
        extractable_page_ratio=page_ratio,
        total_characters=total_characters,
        average_characters_per_nonempty_page=(
            total_characters / pages_with_text if pages_with_text else 0.0
        ),
        replacement_character_count=replacement_count,
        replacement_character_ratio=replacement_ratio,
        warnings=tuple(warnings),
    )


def _validate_quality(report: PDFExtractionQualityReport, file_name: str) -> None:
    reasons: list[str] = []
    if report.pages_with_text == 0:
        reasons.append("no pages contained extractable text")
    if report.extractable_page_ratio < MIN_EXTRACTABLE_PAGE_RATIO:
        reasons.append(
            "extractable page ratio was below "
            f"{MIN_EXTRACTABLE_PAGE_RATIO:.0%}"
        )
    if report.total_characters < MIN_TOTAL_CHARACTERS:
        reasons.append(
            f"fewer than {MIN_TOTAL_CHARACTERS} characters were extracted"
        )
    if report.replacement_character_ratio > MAX_REPLACEMENT_CHARACTER_RATIO:
        reasons.append(
            "Unicode replacement-character ratio exceeded "
            f"{MAX_REPLACEMENT_CHARACTER_RATIO:.0%}"
        )
    if reasons:
        raise PDFQualityError(
            f"PDF text extraction quality is insufficient for {file_name}: "
            + "; ".join(reasons)
            + ". OCR fallback is not supported.",
            report,
        )
