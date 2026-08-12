"""Validated metadata and stable identity for competitor source documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


COMPANY_TICKERS = {
    "gigabyte": "2376",
    "asus": "2357",
    "msi": "2377",
}
ALLOWED_FISCAL_YEARS = frozenset({2024, 2025})
ALLOWED_PERIODS = frozenset({"FY"})
ALLOWED_DOCUMENT_TYPES = frozenset(
    {"annual_report", "consolidated_financial_report"}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_ID_HASH_LENGTH = 16


class CompetitorMetadataError(ValueError):
    """Raised when competitor source metadata is invalid or inconsistent."""


@dataclass(frozen=True)
class CompetitorDocumentMetadata:
    """The minimum validated metadata required for a competitor PDF."""

    company_id: str
    company_name: str
    ticker: str
    fiscal_year: int
    period: str
    document_type: str
    title: str
    language: str
    source_url: str
    source_relative_path: str
    source_document_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        company_id = _required_text("company_id", self.company_id).casefold()
        if company_id not in COMPANY_TICKERS:
            raise CompetitorMetadataError(
                "company_id must be one of: asus, gigabyte, msi."
            )
        ticker = _required_text("ticker", str(self.ticker))
        expected_ticker = COMPANY_TICKERS[company_id]
        if ticker != expected_ticker:
            raise CompetitorMetadataError(
                f"ticker does not match company_id {company_id!r}; "
                f"expected {expected_ticker!r}."
            )
        if isinstance(self.fiscal_year, bool) or self.fiscal_year not in ALLOWED_FISCAL_YEARS:
            raise CompetitorMetadataError("fiscal_year must be 2024 or 2025.")
        period = _required_text("period", self.period).upper()
        if period not in ALLOWED_PERIODS:
            raise CompetitorMetadataError("period must be 'FY'.")
        document_type = _required_text(
            "document_type", self.document_type
        ).casefold()
        if document_type not in ALLOWED_DOCUMENT_TYPES:
            allowed = ", ".join(sorted(ALLOWED_DOCUMENT_TYPES))
            raise CompetitorMetadataError(
                f"document_type must be one of: {allowed}."
            )
        source_relative_path = _safe_relative_pdf_path(self.source_relative_path)
        source_url = _validate_source_url(self.source_url)
        source_sha256 = _required_text("source_sha256", self.source_sha256).casefold()
        if not _SHA256_PATTERN.fullmatch(source_sha256):
            raise CompetitorMetadataError(
                "source_sha256 must be a 64-character lowercase hexadecimal digest."
            )
        expected_id = make_source_document_id(
            company_id, document_type, self.fiscal_year, source_sha256
        )
        if self.source_document_id != expected_id:
            raise CompetitorMetadataError(
                "source_document_id does not match the validated metadata and SHA-256."
            )

        object.__setattr__(self, "company_id", company_id)
        object.__setattr__(self, "company_name", _required_text("company_name", self.company_name))
        object.__setattr__(self, "ticker", ticker)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "document_type", document_type)
        object.__setattr__(self, "title", _required_text("title", self.title))
        object.__setattr__(self, "language", _required_text("language", self.language))
        object.__setattr__(self, "source_url", source_url)
        object.__setattr__(self, "source_relative_path", source_relative_path)
        object.__setattr__(self, "source_sha256", source_sha256)

    @classmethod
    def from_pdf(
        cls,
        pdf_path: str | Path,
        *,
        company_id: str,
        company_name: str,
        ticker: str,
        fiscal_year: int,
        period: str,
        document_type: str,
        title: str,
        language: str,
        source_url: str,
        source_relative_path: str,
    ) -> "CompetitorDocumentMetadata":
        """Build validated metadata using a SHA-256 of the PDF bytes."""
        digest = sha256_file(pdf_path)
        normalized_company = _required_text("company_id", company_id).casefold()
        normalized_type = _required_text("document_type", document_type).casefold()
        return cls(
            company_id=normalized_company,
            company_name=company_name,
            ticker=str(ticker),
            fiscal_year=fiscal_year,
            period=period,
            document_type=normalized_type,
            title=title,
            language=language,
            source_url=source_url,
            source_relative_path=source_relative_path,
            source_document_id=make_source_document_id(
                normalized_company, normalized_type, fiscal_year, digest
            ),
            source_sha256=digest,
        )

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable copy for LoadedDocument metadata."""
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    """Hash a local file without incorporating its machine-specific path."""
    candidate = Path(path)
    if not candidate.exists():
        raise CompetitorMetadataError(
            f"Source file does not exist: {candidate.name or 'unnamed source'}."
        )
    if not candidate.is_file():
        raise CompetitorMetadataError(
            f"Source path is not a file: {candidate.name or 'source directory'}."
        )
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(65_536), b""):
                digest.update(block)
    except OSError as exc:
        raise CompetitorMetadataError(
            f"Could not read source file: {candidate.name}."
        ) from exc
    return digest.hexdigest()


def make_source_document_id(
    company_id: str,
    document_type: str,
    fiscal_year: int,
    source_sha256: str,
) -> str:
    """Create a deterministic identity that never contains a local path."""
    return (
        f"competitor:{company_id}:{document_type}:{fiscal_year}:"
        f"{source_sha256[:_SOURCE_ID_HASH_LENGTH]}"
    )


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompetitorMetadataError(f"{name} must be a non-empty string.")
    return value.strip()


def _safe_relative_pdf_path(value: str) -> str:
    normalized = _required_text("source_relative_path", value).replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or path.anchor or ".." in path.parts or ":" in path.parts[0]:
        raise CompetitorMetadataError(
            "source_relative_path must be a safe relative path without parent traversal."
        )
    if path.suffix.casefold() != ".pdf":
        raise CompetitorMetadataError("source_relative_path must identify a PDF file.")
    return path.as_posix()


def _validate_source_url(value: str) -> str:
    url = _required_text("source_url", value)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CompetitorMetadataError(
            "source_url must be an absolute HTTP or HTTPS URL."
        )
    return url
