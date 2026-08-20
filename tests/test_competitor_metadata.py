"""Offline tests for validated competitor document metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from enterprise_rag.competitor_metadata import (
    CompetitorDocumentMetadata,
    CompetitorMetadataError,
)


COMPANIES = (
    ("gigabyte", "Gigabyte", "2376"),
    ("asus", "ASUS", "2357"),
    ("msi", "MSI", "2377"),
)

EXTENDED_DOCUMENT_TYPES = (
    "earnings_release",
    "investor_presentation",
    "official_press_release",
    "official_product_document",
    "sustainability_report",
)


def metadata_for(
    path: Path,
    *,
    company_id: str = "gigabyte",
    company_name: str = "Gigabyte",
    ticker: str = "2376",
    fiscal_year: int = 2025,
    document_type: str = "annual_report",
    source_relative_path: str = "gigabyte/2025/annual-report.pdf",
) -> CompetitorDocumentMetadata:
    return CompetitorDocumentMetadata.from_pdf(
        path,
        company_id=company_id,
        company_name=company_name,
        ticker=ticker,
        fiscal_year=fiscal_year,
        period="FY",
        document_type=document_type,
        title=f"{company_name} {fiscal_year} report",
        language="en",
        source_url="https://example.com/public-report.pdf",
        source_relative_path=source_relative_path,
    )


@pytest.mark.parametrize("company_id,company_name,ticker", COMPANIES)
def test_valid_company_metadata(
    tmp_path: Path, company_id: str, company_name: str, ticker: str
) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"synthetic-pdf-bytes")

    metadata = metadata_for(
        pdf,
        company_id=company_id,
        company_name=company_name,
        ticker=ticker,
        source_relative_path=f"{company_id}/2025/report.pdf",
    )

    assert metadata.company_id == company_id
    assert metadata.ticker == ticker
    assert metadata.source_document_id.startswith(
        f"competitor:{company_id}:annual_report:2025:"
    )


def test_invalid_ticker_mapping_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"bytes")
    with pytest.raises(CompetitorMetadataError, match="ticker does not match"):
        metadata_for(source, ticker="2357")


def test_invalid_company_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"bytes")
    with pytest.raises(CompetitorMetadataError, match="company_id"):
        metadata_for(source, company_id="unknown")


def test_invalid_fiscal_year_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"bytes")
    with pytest.raises(CompetitorMetadataError, match="2024 or 2025"):
        metadata_for(source, fiscal_year=2023)


def test_invalid_document_type_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"bytes")
    with pytest.raises(CompetitorMetadataError, match="document_type"):
        metadata_for(source, document_type="presentation")


@pytest.mark.parametrize("document_type", EXTENDED_DOCUMENT_TYPES)
def test_controlled_official_document_types_are_accepted(
    tmp_path: Path, document_type: str
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"synthetic-official-source")

    metadata = metadata_for(source, document_type=document_type)

    assert metadata.document_type == document_type
    assert f":{document_type}:" in metadata.source_document_id


def test_source_identity_is_stable_and_changes_with_file_bytes(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"version-one")
    first = metadata_for(source)
    second = metadata_for(source)
    source.write_bytes(b"version-two")
    changed = metadata_for(source)

    assert first.source_document_id == second.source_document_id
    assert first.source_sha256 == second.source_sha256
    assert changed.source_document_id != first.source_document_id
    assert changed.source_sha256 != first.source_sha256


def test_source_identity_does_not_leak_absolute_path(tmp_path: Path) -> None:
    source = tmp_path / "private-user-directory" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"stable")
    metadata = metadata_for(source)

    assert str(tmp_path) not in metadata.source_document_id
    assert "private-user-directory" not in metadata.source_document_id
    assert metadata.source_relative_path == "gigabyte/2025/annual-report.pdf"


@pytest.mark.parametrize(
    "relative",
    ["../secret/report.pdf", "C:/Users/example/report.pdf", "/private/report.pdf"],
)
def test_unsafe_source_relative_path_is_rejected(
    tmp_path: Path, relative: str
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"stable")
    with pytest.raises(CompetitorMetadataError, match="safe relative path"):
        metadata_for(source, source_relative_path=relative)
