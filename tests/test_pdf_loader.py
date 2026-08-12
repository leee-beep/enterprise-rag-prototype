"""Offline tests for page-aware competitor PDF extraction."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.errors import DependencyError
from pypdf.generic import DictionaryObject, NameObject, StreamObject

from enterprise_rag.chunking import split_documents
from enterprise_rag.competitor_metadata import CompetitorDocumentMetadata
from enterprise_rag.config import Settings
from enterprise_rag.pdf_loader import (
    PDFEncryptedError,
    PDFMalformedError,
    PDFMetadataMismatchError,
    PDFPathError,
    PDFQualityError,
    PDFTextExtractionError,
    load_competitor_pdf,
)


def write_pdf(path: Path, pages: list[str | None]) -> None:
    """Generate a tiny text-layer PDF without an additional fixture dependency."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_ref}
                )
            }
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = StreamObject()
        stream.set_data(
            f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
        )
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


def metadata(path: Path) -> CompetitorDocumentMetadata:
    return CompetitorDocumentMetadata.from_pdf(
        path,
        company_id="gigabyte",
        company_name="Gigabyte",
        ticker="2376",
        fiscal_year=2025,
        period="FY",
        document_type="annual_report",
        title="Gigabyte 2025 Annual Report",
        language="en",
        source_url="https://example.com/gigabyte-2025.pdf",
        source_relative_path="gigabyte/2025/annual-report.pdf",
    )


def settings(tmp_path: Path) -> Settings:
    return Settings(
        gemini_api_key=None,
        generation_model="unused",
        embedding_model="unused",
        documents_dir=tmp_path,
        vector_store_dir=tmp_path / "index",
        chunk_size=45,
        chunk_overlap=8,
        top_k=4,
    )


def test_two_page_pdf_produces_page_documents(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(
        source,
        [
            "Annual Report 2025 - Revenue overview and market position.",
            "Strategy - AI servers and enterprise growth initiatives.",
        ],
    )

    result = load_competitor_pdf(source, metadata(source))

    assert len(result.documents) == 2
    assert [doc.metadata["page_number"] for doc in result.documents] == [1, 2]
    assert all(doc.metadata["page_count"] == 2 for doc in result.documents)
    assert "Revenue overview" in result.documents[0].content
    assert "AI servers" in result.documents[1].content
    assert result.documents[0].document_id.endswith(":page-0001")
    assert result.documents[1].document_id.endswith(":page-0002")


def test_empty_page_is_reported_but_not_loaded(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, ["Meaningful annual report content for extraction.", None])

    result = load_competitor_pdf(source, metadata(source))

    assert len(result.documents) == 1
    assert result.skipped_empty_pages == (2,)
    assert result.quality.page_count == 2
    assert result.quality.pages_with_text == 1
    assert result.quality.empty_page_count == 1
    assert result.quality.extractable_page_ratio == pytest.approx(0.5)
    assert result.quality.warnings


def test_quality_statistics_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, ["A sufficiently long first financial report page.", None])
    first = load_competitor_pdf(source, metadata(source)).quality
    second = load_competitor_pdf(source, metadata(source)).quality

    assert first == second
    assert first.total_characters >= 20
    assert first.average_characters_per_nonempty_page == pytest.approx(
        first.total_characters
    )
    assert first.replacement_character_count == 0
    assert first.replacement_character_ratio == 0.0


def test_missing_and_wrong_extension_are_clear(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    placeholder = tmp_path / "placeholder.pdf"
    placeholder.write_bytes(b"metadata source")
    pdf_metadata = metadata(placeholder)
    with pytest.raises(PDFPathError, match="missing.pdf") as error:
        load_competitor_pdf(missing, pdf_metadata)
    assert str(tmp_path) not in str(error.value)

    text = tmp_path / "report.txt"
    text.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(PDFPathError, match="expected .pdf"):
        load_competitor_pdf(text, pdf_metadata)


def test_corrupt_pdf_is_rejected_without_pdf_internals(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.pdf"
    source.write_bytes(b"not a valid PDF")
    with pytest.raises(PDFMalformedError, match="malformed or unreadable") as error:
        load_competitor_pdf(source, metadata(source))
    assert str(tmp_path) not in str(error.value)


def test_textless_pdf_is_rejected_with_quality_report(tmp_path: Path) -> None:
    source = tmp_path / "scanned.pdf"
    write_pdf(source, [None, None])
    with pytest.raises(PDFQualityError, match="no pages") as error:
        load_competitor_pdf(source, metadata(source))

    assert error.value.report.pages_with_text == 0
    assert error.value.report.empty_page_count == 2
    assert error.value.report.extractable_page_ratio == 0.0


def test_extractable_page_ratio_below_threshold_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "mostly-scanned.pdf"
    write_pdf(
        source,
        ["One meaningful text page with enough characters.", None, None, None, None, None],
    )
    with pytest.raises(PDFQualityError, match="below 20%") as error:
        load_competitor_pdf(source, metadata(source))

    assert error.value.report.extractable_page_ratio == pytest.approx(1 / 6)


def test_encrypted_pdf_decryptable_with_empty_password_is_loaded(tmp_path: Path) -> None:
    source = tmp_path / "encrypted.pdf"
    plain = tmp_path / "plain.pdf"
    write_pdf(plain, ["Encrypted annual report page with meaningful text."])
    writer = PdfWriter(clone_from=plain)
    writer.encrypt("", algorithm="AES-256")
    writer.write(source)

    result = load_competitor_pdf(source, metadata(source))

    assert len(result.documents) == 1
    document = result.documents[0]
    assert document.content == "Encrypted annual report page with meaningful text."
    assert document.metadata["company_id"] == "gigabyte"
    assert document.metadata["page_number"] == 1


def test_password_required_pdf_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "password-required.pdf"
    plain = tmp_path / "plain.pdf"
    write_pdf(plain, ["Protected annual report page with meaningful text."])
    writer = PdfWriter(clone_from=plain)
    writer.encrypt("secret-password", algorithm="AES-256")
    writer.write(source)

    with pytest.raises(PDFEncryptedError, match="non-empty password") as error:
        load_competitor_pdf(source, metadata(source))
    assert str(tmp_path) not in str(error.value)
    assert "secret-password" not in str(error.value)


def test_aes_dependency_error_is_domain_friendly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "aes.pdf"
    write_pdf(source, ["Annual report content long enough to pass extraction."])

    def fail_reader(*_args, **_kwargs):
        raise DependencyError("cryptography internal dependency detail")

    monkeypatch.setattr("enterprise_rag.pdf_loader.PdfReader", fail_reader)
    with pytest.raises(PDFEncryptedError, match="encryption support is unavailable") as error:
        load_competitor_pdf(source, metadata(source))

    assert str(tmp_path) not in str(error.value)
    assert "internal dependency detail" not in str(error.value)


def test_encrypted_readable_pdf_integrates_with_chunking(tmp_path: Path) -> None:
    source = tmp_path / "encrypted.pdf"
    plain = tmp_path / "plain.pdf"
    write_pdf(
        plain,
        [
            "Annual report heading. Revenue grew due to enterprise demand. "
            "Management emphasized AI servers and long-term product strategy."
        ],
    )
    writer = PdfWriter(clone_from=plain)
    writer.encrypt("", algorithm="AES-256")
    writer.write(source)

    loaded = load_competitor_pdf(source, metadata(source)).documents
    chunks = split_documents(loaded, settings(tmp_path))

    assert chunks
    assert all(chunk.metadata["page_number"] == 1 for chunk in chunks)
    assert all(chunk.metadata["ticker"] == "2376" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_zero_page_pdf_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "zero-page.pdf"
    writer = PdfWriter()
    with source.open("wb") as output:
        writer.write(output)

    with pytest.raises(PDFQualityError, match="no pages") as error:
        load_competitor_pdf(source, metadata(source))
    assert error.value.report.page_count == 0


def test_page_extraction_exception_is_domain_friendly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, ["Original text long enough for extraction."])

    def fail_extract(_page):
        raise RuntimeError("internal parser detail")

    monkeypatch.setattr(PdfReader(source).pages[0].__class__, "extract_text", fail_extract)
    with pytest.raises(PDFTextExtractionError, match="page 1 of report.pdf") as error:
        load_competitor_pdf(source, metadata(source))
    assert "internal parser detail" not in str(error.value)


def test_unicode_text_is_preserved_at_loader_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "unicode.pdf"
    write_pdf(source, ["Placeholder text long enough for extraction."])
    chinese = "年度報告營收成長，管理層持續投入人工智慧伺服器與企業市場。"
    monkeypatch.setattr(
        PdfReader(source).pages[0].__class__, "extract_text", lambda _page: chinese
    )

    result = load_competitor_pdf(source, metadata(source))

    assert result.documents[0].content == chinese
    assert result.quality.replacement_character_count == 0


def test_changed_file_is_rejected_against_stale_metadata(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, ["Original annual report content long enough to extract."])
    original_metadata = metadata(source)
    with source.open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(PDFMetadataMismatchError, match="no longer match"):
        load_competitor_pdf(source, original_metadata)


def test_safe_relative_source_and_no_absolute_metadata(tmp_path: Path) -> None:
    source = tmp_path / "private-user-folder" / "report.pdf"
    source.parent.mkdir()
    write_pdf(source, ["Public report text with enough meaningful characters."])
    result = load_competitor_pdf(source, metadata(source))
    document = result.documents[0]

    assert document.source == "gigabyte/2025/annual-report.pdf"
    assert document.file_name == "annual-report.pdf"
    assert document.file_type == ".pdf"
    assert str(tmp_path) not in repr(document)
    assert all(str(tmp_path) not in str(value) for value in document.metadata.values())


def test_pdf_pages_integrate_with_structure_aware_chunking(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(
        source,
        [
            "Annual report heading. Revenue grew due to enterprise demand. "
            "Management emphasized AI servers and long-term product strategy."
        ],
    )
    loaded = load_competitor_pdf(source, metadata(source)).documents

    first = split_documents(loaded, settings(tmp_path))
    second = split_documents(loaded, settings(tmp_path))

    assert first
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    for chunk in first:
        assert chunk.metadata["company_id"] == "gigabyte"
        assert chunk.metadata["ticker"] == "2376"
        assert chunk.metadata["fiscal_year"] == 2025
        assert chunk.metadata["document_type"] == "annual_report"
        assert chunk.metadata["page_number"] == 1
        assert chunk.metadata["source_document_id"].startswith("competitor:gigabyte:")


def test_replacement_character_quality_rule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    write_pdf(source, ["Original page content long enough for the fixture."])
    pdf_metadata = metadata(source)

    monkeypatch.setattr(
        PdfReader(source).pages[0].__class__,
        "extract_text",
        lambda self: "Useful text " + "\ufffd" * 20,
    )
    with pytest.raises(PDFQualityError, match="replacement-character") as error:
        load_competitor_pdf(source, pdf_metadata)
    assert error.value.report.replacement_character_count == 20
    assert error.value.report.replacement_character_ratio > 0.05
