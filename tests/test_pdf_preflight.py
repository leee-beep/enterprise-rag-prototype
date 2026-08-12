"""Offline tests for the read-only competitor PDF preflight command."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.preflight_competitor_pdfs import main
from tests.test_pdf_loader import write_pdf


def manifest_entry(file_name: str = "gigabyte/report.pdf") -> dict[str, object]:
    return {
        "file": file_name,
        "company_id": "gigabyte",
        "company_name": "Gigabyte",
        "ticker": "2376",
        "fiscal_year": 2025,
        "period": "FY",
        "document_type": "annual_report",
        "title": "Gigabyte 2025 Annual Report",
        "language": "en",
        "source_url": "https://example.com/gigabyte-2025.pdf",
        "source_relative_path": "gigabyte/2025/annual-report.pdf",
    }


def test_preflight_reports_quality_without_writing_outputs(
    tmp_path: Path, capsys
) -> None:
    sources = tmp_path / "sources"
    pdf = sources / "gigabyte" / "report.pdf"
    pdf.parent.mkdir(parents=True)
    write_pdf(pdf, ["Annual report content with enough extractable text."])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([manifest_entry()]), encoding="utf-8")

    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    exit_code = main(
        ["--source-root", str(sources), "--manifest", str(manifest)]
    )
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "PASS gigabyte/report.pdf" in output
    assert "passed=1, failed=0" in output
    assert after == before


def test_preflight_reports_invalid_pdf_and_returns_two(tmp_path: Path, capsys) -> None:
    sources = tmp_path / "sources"
    pdf = sources / "gigabyte" / "report.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"not a PDF")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([manifest_entry()]), encoding="utf-8")

    assert main(["--source-root", str(sources), "--manifest", str(manifest)]) == 2
    output = capsys.readouterr().out
    assert "FAIL entry 1" in output
    assert "passed=0, failed=1" in output
    assert str(tmp_path) not in output


def test_preflight_rejects_parent_traversal(tmp_path: Path, capsys) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([manifest_entry("../outside.pdf")]), encoding="utf-8"
    )

    assert main(["--source-root", str(sources), "--manifest", str(manifest)]) == 2
    assert "safe relative path" in capsys.readouterr().out
