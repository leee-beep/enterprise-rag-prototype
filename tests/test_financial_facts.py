"""Offline tests for deterministic structured financial facts."""

from __future__ import annotations

import csv
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from enterprise_rag.financial_facts import (
    CSV_COLUMNS,
    FinancialFact,
    FinancialFactCollection,
    FinancialFactConflictError,
    FinancialFactNotFoundError,
    FinancialFactValidationError,
    load_financial_facts_csv,
    normalize_to_twd,
)


FIXTURE = Path(__file__).parent / "fixtures" / "financial_facts_synthetic.csv"


def fact(**changes: object) -> FinancialFact:
    values = {
        "company_id": "gigabyte",
        "company_name": "Gigabyte",
        "ticker": "2376",
        "fiscal_year": 2025,
        "period": "FY",
        "metric": "revenue",
        "value": Decimal("123.45"),
        "currency": "TWD",
        "unit": "million_TWD",
        "reporting_scope": "consolidated",
        "document_type": "consolidated_financial_report",
        "source_document_id": "synthetic:gigabyte:2025",
        "source_title": "Synthetic Gigabyte Financial Report",
        "page_number": 12,
    }
    values.update(changes)
    return FinancialFact(**values)


@pytest.mark.parametrize(
    "company_id,company_name,ticker", (
        ("gigabyte", "Gigabyte", "2376"),
        ("asus", "ASUS", "2357"),
        ("msi", "MSI", "2377"),
    )
)
def test_valid_fact_and_supported_companies(company_id: str, company_name: str, ticker: str) -> None:
    value = fact(company_id=company_id, company_name=company_name, ticker=ticker)
    assert value.company_id == company_id
    assert value.canonical_value_twd == Decimal("123450000.00")


@pytest.mark.parametrize("value", (Decimal("10"), Decimal("0"), Decimal("-10")))
def test_positive_zero_and_negative_values_are_valid(value: Decimal) -> None:
    assert fact(value=value).value == value


def test_decimal_eps_precision() -> None:
    value = fact(metric="eps", value="-1.23456789", unit="TWD")
    assert value.value == Decimal("-1.23456789")
    assert value.canonical_value_twd == Decimal("-1.23456789")


@pytest.mark.parametrize(
    "changes,match", (
        ({"company_id": "other"}, "company_id"),
        ({"company_name": "Wrong"}, "inconsistent"),
        ({"ticker": "0000"}, "inconsistent"),
        ({"fiscal_year": 2023}, "2024 or 2025"),
        ({"period": "Q4"}, "period"),
        ({"metric": "assets"}, "metric"),
        ({"currency": "USD"}, "currency"),
        ({"unit": "dollars"}, "unit"),
        ({"reporting_scope": "standalone"}, "reporting_scope"),
        ({"document_type": "annual_report"}, "document_type"),
        ({"page_number": 0}, "page_number"),
        ({"source_document_id": ""}, "source_document_id"),
        ({"source_document_id": "C:\\private\\report.pdf"}, "not a local path"),
        ({"source_title": "C:\\private\\report.pdf"}, "safe display title"),
        ({"value": "NaN"}, "finite"),
        ({"value": "Infinity"}, "finite"),
        ({"value": "not-a-number"}, "finite"),
        ({"metric": "eps", "unit": "thousand_TWD"}, "eps must use"),
    )
)
def test_model_validation(changes: dict[str, object], match: str) -> None:
    with pytest.raises(FinancialFactValidationError, match=match):
        fact(**changes)


@pytest.mark.parametrize(
    "value,unit,expected", (
        ("123.45", "TWD", Decimal("123.45")),
        ("123.45", "thousand_TWD", Decimal("123450.00")),
        ("123.45", "million_TWD", Decimal("123450000.00")),
        ("-2.5", "million_TWD", Decimal("-2500000.0")),
    )
)
def test_exact_twd_normalization(value: str, unit: str, expected: Decimal) -> None:
    assert normalize_to_twd(value, unit) == expected


def test_loader_reads_synthetic_csv_and_orders_deterministically() -> None:
    dataset = load_financial_facts_csv(FIXTURE)
    assert len(dataset) == 4
    assert [item.logical_key for item in dataset] == sorted(
        item.logical_key for item in dataset
    )
    assert dataset.get_fact("gigabyte", 2024, "revenue").canonical_value_twd == Decimal("123456000")
    assert dataset.get_fact("msi", 2025, "eps").value == Decimal("12.3456")


def test_lookup_and_filters() -> None:
    dataset = load_financial_facts_csv(FIXTURE)
    assert dataset.get_fact("asus", 2025, "operating_income").value < 0
    assert len(dataset.list_company_facts("gigabyte")) == 2
    assert len(dataset.list_company_facts("gigabyte", 2024)) == 1
    assert len(dataset.list_metric_facts("revenue")) == 2
    with pytest.raises(FinancialFactNotFoundError, match="not found"):
        dataset.get_fact("asus", 2024, "net_income")


def test_exact_normalized_duplicate_is_deduplicated() -> None:
    first = fact(value="1000", unit="thousand_TWD")
    same = replace(first, value=Decimal("1"), unit="million_TWD", page_number=13)
    dataset = FinancialFactCollection((first, same))
    assert len(dataset) == 1
    assert dataset.facts[0] is first


def test_conflicting_fact_is_rejected() -> None:
    first = fact(value="1", unit="million_TWD")
    conflicting = replace(first, value=Decimal("2"), page_number=13)
    with pytest.raises(FinancialFactConflictError, match="Conflicting values"):
        FinancialFactCollection((first, conflicting))


def write_rows(path: Path, rows: list[dict[str, str]], columns=CSV_COLUMNS) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def base_row() -> dict[str, str]:
    return {
        "company_id": "gigabyte",
        "company_name": "Gigabyte",
        "ticker": "2376",
        "fiscal_year": "2025",
        "period": "FY",
        "metric": "revenue",
        "value": "1000",
        "currency": "TWD",
        "unit": "thousand_TWD",
        "reporting_scope": "consolidated",
        "document_type": "consolidated_financial_report",
        "source_document_id": "synthetic:gigabyte:2025",
        "source_title": "Synthetic report",
        "page_number": "8",
        "source_label": "Revenue",
        "source_sha256": "",
        "notes": "Fictional",
    }


def test_loader_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "facts.csv"
    columns = tuple(name for name in CSV_COLUMNS if name != "metric")
    write_rows(path, [base_row()], columns)
    with pytest.raises(FinancialFactValidationError, match="missing columns: metric"):
        load_financial_facts_csv(path)


@pytest.mark.parametrize(
    "column,value,match", (
        ("value", "broken", "finite decimal"),
        ("metric", "assets", "metric"),
        ("fiscal_year", "2025.0", "integer"),
    )
)
def test_loader_reports_invalid_rows_without_path_leakage(
    tmp_path: Path, column: str, value: str, match: str
) -> None:
    private = tmp_path / "private-user-directory"
    private.mkdir()
    path = private / "facts.csv"
    row = base_row()
    row[column] = value
    write_rows(path, [row])
    with pytest.raises(FinancialFactValidationError, match=match) as caught:
        load_financial_facts_csv(path)
    assert str(tmp_path) not in str(caught.value)
    assert "private-user-directory" not in str(caught.value)


def test_loader_detects_duplicate_and_conflict(tmp_path: Path) -> None:
    path = tmp_path / "facts.csv"
    first = base_row()
    duplicate = dict(first, value="1", unit="million_TWD", page_number="9")
    write_rows(path, [first, duplicate])
    assert len(load_financial_facts_csv(path)) == 1

    conflict = dict(first, value="2", unit="million_TWD", page_number="10")
    write_rows(path, [first, conflict])
    with pytest.raises(FinancialFactConflictError, match="Conflicting values"):
        load_financial_facts_csv(path)


def test_fixture_contains_only_synthetic_provenance() -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    assert "Synthetic" in text
    assert "Fictional" in text
    assert "data/private" not in text
    assert "C:\\Users" not in text
