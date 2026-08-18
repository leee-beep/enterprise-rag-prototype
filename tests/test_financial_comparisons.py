"""Synthetic offline tests for deterministic financial comparisons."""

from __future__ import annotations

from decimal import Decimal

import pytest

from enterprise_rag.financial_calculations import (
    FinancialCalculationEngine,
    FinancialCalculationMissingFactError,
)
from enterprise_rag.financial_comparisons import (
    FinancialComparisonEngine,
    FinancialComparisonValidationError,
)
from enterprise_rag.financial_facts import FinancialFact, FinancialFactCollection


COMPANY_DETAILS = {
    "gigabyte": ("Gigabyte", "2376"),
    "asus": ("ASUS", "2357"),
    "msi": ("MSI", "2377"),
}


def fact(
    company_id: str,
    year: int,
    metric: str,
    value: str,
    *,
    unit: str = "million_TWD",
) -> FinancialFact:
    company_name, ticker = COMPANY_DETAILS[company_id]
    return FinancialFact(
        company_id=company_id,
        company_name=company_name,
        ticker=ticker,
        fiscal_year=year,
        period="FY",
        metric=metric,
        value=Decimal(value),
        currency="TWD",
        unit=unit,
        reporting_scope="consolidated",
        document_type="consolidated_financial_report",
        source_document_id=f"synthetic:{company_id}:{year}",
        source_title=f"Synthetic {company_name} {year} report",
        page_number=10 + year - 2024,
        source_sha256="a" * 64,
        notes="Fictional test value",
    )


def comparison(*facts: FinancialFact) -> FinancialComparisonEngine:
    calculations = FinancialCalculationEngine(FinancialFactCollection(facts))
    return FinancialComparisonEngine(calculations)


def margin_facts(
    company_id: str, year: int, margin: str, *, metric: str = "gross_profit"
) -> tuple[FinancialFact, FinancialFact]:
    return (
        fact(company_id, year, "revenue", "100"),
        fact(company_id, year, metric, margin),
    )


def test_three_company_descending_ranking() -> None:
    engine = comparison(
        *margin_facts("gigabyte", 2025, "30"),
        *margin_facts("asus", 2025, "50"),
        *margin_facts("msi", 2025, "40"),
    )
    result = engine.rank_companies(
        "gross_margin", 2025, ("gigabyte", "asus", "msi")
    )
    assert [entry.company_id for entry in result.ranked_entries] == ["asus", "msi", "gigabyte"]
    assert [entry.rank for entry in result.ranked_entries] == [1, 2, 3]
    assert result.status == "complete"
    assert result.ranking_direction == "higher_value_first"


def test_two_company_subset() -> None:
    engine = comparison(
        *margin_facts("gigabyte", 2025, "30"),
        *margin_facts("asus", 2025, "50"),
        *margin_facts("msi", 2025, "40"),
    )
    result = engine.rank_companies("gross_margin", 2025, ("msi", "gigabyte"))
    assert result.requested_companies == ("msi", "gigabyte")
    assert [entry.company_id for entry in result.ranked_entries] == ["msi", "gigabyte"]


def test_negative_values_rank_by_numeric_value() -> None:
    engine = comparison(
        *margin_facts("gigabyte", 2025, "-20", metric="net_income"),
        *margin_facts("asus", 2025, "-5", metric="net_income"),
        *margin_facts("msi", 2025, "-10", metric="net_income"),
    )
    result = engine.rank_companies("net_margin", 2025, ("gigabyte", "asus", "msi"))
    assert [entry.company_id for entry in result.ranked_entries] == ["asus", "msi", "gigabyte"]


def test_ties_use_competition_rank_and_requested_order() -> None:
    engine = comparison(
        *margin_facts("gigabyte", 2025, "50"),
        *margin_facts("asus", 2025, "40"),
        *margin_facts("msi", 2025, "40"),
    )
    result = engine.rank_companies("gross_margin", 2025, ("gigabyte", "msi", "asus"))
    assert [entry.rank for entry in result.ranked_entries] == [1, 2, 2]
    assert [entry.company_id for entry in result.ranked_entries] == ["gigabyte", "msi", "asus"]


@pytest.mark.parametrize(
    "companies,match",
    (
        ((), "non-empty tuple"),
        (("gigabyte", "gigabyte"), "duplicate"),
        (("unknown",), "one of"),
    ),
)
def test_invalid_company_selection(companies: tuple[str, ...], match: str) -> None:
    with pytest.raises(FinancialComparisonValidationError, match=match):
        comparison().rank_companies("gross_margin", 2025, companies)


def test_partial_comparison_preserves_missing_company() -> None:
    engine = comparison(
        *margin_facts("gigabyte", 2025, "30"),
        *margin_facts("asus", 2025, "50"),
    )
    result = engine.rank_companies("gross_margin", 2025, ("gigabyte", "asus", "msi"))
    assert result.status == "partial"
    assert result.missing_companies == ("msi",)
    assert len(result.ranked_entries) == 2


def test_all_missing_returns_explicit_insufficient_result() -> None:
    result = comparison().rank_companies(
        "gross_margin", 2025, ("gigabyte", "asus", "msi")
    )
    assert result.status == "insufficient"
    assert not result.ranked_entries
    assert result.missing_companies == ("gigabyte", "asus", "msi")


@pytest.mark.parametrize(
    "earlier,later,direction,change",
    (
        ("20", "25", "increase", Decimal("5.00")),
        ("25", "20", "decrease", Decimal("-5.00")),
        ("20", "20", "unchanged", Decimal("0.00")),
    ),
)
def test_margin_year_change(
    earlier: str, later: str, direction: str, change: Decimal
) -> None:
    engine = comparison(
        *margin_facts("gigabyte", 2024, earlier),
        *margin_facts("gigabyte", 2025, later),
    )
    result = engine.compare_company_years("gigabyte", "gross_margin", 2024, 2025)
    assert result.direction == direction
    assert result.percentage_point_change == change
    assert result.unit == "percentage_points"
    assert isinstance(result.percentage_point_change, Decimal)


def test_change_uses_exact_rounded_decimal_subtraction() -> None:
    engine = comparison(
        *margin_facts("gigabyte", 2024, "33.333"),
        *margin_facts("gigabyte", 2025, "33.338"),
    )
    result = engine.compare_company_years("gigabyte", "gross_margin", 2024, 2025)
    assert result.earlier_value == Decimal("33.33")
    assert result.later_value == Decimal("33.34")
    assert result.percentage_point_change == Decimal("0.01")


@pytest.mark.parametrize("missing_year", (2024, 2025))
def test_change_rejects_missing_year(missing_year: int) -> None:
    available_year = 2025 if missing_year == 2024 else 2024
    engine = comparison(*margin_facts("gigabyte", available_year, "20"))
    with pytest.raises(FinancialCalculationMissingFactError, match=str(missing_year)):
        engine.compare_company_years("gigabyte", "gross_margin", 2024, 2025)


def test_revenue_growth_year_change_is_explicitly_unsupported() -> None:
    with pytest.raises(FinancialComparisonValidationError, match="already a cross-year"):
        comparison().compare_company_years(
            "gigabyte", "revenue_yoy_growth", 2024, 2025
        )


def test_invalid_year_order_is_rejected() -> None:
    with pytest.raises(FinancialComparisonValidationError, match="earlier_year < later_year"):
        comparison().compare_company_years("gigabyte", "gross_margin", 2025, 2024)


@pytest.mark.parametrize("year", (2023, 2026, 2025.0, True))
def test_ranking_rejects_unsupported_fiscal_year(year: object) -> None:
    with pytest.raises(FinancialComparisonValidationError, match="2024 or 2025"):
        comparison().rank_companies("gross_margin", year, ("gigabyte",))


def test_change_rejects_non_string_metric_with_domain_error() -> None:
    with pytest.raises(FinancialComparisonValidationError, match="metric"):
        comparison().compare_company_years("gigabyte", None, 2024, 2025)


def test_ranking_preserves_calculation_provenance() -> None:
    result = comparison(*margin_facts("asus", 2025, "40")).rank_companies(
        "gross_margin", 2025, ("asus",)
    )
    entry = result.ranked_entries[0]
    assert entry.provenance is entry.calculation_result.provenance
    assert len(entry.provenance) == 2
    assert all(item.source_document_id.startswith("synthetic:") for item in entry.provenance)


def test_change_preserves_both_years_provenance() -> None:
    result = comparison(
        *margin_facts("msi", 2024, "20"),
        *margin_facts("msi", 2025, "25"),
    ).compare_company_years("msi", "gross_margin", 2024, 2025)
    assert len(result.provenance) == 4
    assert result.provenance == result.earlier_result.provenance + result.later_result.provenance
    assert {item.fiscal_year for item in result.provenance} == {2024, 2025}


def test_repeated_comparisons_are_deterministic() -> None:
    engine = comparison(
        *margin_facts("gigabyte", 2024, "20"),
        *margin_facts("gigabyte", 2025, "25"),
        *margin_facts("asus", 2025, "30"),
    )
    first_ranking = engine.rank_companies("gross_margin", 2025, ("gigabyte", "asus"))
    second_ranking = engine.rank_companies("gross_margin", 2025, ("gigabyte", "asus"))
    first_change = engine.compare_company_years("gigabyte", "gross_margin", 2024, 2025)
    second_change = engine.compare_company_years("gigabyte", "gross_margin", 2024, 2025)
    assert first_ranking == second_ranking
    assert first_change == second_change


def test_unsupported_metric_is_rejected() -> None:
    with pytest.raises(FinancialComparisonValidationError, match="metric must be one of"):
        comparison().rank_companies("eps", 2025, ("gigabyte",))
