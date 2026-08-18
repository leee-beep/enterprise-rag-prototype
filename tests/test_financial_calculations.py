"""Synthetic offline tests for deterministic financial calculations."""

from __future__ import annotations

from decimal import Decimal

import pytest

from enterprise_rag.financial_calculations import (
    FORMULAS,
    FinancialCalculationEngine,
    FinancialCalculationMissingFactError,
    FinancialCalculationZeroDenominatorError,
    UnsupportedFinancialCalculationError,
)
from enterprise_rag.financial_facts import FinancialFact, FinancialFactCollection


def fact(
    metric: str,
    value: str,
    *,
    year: int = 2025,
    unit: str = "thousand_TWD",
    company_id: str = "gigabyte",
    company_name: str = "Gigabyte",
    ticker: str = "2376",
    page: int = 10,
) -> FinancialFact:
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
        page_number=page,
        source_sha256="a" * 64,
        notes="Fictional test value",
    )


def engine(*facts: FinancialFact) -> FinancialCalculationEngine:
    return FinancialCalculationEngine(FinancialFactCollection(facts))


@pytest.mark.parametrize(
    "prior,current,expected",
    (
        ("100", "125", Decimal("25.00")),
        ("100", "75", Decimal("-25.00")),
        ("100", "100", Decimal("0.00")),
    ),
)
def test_revenue_yoy_growth(prior: str, current: str, expected: Decimal) -> None:
    result = engine(
        fact("revenue", prior, year=2024),
        fact("revenue", current, year=2025),
    ).revenue_yoy_growth("gigabyte", 2025)
    assert result.value == expected
    assert result.metric == "revenue_yoy_growth"
    assert result.fiscal_year == 2025


def test_yoy_normalizes_mixed_source_units_before_arithmetic() -> None:
    result = engine(
        fact("revenue", "100000", year=2024, unit="thousand_TWD"),
        fact("revenue", "125", year=2025, unit="million_TWD"),
    ).revenue_yoy_growth("gigabyte", 2025)
    assert result.value == Decimal("25.00")


@pytest.mark.parametrize(
    "method,numerator_metric,numerator,expected",
    (
        ("gross_margin", "gross_profit", "40", Decimal("40.00")),
        ("operating_margin", "operating_income", "12.3456", Decimal("12.35")),
        ("net_margin", "net_income", "8", Decimal("8.00")),
        ("net_margin", "net_income", "-2.555", Decimal("-2.56")),
    ),
)
def test_margin_calculations(
    method: str, numerator_metric: str, numerator: str, expected: Decimal
) -> None:
    calculation = engine(
        fact("revenue", "100", unit="million_TWD"),
        fact(numerator_metric, numerator, unit="million_TWD"),
    )
    result = getattr(calculation, method)("gigabyte", 2025)
    assert result.value == expected
    assert result.metric == method


def test_margin_normalizes_mixed_units() -> None:
    result = engine(
        fact("revenue", "2", unit="million_TWD"),
        fact("gross_profit", "500", unit="thousand_TWD"),
    ).gross_margin("gigabyte", 2025)
    assert result.value == Decimal("25.00")


def test_round_half_up_and_decimal_output() -> None:
    result = engine(
        fact("revenue", "8"),
        fact("operating_income", "1"),
    ).operating_margin("gigabyte", 2025)
    assert result.value == Decimal("12.50")
    assert isinstance(result.value, Decimal)
    assert result.value.as_tuple().exponent == -2


@pytest.mark.parametrize(
    "method,facts,missing_metric",
    (
        ("gross_margin", (fact("revenue", "100"),), "gross_profit"),
        ("gross_margin", (fact("gross_profit", "40"),), "revenue"),
        ("revenue_yoy_growth", (fact("revenue", "100", year=2025),), "revenue"),
    ),
)
def test_missing_required_fact_is_explicit(
    method: str, facts: tuple[FinancialFact, ...], missing_metric: str
) -> None:
    with pytest.raises(FinancialCalculationMissingFactError, match=missing_metric):
        getattr(engine(*facts), method)("gigabyte", 2025)


def test_2024_yoy_does_not_infer_2023() -> None:
    calculation = engine(fact("revenue", "100", year=2024))
    with pytest.raises(FinancialCalculationMissingFactError, match="2023"):
        calculation.revenue_yoy_growth("gigabyte", 2024)


def test_zero_margin_denominator_is_rejected() -> None:
    calculation = engine(
        fact("revenue", "0"),
        fact("gross_profit", "10"),
    )
    with pytest.raises(FinancialCalculationZeroDenominatorError, match="revenue is zero"):
        calculation.gross_margin("gigabyte", 2025)


def test_zero_prior_revenue_is_rejected() -> None:
    calculation = engine(
        fact("revenue", "0", year=2024),
        fact("revenue", "10", year=2025),
    )
    with pytest.raises(FinancialCalculationZeroDenominatorError, match="prior-year"):
        calculation.revenue_yoy_growth("gigabyte", 2025)


def test_result_preserves_company_formula_and_provenance() -> None:
    revenue = fact(
        "revenue", "200", company_id="asus", company_name="ASUS", ticker="2357", page=20
    )
    profit = fact(
        "net_income", "20", company_id="asus", company_name="ASUS", ticker="2357", page=21
    )
    result = engine(revenue, profit).net_margin("asus", 2025)
    assert (result.company_id, result.company_name, result.ticker) == ("asus", "ASUS", "2357")
    assert result.formula == FORMULAS["net_margin"]
    assert result.input_facts == (profit, revenue)
    assert len(result.provenance) == 2
    assert [item.page_number for item in result.provenance] == [21, 20]
    assert all(item.source_document_id.startswith("synthetic:") for item in result.provenance)


def test_repeated_calculation_is_deterministic() -> None:
    calculation = engine(fact("revenue", "3"), fact("gross_profit", "1"))
    first = calculation.gross_margin("gigabyte", 2025)
    second = calculation.gross_margin("gigabyte", 2025)
    assert first == second
    assert first.value == Decimal("33.33")


@pytest.mark.parametrize(
    "metric,expected",
    (
        ("gross_margin", Decimal("40.00")),
        ("operating_margin", Decimal("20.00")),
        ("net_margin", Decimal("10.00")),
    ),
)
def test_calculate_dispatch(metric: str, expected: Decimal) -> None:
    calculation = engine(
        fact("revenue", "100"),
        fact("gross_profit", "40"),
        fact("operating_income", "20"),
        fact("net_income", "10"),
    )
    assert calculation.calculate("gigabyte", 2025, metric).value == expected


def test_unsupported_calculation_is_rejected() -> None:
    with pytest.raises(UnsupportedFinancialCalculationError, match="calculated metric"):
        engine().calculate("gigabyte", 2025, "ranking")


def test_errors_do_not_contain_private_paths() -> None:
    with pytest.raises(FinancialCalculationMissingFactError) as caught:
        engine().gross_margin("gigabyte", 2025)
    message = str(caught.value)
    assert "C:\\Users" not in message
    assert "data/private" not in message
