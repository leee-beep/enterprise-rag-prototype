"""Deterministic Decimal calculations over validated financial facts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable

from enterprise_rag.financial_facts import (
    FinancialFact,
    FinancialFactCollection,
    FinancialFactNotFoundError,
)


PERCENT_QUANTUM = Decimal("0.01")
PERCENT_UNIT = "percent"
SUPPORTED_CALCULATED_METRICS = frozenset(
    {"revenue_yoy_growth", "gross_margin", "operating_margin", "net_margin"}
)
FORMULAS = {
    "revenue_yoy_growth": (
        "(current_revenue - prior_revenue) / prior_revenue * 100"
    ),
    "gross_margin": "gross_profit / revenue * 100",
    "operating_margin": "operating_income / revenue * 100",
    "net_margin": "net_income / revenue * 100",
}


class FinancialCalculationError(RuntimeError):
    """Base error for deterministic financial calculations."""


class FinancialCalculationMissingFactError(FinancialCalculationError):
    """Raised when a calculation's exact required source fact is unavailable."""


class FinancialCalculationZeroDenominatorError(FinancialCalculationError):
    """Raised when a required revenue denominator is zero."""


class UnsupportedFinancialCalculationError(FinancialCalculationError):
    """Raised when a calculated metric is outside the initial vocabulary."""


@dataclass(frozen=True)
class FinancialFactProvenance:
    """Safe source identity retained from one calculation input fact."""

    company_id: str
    fiscal_year: int
    metric: str
    source_document_id: str
    source_title: str
    page_number: int
    source_sha256: str | None

    @classmethod
    def from_fact(cls, fact: FinancialFact) -> "FinancialFactProvenance":
        return cls(
            company_id=fact.company_id,
            fiscal_year=fact.fiscal_year,
            metric=fact.metric,
            source_document_id=fact.source_document_id,
            source_title=fact.source_title,
            page_number=fact.page_number,
            source_sha256=fact.source_sha256,
        )


@dataclass(frozen=True)
class FinancialCalculationResult:
    """One rounded percentage with its exact formula and source facts."""

    company_id: str
    company_name: str
    ticker: str
    fiscal_year: int
    metric: str
    value: Decimal
    unit: str
    input_facts: tuple[FinancialFact, FinancialFact]
    formula: str
    provenance: tuple[FinancialFactProvenance, FinancialFactProvenance]


class FinancialCalculationEngine:
    """Calculate a deliberately small metric vocabulary without LLMs or floats."""

    def __init__(self, facts: FinancialFactCollection) -> None:
        if not isinstance(facts, FinancialFactCollection):
            raise TypeError("facts must be a FinancialFactCollection.")
        self._facts = facts

    def revenue_yoy_growth(
        self, company_id: str, fiscal_year: int
    ) -> FinancialCalculationResult:
        current = self._get_fact(company_id, fiscal_year, "revenue")
        prior = self._get_fact(company_id, fiscal_year - 1, "revenue")
        denominator = prior.canonical_value_twd
        if denominator == 0:
            raise FinancialCalculationZeroDenominatorError(
                f"Cannot calculate revenue_yoy_growth for {current.company_id}/"
                f"{fiscal_year}: prior-year revenue is zero."
            )
        percentage = (
            (current.canonical_value_twd - denominator) / denominator
        ) * Decimal("100")
        return self._result(
            current,
            prior,
            metric="revenue_yoy_growth",
            percentage=percentage,
        )

    def gross_margin(
        self, company_id: str, fiscal_year: int
    ) -> FinancialCalculationResult:
        return self._margin(company_id, fiscal_year, "gross_profit", "gross_margin")

    def operating_margin(
        self, company_id: str, fiscal_year: int
    ) -> FinancialCalculationResult:
        return self._margin(
            company_id, fiscal_year, "operating_income", "operating_margin"
        )

    def net_margin(
        self, company_id: str, fiscal_year: int
    ) -> FinancialCalculationResult:
        return self._margin(company_id, fiscal_year, "net_income", "net_margin")

    def calculate(
        self, company_id: str, fiscal_year: int, metric: str
    ) -> FinancialCalculationResult:
        normalized = metric.strip().casefold()
        methods: dict[str, Callable[[str, int], FinancialCalculationResult]] = {
            "revenue_yoy_growth": self.revenue_yoy_growth,
            "gross_margin": self.gross_margin,
            "operating_margin": self.operating_margin,
            "net_margin": self.net_margin,
        }
        try:
            method = methods[normalized]
        except KeyError as exc:
            raise UnsupportedFinancialCalculationError(
                "calculated metric must be one of: "
                + ", ".join(sorted(SUPPORTED_CALCULATED_METRICS))
                + "."
            ) from exc
        return method(company_id, fiscal_year)

    def _margin(
        self,
        company_id: str,
        fiscal_year: int,
        numerator_metric: str,
        result_metric: str,
    ) -> FinancialCalculationResult:
        numerator = self._get_fact(company_id, fiscal_year, numerator_metric)
        revenue = self._get_fact(company_id, fiscal_year, "revenue")
        denominator = revenue.canonical_value_twd
        if denominator == 0:
            raise FinancialCalculationZeroDenominatorError(
                f"Cannot calculate {result_metric} for {revenue.company_id}/"
                f"{fiscal_year}: revenue is zero."
            )
        percentage = numerator.canonical_value_twd / denominator * Decimal("100")
        return self._result(
            numerator,
            revenue,
            metric=result_metric,
            percentage=percentage,
        )

    def _get_fact(
        self, company_id: str, fiscal_year: int, metric: str
    ) -> FinancialFact:
        try:
            return self._facts.get_fact(company_id, fiscal_year, metric)
        except FinancialFactNotFoundError as exc:
            safe_company = str(company_id).strip().casefold() or "unknown-company"
            raise FinancialCalculationMissingFactError(
                "Required financial fact is missing for "
                f"{safe_company}/{fiscal_year}/FY/{metric}/consolidated."
            ) from exc

    @staticmethod
    def _result(
        primary: FinancialFact,
        secondary: FinancialFact,
        *,
        metric: str,
        percentage: Decimal,
    ) -> FinancialCalculationResult:
        inputs = (primary, secondary)
        return FinancialCalculationResult(
            company_id=primary.company_id,
            company_name=primary.company_name,
            ticker=primary.ticker,
            fiscal_year=primary.fiscal_year,
            metric=metric,
            value=percentage.quantize(PERCENT_QUANTUM, rounding=ROUND_HALF_UP),
            unit=PERCENT_UNIT,
            input_facts=inputs,
            formula=FORMULAS[metric],
            provenance=tuple(
                FinancialFactProvenance.from_fact(fact) for fact in inputs
            ),
        )
