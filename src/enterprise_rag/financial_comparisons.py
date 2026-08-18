"""Deterministic comparisons over trusted financial calculation results."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from enterprise_rag.financial_calculations import (
    FinancialCalculationEngine,
    FinancialCalculationMissingFactError,
    FinancialCalculationResult,
    FinancialFactProvenance,
    SUPPORTED_CALCULATED_METRICS,
)
from enterprise_rag.financial_facts import COMPANIES, SUPPORTED_FISCAL_YEARS


RANKING_DIRECTION = "higher_value_first"
COMPARISON_STATUSES = frozenset({"complete", "partial", "insufficient"})
CHANGE_DIRECTIONS = frozenset({"increase", "decrease", "unchanged"})
MARGIN_METRICS = frozenset({"gross_margin", "operating_margin", "net_margin"})


class FinancialComparisonError(RuntimeError):
    """Base error for deterministic financial comparison operations."""


class FinancialComparisonValidationError(FinancialComparisonError):
    """Raised when a comparison request is malformed or unsupported."""


@dataclass(frozen=True)
class FinancialRankingEntry:
    """One company-local calculation placed in a neutral numeric ranking."""

    rank: int
    company_id: str
    company_name: str
    ticker: str
    fiscal_year: int
    metric: str
    value: Decimal
    unit: str
    calculation_result: FinancialCalculationResult
    provenance: tuple[FinancialFactProvenance, FinancialFactProvenance]


@dataclass(frozen=True)
class FinancialComparisonResult:
    """A complete, partial, or insufficient cross-company ranking."""

    metric: str
    fiscal_year: int
    requested_companies: tuple[str, ...]
    ranked_entries: tuple[FinancialRankingEntry, ...]
    missing_companies: tuple[str, ...]
    ranking_direction: str
    status: str


@dataclass(frozen=True)
class FinancialChangeResult:
    """A percentage-point change between two same-company margin results."""

    company_id: str
    company_name: str
    ticker: str
    metric: str
    earlier_year: int
    later_year: int
    earlier_value: Decimal
    later_value: Decimal
    percentage_point_change: Decimal
    direction: str
    unit: str
    earlier_result: FinancialCalculationResult
    later_result: FinancialCalculationResult
    provenance: tuple[FinancialFactProvenance, ...]


class FinancialComparisonEngine:
    """Rank calculated values and compare margin years without interpretation."""

    def __init__(self, calculation_engine: FinancialCalculationEngine) -> None:
        if not isinstance(calculation_engine, FinancialCalculationEngine):
            raise TypeError("calculation_engine must be a FinancialCalculationEngine.")
        self._calculations = calculation_engine

    def rank_companies(
        self,
        metric: str,
        fiscal_year: int,
        companies: tuple[str, ...],
    ) -> FinancialComparisonResult:
        normalized_metric = self._validate_metric(metric)
        validated_year = self._validate_year("fiscal_year", fiscal_year)
        requested = self._validate_companies(companies)
        available: list[tuple[int, FinancialCalculationResult]] = []
        missing: list[str] = []
        for request_index, company_id in enumerate(requested):
            try:
                result = self._calculations.calculate(
                    company_id, validated_year, normalized_metric
                )
            except FinancialCalculationMissingFactError:
                missing.append(company_id)
                continue
            available.append((request_index, result))

        ordered = sorted(available, key=lambda item: (-item[1].value, item[0]))
        ranked: list[FinancialRankingEntry] = []
        previous_value: Decimal | None = None
        previous_rank = 0
        for position, (_, result) in enumerate(ordered, start=1):
            rank = previous_rank if previous_value == result.value else position
            ranked.append(
                FinancialRankingEntry(
                    rank=rank,
                    company_id=result.company_id,
                    company_name=result.company_name,
                    ticker=result.ticker,
                    fiscal_year=result.fiscal_year,
                    metric=result.metric,
                    value=result.value,
                    unit=result.unit,
                    calculation_result=result,
                    provenance=result.provenance,
                )
            )
            previous_value = result.value
            previous_rank = rank

        status = (
            "insufficient"
            if not ranked
            else "partial"
            if missing
            else "complete"
        )
        return FinancialComparisonResult(
            metric=normalized_metric,
            fiscal_year=validated_year,
            requested_companies=requested,
            ranked_entries=tuple(ranked),
            missing_companies=tuple(missing),
            ranking_direction=RANKING_DIRECTION,
            status=status,
        )

    def compare_company_years(
        self,
        company_id: str,
        metric: str,
        earlier_year: int,
        later_year: int,
    ) -> FinancialChangeResult:
        normalized_company = self._validate_companies((company_id,))[0]
        normalized_metric = self._validate_metric(metric)
        if normalized_metric not in MARGIN_METRICS:
            raise FinancialComparisonValidationError(
                "year-change metric must be one of: "
                + ", ".join(sorted(MARGIN_METRICS))
                + "; revenue_yoy_growth is already a cross-year calculation."
            )
        validated_earlier = self._validate_year("earlier_year", earlier_year)
        validated_later = self._validate_year("later_year", later_year)
        if validated_earlier >= validated_later:
            raise FinancialComparisonValidationError(
                "earlier_year and later_year must be integers with earlier_year < later_year."
            )
        earlier = self._calculations.calculate(
            normalized_company, validated_earlier, normalized_metric
        )
        later = self._calculations.calculate(
            normalized_company, validated_later, normalized_metric
        )
        change = later.value - earlier.value
        direction = (
            "increase" if change > 0 else "decrease" if change < 0 else "unchanged"
        )
        return FinancialChangeResult(
            company_id=later.company_id,
            company_name=later.company_name,
            ticker=later.ticker,
            metric=normalized_metric,
            earlier_year=validated_earlier,
            later_year=validated_later,
            earlier_value=earlier.value,
            later_value=later.value,
            percentage_point_change=change,
            direction=direction,
            unit="percentage_points",
            earlier_result=earlier,
            later_result=later,
            provenance=earlier.provenance + later.provenance,
        )

    @staticmethod
    def _validate_metric(metric: str) -> str:
        if not isinstance(metric, str) or not metric.strip():
            raise FinancialComparisonValidationError(
                "metric must be a supported non-empty calculated metric."
            )
        normalized = metric.strip().casefold()
        if normalized not in SUPPORTED_CALCULATED_METRICS:
            raise FinancialComparisonValidationError(
                "metric must be one of: "
                + ", ".join(sorted(SUPPORTED_CALCULATED_METRICS))
                + "."
            )
        return normalized

    @staticmethod
    def _validate_year(name: str, value: int) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in SUPPORTED_FISCAL_YEARS
        ):
            raise FinancialComparisonValidationError(
                f"{name} must be 2024 or 2025."
            )
        return value

    @staticmethod
    def _validate_companies(companies: tuple[str, ...]) -> tuple[str, ...]:
        if not isinstance(companies, tuple) or not companies:
            raise FinancialComparisonValidationError(
                "companies must be a non-empty tuple of company IDs."
            )
        normalized: list[str] = []
        for company in companies:
            if not isinstance(company, str) or not company.strip():
                raise FinancialComparisonValidationError(
                    "every company ID must be a non-empty string."
                )
            company_id = company.strip().casefold()
            if company_id not in COMPANIES:
                raise FinancialComparisonValidationError(
                    "company must be one of: asus, gigabyte, msi."
                )
            if company_id in normalized:
                raise FinancialComparisonValidationError(
                    f"duplicate company is not allowed: {company_id}."
                )
            normalized.append(company_id)
        return tuple(normalized)
