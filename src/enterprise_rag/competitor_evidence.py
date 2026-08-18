"""Immutable, provider-free evidence normalization for competitor analysis."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import TypeAlias, overload

from enterprise_rag.competitor_analysis import CitationReadyEvidence
from enterprise_rag.financial_calculations import FinancialCalculationResult
from enterprise_rag.financial_comparisons import (
    FinancialChangeResult,
    FinancialComparisonResult,
)
from enterprise_rag.financial_facts import FinancialFact


_EVIDENCE_ID = re.compile(r"E[1-9]\d*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_WINDOWS_ABSOLUTE = re.compile(r"(?i)(?<![0-9a-z])[a-z]:[\\/]")
_UNC_BACKSLASH = re.compile(r"(?<![\\\w])\\\\[^\\\s]+\\")
_UNC_FORWARD_SLASH = re.compile(r"(?<![:/\w])//[^/\s]+/")
_POSIX_ABSOLUTE = re.compile(
    r"(?<![0-9A-Za-z_:/])/[0-9A-Za-z._~-]+(?:/[0-9A-Za-z._~-]+)*"
)
_SECRET_VALUE = re.compile(r"(?:AIza[0-9A-Za-z_-]{20,}|sk-[0-9A-Za-z_-]{20,})")
_PRIVATE_MARKERS = ("data/private", "data\\private", "data/vector_store", "data\\vector_store")


class EvidenceValidationError(ValueError):
    """Raised when unified evidence could expose unsafe or invalid metadata."""


class EvidenceNotFoundError(LookupError):
    """Raised when an evidence ID is absent from a unified evidence set."""


class EvidenceType(str, Enum):
    QUALITATIVE = "qualitative"
    FINANCIAL_FACT = "financial_fact"
    FINANCIAL_CALCULATION = "financial_calculation"
    FINANCIAL_COMPARISON = "financial_comparison"
    FINANCIAL_CHANGE = "financial_change"


@dataclass(frozen=True)
class SourceReference:
    """Trusted, display-safe source identity for one evidence input."""

    source_document_id: str | None
    source_title: str
    page_number: int | None
    source_metric: str | None = None
    fiscal_year: int | None = None
    source_sha256: str | None = None
    chunk_id: str | None = None
    document_type: str | None = None

    def __post_init__(self) -> None:
        if self.source_document_id is not None:
            identity = _safe_identifier(
                "source_document_id", self.source_document_id
            )
            object.__setattr__(self, "source_document_id", identity)
        object.__setattr__(self, "source_title", _safe_title(self.source_title))
        if (
            self.page_number is not None
            and (
                isinstance(self.page_number, bool)
                or not isinstance(self.page_number, int)
                or self.page_number <= 0
            )
        ):
            raise EvidenceValidationError(
                "page_number must be a positive integer when provided."
            )
        if self.source_sha256 is not None:
            digest = _required_text("source_sha256", self.source_sha256).casefold()
            if not _SHA256.fullmatch(digest):
                raise EvidenceValidationError(
                    "source_sha256 must be a 64-character hexadecimal digest."
                )
            object.__setattr__(self, "source_sha256", digest)
        for name in ("source_metric", "document_type"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _safe_source_text(name, value))
        if self.chunk_id is not None:
            object.__setattr__(
                self, "chunk_id", _safe_identifier("chunk_id", self.chunk_id)
            )


@dataclass(frozen=True)
class QualitativeEvidenceData:
    text: str
    document_type: str | None
    chunk_id: str
    retrieval_score: float
    quality_score: float
    original_candidate_rank: int
    final_company_rank: int

    def __post_init__(self) -> None:
        _safe_display_text(self.text)
        _safe_identifier("chunk_id", self.chunk_id)
        if self.document_type is not None:
            _safe_source_text("document_type", self.document_type)


@dataclass(frozen=True)
class FinancialInputFact:
    metric: str
    fiscal_year: int
    source_value: Decimal
    source_unit: str
    currency: str
    canonical_value_twd: Decimal
    source_reference: SourceReference

    def __post_init__(self) -> None:
        _safe_source_text("metric", self.metric)
        _safe_source_text("source_unit", self.source_unit)
        _safe_source_text("currency", self.currency)
        if not isinstance(self.source_reference, SourceReference):
            raise EvidenceValidationError(
                "source_reference must be a SourceReference value."
            )


@dataclass(frozen=True)
class FinancialFactEvidenceData:
    metric: str
    source_value: Decimal
    source_unit: str
    currency: str
    canonical_value_twd: Decimal
    period: str
    reporting_scope: str
    document_type: str
    source_label: str | None

    def __post_init__(self) -> None:
        for name in (
            "metric",
            "source_unit",
            "currency",
            "period",
            "reporting_scope",
            "document_type",
        ):
            _safe_source_text(name, getattr(self, name))
        if self.source_label is not None:
            _safe_source_text("source_label", self.source_label)


@dataclass(frozen=True)
class FinancialCalculationEvidenceData:
    metric: str
    value: Decimal
    unit: str
    formula: str
    input_facts: tuple[FinancialInputFact, FinancialInputFact]

    def __post_init__(self) -> None:
        _safe_source_text("metric", self.metric)
        _safe_source_text("unit", self.unit)
        _safe_source_text("formula", self.formula)
        if not isinstance(self.input_facts, tuple) or len(self.input_facts) != 2 or not all(
            isinstance(item, FinancialInputFact) for item in self.input_facts
        ):
            raise EvidenceValidationError(
                "input_facts must contain exactly two FinancialInputFact values."
            )


@dataclass(frozen=True)
class FinancialRankingEvidenceEntry:
    rank: int
    company_id: str
    company_name: str
    ticker: str
    fiscal_year: int
    metric: str
    value: Decimal
    unit: str
    calculation: FinancialCalculationEvidenceData

    def __post_init__(self) -> None:
        _safe_identifier("company_id", self.company_id)
        _safe_source_text("company_name", self.company_name)
        _safe_source_text("ticker", self.ticker)
        _safe_source_text("metric", self.metric)
        _safe_source_text("unit", self.unit)
        if not isinstance(self.calculation, FinancialCalculationEvidenceData):
            raise EvidenceValidationError(
                "calculation must be FinancialCalculationEvidenceData."
            )


@dataclass(frozen=True)
class FinancialComparisonEvidenceData:
    metric: str
    fiscal_year: int
    requested_companies: tuple[str, ...]
    missing_companies: tuple[str, ...]
    ranking_direction: str
    status: str
    ranked_entries: tuple[FinancialRankingEvidenceEntry, ...]

    def __post_init__(self) -> None:
        _safe_source_text("metric", self.metric)
        _safe_source_text("ranking_direction", self.ranking_direction)
        _safe_source_text("status", self.status)
        for company_id in self.requested_companies + self.missing_companies:
            _safe_identifier("company_id", company_id)
        if not isinstance(self.ranked_entries, tuple) or not all(
            isinstance(item, FinancialRankingEvidenceEntry)
            for item in self.ranked_entries
        ):
            raise EvidenceValidationError(
                "ranked_entries must be a tuple of FinancialRankingEvidenceEntry values."
            )


@dataclass(frozen=True)
class FinancialChangeEvidenceData:
    metric: str
    earlier_year: int
    later_year: int
    earlier_value: Decimal
    later_value: Decimal
    percentage_point_change: Decimal
    direction: str
    unit: str
    earlier_calculation: FinancialCalculationEvidenceData
    later_calculation: FinancialCalculationEvidenceData

    def __post_init__(self) -> None:
        _safe_source_text("metric", self.metric)
        _safe_source_text("direction", self.direction)
        _safe_source_text("unit", self.unit)
        if not isinstance(
            self.earlier_calculation, FinancialCalculationEvidenceData
        ) or not isinstance(
            self.later_calculation, FinancialCalculationEvidenceData
        ):
            raise EvidenceValidationError(
                "change calculations must be FinancialCalculationEvidenceData values."
            )


EvidenceData: TypeAlias = (
    QualitativeEvidenceData
    | FinancialFactEvidenceData
    | FinancialCalculationEvidenceData
    | FinancialComparisonEvidenceData
    | FinancialChangeEvidenceData
)
EvidenceInput: TypeAlias = (
    CitationReadyEvidence
    | FinancialFact
    | FinancialCalculationResult
    | FinancialComparisonResult
    | FinancialChangeResult
)


@dataclass(frozen=True)
class UnifiedEvidence:
    """One normalized evidence item with explicit type and trusted provenance."""

    evidence_id: str
    evidence_type: EvidenceType
    company_id: str | None
    company_name: str | None
    ticker: str | None
    fiscal_year: int | None
    display_text: str
    source_references: tuple[SourceReference, ...]
    data: EvidenceData

    def __post_init__(self) -> None:
        if not _EVIDENCE_ID.fullmatch(self.evidence_id):
            raise EvidenceValidationError("evidence_id must use the E1, E2, ... format.")
        if not isinstance(self.evidence_type, EvidenceType):
            raise EvidenceValidationError(
                "evidence_type must be an EvidenceType instance."
            )
        for name in ("company_id", "company_name", "ticker"):
            value = getattr(self, name)
            if value is not None:
                _safe_source_text(name, value)
        _safe_display_text(self.display_text)
        if not isinstance(self.source_references, tuple) or not all(
            isinstance(item, SourceReference) for item in self.source_references
        ):
            raise EvidenceValidationError("source_references must be a tuple of SourceReference values.")
        expected_data_type = {
            EvidenceType.QUALITATIVE: QualitativeEvidenceData,
            EvidenceType.FINANCIAL_FACT: FinancialFactEvidenceData,
            EvidenceType.FINANCIAL_CALCULATION: FinancialCalculationEvidenceData,
            EvidenceType.FINANCIAL_COMPARISON: FinancialComparisonEvidenceData,
            EvidenceType.FINANCIAL_CHANGE: FinancialChangeEvidenceData,
        }[self.evidence_type]
        if not isinstance(self.data, expected_data_type):
            raise EvidenceValidationError(
                "evidence_type must match the unified evidence data model."
            )


class UnifiedEvidenceSet:
    """Immutable, ordered evidence with deterministic ID lookup and filtering."""

    def __init__(self, evidence: Iterable[UnifiedEvidence]) -> None:
        values = tuple(evidence)
        if not all(isinstance(item, UnifiedEvidence) for item in values):
            raise EvidenceValidationError(
                "UnifiedEvidenceSet accepts only UnifiedEvidence values."
            )
        ids = tuple(item.evidence_id for item in values)
        if len(ids) != len(set(ids)):
            raise EvidenceValidationError("Duplicate evidence IDs are not allowed.")
        self._evidence = values
        self._by_id = {item.evidence_id: item for item in values}

    def __iter__(self) -> Iterator[UnifiedEvidence]:
        return iter(self._evidence)

    def __len__(self) -> int:
        return len(self._evidence)

    @overload
    def __getitem__(self, index: int) -> UnifiedEvidence: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[UnifiedEvidence, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> UnifiedEvidence | tuple[UnifiedEvidence, ...]:
        return self._evidence[index]

    @property
    def evidence(self) -> tuple[UnifiedEvidence, ...]:
        return self._evidence

    def get(self, evidence_id: str) -> UnifiedEvidence:
        try:
            return self._by_id[evidence_id]
        except KeyError as exc:
            raise EvidenceNotFoundError(
                f"Unknown unified evidence ID: {evidence_id}."
            ) from exc

    def filter_by_type(self, evidence_type: EvidenceType) -> tuple[UnifiedEvidence, ...]:
        return tuple(item for item in self._evidence if item.evidence_type is evidence_type)

    def filter_by_company(self, company_id: str) -> tuple[UnifiedEvidence, ...]:
        normalized = _required_text("company_id", company_id).casefold()
        return tuple(
            item
            for item in self._evidence
            if item.company_id == normalized
            or (
                isinstance(item.data, FinancialComparisonEvidenceData)
                and normalized in item.data.requested_companies
            )
        )


class UnifiedEvidenceBuilder:
    """Adapt trusted evidence models without retrieval, calculation, or generation."""

    def build(self, inputs: Iterable[EvidenceInput]) -> UnifiedEvidenceSet:
        built = tuple(
            self._adapt(item, f"E{number}")
            for number, item in enumerate(inputs, start=1)
        )
        return UnifiedEvidenceSet(built)

    def _adapt(self, item: EvidenceInput, evidence_id: str) -> UnifiedEvidence:
        if isinstance(item, CitationReadyEvidence):
            return _adapt_qualitative(item, evidence_id)
        if isinstance(item, FinancialFact):
            return _adapt_financial_fact(item, evidence_id)
        if isinstance(item, FinancialCalculationResult):
            return _adapt_financial_calculation(item, evidence_id)
        if isinstance(item, FinancialComparisonResult):
            return _adapt_financial_comparison(item, evidence_id)
        if isinstance(item, FinancialChangeResult):
            return _adapt_financial_change(item, evidence_id)
        raise TypeError(f"Unsupported evidence input type: {type(item).__name__}.")


def _adapt_qualitative(item: CitationReadyEvidence, evidence_id: str) -> UnifiedEvidence:
    reference = SourceReference(
        item.source_document_id,
        item.source_title,
        item.page_number,
        fiscal_year=item.fiscal_year if isinstance(item.fiscal_year, int) else None,
        chunk_id=item.chunk_id,
        document_type=item.document_type,
    )
    data = QualitativeEvidenceData(
        item.text,
        item.document_type,
        item.chunk_id,
        item.retrieval_score,
        item.quality_score,
        item.original_candidate_rank,
        item.final_company_rank,
    )
    return UnifiedEvidence(
        evidence_id,
        EvidenceType.QUALITATIVE,
        item.company_id,
        item.company_name,
        item.ticker,
        item.fiscal_year if isinstance(item.fiscal_year, int) else None,
        item.text,
        (reference,),
        data,
    )


def _adapt_financial_fact(item: FinancialFact, evidence_id: str) -> UnifiedEvidence:
    reference = _source_from_fact(item)
    data = FinancialFactEvidenceData(
        item.metric,
        item.value,
        item.unit,
        item.currency,
        item.canonical_value_twd,
        item.period,
        item.reporting_scope,
        item.document_type,
        _safe_optional_label(item.source_label),
    )
    display = f"{item.company_name} {item.fiscal_year} {item.metric}: {item.value} {item.unit}"
    return UnifiedEvidence(
        evidence_id,
        EvidenceType.FINANCIAL_FACT,
        item.company_id,
        item.company_name,
        item.ticker,
        item.fiscal_year,
        display,
        (reference,),
        data,
    )


def _adapt_financial_calculation(
    item: FinancialCalculationResult, evidence_id: str
) -> UnifiedEvidence:
    data = _calculation_data(item)
    references = tuple(value.source_reference for value in data.input_facts)
    display = (
        f"{item.company_name} {item.fiscal_year} {item.metric}: {item.value} "
        f"{item.unit} (deterministic calculation)"
    )
    return UnifiedEvidence(
        evidence_id,
        EvidenceType.FINANCIAL_CALCULATION,
        item.company_id,
        item.company_name,
        item.ticker,
        item.fiscal_year,
        display,
        references,
        data,
    )


def _adapt_financial_comparison(
    item: FinancialComparisonResult, evidence_id: str
) -> UnifiedEvidence:
    entries = tuple(
        FinancialRankingEvidenceEntry(
            entry.rank,
            entry.company_id,
            entry.company_name,
            entry.ticker,
            entry.fiscal_year,
            entry.metric,
            entry.value,
            entry.unit,
            _calculation_data(entry.calculation_result),
        )
        for entry in item.ranked_entries
    )
    data = FinancialComparisonEvidenceData(
        item.metric,
        item.fiscal_year,
        item.requested_companies,
        item.missing_companies,
        item.ranking_direction,
        item.status,
        entries,
    )
    references = tuple(
        input_fact.source_reference
        for entry in entries
        for input_fact in entry.calculation.input_facts
    )
    ranking = "; ".join(
        f"rank {entry.rank}: {entry.company_name} {entry.value} {entry.unit}"
        for entry in entries
    ) or "no ranked entries"
    display = (
        f"{item.fiscal_year} {item.metric} ranking "
        f"({item.ranking_direction}, {item.status}): {ranking}"
    )
    return UnifiedEvidence(
        evidence_id,
        EvidenceType.FINANCIAL_COMPARISON,
        None,
        None,
        None,
        item.fiscal_year,
        display,
        references,
        data,
    )


def _adapt_financial_change(
    item: FinancialChangeResult, evidence_id: str
) -> UnifiedEvidence:
    earlier = _calculation_data(item.earlier_result)
    later = _calculation_data(item.later_result)
    data = FinancialChangeEvidenceData(
        item.metric,
        item.earlier_year,
        item.later_year,
        item.earlier_value,
        item.later_value,
        item.percentage_point_change,
        item.direction,
        item.unit,
        earlier,
        later,
    )
    references = tuple(
        value.source_reference
        for calculation in (earlier, later)
        for value in calculation.input_facts
    )
    display = (
        f"{item.company_name} {item.metric}: {item.earlier_year} {item.earlier_value} "
        f"percent; {item.later_year} {item.later_value} percent; change "
        f"{item.percentage_point_change} percentage_points ({item.direction})"
    )
    return UnifiedEvidence(
        evidence_id,
        EvidenceType.FINANCIAL_CHANGE,
        item.company_id,
        item.company_name,
        item.ticker,
        item.later_year,
        display,
        references,
        data,
    )


def _calculation_data(item: FinancialCalculationResult) -> FinancialCalculationEvidenceData:
    return FinancialCalculationEvidenceData(
        item.metric,
        item.value,
        item.unit,
        item.formula,
        tuple(_input_fact(fact) for fact in item.input_facts),
    )


def _input_fact(item: FinancialFact) -> FinancialInputFact:
    return FinancialInputFact(
        item.metric,
        item.fiscal_year,
        item.value,
        item.unit,
        item.currency,
        item.canonical_value_twd,
        _source_from_fact(item),
    )


def _source_from_fact(item: FinancialFact) -> SourceReference:
    return SourceReference(
        item.source_document_id,
        item.source_title,
        item.page_number,
        source_metric=item.metric,
        fiscal_year=item.fiscal_year,
        source_sha256=item.source_sha256,
        document_type=item.document_type,
    )


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError(f"{name} must be a non-empty string.")
    return value.strip()


def _safe_title(value: object) -> str:
    title = _required_text("source_title", value)
    if (
        PureWindowsPath(title).is_absolute()
        or PurePosixPath(title).is_absolute()
        or _contains_private_path(title)
    ):
        raise EvidenceValidationError(
            "source_title must be a safe display title, not a private local path."
        )
    return _validate_private_text("source_title", title)


def _safe_display_text(value: object) -> str:
    text = _required_text("display_text", value)
    return _validate_private_text("display_text", text)


def _contains_private_path(value: str) -> bool:
    normalized = value.casefold()
    return any(marker in normalized for marker in _PRIVATE_MARKERS)


def _safe_optional_label(value: str | None) -> str | None:
    if value is None:
        return None
    return _safe_source_text("source_label", value)


def _safe_identifier(name: str, value: object) -> str:
    identity = _safe_source_text(name, value)
    if "/" in identity or "\\" in identity:
        raise EvidenceValidationError(
            f"{name} must be a safe identity, not a local path."
        )
    return identity


def _safe_source_text(name: str, value: object) -> str:
    text = _required_text(name, value)
    return _validate_private_text(name, text)


def _validate_private_text(name: str, text: str) -> str:
    if _looks_like_absolute_path(text) or _contains_private_path(text):
        raise EvidenceValidationError(
            f"{name} must not contain a local or private path."
        )
    if _SECRET_VALUE.search(text):
        raise EvidenceValidationError(f"{name} must not contain a secret value.")
    return text


def _looks_like_absolute_path(text: str) -> bool:
    return any(
        pattern.search(text) is not None
        for pattern in (
            _WINDOWS_ABSOLUTE,
            _UNC_BACKSLASH,
            _UNC_FORWARD_SLASH,
            _POSIX_ABSOLUTE,
        )
    )
