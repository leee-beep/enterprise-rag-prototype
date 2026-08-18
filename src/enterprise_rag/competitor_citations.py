"""Deterministic citation and provenance rendering for competitor answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from enterprise_rag.competitor_evidence import (
    EvidenceNotFoundError,
    EvidenceType,
    FinancialCalculationEvidenceData,
    FinancialChangeEvidenceData,
    FinancialComparisonEvidenceData,
    FinancialFactEvidenceData,
    QualitativeEvidenceData,
    SourceReference,
    UnifiedEvidence,
    UnifiedEvidenceSet,
)
from enterprise_rag.competitor_grounded_synthesis import (
    GroundedSynthesisResult,
    GroundedSynthesisStatus,
    ValidatedFinancialClaim,
)


class CitationRenderingError(ValueError):
    """Raised when trusted synthesis and evidence inputs are incompatible."""


@dataclass(frozen=True)
class RenderedSourceReference:
    source_title: str
    page_number: int | None
    source_metric: str | None
    fiscal_year: int | None
    document_type: str | None


@dataclass(frozen=True)
class RenderedFinancialInput:
    metric: str
    fiscal_year: int
    source_unit: str
    source: RenderedSourceReference


@dataclass(frozen=True)
class RenderedQualitativeDetails:
    document_type: str | None
    chunk_id: str


@dataclass(frozen=True)
class RenderedFinancialFactDetails:
    metric: str
    value: str
    unit: str
    currency: str
    source_label: str | None


@dataclass(frozen=True)
class RenderedCalculationDetails:
    metric: str
    value: str
    unit: str
    formula: str
    inputs: tuple[RenderedFinancialInput, ...]


@dataclass(frozen=True)
class RenderedRankingEntry:
    rank: int
    company_id: str
    company_name: str
    value: str
    unit: str
    calculation: RenderedCalculationDetails


@dataclass(frozen=True)
class RenderedComparisonDetails:
    metric: str
    fiscal_year: int
    requested_companies: tuple[str, ...]
    ranking_direction: str
    status: str
    missing_companies: tuple[str, ...]
    ranked_entries: tuple[RenderedRankingEntry, ...]


@dataclass(frozen=True)
class RenderedChangeDetails:
    metric: str
    earlier_year: int
    later_year: int
    earlier_value: str
    later_value: str
    percentage_point_change: str
    direction: str
    unit: str
    earlier_calculation: RenderedCalculationDetails
    later_calculation: RenderedCalculationDetails


RenderedEvidenceDetails: TypeAlias = (
    RenderedQualitativeDetails
    | RenderedFinancialFactDetails
    | RenderedCalculationDetails
    | RenderedComparisonDetails
    | RenderedChangeDetails
)


@dataclass(frozen=True)
class RenderedCitation:
    evidence_id: str
    evidence_type: EvidenceType
    company_id: str | None
    company_name: str | None
    fiscal_year: int | None
    source_references: tuple[RenderedSourceReference, ...]
    financial_claims: tuple[ValidatedFinancialClaim, ...]
    details: RenderedEvidenceDetails


@dataclass(frozen=True)
class RenderedCompetitorAnswer:
    question: str
    answer_text: str
    citations: tuple[RenderedCitation, ...]
    status: GroundedSynthesisStatus
    financial_claims: tuple[ValidatedFinancialClaim, ...]
    generation_provider: str | None
    generation_model: str | None

    def render_text(self) -> str:
        """Render deterministic, concise human-facing answer and provenance text."""
        sections = ["ANSWER", self.answer_text, "SOURCES"]
        sections.extend(_render_citation(citation) for citation in self.citations)
        return "\n\n".join(sections)


def render_competitor_answer(
    result: GroundedSynthesisResult,
    evidence: UnifiedEvidenceSet,
) -> RenderedCompetitorAnswer:
    """Render already-validated synthesis and evidence without inference or I/O."""
    if not isinstance(result, GroundedSynthesisResult):
        raise TypeError("result must be a GroundedSynthesisResult.")
    if not isinstance(evidence, UnifiedEvidenceSet):
        raise TypeError("evidence must be a UnifiedEvidenceSet.")
    if not all(
        isinstance(claim, ValidatedFinancialClaim)
        for claim in result.financial_claims
    ):
        raise CitationRenderingError(
            "financial_claims must contain validated typed claims."
        )

    claims_by_id: dict[str, list[ValidatedFinancialClaim]] = {}
    for claim in result.financial_claims:
        if claim.evidence_id not in result.cited_evidence_ids:
            raise CitationRenderingError(
                "A financial claim references evidence outside validated citations."
            )
        claims_by_id.setdefault(claim.evidence_id, []).append(claim)

    citations: list[RenderedCitation] = []
    seen: set[str] = set()
    for evidence_id in result.cited_evidence_ids:
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        try:
            item = evidence.get(evidence_id)
        except EvidenceNotFoundError as exc:
            raise CitationRenderingError(
                "Validated citation evidence is unavailable for rendering."
            ) from exc
        citations.append(
            _render_citation_model(
                item, tuple(claims_by_id.get(evidence_id, ()))
            )
        )

    return RenderedCompetitorAnswer(
        result.question,
        result.answer_text,
        tuple(citations),
        result.status,
        result.financial_claims,
        result.generation_provider,
        result.generation_model,
    )


def _render_citation_model(
    item: UnifiedEvidence,
    claims: tuple[ValidatedFinancialClaim, ...],
) -> RenderedCitation:
    references = tuple(_source_reference(value) for value in item.source_references)
    data = item.data
    if isinstance(data, QualitativeEvidenceData):
        if claims:
            raise CitationRenderingError(
                "Qualitative evidence cannot render financial claims."
            )
        details: RenderedEvidenceDetails = RenderedQualitativeDetails(
            data.document_type, data.chunk_id
        )
    elif isinstance(data, FinancialFactEvidenceData):
        details = RenderedFinancialFactDetails(
            data.metric,
            str(data.source_value),
            data.source_unit,
            data.currency,
            data.source_label,
        )
    elif isinstance(data, FinancialCalculationEvidenceData):
        details = _calculation_details(data)
    elif isinstance(data, FinancialComparisonEvidenceData):
        details = RenderedComparisonDetails(
            data.metric,
            data.fiscal_year,
            data.requested_companies,
            data.ranking_direction,
            data.status,
            data.missing_companies,
            tuple(
                RenderedRankingEntry(
                    entry.rank,
                    entry.company_id,
                    entry.company_name,
                    str(entry.value),
                    entry.unit,
                    _calculation_details(entry.calculation),
                )
                for entry in data.ranked_entries
            ),
        )
    elif isinstance(data, FinancialChangeEvidenceData):
        details = RenderedChangeDetails(
            data.metric,
            data.earlier_year,
            data.later_year,
            str(data.earlier_value),
            str(data.later_value),
            str(data.percentage_point_change),
            data.direction,
            data.unit,
            _calculation_details(data.earlier_calculation),
            _calculation_details(data.later_calculation),
        )
    else:  # pragma: no cover - UnifiedEvidence validates its data union.
        raise CitationRenderingError("Unsupported evidence type for rendering.")
    return RenderedCitation(
        item.evidence_id,
        item.evidence_type,
        item.company_id,
        item.company_name,
        item.fiscal_year,
        references,
        claims,
        details,
    )


def _calculation_details(
    data: FinancialCalculationEvidenceData,
) -> RenderedCalculationDetails:
    return RenderedCalculationDetails(
        data.metric,
        str(data.value),
        data.unit,
        data.formula,
        tuple(
            RenderedFinancialInput(
                value.metric,
                value.fiscal_year,
                value.source_unit,
                _source_reference(value.source_reference),
            )
            for value in data.input_facts
        ),
    )


def _source_reference(reference: SourceReference) -> RenderedSourceReference:
    return RenderedSourceReference(
        reference.source_title,
        reference.page_number,
        reference.source_metric,
        reference.fiscal_year,
        reference.document_type,
    )


def _render_citation(citation: RenderedCitation) -> str:
    data = citation.details
    if isinstance(data, RenderedQualitativeDetails):
        return _citation_heading(citation, citation.source_references[0])
    if isinstance(data, RenderedFinancialFactDetails):
        lines = [_citation_heading(citation, citation.source_references[0])]
        lines.extend(
            (
                "Type: Reported source fact",
                f"Metric: {data.metric}",
                f"Value: {data.value} {data.unit}",
            )
        )
        if data.source_label:
            lines.append(f"Source label: {data.source_label}")
        return "\n".join(lines)
    if isinstance(data, RenderedCalculationDetails):
        return "\n".join(
            (
                _metric_heading(citation, data.metric),
                "Type: Python-calculated metric",
                f"Value: {data.value} {data.unit}",
                f"Formula: {data.formula}",
                "Inputs:",
                *(_render_input(value) for value in data.inputs),
            )
        )
    if isinstance(data, RenderedComparisonDetails):
        lines = [
            f"[{citation.evidence_id}] {data.fiscal_year} {data.metric} comparison",
            "Type: Supplied deterministic comparison",
            "Requested companies: " + ", ".join(data.requested_companies),
            f"Direction: {data.ranking_direction}",
            f"Status: {data.status}",
        ]
        if data.missing_companies:
            lines.append("Missing: " + ", ".join(data.missing_companies))
        if not data.ranked_entries:
            lines.append("No comparison provenance available.")
        for entry in data.ranked_entries:
            lines.append(
                f"{entry.rank}. {entry.company_name} — {entry.value} {entry.unit}"
            )
            lines.append(
                "   Inputs: "
                + "; ".join(_render_input(value, prefix="") for value in entry.calculation.inputs)
            )
        return "\n".join(lines)
    if isinstance(data, RenderedChangeDetails):
        lines = [
            _metric_heading(citation, f"{data.metric} change, {data.earlier_year}–{data.later_year}"),
            "Type: Supplied deterministic financial change",
            f"{data.earlier_year}: {data.earlier_value} {data.unit}",
            f"{data.later_year}: {data.later_value} {data.unit}",
            f"Change: {data.percentage_point_change} percentage points",
            f"Direction: {data.direction}",
            f"{data.earlier_year} inputs:",
            *(_render_input(value) for value in data.earlier_calculation.inputs),
            f"{data.later_year} inputs:",
            *(_render_input(value) for value in data.later_calculation.inputs),
        ]
        return "\n".join(lines)
    raise CitationRenderingError("Unsupported rendered citation details.")


def _citation_heading(
    citation: RenderedCitation, reference: RenderedSourceReference
) -> str:
    parts = [f"[{citation.evidence_id}]", citation.company_name or "Company unavailable"]
    if citation.fiscal_year is not None:
        parts.append(str(citation.fiscal_year))
    parts.extend((reference.source_title, _page_label(reference.page_number)))
    return " — ".join(parts)


def _metric_heading(citation: RenderedCitation, metric: str) -> str:
    parts = [f"[{citation.evidence_id}]", citation.company_name or "Company unavailable"]
    if citation.fiscal_year is not None:
        parts.append(str(citation.fiscal_year))
    parts.append(metric)
    return " — ".join(parts)


def _render_input(value: RenderedFinancialInput, *, prefix: str = "- ") -> str:
    return (
        f"{prefix}{value.metric} — {value.fiscal_year} {value.source.source_title} — "
        f"{_page_label(value.source.page_number)}"
    )


def _page_label(page_number: int | None) -> str:
    return "PDF page unavailable" if page_number is None else f"PDF p.{page_number}"
