"""Grounded synthesis over already-prepared unified competitor evidence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

from enterprise_rag.competitor_evidence import (
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
from enterprise_rag.competitor_planning import AnalysisPlan, AnalysisRoute, PlanStatus
from enterprise_rag.generation import (
    GenerationClient,
    StructuredGenerationClient,
    validate_generated_text,
)


QUALITATIVE_SYNTHESIS_INSTRUCTIONS = """Write a concise competitor-analysis narrative.
Use only the supplied qualitative evidence. Do not introduce external knowledge, cite
evidence IDs, pages, paths, URLs, or sources, and do not characterize a company for which
no evidence was supplied. Return exactly one JSON object with a single non-empty `text`
field. Numeric text is permitted, but it is never authoritative financial output."""

FINANCIAL_SYNTHESIS_INSTRUCTIONS = """Select structured financial claims from only the
supplied trusted financial evidence. Do not write prose. Never calculate, recalculate,
round, convert, alter, or infer a value. Copy the exact evidence ID, claim type, role,
Decimal value, company ID, and rank where applicable. Return exactly one JSON object
with a `claims` array and no other root fields."""

_EVIDENCE_ID = re.compile(r"E[1-9]\d*")
_CITATION_GROUP = re.compile(r"\[((?:E\d+\s*(?:,\s*E\d+\s*)*))\]")
_DECIMAL_TEXT = re.compile(r"[-+]?\d+(?:\.\d+)?")
_QUALITATIVE_RESPONSE_KEYS = frozenset({"text"})
_FINANCIAL_RESPONSE_KEYS = frozenset({"claims"})
_BASIC_CLAIM_KEYS = frozenset({"evidence_id", "claim_type", "role", "value"})
_COMPARISON_CLAIM_KEYS = _BASIC_CLAIM_KEYS | {"company_id", "rank"}

QUALITATIVE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
    "additionalProperties": False,
}

FINANCIAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "evidence_id": {"type": "string"},
                            "claim_type": {"type": "string", "enum": ["reported_fact"]},
                            "role": {"type": "string", "enum": ["reported_value"]},
                            "value": {"type": "string"},
                        },
                        "required": ["evidence_id", "claim_type", "role", "value"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "evidence_id": {"type": "string"},
                            "claim_type": {"type": "string", "enum": ["calculated_metric"]},
                            "role": {"type": "string", "enum": ["calculated_value"]},
                            "value": {"type": "string"},
                        },
                        "required": ["evidence_id", "claim_type", "role", "value"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "evidence_id": {"type": "string"},
                            "claim_type": {"type": "string", "enum": ["comparison_entry"]},
                            "role": {"type": "string", "enum": ["ranked_entry"]},
                            "value": {"type": "string"},
                            "company_id": {"type": "string"},
                            "rank": {"type": "integer"},
                        },
                        "required": [
                            "evidence_id", "claim_type", "role", "value",
                            "company_id", "rank",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "evidence_id": {"type": "string"},
                            "claim_type": {"type": "string", "enum": ["financial_change_value"]},
                            "role": {
                                "type": "string",
                                "enum": [
                                    "earlier_value", "later_value",
                                    "percentage_point_change",
                                ],
                            },
                            "value": {"type": "string"},
                        },
                        "required": ["evidence_id", "claim_type", "role", "value"],
                        "additionalProperties": False,
                    },
                ]
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}


class GroundedSynthesisError(RuntimeError):
    """Base error for deterministic grounded-synthesis safeguards."""


class GroundedGenerationError(GroundedSynthesisError):
    """Raised when the provider cannot return a usable response."""


class GroundedResponseFormatError(GroundedSynthesisError):
    """Raised when generated output violates the strict response contract."""


class UnknownEvidenceCitationError(GroundedSynthesisError):
    """Raised when generated output cites evidence that was not supplied."""


class FinancialGroundingError(GroundedSynthesisError):
    """Raised when generated financial numbers are not exactly evidence-backed."""


class GroundedSynthesisStatus(str, Enum):
    GROUNDED = "grounded"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class QualitativeCoverage(str, Enum):
    """Python-owned company coverage for qualitative generation."""

    NOT_REQUESTED = "not_requested"
    INSUFFICIENT = "insufficient"
    PARTIAL = "partial"
    COMPLETE = "complete"


class FinancialClaimType(str, Enum):
    REPORTED_FACT = "reported_fact"
    CALCULATED_METRIC = "calculated_metric"
    COMPARISON_ENTRY = "comparison_entry"
    FINANCIAL_CHANGE_VALUE = "financial_change_value"


@dataclass(frozen=True)
class ValidatedFinancialClaim:
    evidence_id: str
    claim_type: FinancialClaimType
    role: str
    value: str
    company_id: str | None = None
    rank: int | None = None


@dataclass(frozen=True)
class GroundedSynthesisResult:
    question: str
    answer_text: str
    cited_evidence_ids: tuple[str, ...]
    status: GroundedSynthesisStatus
    evidence_count: int
    financial_claims: tuple[ValidatedFinancialClaim, ...]
    generation_provider: str | None
    generation_model: str | None
    qualitative_coverage: QualitativeCoverage = QualitativeCoverage.NOT_REQUESTED
    missing_qualitative_companies: tuple[str, ...] = ()


class GroundedCompetitorSynthesizer:
    """Split qualitative prose and financial claims, then combine in Python.

    Qualitative provenance is scope-level: every selected qualitative item supplied to
    generation authorizes the narrative as a whole. Financial provenance remains
    claim-level through validated structured claims.
    """

    def __init__(self, generation_client: GenerationClient) -> None:
        self._generation_client = generation_client

    def synthesize(
        self,
        question: str,
        evidence: UnifiedEvidenceSet,
        analysis_plan: AnalysisPlan | None = None,
    ) -> GroundedSynthesisResult:
        normalized_question = _required_text("question", question)
        if not isinstance(evidence, UnifiedEvidenceSet):
            raise TypeError("evidence must be a UnifiedEvidenceSet.")
        if analysis_plan is not None and not isinstance(analysis_plan, AnalysisPlan):
            raise TypeError("analysis_plan must be an AnalysisPlan or None.")
        if analysis_plan is not None and analysis_plan.status is not PlanStatus.READY:
            return _insufficient_result(normalized_question, len(evidence))
        if not evidence:
            coverage, missing = _qualitative_coverage((), analysis_plan)
            return _insufficient_result(
                normalized_question,
                0,
                qualitative_coverage=coverage,
                missing_qualitative_companies=missing,
            )
        if _only_insufficient_comparisons(evidence):
            return _insufficient_result(normalized_question, len(evidence))

        qualitative = tuple(
            item for item in evidence if item.evidence_type is EvidenceType.QUALITATIVE
        )
        financial = tuple(
            item for item in evidence if item.evidence_type is not EvidenceType.QUALITATIVE
        )
        coverage, missing = _qualitative_coverage(qualitative, analysis_plan)
        qualitative_requested = _qualitative_requested(analysis_plan, qualitative)
        if qualitative_requested and not qualitative and not financial:
            return _insufficient_result(
                normalized_question,
                len(evidence),
                qualitative_coverage=QualitativeCoverage.INSUFFICIENT,
                missing_qualitative_companies=missing,
            )

        blocks: list[str] = []
        cited_ids: list[str] = []
        if qualitative:
            qualitative_text = self._generate_qualitative(
                normalized_question, qualitative, analysis_plan
            )
            qualitative_ids = tuple(item.evidence_id for item in qualitative)
            blocks.append(qualitative_text + " [" + ", ".join(qualitative_ids) + "]")
            cited_ids.extend(qualitative_ids)

        claims: tuple[ValidatedFinancialClaim, ...] = ()
        if financial:
            claims = self._generate_financial(
                normalized_question, financial, analysis_plan
            )
            available = {item.evidence_id: item for item in financial}
            for claim in claims:
                _validate_claim_against_evidence(
                    claim, (claim.evidence_id,), available
                )
                blocks.append(
                    _render_financial_claim(claim, available[claim.evidence_id])
                    + f" [{claim.evidence_id}]"
                )
                if claim.evidence_id not in cited_ids:
                    cited_ids.append(claim.evidence_id)

        if not blocks:
            return _insufficient_result(normalized_question, len(evidence))
        status = (
            GroundedSynthesisStatus.PARTIAL
            if coverage in (QualitativeCoverage.PARTIAL, QualitativeCoverage.INSUFFICIENT)
            and qualitative_requested
            else GroundedSynthesisStatus.GROUNDED
        )
        return GroundedSynthesisResult(
            normalized_question,
            "\n\n".join(blocks),
            tuple(cited_ids),
            status,
            len(evidence),
            claims,
            self._generation_client.provider,
            self._generation_client.model,
            coverage,
            missing,
        )

    def _generate_qualitative(
        self,
        question: str,
        evidence: tuple[UnifiedEvidence, ...],
        plan: AnalysisPlan | None,
    ) -> str:
        raw = self._generate_response(
            build_qualitative_synthesis_prompt(question, evidence, plan),
            QUALITATIVE_RESPONSE_SCHEMA,
        )
        return _parse_qualitative_response(raw)

    def _generate_financial(
        self,
        question: str,
        evidence: tuple[UnifiedEvidence, ...],
        plan: AnalysisPlan | None,
    ) -> tuple[ValidatedFinancialClaim, ...]:
        raw = self._generate_response(
            build_financial_synthesis_prompt(question, evidence, plan),
            _financial_response_schema(evidence),
        )
        claims = _parse_financial_response(raw)
        if not claims:
            raise FinancialGroundingError(
                "Financial generation must return at least one trusted claim."
            )
        return claims

    def _generate_response(self, prompt: str, schema: dict[str, object]) -> str:
        try:
            if isinstance(self._generation_client, StructuredGenerationClient):
                generated = self._generation_client.generate_structured(
                    prompt, schema
                )
            else:
                generated = self._generation_client.generate(prompt)
            return validate_generated_text(generated)
        except Exception as exc:
            raise GroundedGenerationError(
                "Grounded competitor generation failed safely."
            ) from exc


def build_grounded_synthesis_prompt(
    question: str,
    evidence: UnifiedEvidenceSet,
    analysis_plan: AnalysisPlan | None = None,
) -> str:
    """Backward-compatible prompt helper for all supplied evidence."""
    normalized_question = _required_text("question", question)
    if not isinstance(evidence, UnifiedEvidenceSet):
        raise TypeError("evidence must be a UnifiedEvidenceSet.")
    plan_record = _plan_record(analysis_plan) if analysis_plan is not None else None
    sections = ["QUESTION:\n" + normalized_question]
    if plan_record is not None:
        sections.append("ANALYSIS_PLAN:\n" + _json(plan_record))
    sections.append("UNIFIED_EVIDENCE_IN_CALLER_ORDER:\n" + _json([_evidence_record(item) for item in evidence]))
    return "\n\n".join(sections)


def build_qualitative_synthesis_prompt(
    question: str,
    evidence: tuple[UnifiedEvidence, ...],
    analysis_plan: AnalysisPlan | None = None,
) -> str:
    """Build a prose-only prompt from the exact Python-owned evidence scope."""
    return QUALITATIVE_SYNTHESIS_INSTRUCTIONS + "\n\n" + build_grounded_synthesis_prompt(
        question, UnifiedEvidenceSet(evidence), analysis_plan
    )


def build_financial_synthesis_prompt(
    question: str,
    evidence: tuple[UnifiedEvidence, ...],
    analysis_plan: AnalysisPlan | None = None,
) -> str:
    """Build a claim-only prompt from trusted financial evidence."""
    return (
        FINANCIAL_SYNTHESIS_INSTRUCTIONS
        + "\n\nAUTHORIZED_CLAIMS_IN_EVIDENCE_ORDER:\n"
        + _json(
            [claim for item in evidence for claim in _authorized_claim_records(item)]
        )
        + "\n\n"
        + build_grounded_synthesis_prompt(
            question, UnifiedEvidenceSet(evidence), analysis_plan
        )
    )


def _parse_json_object(raw_response: str, expected_keys: frozenset[str]) -> dict[str, object]:
    try:
        payload = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as exc:
        raise GroundedResponseFormatError(
            "Grounded generation response must be one JSON object."
        ) from exc
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise GroundedResponseFormatError(
            "Grounded generation response has an invalid JSON schema."
        )
    return payload


def _parse_qualitative_response(raw_response: str) -> str:
    payload = _parse_json_object(raw_response, _QUALITATIVE_RESPONSE_KEYS)
    text = _required_generated_text(payload.get("text"))
    if _CITATION_GROUP.search(text):
        raise GroundedResponseFormatError(
            "Qualitative generation must not contain evidence citation syntax."
        )
    return text


def _parse_financial_response(
    raw_response: str,
) -> tuple[ValidatedFinancialClaim, ...]:
    payload = _parse_json_object(raw_response, _FINANCIAL_RESPONSE_KEYS)
    return _parse_financial_claims(payload.get("claims"))


def _financial_response_schema(
    evidence: tuple[UnifiedEvidence, ...],
) -> dict[str, object]:
    """Return one strict union schema for every supported financial evidence type."""
    if any(not _authorized_claim_records(item) for item in evidence):
        raise FinancialGroundingError(
            "Financial evidence has no supported structured claim family."
        )
    return FINANCIAL_RESPONSE_SCHEMA


def _authorized_claim_records(item: UnifiedEvidence) -> tuple[dict[str, object], ...]:
    """Serialize exact trusted claim choices without asking the model to infer roles."""
    data = item.data
    if isinstance(data, FinancialFactEvidenceData):
        return ({
            "evidence_id": item.evidence_id,
            "claim_type": FinancialClaimType.REPORTED_FACT.value,
            "role": "reported_value",
            "value": str(data.source_value),
        },)
    if isinstance(data, FinancialCalculationEvidenceData):
        return ({
            "evidence_id": item.evidence_id,
            "claim_type": FinancialClaimType.CALCULATED_METRIC.value,
            "role": "calculated_value",
            "value": str(data.value),
        },)
    if isinstance(data, FinancialComparisonEvidenceData):
        return tuple(
            {
                "evidence_id": item.evidence_id,
                "claim_type": FinancialClaimType.COMPARISON_ENTRY.value,
                "role": "ranked_entry",
                "value": str(entry.value),
                "company_id": entry.company_id,
                "rank": entry.rank,
            }
            for entry in data.ranked_entries
        )
    if isinstance(data, FinancialChangeEvidenceData):
        return tuple(
            {
                "evidence_id": item.evidence_id,
                "claim_type": FinancialClaimType.FINANCIAL_CHANGE_VALUE.value,
                "role": role,
                "value": value,
            }
            for role, value in (
                ("earlier_value", str(data.earlier_value)),
                ("later_value", str(data.later_value)),
                ("percentage_point_change", str(data.percentage_point_change)),
            )
        )
    return ()


def _parse_financial_claims(
    value: object,
) -> tuple[ValidatedFinancialClaim, ...]:
    if not isinstance(value, list):
        raise GroundedResponseFormatError("financial_claims must be a list.")
    claims = tuple(_parse_financial_claim(item) for item in value)
    if len(claims) != len(set(claims)):
        raise FinancialGroundingError("Duplicate financial claims are not allowed.")
    return claims


def _parse_financial_claim(value: object) -> ValidatedFinancialClaim:
    if not isinstance(value, dict):
        raise GroundedResponseFormatError("Each financial claim must be an object.")
    claim_type_value = value.get("claim_type")
    try:
        claim_type = FinancialClaimType(claim_type_value)
    except (TypeError, ValueError) as exc:
        raise GroundedResponseFormatError(
            "Financial claim has an unsupported claim_type."
        ) from exc
    expected_keys = (
        _COMPARISON_CLAIM_KEYS
        if claim_type is FinancialClaimType.COMPARISON_ENTRY
        else _BASIC_CLAIM_KEYS
    )
    if set(value) != expected_keys:
        raise GroundedResponseFormatError("Financial claim has an invalid schema.")
    evidence_id = value.get("evidence_id")
    role = value.get("role")
    exact_value = value.get("value")
    if not isinstance(evidence_id, str) or not _EVIDENCE_ID.fullmatch(evidence_id):
        raise GroundedResponseFormatError("Financial claim evidence_id is invalid.")
    if not isinstance(role, str):
        raise GroundedResponseFormatError("Financial claim role must be a string.")
    if not isinstance(exact_value, str) or not _DECIMAL_TEXT.fullmatch(exact_value):
        raise GroundedResponseFormatError(
            "Financial claim value must be an exact decimal string."
        )
    company_id: str | None = None
    rank: int | None = None
    if claim_type is FinancialClaimType.COMPARISON_ENTRY:
        company_id = value.get("company_id")
        rank = value.get("rank")
        if not isinstance(company_id, str) or not company_id:
            raise GroundedResponseFormatError(
                "Comparison claim company_id must be a non-empty string."
            )
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise GroundedResponseFormatError("Comparison claim rank must be an integer.")
    return ValidatedFinancialClaim(
        evidence_id, claim_type, role, exact_value, company_id, rank
    )


def _validate_claim_against_evidence(
    claim: ValidatedFinancialClaim,
    cited_ids: tuple[str, ...],
    available: dict[str, UnifiedEvidence],
) -> None:
    if claim.evidence_id not in available:
        raise UnknownEvidenceCitationError(
            "Generated response referenced an unknown evidence ID."
        )
    if claim.evidence_id not in cited_ids:
        raise FinancialGroundingError(
            "A financial claim must reference normally cited evidence."
        )
    item = available[claim.evidence_id]
    if item.evidence_type is EvidenceType.QUALITATIVE:
        raise FinancialGroundingError(
            "Qualitative evidence cannot authorize a financial claim."
        )
    if not _claim_matches(item, claim):
        raise FinancialGroundingError(
            "Financial claim role or value changed or invented trusted evidence."
        )


def _claim_matches(item: UnifiedEvidence, claim: ValidatedFinancialClaim) -> bool:
    data = item.data
    if isinstance(data, FinancialFactEvidenceData):
        return (
            claim.claim_type is FinancialClaimType.REPORTED_FACT
            and claim.role == "reported_value"
            and claim.value == str(data.source_value)
            and claim.company_id is None
            and claim.rank is None
        )
    if isinstance(data, FinancialCalculationEvidenceData):
        return (
            claim.claim_type is FinancialClaimType.CALCULATED_METRIC
            and claim.role == "calculated_value"
            and claim.value == str(data.value)
            and claim.company_id is None
            and claim.rank is None
        )
    if isinstance(data, FinancialComparisonEvidenceData):
        if claim.claim_type is not FinancialClaimType.COMPARISON_ENTRY or claim.role != "ranked_entry":
            return False
        return any(
            (entry.company_id, entry.rank, str(entry.value))
            == (claim.company_id, claim.rank, claim.value)
            for entry in data.ranked_entries
        )
    if isinstance(data, FinancialChangeEvidenceData):
        trusted = {
            "earlier_value": str(data.earlier_value),
            "later_value": str(data.later_value),
            "percentage_point_change": str(data.percentage_point_change),
        }
        return (
            claim.claim_type is FinancialClaimType.FINANCIAL_CHANGE_VALUE
            and claim.role in trusted
            and claim.value == trusted[claim.role]
            and claim.company_id is None
            and claim.rank is None
        )
    return False


def _render_financial_claim(
    claim: ValidatedFinancialClaim,
    item: UnifiedEvidence,
) -> str:
    """Render only trusted, already-validated evidence fields; never recalculate."""
    data = item.data
    if isinstance(data, FinancialFactEvidenceData):
        return (
            f"{item.company_name} {item.fiscal_year} {data.metric} reported value: "
            f"{claim.value} {data.source_unit}"
        )
    if isinstance(data, FinancialCalculationEvidenceData):
        return (
            f"{item.company_name} {item.fiscal_year} {data.metric} calculated value: "
            f"{claim.value} {data.unit}"
        )
    if isinstance(data, FinancialComparisonEvidenceData):
        entry = next(
            entry
            for entry in data.ranked_entries
            if (entry.company_id, entry.rank, str(entry.value))
            == (claim.company_id, claim.rank, claim.value)
        )
        return (
            f"Rank {entry.rank} - {entry.company_name} {entry.fiscal_year} "
            f"{entry.metric}: {claim.value} {entry.unit}"
        )
    if isinstance(data, FinancialChangeEvidenceData):
        if claim.role == "earlier_value":
            return (
                f"{item.company_name} {data.metric} in {data.earlier_year}: "
                f"{claim.value} {data.unit}"
            )
        if claim.role == "later_value":
            return (
                f"{item.company_name} {data.metric} in {data.later_year}: "
                f"{claim.value} {data.unit}"
            )
        return (
            f"{item.company_name} {data.metric} change from {data.earlier_year} to "
            f"{data.later_year}: {claim.value} percentage points"
        )
    raise FinancialGroundingError(
        "Financial claim cannot be rendered from the supplied evidence type."
    )


def _evidence_record(item: UnifiedEvidence) -> dict[str, object]:
    record: dict[str, object] = {
        "evidence_id": item.evidence_id,
        "evidence_type": item.evidence_type.value,
        "company_id": item.company_id,
        "company_name": item.company_name,
        "fiscal_year": item.fiscal_year,
        "display_text": item.display_text,
        "source_references": [_source_record(value) for value in item.source_references],
    }
    data = item.data
    if isinstance(data, QualitativeEvidenceData):
        record["evidence"] = {
            "text": data.text,
            "document_type": data.document_type,
            "chunk_id": data.chunk_id,
        }
    elif isinstance(data, FinancialFactEvidenceData):
        record["evidence"] = {
            "metric": data.metric,
            "reported_source_value": str(data.source_value),
            "source_unit": data.source_unit,
            "currency": data.currency,
            "canonical_value_twd": str(data.canonical_value_twd),
            "period": data.period,
            "reporting_scope": data.reporting_scope,
        }
    elif isinstance(data, FinancialCalculationEvidenceData):
        record["evidence"] = _calculation_record(data)
    elif isinstance(data, FinancialComparisonEvidenceData):
        record["evidence"] = {
            "metric": data.metric,
            "fiscal_year": data.fiscal_year,
            "requested_companies": data.requested_companies,
            "missing_companies": data.missing_companies,
            "ranking_direction": data.ranking_direction,
            "comparison_status": data.status,
            "ranked_entries": [
                {
                    "rank": entry.rank,
                    "company_id": entry.company_id,
                    "company_name": entry.company_name,
                    "value": str(entry.value),
                    "unit": entry.unit,
                    "calculation": _calculation_record(entry.calculation),
                }
                for entry in data.ranked_entries
            ],
        }
    elif isinstance(data, FinancialChangeEvidenceData):
        record["evidence"] = {
            "metric": data.metric,
            "earlier_year": data.earlier_year,
            "later_year": data.later_year,
            "earlier_value": str(data.earlier_value),
            "later_value": str(data.later_value),
            "percentage_point_change": str(data.percentage_point_change),
            "direction": data.direction,
            "unit": data.unit,
            "earlier_calculation": _calculation_record(data.earlier_calculation),
            "later_calculation": _calculation_record(data.later_calculation),
        }
    return record


def _calculation_record(data: FinancialCalculationEvidenceData) -> dict[str, object]:
    return {
        "metric": data.metric,
        "calculated_value": str(data.value),
        "unit": data.unit,
        "formula": data.formula,
        "inputs": [
            {
                "metric": fact.metric,
                "fiscal_year": fact.fiscal_year,
                "source_value": str(fact.source_value),
                "source_unit": fact.source_unit,
                "canonical_value_twd": str(fact.canonical_value_twd),
                "source": _source_record(fact.source_reference),
            }
            for fact in data.input_facts
        ],
    }


def _source_record(reference: SourceReference) -> dict[str, object]:
    return {
        "source_document_id": reference.source_document_id,
        "source_title": reference.source_title,
        "page_number": reference.page_number,
        "source_metric": reference.source_metric,
        "fiscal_year": reference.fiscal_year,
        "chunk_id": reference.chunk_id,
        "document_type": reference.document_type,
    }


def _plan_record(plan: AnalysisPlan) -> dict[str, object]:
    if not isinstance(plan, AnalysisPlan):
        raise TypeError("analysis_plan must be an AnalysisPlan.")
    return {
        "route": plan.route.value if plan.route is not None else None,
        "status": plan.status.value,
        "requested_companies": plan.requested_companies,
        "qualitative_intents": plan.qualitative_intents,
        "financial_intents": plan.financial_intents,
        "fiscal_years": plan.fiscal_years,
        "unsupported_intents": plan.unsupported_intents,
    }


def _only_insufficient_comparisons(evidence: UnifiedEvidenceSet) -> bool:
    return bool(evidence) and all(
        isinstance(item.data, FinancialComparisonEvidenceData)
        and item.data.status == "insufficient"
        and not item.data.ranked_entries
        for item in evidence
    )


def _qualitative_requested(
    plan: AnalysisPlan | None,
    qualitative: tuple[UnifiedEvidence, ...],
) -> bool:
    if plan is None:
        return bool(qualitative)
    return plan.route in (AnalysisRoute.QUALITATIVE, AnalysisRoute.COMBINED)


def _qualitative_coverage(
    qualitative: tuple[UnifiedEvidence, ...],
    plan: AnalysisPlan | None,
) -> tuple[QualitativeCoverage, tuple[str, ...]]:
    if not _qualitative_requested(plan, qualitative):
        return QualitativeCoverage.NOT_REQUESTED, ()
    requested = plan.requested_companies if plan is not None else ()
    present = {item.company_id for item in qualitative if item.company_id is not None}
    missing = tuple(company for company in requested if company not in present)
    if not qualitative:
        return QualitativeCoverage.INSUFFICIENT, missing
    if missing:
        return QualitativeCoverage.PARTIAL, missing
    return QualitativeCoverage.COMPLETE, ()


def _insufficient_result(
    question: str,
    evidence_count: int,
    *,
    qualitative_coverage: QualitativeCoverage = QualitativeCoverage.NOT_REQUESTED,
    missing_qualitative_companies: tuple[str, ...] = (),
) -> GroundedSynthesisResult:
    return GroundedSynthesisResult(
        question,
        "The supplied evidence is insufficient to answer this question.",
        (),
        GroundedSynthesisStatus.INSUFFICIENT,
        evidence_count,
        (),
        None,
        None,
        qualitative_coverage,
        missing_qualitative_companies,
    )


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GroundedResponseFormatError(f"{name} must be a non-empty string.")
    return value.strip()


def _required_generated_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GroundedResponseFormatError("answer_text must be a non-empty string.")
    return value.strip()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
