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
from enterprise_rag.competitor_planning import AnalysisPlan, PlanStatus
from enterprise_rag.generation import GenerationClient, validate_generated_text


GROUNDED_SYNTHESIS_INSTRUCTIONS = """You are a grounded competitor-analysis writer.
Use ONLY the supplied evidence. Every factual claim must cite one or more supplied
evidence IDs using [E1] or [E1, E2]. Do not retrieve or introduce external knowledge.
Do not invent evidence IDs, sources, companies, rankings, values, or missing facts.
If evidence is insufficient, say so instead of guessing.

Evidence types have different authority:
- qualitative: source text from a cited document.
- financial_fact: a value reported directly by the source statement.
- financial_calculation: a deterministic Python-calculated value, not a reported fact.
- financial_comparison: a deterministic supplied ranking; never rerank it.
- financial_change: a supplied percentage-point change; never recompute it.

Copy every supplied financial value and every marker field exactly. Never calculate, recalculate,
round, convert, alter, or infer a financial value. Use only these markers:
- reported fact: [[E1:reported_value:123.45]]
- Python-calculated metric: [[E2:calculated_value:25.00]]
- supplied ranking entry: [[E3:ranked_entry:example_company:1:25.00]]
- year change: [[E4:earlier_value:20.00]], [[E4:later_value:25.00]], or
  [[E4:percentage_point_change:5.00]]
The examples are synthetic. For every marker, return a matching financial_claims entry.
Use claim_type reported_fact for reported_value, calculated_metric for calculated_value,
comparison_entry for ranked_entry, and financial_change_value for change roles. Never
swap a ranking company, rank, or value. Never use a
calculation input as its calculated result. Never relabel a reported fact as calculated,
or an earlier/later value as the percentage-point change. Do not write bare numbers when
citing financial evidence, including years, ranks, and tickers. Every marker's evidence
must also appear as a normal citation. The application validates and removes markers.

Return exactly one JSON object with these keys and no others:
{"answer_text":"grounded prose with citations","cited_evidence_ids":["E1"],
"financial_claims":[{"evidence_id":"E1","claim_type":"reported_fact",
"role":"reported_value","value":"123.45"}],"insufficient":false}
For comparison_entry only, also include company_id and integer rank. For qualitative-only
answers, financial_claims must be an empty list. Copy fields exactly; do not add fields.
Do not wrap the JSON in Markdown fences."""

_EVIDENCE_ID = re.compile(r"E[1-9]\d*")
_CITATION_GROUP = re.compile(r"\[((?:E\d+\s*(?:,\s*E\d+\s*)*))\]")
_MARKER_TOKEN = re.compile(r"\[\[([^\[\]\r\n]*)\]\]")
_DECIMAL_TEXT = re.compile(r"[-+]?\d+(?:\.\d+)?")
_BARE_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?!\w)")
_RESPONSE_KEYS = frozenset(
    {"answer_text", "cited_evidence_ids", "financial_claims", "insufficient"}
)
_BASIC_CLAIM_KEYS = frozenset({"evidence_id", "claim_type", "role", "value"})
_COMPARISON_CLAIM_KEYS = _BASIC_CLAIM_KEYS | {"company_id", "rank"}


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
    INSUFFICIENT = "insufficient"


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


class GroundedCompetitorSynthesizer:
    """Generate prose from trusted evidence without orchestration or calculation."""

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
        if not evidence or _only_insufficient_comparisons(evidence):
            return _insufficient_result(normalized_question, len(evidence))

        prompt = build_grounded_synthesis_prompt(
            normalized_question, evidence, analysis_plan
        )
        try:
            raw_response = validate_generated_text(
                self._generation_client.generate(prompt)
            )
        except Exception as exc:
            raise GroundedGenerationError(
                "Grounded competitor generation failed safely."
            ) from exc
        parsed = _parse_response(raw_response, evidence)
        status = (
            GroundedSynthesisStatus.INSUFFICIENT
            if parsed.insufficient
            else GroundedSynthesisStatus.GROUNDED
        )
        return GroundedSynthesisResult(
            normalized_question,
            parsed.answer_text,
            parsed.cited_evidence_ids,
            status,
            len(evidence),
            parsed.financial_claims,
            self._generation_client.provider,
            self._generation_client.model,
        )


@dataclass(frozen=True)
class _ParsedResponse:
    answer_text: str
    cited_evidence_ids: tuple[str, ...]
    financial_claims: tuple[ValidatedFinancialClaim, ...]
    insufficient: bool


def build_grounded_synthesis_prompt(
    question: str,
    evidence: UnifiedEvidenceSet,
    analysis_plan: AnalysisPlan | None = None,
) -> str:
    """Build a deterministic prompt without executing planning or evidence creation."""
    normalized_question = _required_text("question", question)
    if not isinstance(evidence, UnifiedEvidenceSet):
        raise TypeError("evidence must be a UnifiedEvidenceSet.")
    plan_record = _plan_record(analysis_plan) if analysis_plan is not None else None
    sections = [
        GROUNDED_SYNTHESIS_INSTRUCTIONS,
        "QUESTION:\n" + normalized_question,
    ]
    if plan_record is not None:
        sections.append("ANALYSIS_PLAN:\n" + _json(plan_record))
    sections.append(
        "UNIFIED_EVIDENCE_IN_CALLER_ORDER:\n"
        + _json([_evidence_record(item) for item in evidence])
    )
    return "\n\n".join(sections)


def _parse_response(raw_response: str, evidence: UnifiedEvidenceSet) -> _ParsedResponse:
    try:
        payload = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as exc:
        raise GroundedResponseFormatError(
            "Grounded generation response must be one JSON object."
        ) from exc
    if not isinstance(payload, dict) or set(payload) != _RESPONSE_KEYS:
        raise GroundedResponseFormatError(
            "Grounded generation response has an invalid JSON schema."
        )
    answer_text = _required_generated_text(payload.get("answer_text"))
    declared = payload.get("cited_evidence_ids")
    insufficient = payload.get("insufficient")
    if (
        not isinstance(declared, list)
        or not all(isinstance(item, str) and _EVIDENCE_ID.fullmatch(item) for item in declared)
        or len(declared) != len(set(declared))
    ):
        raise GroundedResponseFormatError(
            "cited_evidence_ids must be a unique list of evidence IDs."
        )
    if not isinstance(insufficient, bool):
        raise GroundedResponseFormatError("insufficient must be a boolean.")

    available = {item.evidence_id: item for item in evidence}
    unknown_declared = tuple(item for item in declared if item not in available)
    cited_in_text = _citation_ids(answer_text)
    unknown_text = tuple(item for item in cited_in_text if item not in available)
    if unknown_declared or unknown_text:
        raise UnknownEvidenceCitationError(
            "Generated response referenced an unknown evidence ID."
        )
    if tuple(declared) != cited_in_text:
        raise GroundedResponseFormatError(
            "Declared evidence IDs must match citations in answer order."
        )
    if not insufficient and not cited_in_text:
        raise GroundedResponseFormatError(
            "A grounded response must cite at least one supplied evidence ID."
        )

    validated_answer, marker_claims = _validate_financial_values(
        answer_text, cited_in_text, available
    )
    financial_claims = _parse_financial_claims(
        payload.get("financial_claims"), cited_in_text, available
    )
    if marker_claims != financial_claims:
        raise FinancialGroundingError(
            "Financial claims must match answer markers exactly and in order."
        )
    return _ParsedResponse(
        validated_answer, cited_in_text, financial_claims, insufficient
    )


def _validate_financial_values(
    answer_text: str,
    cited_ids: tuple[str, ...],
    available: dict[str, UnifiedEvidence],
) -> tuple[str, tuple[ValidatedFinancialClaim, ...]]:
    has_financial_evidence = any(
        item.evidence_type is not EvidenceType.QUALITATIVE
        for item in available.values()
    )
    replacements: dict[tuple[int, int], str] = {}
    marker_claims: list[ValidatedFinancialClaim] = []
    for marker in _MARKER_TOKEN.finditer(answer_text):
        claim = _validate_financial_marker(marker.group(1), cited_ids, available)
        marker_claims.append(claim)
        replacements[marker.span()] = claim.value

    if len(marker_claims) != len(set(marker_claims)):
        raise FinancialGroundingError("Duplicate financial markers are not allowed.")

    text_without_markers = _replace_markers(answer_text, replacements, replacement="")
    if "[[" in text_without_markers or "]]" in text_without_markers:
        raise FinancialGroundingError("Generated response contains a malformed marker.")
    text_without_citations = _CITATION_GROUP.sub("", text_without_markers)
    if has_financial_evidence and _BARE_NUMBER.search(text_without_citations):
        raise FinancialGroundingError(
            "Financial responses must bind every numeric claim to evidence."
        )
    return _replace_markers(answer_text, replacements), tuple(marker_claims)


def _validate_financial_marker(
    marker_text: str,
    cited_ids: tuple[str, ...],
    available: dict[str, UnifiedEvidence],
) -> ValidatedFinancialClaim:
    fields = marker_text.split(":")
    if len(fields) < 2 or not _EVIDENCE_ID.fullmatch(fields[0]):
        raise FinancialGroundingError("Generated response contains a malformed marker.")
    evidence_id, role = fields[:2]
    if evidence_id not in available:
        raise UnknownEvidenceCitationError(
            "Generated response referenced an unknown evidence ID."
        )
    if evidence_id not in cited_ids:
        raise FinancialGroundingError(
            "A financial value marker must reference cited evidence."
        )
    expected_fields = 5 if role == "ranked_entry" else 3
    if len(fields) != expected_fields:
        raise FinancialGroundingError("Financial marker has the wrong field count.")
    if role == "ranked_entry":
        rank_text = fields[3]
        if not rank_text.isascii() or not rank_text.isdecimal():
            raise FinancialGroundingError("Ranking marker rank must be an integer.")
    value = fields[-1]
    if not _DECIMAL_TEXT.fullmatch(value):
        raise FinancialGroundingError("Financial marker has an invalid decimal value.")
    claim_type = _claim_type_for_role(role)
    claim = ValidatedFinancialClaim(
        evidence_id=evidence_id,
        claim_type=claim_type,
        role=role,
        value=value,
        company_id=fields[2] if role == "ranked_entry" else None,
        rank=int(fields[3]) if role == "ranked_entry" else None,
    )
    _validate_claim_against_evidence(claim, cited_ids, available)
    return claim


def _parse_financial_claims(
    value: object,
    cited_ids: tuple[str, ...],
    available: dict[str, UnifiedEvidence],
) -> tuple[ValidatedFinancialClaim, ...]:
    if not isinstance(value, list):
        raise GroundedResponseFormatError("financial_claims must be a list.")
    claims = tuple(_parse_financial_claim(item) for item in value)
    if len(claims) != len(set(claims)):
        raise FinancialGroundingError("Duplicate financial claims are not allowed.")
    for claim in claims:
        _validate_claim_against_evidence(claim, cited_ids, available)
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


def _claim_type_for_role(role: str) -> FinancialClaimType:
    mapping = {
        "reported_value": FinancialClaimType.REPORTED_FACT,
        "calculated_value": FinancialClaimType.CALCULATED_METRIC,
        "ranked_entry": FinancialClaimType.COMPARISON_ENTRY,
        "earlier_value": FinancialClaimType.FINANCIAL_CHANGE_VALUE,
        "later_value": FinancialClaimType.FINANCIAL_CHANGE_VALUE,
        "percentage_point_change": FinancialClaimType.FINANCIAL_CHANGE_VALUE,
    }
    try:
        return mapping[role]
    except KeyError as exc:
        raise FinancialGroundingError(
            "Financial marker role or value is unsupported."
        ) from exc


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


def _replace_markers(
    text: str,
    replacements: dict[tuple[int, int], str],
    *,
    replacement: str | None = None,
) -> str:
    parts: list[str] = []
    cursor = 0
    for (start, end), value in replacements.items():
        parts.append(text[cursor:start])
        parts.append(value if replacement is None else replacement)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _citation_ids(answer_text: str) -> tuple[str, ...]:
    ordered: list[str] = []
    for group in _CITATION_GROUP.findall(answer_text):
        for evidence_id in _EVIDENCE_ID.findall(group):
            if evidence_id not in ordered:
                ordered.append(evidence_id)
    return tuple(ordered)


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


def _insufficient_result(question: str, evidence_count: int) -> GroundedSynthesisResult:
    return GroundedSynthesisResult(
        question,
        "The supplied evidence is insufficient to answer this question.",
        (),
        GroundedSynthesisStatus.INSUFFICIENT,
        evidence_count,
        (),
        None,
        None,
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
