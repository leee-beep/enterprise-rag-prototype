"""Citation-ready, guarded synthesis for competitor retrieval evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from enterprise_rag.competitor_retrieval import (
    BalancedCompetitorRetriever,
    BalancedRetrievalResponse,
    CompetitorRetrievalResult,
)
from enterprise_rag.generation import GenerationClient, validate_generated_text


GUARDED_COMPETITOR_INSTRUCTIONS = """You are a competitor-analysis assistant.
Use only the supplied evidence. Do not add company facts from background knowledge.
Cite supported claims using only the provided evidence IDs in square brackets.
Never invent evidence IDs or page numbers. Do not characterize a company listed as
insufficient. State when evidence cannot support a requested comparison. Separate
companies clearly, preserve uncertainty, and do not infer causality or financial
performance unless the evidence explicitly supports it. Respond in the user's
language when practical."""

_CITATION_GROUP = re.compile(r"\[((?:E\d+\s*(?:,\s*E\d+\s*)*))\]")
_EVIDENCE_ID = re.compile(r"E\d+")


class CompetitorAnalysisError(RuntimeError):
    """Base error for guarded competitor synthesis."""


class CitationValidationError(CompetitorAnalysisError):
    """Raised when generated text refers to evidence that was not supplied."""


class CompetitorGenerationError(CompetitorAnalysisError):
    """Raised when guarded generation cannot return usable text."""


@dataclass(frozen=True)
class CitationReadyEvidence:
    evidence_id: str
    company_id: str
    company_name: str
    ticker: str | None
    fiscal_year: int | str | None
    document_type: str | None
    source_document_id: str | None
    source_title: str
    page_number: int | str | None
    chunk_id: str
    text: str
    retrieval_score: float
    quality_score: float
    original_candidate_rank: int
    final_company_rank: int


@dataclass(frozen=True)
class CompetitorCitation:
    evidence_id: str
    company_name: str
    ticker: str | None
    fiscal_year: int | str | None
    document_type: str | None
    source_title: str
    page_number: int | str | None
    chunk_id: str


@dataclass(frozen=True)
class CompanyEvidenceStatus:
    company_id: str
    company_name: str
    evidence_count: int
    sufficient: bool


@dataclass(frozen=True)
class CompetitorAnswer:
    question: str
    answer_text: str
    requested_companies: tuple[str, ...]
    answered_companies: tuple[str, ...]
    insufficient_companies: tuple[str, ...]
    evidence_status: tuple[CompanyEvidenceStatus, ...]
    evidence: tuple[CitationReadyEvidence, ...]
    citations: tuple[CompetitorCitation, ...]
    generation_model: str | None
    grounding_status: str


def build_citation_ready_evidence(
    response: BalancedRetrievalResponse,
) -> tuple[CitationReadyEvidence, ...]:
    """Convert selected results to deterministic, request-local evidence IDs."""
    evidence: list[CitationReadyEvidence] = []
    for number, selected in enumerate(response.results, start=1):
        result = selected.retrieval_result
        chunk = result.embedded_chunk.chunk
        metadata = result.metadata
        evidence.append(
            CitationReadyEvidence(
                evidence_id=f"E{number}",
                company_id=selected.company_id,
                company_name=selected.company_name,
                ticker=_optional_text(metadata.get("ticker")),
                fiscal_year=metadata.get("fiscal_year"),
                document_type=_optional_text(metadata.get("document_type")),
                source_document_id=_optional_text(metadata.get("source_document_id")),
                source_title=_safe_title(metadata.get("title"), chunk.file_name),
                page_number=metadata.get("page_number"),
                chunk_id=chunk.chunk_id,
                text=chunk.content,
                retrieval_score=result.score,
                quality_score=selected.quality_score,
                original_candidate_rank=selected.original_candidate_rank,
                final_company_rank=selected.company_rank,
            )
        )
    return tuple(evidence)


def build_evidence_status(
    response: BalancedRetrievalResponse,
) -> tuple[CompanyEvidenceStatus, ...]:
    """Require at least one selected usable item independently per company."""
    return tuple(
        CompanyEvidenceStatus(
            group.company_id,
            group.company_name,
            group.returned_count,
            group.returned_count >= 1,
        )
        for group in response.company_evidence
    )


def build_guarded_prompt(
    question: str,
    evidence: Sequence[CitationReadyEvidence],
    insufficient_companies: Sequence[str],
) -> str:
    sections = [GUARDED_COMPETITOR_INSTRUCTIONS, f"QUESTION:\n{question.strip()}", "EVIDENCE:"]
    for item in evidence:
        sections.append(
            f"[{item.evidence_id}]\n"
            f"Company: {item.company_name}\n"
            f"Year: {item.fiscal_year if item.fiscal_year is not None else 'unknown'}\n"
            f"Document: {item.source_title}\n"
            f"PDF Page: {item.page_number if item.page_number is not None else 'unknown'}\n"
            f"Content:\n{item.text}"
        )
    sections.append(
        "INSUFFICIENT EVIDENCE:\n"
        + (", ".join(insufficient_companies) if insufficient_companies else "None")
    )
    return "\n\n".join(sections)


def validate_and_build_citations(
    answer_text: str,
    evidence: Sequence[CitationReadyEvidence],
) -> tuple[CompetitorCitation, ...]:
    """Validate model references and render citations from trusted metadata."""
    lookup = {item.evidence_id: item for item in evidence}
    referenced: list[str] = []
    for group in _CITATION_GROUP.findall(answer_text):
        for evidence_id in _EVIDENCE_ID.findall(group):
            if evidence_id not in lookup:
                raise CitationValidationError(
                    f"Generated answer referenced unknown evidence ID [{evidence_id}]."
                )
            if evidence_id not in referenced:
                referenced.append(evidence_id)
    if evidence and not referenced:
        raise CitationValidationError(
            "Generated answer did not cite any supplied evidence ID."
        )
    return tuple(_citation_from_evidence(lookup[item]) for item in referenced)


def render_citation(citation: CompetitorCitation) -> str:
    year = citation.fiscal_year if citation.fiscal_year is not None else "unknown year"
    page = citation.page_number if citation.page_number is not None else "unknown"
    return f"[{citation.evidence_id}] {citation.company_name} — {year} {citation.source_title} — PDF p. {page}"


class CompetitorAnalysisPipeline:
    """Orchestrate balanced retrieval and guarded provider-neutral generation."""

    def __init__(
        self,
        retriever: BalancedCompetitorRetriever,
        generation_client: GenerationClient,
    ) -> None:
        self._retriever = retriever
        self._generation_client = generation_client

    def answer(
        self,
        question: str,
        company_ids: Sequence[str],
        top_k_per_company: int = 2,
    ) -> CompetitorAnswer:
        response = self._retriever.retrieve(question, company_ids, top_k_per_company)
        evidence = build_citation_ready_evidence(response)
        statuses = build_evidence_status(response)
        insufficient = tuple(item.company_name for item in statuses if not item.sufficient)
        answered = tuple(item.company_name for item in statuses if item.sufficient)
        requested = tuple(item.company_name for item in statuses)
        if not evidence:
            return CompetitorAnswer(
                question.strip(),
                "The retrieved evidence is insufficient to answer this question.",
                requested,
                (),
                insufficient,
                statuses,
                evidence,
                (),
                None,
                "insufficient",
            )
        prompt = build_guarded_prompt(question, evidence, insufficient)
        try:
            answer_text = validate_generated_text(self._generation_client.generate(prompt))
        except CitationValidationError:
            raise
        except Exception as exc:
            raise CompetitorGenerationError("Competitor answer generation failed safely.") from exc
        citations = validate_and_build_citations(answer_text, evidence)
        grounding = "partial" if insufficient else "grounded"
        return CompetitorAnswer(
            question.strip(), answer_text, requested, answered, insufficient,
            statuses, evidence, citations, self._generation_client.model, grounding,
        )


def _citation_from_evidence(item: CitationReadyEvidence) -> CompetitorCitation:
    return CompetitorCitation(
        item.evidence_id, item.company_name, item.ticker, item.fiscal_year,
        item.document_type, item.source_title, item.page_number, item.chunk_id,
    )


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _safe_title(value: object, fallback: str) -> str:
    title = _optional_text(value) or fallback
    return title.replace("\\", "/").rsplit("/", 1)[-1]
