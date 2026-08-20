"""Thin local FastAPI boundary for competitor-intelligence analysis."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator

from enterprise_rag.competitor_application import (
    CompetitorApplicationConfigurationError,
    CompetitorApplicationReadiness,
    CompetitorIntelligenceApplication,
    create_local_competitor_intelligence_application,
)
from enterprise_rag.competitor_citations import (
    RenderedCitation,
    RenderedCompetitorAnswer,
    RenderedSourceReference,
)
from enterprise_rag.competitor_grounded_synthesis import (
    CompanyObservation,
    ComparisonDimension,
    CompanyStrategyProfile,
    GroundedKeyTakeaway,
    GroundedGenerationError,
    GroundedSynthesisError,
    ValidatedFinancialClaim,
    StructuredComparison,
)
from enterprise_rag.competitor_orchestration import (
    CompetitorIntelligenceResult,
    CompetitorOrchestrationError,
)
from enterprise_rag.competitor_retrieval import RetrievalError
from enterprise_rag.financial_calculations import FinancialCalculationError
from enterprise_rag.financial_comparisons import FinancialComparisonError
from enterprise_rag.financial_facts import (
    FinancialFactNotFoundError,
    FinancialFactValidationError,
)
from enterprise_rag.generation import GenerationError
from enterprise_rag.providers.ollama import OllamaError
from enterprise_rag.config import load_settings


APIStatus = Literal[
    "completed", "partial", "ambiguous", "unsupported", "insufficient"
]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CompetitorAnalyzeRequest(APIModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must be a non-empty string")
        return normalized


class GenerationMetadataResponse(APIModel):
    provider: str | None
    model: str | None


class CitationSourceResponse(APIModel):
    source_title: str
    page_number: int | None
    source_metric: str | None
    fiscal_year: int | None
    document_type: str | None


class CitationResponse(APIModel):
    evidence_id: str
    evidence_type: str
    company_id: str | None
    company_name: str | None
    fiscal_year: int | None
    sources: tuple[CitationSourceResponse, ...]


class FinancialClaimResponse(APIModel):
    evidence_id: str
    claim_type: str
    role: str
    value: str
    company_id: str | None
    rank: int | None


class CompanyStrategyProfileResponse(APIModel):
    company_id: str
    summary: str
    evidence_ids: tuple[str, ...]


class CompanyObservationResponse(APIModel):
    company_id: str
    text: str
    evidence_ids: tuple[str, ...]


class ComparisonDimensionResponse(APIModel):
    label: str
    observations: tuple[CompanyObservationResponse, ...]


class GroundedKeyTakeawayResponse(APIModel):
    text: str
    evidence_ids: tuple[str, ...]


class StructuredComparisonResponse(APIModel):
    requested_companies: tuple[str, ...]
    covered_companies: tuple[str, ...]
    missing_companies: tuple[str, ...]
    company_profiles: tuple[CompanyStrategyProfileResponse, ...]
    comparison_dimensions: tuple[ComparisonDimensionResponse, ...]
    key_takeaway: GroundedKeyTakeawayResponse | None


class CompetitorAnalyzeResponse(APIModel):
    question: str
    status: APIStatus
    answer_text: str | None
    reasons: tuple[str, ...]
    citations: tuple[CitationResponse, ...]
    financial_claims: tuple[FinancialClaimResponse, ...]
    generation: GenerationMetadataResponse | None
    comparison: StructuredComparisonResponse | None = None
    response_language: Literal["zh-TW", "en"] | None = None


class HealthResponse(APIModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(APIModel):
    status: Literal["ready"] = "ready"
    companies: tuple[str, ...]
    financial_facts_loaded: bool
    embedding_provider: str
    embedding_model: str
    generation_provider: str
    generation_model: str


class APIErrorDetail(APIModel):
    code: str
    message: str


class APIErrorResponse(APIModel):
    error: APIErrorDetail


class APIResponseContractError(RuntimeError):
    """Raised when a domain result contradicts the public API contract."""


def create_competitor_api(
    application: CompetitorIntelligenceApplication | None = None,
    *,
    cors_origins: tuple[str, ...] | None = None,
) -> FastAPI:
    """Create a local API whose lifespan constructs at most one backend."""

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        api.state.competitor_application = application
        api.state.application_available = application is not None
        if application is None:
            try:
                api.state.competitor_application = await run_in_threadpool(
                    create_local_competitor_intelligence_application
                )
                api.state.application_available = True
            except Exception:
                # Startup remains available for safe health/readiness diagnostics.
                api.state.competitor_application = None
                api.state.application_available = False
        yield

    api = FastAPI(
        title="Local Competitor Intelligence API",
        version="0.1.0",
        lifespan=lifespan,
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=list(
            cors_origins
            if cors_origins is not None
            else load_settings().competitor_api_cors_origins
        ),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @api.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @api.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": APIErrorResponse}},
    )
    async def ready() -> ReadinessResponse | JSONResponse:
        backend = _backend(api)
        if backend is None:
            return _error_response(
                503,
                "application_unavailable",
                "The local competitor application is unavailable.",
            )
        return _readiness_response(backend.readiness)

    @api.post(
        "/api/competitor/analyze",
        response_model=CompetitorAnalyzeResponse,
        responses={
            500: {"model": APIErrorResponse},
            503: {"model": APIErrorResponse},
        },
    )
    async def analyze(
        request: CompetitorAnalyzeRequest,
    ) -> CompetitorAnalyzeResponse | JSONResponse:
        backend = _backend(api)
        if backend is None:
            return _error_response(
                503,
                "application_unavailable",
                "The local competitor application is unavailable.",
            )
        try:
            result = await run_in_threadpool(backend.run, request.question)
            return _analyze_response(result)
        except CompetitorApplicationConfigurationError:
            return _error_response(
                503,
                "application_unavailable",
                "The local competitor application is unavailable.",
            )
        except (OllamaError, GroundedGenerationError, GenerationError):
            return _error_response(
                503,
                "local_provider_unavailable",
                "The local model service is unavailable.",
            )
        except (
            CompetitorOrchestrationError,
            GroundedSynthesisError,
            RetrievalError,
            FinancialCalculationError,
            FinancialComparisonError,
            FinancialFactNotFoundError,
            FinancialFactValidationError,
        ):
            return _error_response(
                500,
                "analysis_failed",
                "Competitor analysis could not be completed.",
            )
        except Exception:
            return _error_response(
                500,
                "internal_error",
                "An unexpected internal error occurred.",
            )

    return api


def _backend(api: FastAPI) -> CompetitorIntelligenceApplication | None:
    value = getattr(api.state, "competitor_application", None)
    return value if isinstance(value, CompetitorIntelligenceApplication) else value


def _readiness_response(
    readiness: CompetitorApplicationReadiness,
) -> ReadinessResponse:
    return ReadinessResponse(
        companies=readiness.company_ids,
        financial_facts_loaded=readiness.financial_facts_loaded,
        embedding_provider=readiness.embedding_provider,
        embedding_model=readiness.embedding_model,
        generation_provider=readiness.generation_provider,
        generation_model=readiness.generation_model,
    )


def _analyze_response(result: CompetitorIntelligenceResult) -> CompetitorAnalyzeResponse:
    rendered = result.rendered_answer
    if result.status.value == "completed" and rendered is None:
        raise APIResponseContractError(
            "A completed competitor result requires a rendered answer."
        )
    if rendered is None:
        return CompetitorAnalyzeResponse(
            question=result.question,
            status=result.status.value,
            answer_text=None,
            reasons=result.reasons,
            citations=(),
            financial_claims=(),
            generation=None,
            comparison=None,
            response_language=None,
        )
    return CompetitorAnalyzeResponse(
        question=result.question,
        status=result.status.value,
        answer_text=rendered.answer_text,
        reasons=result.reasons,
        citations=tuple(_citation_response(item) for item in rendered.citations),
        financial_claims=tuple(
            _financial_claim_response(item) for item in rendered.financial_claims
        ),
        generation=GenerationMetadataResponse(
            provider=rendered.generation_provider,
            model=rendered.generation_model,
        ),
        comparison=_comparison_response(rendered.comparison),
        response_language=rendered.response_language.value,
    )


def _comparison_response(
    comparison: StructuredComparison | None,
) -> StructuredComparisonResponse | None:
    if comparison is None:
        return None
    return StructuredComparisonResponse(
        requested_companies=comparison.requested_companies,
        covered_companies=comparison.covered_companies,
        missing_companies=comparison.missing_companies,
        company_profiles=tuple(
            CompanyStrategyProfileResponse(
                company_id=item.company_id,
                summary=item.summary,
                evidence_ids=item.evidence_ids,
            ) for item in comparison.company_profiles
        ),
        comparison_dimensions=tuple(
            ComparisonDimensionResponse(
                label=item.label,
                observations=tuple(
                    CompanyObservationResponse(
                        company_id=observation.company_id,
                        text=observation.text,
                        evidence_ids=observation.evidence_ids,
                    ) for observation in item.observations
                ),
            ) for item in comparison.comparison_dimensions
        ),
        key_takeaway=(
            GroundedKeyTakeawayResponse(
                text=comparison.key_takeaway.text,
                evidence_ids=comparison.key_takeaway.evidence_ids,
            ) if comparison.key_takeaway else None
        ),
    )


def _citation_response(citation: RenderedCitation) -> CitationResponse:
    return CitationResponse(
        evidence_id=citation.evidence_id,
        evidence_type=citation.evidence_type.value,
        company_id=citation.company_id,
        company_name=citation.company_name,
        fiscal_year=citation.fiscal_year,
        sources=tuple(_source_response(item) for item in citation.source_references),
    )


def _source_response(source: RenderedSourceReference) -> CitationSourceResponse:
    return CitationSourceResponse(
        source_title=source.source_title,
        page_number=source.page_number,
        source_metric=source.source_metric,
        fiscal_year=source.fiscal_year,
        document_type=source.document_type,
    )


def _financial_claim_response(
    claim: ValidatedFinancialClaim,
) -> FinancialClaimResponse:
    return FinancialClaimResponse(
        evidence_id=claim.evidence_id,
        claim_type=claim.claim_type.value,
        role=claim.role,
        value=claim.value,
        company_id=claim.company_id,
        rank=claim.rank,
    )


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = APIErrorResponse(error=APIErrorDetail(code=code, message=message))
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


# Lightweight module entry point. Resource construction occurs in lifespan startup.
app = create_competitor_api()
