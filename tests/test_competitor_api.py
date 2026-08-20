"""Offline HTTP-contract tests for the local competitor API."""

from __future__ import annotations

import importlib
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import enterprise_rag.competitor_api as api_module
from enterprise_rag.competitor_api import create_competitor_api
from enterprise_rag.competitor_application import (
    CompetitorApplicationConfigurationError,
    CompetitorApplicationReadiness,
)
from enterprise_rag.competitor_citations import (
    RenderedCitation,
    RenderedCompetitorAnswer,
    RenderedFinancialFactDetails,
    RenderedQualitativeDetails,
    RenderedSourceReference,
)
from enterprise_rag.competitor_evidence import EvidenceType
from enterprise_rag.competitor_grounded_synthesis import (
    CompanyObservation,
    CompanyStrategyProfile,
    ComparisonDimension,
    FinancialClaimType,
    GroundedSynthesisStatus,
    ValidatedFinancialClaim,
    GroundedKeyTakeaway,
    ResponseLanguage,
    StructuredComparison,
)
from enterprise_rag.competitor_orchestration import (
    CompetitorIntelligenceResult,
    CompetitorOrchestrationError,
    OrchestrationStatus,
)
from enterprise_rag.competitor_planning import (
    AnalysisPlan,
    AnalysisRoute,
    PlanStatus,
)
from enterprise_rag.providers.ollama import OllamaTimeoutError


READINESS = CompetitorApplicationReadiness(
    ("gigabyte", "asus", "msi"),
    True,
    "ollama",
    "synthetic-embedding",
    "ollama",
    "synthetic-generation",
)


class FakeApplication:
    def __init__(self, result: CompetitorIntelligenceResult | Exception) -> None:
        self.readiness = READINESS
        self.result = result
        self.questions: list[str] = []

    def run(self, question: str) -> CompetitorIntelligenceResult:
        self.questions.append(question)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def plan(question: str, status: PlanStatus = PlanStatus.READY) -> AnalysisPlan:
    return AnalysisPlan(
        question=question,
        route=AnalysisRoute.COMBINED if status is PlanStatus.READY else None,
        status=status,
        requested_companies=("asus",),
        qualitative_intents=("ai_strategy",) if status is PlanStatus.READY else (),
        financial_intents=(),
        financial_request=None,
        fiscal_years=(2025,),
        unsupported_intents=(),
        reasons=(),
    )


def completed_result() -> CompetitorIntelligenceResult:
    question = "Compare ASUS strategy and margin."
    claim = ValidatedFinancialClaim(
        "E2", FinancialClaimType.CALCULATED_METRIC, "calculated_value", "40.00"
    )
    qualitative_source = RenderedSourceReference(
        "Synthetic Annual Report", 42, None, 2025, "annual_report"
    )
    financial_source = RenderedSourceReference(
        "Synthetic Financial Report", None, "gross_margin", 2025,
        "consolidated_financial_report",
    )
    answer = RenderedCompetitorAnswer(
        question=question,
        answer_text="Synthetic grounded answer.",
        citations=(
            RenderedCitation(
                "E1", EvidenceType.QUALITATIVE, "asus", "ASUS", 2025,
                (qualitative_source,), (),
                RenderedQualitativeDetails("annual_report", "internal-chunk-id"),
            ),
            RenderedCitation(
                "E2", EvidenceType.FINANCIAL_CALCULATION, "asus", "ASUS", 2025,
                (financial_source,), (claim,),
                RenderedFinancialFactDetails(
                    "gross_margin", "40.00", "percent", "TWD", None
                ),
            ),
        ),
        status=GroundedSynthesisStatus.GROUNDED,
        financial_claims=(claim,),
        generation_provider="ollama",
        generation_model="synthetic-generation",
    )
    return CompetitorIntelligenceResult(
        question, plan(question), OrchestrationStatus.COMPLETED, answer, ()
    )


def domain_result(status: OrchestrationStatus) -> CompetitorIntelligenceResult:
    question = "Synthetic domain question"
    plan_status = (
        PlanStatus.AMBIGUOUS
        if status is OrchestrationStatus.AMBIGUOUS
        else PlanStatus.UNSUPPORTED
        if status is OrchestrationStatus.UNSUPPORTED
        else PlanStatus.READY
    )
    return CompetitorIntelligenceResult(
        question, plan(question, plan_status), status, None, ("synthetic_reason",)
    )


def client(application: FakeApplication) -> TestClient:
    return TestClient(
        create_competitor_api(
            application,  # type: ignore[arg-type]
            cors_origins=("http://localhost:3000",),
        )
    )


def test_cors_allows_only_configured_origin_and_supports_analyze_preflight() -> None:
    backend = FakeApplication(completed_result())
    with client(backend) as http:
        allowed = http.get("/ready", headers={"Origin": "http://localhost:3000"})
        unlisted = http.get("/ready", headers={"Origin": "http://example.test"})
        preflight = http.options(
            "/api/competitor/analyze",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in unlisted.headers
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "POST" in preflight.headers["access-control-allow-methods"]
    assert "access-control-allow-credentials" not in preflight.headers
    assert backend.questions == []


def test_health_is_stable_and_does_not_run_backend() -> None:
    backend = FakeApplication(completed_result())
    with client(backend) as http:
        response = http.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert backend.questions == []


def test_readiness_exposes_only_safe_operational_metadata() -> None:
    backend = FakeApplication(completed_result())
    with client(backend) as http:
        response = http.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "companies": ["gigabyte", "asus", "msi"],
        "financial_facts_loaded": True,
        "embedding_provider": "ollama",
        "embedding_model": "synthetic-embedding",
        "generation_provider": "ollama",
        "generation_model": "synthetic-generation",
    }
    serialized = response.text.casefold()
    assert "path" not in serialized and "manifest" not in serialized
    assert backend.questions == []


def test_completed_analysis_has_frontend_ready_safe_schema() -> None:
    backend = FakeApplication(completed_result())
    with client(backend) as http:
        response = http.post(
            "/api/competitor/analyze",
            json={"question": "  Compare ASUS strategy and margin.  "},
        )
    assert response.status_code == 200
    body = response.json()
    assert backend.questions == ["Compare ASUS strategy and margin."]
    assert body["status"] == "completed"
    assert body["answer_text"] == "Synthetic grounded answer."
    assert body["generation"] == {
        "provider": "ollama", "model": "synthetic-generation"
    }
    assert body["financial_claims"] == [{
        "evidence_id": "E2",
        "claim_type": "calculated_metric",
        "role": "calculated_value",
        "value": "40.00",
        "company_id": None,
        "rank": None,
    }]
    assert body["citations"][0]["sources"][0]["page_number"] == 42
    assert body["citations"][1]["sources"][0]["page_number"] is None
    assert body["citations"][0]["evidence_type"] == "qualitative"
    assert body["comparison"] is None
    assert body["response_language"] == "en"
    serialized = response.text.casefold()
    assert "chunk_id" not in serialized
    assert "sha256" not in serialized
    assert "source_document" not in serialized
    assert "data/private" not in serialized


def test_structured_comparison_is_exposed_without_parsing_answer_text() -> None:
    base = completed_result()
    comparison = StructuredComparison(
        ("asus", "msi"), ("asus", "msi"), (),
        (CompanyStrategyProfile("asus", "ASUS profile", ("E1",)), CompanyStrategyProfile("msi", "MSI profile", ("E3",))),
        (ComparisonDimension("Positioning", (CompanyObservation("asus", "ASUS observation", ("E1",)), CompanyObservation("msi", "MSI observation", ("E3",)))),),
        GroundedKeyTakeaway("Grounded takeaway", ("E1", "E3")),
    )
    rendered = replace(base.rendered_answer, comparison=comparison, response_language=ResponseLanguage.ZH_TW)
    with client(FakeApplication(replace(base, rendered_answer=rendered))) as http:
        body = http.post("/api/competitor/analyze", json={"question": base.question}).json()
    assert body["answer_text"] == "Synthetic grounded answer."
    assert body["response_language"] == "zh-TW"
    assert len(body["comparison"]["company_profiles"]) == 2
    assert body["comparison"]["key_takeaway"]["evidence_ids"] == ["E1", "E3"]


@pytest.mark.parametrize(
    "status",
    (
        OrchestrationStatus.AMBIGUOUS,
        OrchestrationStatus.UNSUPPORTED,
        OrchestrationStatus.INSUFFICIENT,
    ),
)
def test_domain_states_remain_http_200(status: OrchestrationStatus) -> None:
    with client(FakeApplication(domain_result(status))) as http:
        response = http.post(
            "/api/competitor/analyze", json={"question": "Domain question"}
        )
    assert response.status_code == 200
    assert response.json()["status"] == status.value
    assert response.json()["answer_text"] is None
    assert response.json()["citations"] == []


def test_partial_result_remains_http_200() -> None:
    base = completed_result()
    rendered = replace(
        base.rendered_answer,
        status=GroundedSynthesisStatus.PARTIAL,
    )
    result = replace(
        base,
        status=OrchestrationStatus.PARTIAL,
        rendered_answer=rendered,
        reasons=("qualitative_evidence_unavailable:msi",),
    )
    with client(FakeApplication(result)) as http:
        response = http.post(
            "/api/competitor/analyze", json={"question": "Partial question"}
        )
    assert response.status_code == 200
    assert response.json()["status"] == "partial"
    assert response.json()["answer_text"] == "Synthetic grounded answer."


def test_completed_result_without_rendered_answer_fails_safely() -> None:
    invalid = CompetitorIntelligenceResult(
        "Contradictory question",
        plan("Contradictory question"),
        OrchestrationStatus.COMPLETED,
        None,
        (),
    )
    with client(FakeApplication(invalid)) as http:
        response = http.post(
            "/api/competitor/analyze", json={"question": "Question"}
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "rendered" not in response.text.casefold()


@pytest.mark.parametrize(
    "payload",
    ({}, {"question": ""}, {"question": "   "}, {"question": 123}),
)
def test_request_validation_is_stable(payload: dict[str, object]) -> None:
    backend = FakeApplication(completed_result())
    with client(backend) as http:
        response = http.post("/api/competitor/analyze", json=payload)
    assert response.status_code == 422
    assert backend.questions == []


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    (
        (
            CompetitorApplicationConfigurationError("private path detail"),
            503,
            "application_unavailable",
        ),
        (OllamaTimeoutError("private prompt detail"), 503, "local_provider_unavailable"),
        (CompetitorOrchestrationError("private financial detail"), 500, "analysis_failed"),
        (RuntimeError("private unexpected detail"), 500, "internal_error"),
    ),
)
def test_errors_are_mapped_without_private_details(
    error: Exception, status_code: int, code: str
) -> None:
    with client(FakeApplication(error)) as http:
        response = http.post(
            "/api/competitor/analyze", json={"question": "Synthetic question"}
        )
    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    assert "private" not in response.text.casefold()


def test_startup_constructs_backend_once_and_requests_reuse_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeApplication(completed_result())
    factory_calls: list[None] = []

    def factory() -> FakeApplication:
        factory_calls.append(None)
        return backend

    monkeypatch.setattr(
        api_module, "create_local_competitor_intelligence_application", factory
    )
    api = create_competitor_api()
    with TestClient(api) as http:
        first = http.post("/api/competitor/analyze", json={"question": "First"})
        second = http.post("/api/competitor/analyze", json={"question": "Second"})
    assert first.status_code == second.status_code == 200
    assert factory_calls == [None]
    assert backend.questions == ["First", "Second"]


def test_failed_startup_keeps_health_and_safe_unavailable_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[None] = []

    def fail() -> None:
        calls.append(None)
        raise CompetitorApplicationConfigurationError("private path detail")

    monkeypatch.setattr(
        api_module, "create_local_competitor_intelligence_application", fail
    )
    with TestClient(create_competitor_api()) as http:
        health = http.get("/health")
        ready = http.get("/ready")
        analyze = http.post(
            "/api/competitor/analyze", json={"question": "Question"}
        )
    assert calls == [None]
    assert health.status_code == 200
    assert ready.status_code == analyze.status_code == 503
    assert ready.json()["error"]["code"] == "application_unavailable"
    assert "private" not in ready.text.casefold()


def test_openapi_contains_public_endpoints_without_sensitive_examples() -> None:
    backend = FakeApplication(completed_result())
    with client(backend) as http:
        response = http.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert set(paths) >= {"/health", "/ready", "/api/competitor/analyze"}
    serialized = response.text.casefold()
    assert "data/private" not in serialized and "api_key" not in serialized


def test_import_does_not_construct_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import enterprise_rag.competitor_application as application_module

    def forbidden() -> object:
        raise AssertionError("API import constructed the backend")

    monkeypatch.setattr(
        application_module,
        "create_local_competitor_intelligence_application",
        forbidden,
    )
    importlib.reload(api_module)
