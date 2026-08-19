"""Provider-neutral orchestration for the competitor intelligence backend."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from enterprise_rag.competitor_analysis import (
    CitationReadyEvidence,
    build_citation_ready_evidence,
)
from enterprise_rag.competitor_citations import (
    RenderedCompetitorAnswer,
    render_competitor_answer,
)
from enterprise_rag.competitor_evidence import (
    EvidenceInput,
    UnifiedEvidenceBuilder,
    UnifiedEvidenceSet,
)
from enterprise_rag.competitor_grounded_synthesis import (
    GroundedCompetitorSynthesizer,
    GroundedSynthesisResult,
    GroundedSynthesisStatus,
)
from enterprise_rag.competitor_planning import (
    AnalysisPlan,
    AnalysisRoute,
    DeterministicQuestionRouter,
    FinancialOperation,
    PlanStatus,
)
from enterprise_rag.competitor_retrieval import BalancedCompetitorRetriever
from enterprise_rag.financial_calculations import (
    FinancialCalculationEngine,
    SUPPORTED_CALCULATED_METRICS,
)
from enterprise_rag.financial_comparisons import FinancialComparisonEngine
from enterprise_rag.financial_facts import FinancialFactCollection, SUPPORTED_METRICS


RAW_FINANCIAL_METRICS = SUPPORTED_METRICS - SUPPORTED_CALCULATED_METRICS


class CompetitorOrchestrationError(RuntimeError):
    """Raised when a ready plan cannot map to an existing trusted operation."""


class OrchestrationStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class CompetitorIntelligenceResult:
    question: str
    plan: AnalysisPlan
    status: OrchestrationStatus
    rendered_answer: RenderedCompetitorAnswer | None
    reasons: tuple[str, ...]


class QualitativeEvidenceProvider(Protocol):
    def retrieve(
        self,
        question: str,
        company_ids: tuple[str, ...],
        qualitative_intents: tuple[str, ...],
    ) -> tuple[CitationReadyEvidence, ...]: ...


class BalancedQualitativeEvidenceProvider:
    """Thin adapter from existing balanced retrieval to qualitative evidence."""

    def __init__(
        self,
        retriever: BalancedCompetitorRetriever,
        top_k_per_company: int = 2,
    ) -> None:
        if (
            not isinstance(top_k_per_company, int)
            or isinstance(top_k_per_company, bool)
            or top_k_per_company < 1
        ):
            raise ValueError("top_k_per_company must be a positive integer.")
        self._retriever = retriever
        self._top_k_per_company = top_k_per_company

    def retrieve(
        self,
        question: str,
        company_ids: tuple[str, ...],
        qualitative_intents: tuple[str, ...],
    ) -> tuple[CitationReadyEvidence, ...]:
        del qualitative_intents  # The existing retriever expands the original question.
        response = self._retriever.retrieve(
            question, company_ids, self._top_k_per_company
        )
        return build_citation_ready_evidence(response)


class CompetitorIntelligencePipeline:
    """Coordinate planning, trusted evidence paths, synthesis, and rendering."""

    def __init__(
        self,
        qualitative_provider: QualitativeEvidenceProvider,
        financial_facts: FinancialFactCollection,
        calculation_engine: FinancialCalculationEngine,
        comparison_engine: FinancialComparisonEngine,
        synthesizer: GroundedCompetitorSynthesizer,
        *,
        planner: DeterministicQuestionRouter | None = None,
        evidence_builder: UnifiedEvidenceBuilder | None = None,
        renderer: Callable[
            [GroundedSynthesisResult, UnifiedEvidenceSet],
            RenderedCompetitorAnswer,
        ] = render_competitor_answer,
    ) -> None:
        self._qualitative = qualitative_provider
        self._financial_facts = financial_facts
        self._calculations = calculation_engine
        self._comparisons = comparison_engine
        self._synthesizer = synthesizer
        self._planner = planner or DeterministicQuestionRouter()
        self._evidence_builder = evidence_builder or UnifiedEvidenceBuilder()
        self._renderer = renderer

    def run(self, question: str) -> CompetitorIntelligenceResult:
        """Execute one deterministic plan and return a rendered or non-ready result."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")
        normalized_question = question.strip()
        plan = self._planner.plan(normalized_question)
        if plan.status is not PlanStatus.READY:
            status = (
                OrchestrationStatus.AMBIGUOUS
                if plan.status is PlanStatus.AMBIGUOUS
                else OrchestrationStatus.UNSUPPORTED
            )
            return CompetitorIntelligenceResult(
                normalized_question, plan, status, None, plan.reasons
            )

        evidence_inputs: list[EvidenceInput] = []
        execution_reasons = list(plan.reasons)
        if plan.route in (AnalysisRoute.QUALITATIVE, AnalysisRoute.COMBINED):
            qualitative = self._qualitative.retrieve(
                normalized_question,
                plan.requested_companies,
                plan.qualitative_intents,
            )
            evidence_inputs.extend(qualitative)
            if not qualitative:
                execution_reasons.append("qualitative_evidence_unavailable")
                if plan.route is AnalysisRoute.QUALITATIVE:
                    return CompetitorIntelligenceResult(
                        normalized_question,
                        plan,
                        OrchestrationStatus.INSUFFICIENT,
                        None,
                        tuple(execution_reasons),
                    )

        if plan.route in (AnalysisRoute.FINANCIAL, AnalysisRoute.COMBINED):
            evidence_inputs.extend(self._execute_financial(plan))

        if not evidence_inputs:
            return CompetitorIntelligenceResult(
                normalized_question,
                plan,
                OrchestrationStatus.INSUFFICIENT,
                None,
                tuple(execution_reasons + ["no_evidence_available"]),
            )

        evidence = self._evidence_builder.build(evidence_inputs)
        synthesis = self._synthesizer.synthesize(
            normalized_question, evidence, plan
        )
        execution_reasons.extend(
            f"qualitative_evidence_unavailable:{company_id}"
            for company_id in synthesis.missing_qualitative_companies
            if f"qualitative_evidence_unavailable:{company_id}"
            not in execution_reasons
        )
        rendered = self._renderer(synthesis, evidence)
        if synthesis.status is GroundedSynthesisStatus.INSUFFICIENT:
            status = OrchestrationStatus.INSUFFICIENT
        elif synthesis.status is GroundedSynthesisStatus.PARTIAL:
            status = OrchestrationStatus.PARTIAL
        else:
            status = OrchestrationStatus.COMPLETED
        return CompetitorIntelligenceResult(
            normalized_question,
            plan,
            status,
            rendered,
            tuple(execution_reasons),
        )

    def _execute_financial(self, plan: AnalysisPlan) -> tuple[EvidenceInput, ...]:
        request = plan.financial_request
        if request is None:
            raise CompetitorOrchestrationError(
                "A ready financial route requires a financial request."
            )
        companies = plan.requested_companies
        metrics = request.metrics
        years = request.fiscal_years
        operation = request.operation
        outputs: list[EvidenceInput] = []

        if operation is FinancialOperation.METRIC_LOOKUP:
            fiscal_year = years[0]
            for company_id in companies:
                for metric in metrics:
                    outputs.append(
                        self._metric_lookup(company_id, fiscal_year, metric)
                    )
        elif operation is FinancialOperation.COMPARISON:
            fiscal_year = years[0]
            for metric in metrics:
                if metric in SUPPORTED_CALCULATED_METRICS:
                    outputs.append(
                        self._comparisons.rank_companies(
                            metric, fiscal_year, companies
                        )
                    )
                elif metric in RAW_FINANCIAL_METRICS:
                    outputs.extend(
                        self._financial_facts.get_fact(
                            company_id, fiscal_year, metric
                        )
                        for company_id in companies
                    )
                else:
                    raise CompetitorOrchestrationError(
                        "Planned comparison metric has no trusted execution path."
                    )
        elif operation is FinancialOperation.RANKING:
            fiscal_year = years[0]
            for metric in metrics:
                if metric not in SUPPORTED_CALCULATED_METRICS:
                    raise CompetitorOrchestrationError(
                        "Planned ranking metric is unsupported by the trusted comparison engine."
                    )
                outputs.append(
                    self._comparisons.rank_companies(metric, fiscal_year, companies)
                )
        elif operation is FinancialOperation.YEAR_CHANGE:
            earlier_year, later_year = years
            company_id = companies[0]
            outputs.extend(
                self._comparisons.compare_company_years(
                    company_id, metric, earlier_year, later_year
                )
                for metric in metrics
            )
        else:  # pragma: no cover - FinancialOperation is a closed enum.
            raise CompetitorOrchestrationError(
                "Planned financial operation is unsupported."
            )
        return tuple(outputs)

    def _metric_lookup(
        self, company_id: str, fiscal_year: int, metric: str
    ) -> EvidenceInput:
        if metric in SUPPORTED_CALCULATED_METRICS:
            return self._calculations.calculate(company_id, fiscal_year, metric)
        if metric in RAW_FINANCIAL_METRICS:
            return self._financial_facts.get_fact(company_id, fiscal_year, metric)
        raise CompetitorOrchestrationError(
            "Planned financial metric has no trusted execution path."
        )
