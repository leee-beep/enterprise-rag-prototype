"""Offline end-to-end tests for competitor intelligence orchestration."""

from __future__ import annotations

from decimal import Decimal

import pytest

from enterprise_rag.competitor_analysis import CitationReadyEvidence
from enterprise_rag.competitor_citations import render_competitor_answer
from enterprise_rag.competitor_evidence import UnifiedEvidenceBuilder
from enterprise_rag.competitor_grounded_synthesis import (
    GroundedSynthesisResult,
    GroundedSynthesisStatus,
)
from enterprise_rag.competitor_orchestration import (
    CompetitorIntelligencePipeline,
    OrchestrationStatus,
)
from enterprise_rag.financial_calculations import FinancialCalculationEngine
from enterprise_rag.financial_comparisons import FinancialComparisonEngine
from enterprise_rag.financial_facts import FinancialFact, FinancialFactCollection


COMPANIES = {
    "gigabyte": ("Gigabyte", "2376"),
    "asus": ("ASUS", "2357"),
    "msi": ("MSI", "2377"),
}


def fact(
    company_id: str,
    year: int,
    metric: str,
    value: str,
    *,
    page: int,
) -> FinancialFact:
    company_name, ticker = COMPANIES[company_id]
    return FinancialFact(
        company_id=company_id,
        company_name=company_name,
        ticker=ticker,
        fiscal_year=year,
        period="FY",
        metric=metric,
        value=Decimal(value),
        currency="TWD",
        unit="million_TWD",
        reporting_scope="consolidated",
        document_type="consolidated_financial_report",
        source_document_id=f"synthetic:{company_id}:{year}:{metric}",
        source_title=f"Synthetic {company_name} report",
        page_number=page,
        source_label="Synthetic fixture",
        source_sha256="a" * 64,
        notes="Fictional offline value",
    )


def qualitative(company_id: str = "asus") -> CitationReadyEvidence:
    company_name, ticker = COMPANIES[company_id]
    return CitationReadyEvidence(
        evidence_id="legacy-id",
        company_id=company_id,
        company_name=company_name,
        ticker=ticker,
        fiscal_year=2025,
        document_type="annual_report",
        source_document_id=f"synthetic:{company_id}:2025:annual",
        source_title=f"Synthetic {company_name} Annual Report",
        page_number=42,
        chunk_id=f"synthetic:{company_id}:page-0042:chunk-000001",
        text=f"{company_name} describes a fictional AI server strategy.",
        retrieval_score=0.2,
        quality_score=0.9,
        original_candidate_rank=1,
        final_company_rank=1,
    )


def all_facts() -> FinancialFactCollection:
    values: list[FinancialFact] = []
    for index, company_id in enumerate(COMPANIES, start=1):
        for year, revenue, gross_profit in (
            (2024, "100", str(30 + index)),
            (2025, "120", str(40 + index)),
        ):
            values.extend(
                (
                    fact(company_id, year, "revenue", revenue, page=10),
                    fact(company_id, year, "gross_profit", gross_profit, page=11),
                )
            )
    return FinancialFactCollection(values)


class FakeQualitativeProvider:
    def __init__(self, items: tuple[CitationReadyEvidence, ...] = ()) -> None:
        self.items = items
        self.calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def retrieve(self, question, company_ids, qualitative_intents):
        self.calls.append((question, company_ids, qualitative_intents))
        return self.items


class RecordingCalculationEngine:
    def __init__(self, delegate: FinancialCalculationEngine) -> None:
        self.delegate = delegate
        self.calls: list[tuple[str, int, str]] = []

    def calculate(self, company_id, fiscal_year, metric):
        self.calls.append((company_id, fiscal_year, metric))
        return self.delegate.calculate(company_id, fiscal_year, metric)


class RecordingComparisonEngine:
    def __init__(self, delegate: FinancialComparisonEngine) -> None:
        self.delegate = delegate
        self.rank_calls: list[tuple[str, int, tuple[str, ...]]] = []
        self.change_calls: list[tuple[str, str, int, int]] = []

    def rank_companies(self, metric, fiscal_year, companies):
        self.rank_calls.append((metric, fiscal_year, companies))
        return self.delegate.rank_companies(metric, fiscal_year, companies)

    def compare_company_years(self, company_id, metric, earlier_year, later_year):
        self.change_calls.append((company_id, metric, earlier_year, later_year))
        return self.delegate.compare_company_years(
            company_id, metric, earlier_year, later_year
        )


class RecordingBuilder:
    def __init__(self) -> None:
        self.delegate = UnifiedEvidenceBuilder()
        self.calls: list[tuple[object, ...]] = []
        self.results = []

    def build(self, inputs):
        captured = tuple(inputs)
        self.calls.append(captured)
        result = self.delegate.build(captured)
        self.results.append(result)
        return result


class FakeSynthesizer:
    def __init__(
        self, status: GroundedSynthesisStatus = GroundedSynthesisStatus.GROUNDED
    ) -> None:
        self.status = status
        self.calls = []

    def synthesize(self, question, evidence, analysis_plan=None):
        self.calls.append((question, evidence, analysis_plan))
        ids = tuple(item.evidence_id for item in evidence)
        return GroundedSynthesisResult(
            question=question,
            answer_text="Synthetic grounded answer " + " ".join(f"[{i}]" for i in ids),
            cited_evidence_ids=ids,
            status=self.status,
            evidence_count=len(evidence),
            financial_claims=(),
            generation_provider="fake",
            generation_model="fake-model",
        )


class RecordingRenderer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, result, evidence):
        self.calls.append((result, evidence))
        return render_competitor_answer(result, evidence)


def pipeline(*, qualitative_items=(), synthesis_status=GroundedSynthesisStatus.GROUNDED):
    facts = all_facts()
    calculation_delegate = FinancialCalculationEngine(facts)
    qualitative_provider = FakeQualitativeProvider(qualitative_items)
    calculations = RecordingCalculationEngine(calculation_delegate)
    comparisons = RecordingComparisonEngine(
        FinancialComparisonEngine(calculation_delegate)
    )
    builder = RecordingBuilder()
    synthesizer = FakeSynthesizer(synthesis_status)
    renderer = RecordingRenderer()
    subject = CompetitorIntelligencePipeline(
        qualitative_provider,
        facts,
        calculations,
        comparisons,
        synthesizer,
        evidence_builder=builder,
        renderer=renderer,
    )
    return subject, qualitative_provider, calculations, comparisons, builder, synthesizer, renderer


def test_qualitative_route_uses_only_retrieval_and_preserves_provenance() -> None:
    subject, q, calc, comp, builder, synth, renderer = pipeline(
        qualitative_items=(qualitative(),)
    )
    result = subject.run("What is ASUS's AI server strategy?")
    assert result.status is OrchestrationStatus.COMPLETED
    assert len(q.calls) == 1 and calc.calls == []
    assert comp.rank_calls == [] and comp.change_calls == []
    assert len(builder.calls) == len(synth.calls) == len(renderer.calls) == 1
    assert result.rendered_answer.citations[0].source_references[0].page_number == 42


def test_financial_lookup_uses_calculation_only() -> None:
    subject, q, calc, comp, builder, synth, renderer = pipeline()
    result = subject.run("What was ASUS gross margin in 2025?")
    assert result.status is OrchestrationStatus.COMPLETED
    assert q.calls == []
    assert calc.calls == [("asus", 2025, "gross_margin")]
    assert comp.rank_calls == [] and len(result.rendered_answer.citations) == 1
    assert len(builder.calls) == len(synth.calls) == len(renderer.calls) == 1


def test_ranking_preserves_requested_company_order() -> None:
    subject, q, calc, comp, *_ = pipeline()
    result = subject.run("Rank Gigabyte, ASUS, and MSI by gross margin in 2025.")
    assert result.status is OrchestrationStatus.COMPLETED and q.calls == []
    assert calc.calls == []
    assert comp.rank_calls == [
        ("gross_margin", 2025, ("gigabyte", "asus", "msi"))
    ]
    details = result.rendered_answer.citations[0].details
    assert details.requested_companies == ("gigabyte", "asus", "msi")


def test_year_change_uses_existing_comparison_engine() -> None:
    subject, q, calc, comp, *_ = pipeline()
    result = subject.run("How did ASUS gross margin change from 2024 to 2025?")
    assert result.status is OrchestrationStatus.COMPLETED and q.calls == []
    assert calc.calls == []
    assert comp.change_calls == [("asus", "gross_margin", 2024, 2025)]
    assert len(result.rendered_answer.citations[0].source_references) == 4


def test_combined_route_builds_once_and_shares_exact_evidence_object() -> None:
    subject, q, calc, comp, builder, synth, renderer = pipeline(
        qualitative_items=(qualitative(),)
    )
    result = subject.run(
        "Compare ASUS and MSI AI server strategy and gross margin in 2025."
    )
    assert result.status is OrchestrationStatus.COMPLETED
    assert len(q.calls) == 1 and calc.calls == [] and len(comp.rank_calls) == 1
    assert len(builder.calls) == 1
    assert type(builder.calls[0][0]).__name__ == "CitationReadyEvidence"
    assert type(builder.calls[0][1]).__name__ == "FinancialComparisonResult"
    evidence = builder.results[0]
    assert synth.calls[0][1] is evidence
    assert renderer.calls[0][1] is evidence
    assert tuple(item.evidence_id for item in evidence) == ("E1", "E2")
    assert len(result.rendered_answer.citations) == 2


@pytest.mark.parametrize(
    ("question", "expected_status"),
    (
        ("What about ASUS?", OrchestrationStatus.AMBIGUOUS),
        ("What was ASUS ROE in 2025?", OrchestrationStatus.UNSUPPORTED),
    ),
)
def test_non_ready_plan_stops_all_downstream_work(question, expected_status) -> None:
    subject, q, calc, comp, builder, synth, renderer = pipeline(
        qualitative_items=(qualitative(),)
    )
    result = subject.run(question)
    assert result.status is expected_status and result.rendered_answer is None
    assert q.calls == [] and calc.calls == []
    assert comp.rank_calls == [] and comp.change_calls == []
    assert builder.calls == [] and synth.calls == [] and renderer.calls == []


def test_empty_qualitative_evidence_stops_before_synthesis() -> None:
    subject, q, calc, comp, builder, synth, renderer = pipeline()
    result = subject.run("What is ASUS's AI server strategy?")
    assert result.status is OrchestrationStatus.INSUFFICIENT
    assert "qualitative_evidence_unavailable" in result.reasons
    assert len(q.calls) == 1 and calc.calls == [] and comp.rank_calls == []
    assert builder.calls == [] and synth.calls == [] and renderer.calls == []


def test_combined_route_can_continue_with_trusted_financial_evidence() -> None:
    subject, q, calc, comp, builder, synth, renderer = pipeline()
    result = subject.run(
        "Compare ASUS and MSI AI server strategy and gross margin in 2025."
    )
    assert result.status is OrchestrationStatus.COMPLETED
    assert "qualitative_evidence_unavailable" in result.reasons
    assert len(q.calls) == 1 and len(comp.rank_calls) == 1
    assert len(builder.calls[0]) == 1
    assert len(synth.calls) == len(renderer.calls) == 1


def test_raw_metric_lookup_uses_trusted_fact_collection() -> None:
    subject, q, calc, comp, builder, *_ = pipeline()
    result = subject.run("What was ASUS revenue in 2025?")
    assert result.status is OrchestrationStatus.COMPLETED and q.calls == []
    assert calc.calls == [] and comp.rank_calls == []
    assert isinstance(builder.calls[0][0], FinancialFact)


def test_raw_metric_comparison_does_not_invent_a_ranking() -> None:
    subject, q, calc, comp, builder, *_ = pipeline()
    result = subject.run("Compare ASUS and MSI revenue in 2025.")
    assert result.status is OrchestrationStatus.COMPLETED and q.calls == []
    assert calc.calls == [] and comp.rank_calls == []
    assert [item.company_id for item in builder.calls[0]] == ["asus", "msi"]


def test_repeated_execution_is_deterministic() -> None:
    subject, *_ = pipeline(qualitative_items=(qualitative(),))
    first = subject.run("What is ASUS's AI server strategy?")
    second = subject.run("What is ASUS's AI server strategy?")
    assert first.plan == second.plan
    assert first.rendered_answer == second.rendered_answer


def test_synthesis_insufficient_status_propagates_to_result() -> None:
    subject, *_ = pipeline(
        qualitative_items=(qualitative(),),
        synthesis_status=GroundedSynthesisStatus.INSUFFICIENT,
    )
    result = subject.run("What is ASUS's AI server strategy?")
    assert result.status is OrchestrationStatus.INSUFFICIENT
    assert result.rendered_answer.status is GroundedSynthesisStatus.INSUFFICIENT


def test_empty_question_is_rejected_before_planning() -> None:
    subject, *_ = pipeline()
    with pytest.raises(ValueError, match="non-empty"):
        subject.run("  ")
