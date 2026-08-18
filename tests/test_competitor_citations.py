"""Offline tests for deterministic competitor citation rendering."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from enterprise_rag.competitor_analysis import CitationReadyEvidence
from enterprise_rag.competitor_citations import (
    CitationRenderingError,
    RenderedCalculationDetails,
    RenderedChangeDetails,
    RenderedComparisonDetails,
    RenderedFinancialFactDetails,
    RenderedQualitativeDetails,
    render_competitor_answer,
)
from enterprise_rag.competitor_evidence import UnifiedEvidenceBuilder, UnifiedEvidenceSet
from enterprise_rag.competitor_grounded_synthesis import (
    FinancialClaimType,
    GroundedSynthesisResult,
    GroundedSynthesisStatus,
    ValidatedFinancialClaim,
)
from enterprise_rag.financial_calculations import FinancialCalculationEngine
from enterprise_rag.financial_comparisons import FinancialComparisonEngine
from enterprise_rag.financial_facts import FinancialFact, FinancialFactCollection


COMPANIES = {
    "gigabyte": ("Gigabyte", "2376"),
    "asus": ("ASUS", "2357"),
    "msi": ("MSI", "2377"),
}


def qualitative(
    company_id: str = "asus", page: int | None = 42
) -> CitationReadyEvidence:
    company_name, ticker = COMPANIES[company_id]
    return CitationReadyEvidence(
        evidence_id="upstream-id",
        company_id=company_id,
        company_name=company_name,
        ticker=ticker,
        fiscal_year=2025,
        document_type="annual_report",
        source_document_id=f"synthetic-{company_id}-2025",
        source_title=f"Synthetic {company_name} Annual Report",
        page_number=page,
        chunk_id=f"synthetic-{company_id}:page-{page or 0:04}:chunk-000001",
        text=f"{company_name} describes a fictional strategy.",
        retrieval_score=0.123,
        quality_score=0.987,
        original_candidate_rank=4,
        final_company_rank=1,
    )


def fact(
    company_id: str = "asus",
    year: int = 2025,
    metric: str = "revenue",
    value: str = "100",
    *,
    page: int = 10,
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
        source_title=f"Synthetic {company_name} Financial Report",
        page_number=page,
        source_label="Synthetic fixture",
        source_sha256="b" * 64,
        notes="Fictional value",
    )


def margin_facts(company_id: str, year: int, numerator: str):
    return (
        fact(company_id, year, "revenue", "100", page=20),
        fact(company_id, year, "gross_profit", numerator, page=21),
    )


def calculation(company_id: str = "asus", year: int = 2025, numerator: str = "40"):
    return FinancialCalculationEngine(
        FinancialFactCollection(margin_facts(company_id, year, numerator))
    ).gross_margin(company_id, year)


def comparison_engine(*facts: FinancialFact) -> FinancialComparisonEngine:
    return FinancialComparisonEngine(
        FinancialCalculationEngine(FinancialFactCollection(facts))
    )


def unified(*items) -> UnifiedEvidenceSet:
    return UnifiedEvidenceBuilder().build(items)


def result(
    evidence: UnifiedEvidenceSet,
    cited: tuple[str, ...],
    *,
    claims: tuple[ValidatedFinancialClaim, ...] = (),
    status: GroundedSynthesisStatus = GroundedSynthesisStatus.GROUNDED,
) -> GroundedSynthesisResult:
    return GroundedSynthesisResult(
        "Synthetic question?",
        "Synthetic validated answer " + " ".join(f"[{value}]" for value in cited),
        cited,
        status,
        len(evidence),
        claims,
        "fake",
        "fake-model",
    )


def claim(
    evidence_id: str,
    claim_type: FinancialClaimType,
    role: str,
    value: str,
    *,
    company_id: str | None = None,
    rank: int | None = None,
) -> ValidatedFinancialClaim:
    return ValidatedFinancialClaim(
        evidence_id, claim_type, role, value, company_id, rank
    )


def test_qualitative_citation_renders_company_title_page_without_scores() -> None:
    evidence = unified(qualitative())
    rendered = render_competitor_answer(result(evidence, ("E1",)), evidence)
    citation = rendered.citations[0]
    assert isinstance(citation.details, RenderedQualitativeDetails)
    assert citation.company_name == "ASUS"
    assert citation.source_references[0].source_title == "Synthetic ASUS Annual Report"
    assert citation.source_references[0].page_number == 42
    text = rendered.render_text()
    assert "[E1] — ASUS — 2025 — Synthetic ASUS Annual Report — PDF p.42" in text
    assert "0.123" not in text and "retrieval" not in text.casefold()


def test_citation_order_follows_result_and_duplicates_are_deduplicated() -> None:
    evidence = unified(qualitative("asus"), qualitative("msi"))
    rendered = render_competitor_answer(result(evidence, ("E2", "E1", "E2")), evidence)
    assert tuple(value.evidence_id for value in rendered.citations) == ("E2", "E1")
    assert rendered.render_text().index("[E2]") < rendered.render_text().index("[E1]")


def test_missing_pdf_page_is_rendered_explicitly_without_invention() -> None:
    evidence = unified(qualitative(page=None))
    rendered = render_competitor_answer(result(evidence, ("E1",)), evidence)
    assert rendered.citations[0].source_references[0].page_number is None
    assert "PDF page unavailable" in rendered.render_text()


def test_unknown_evidence_id_is_rejected_safely() -> None:
    evidence = unified(qualitative())
    with pytest.raises(CitationRenderingError, match="unavailable"):
        render_competitor_answer(result(evidence, ("E99",)), evidence)


def test_financial_fact_is_labeled_reported_and_preserves_provenance() -> None:
    evidence = unified(fact())
    claims = (
        claim("E1", FinancialClaimType.REPORTED_FACT, "reported_value", "100"),
    )
    rendered = render_competitor_answer(result(evidence, ("E1",), claims=claims), evidence)
    details = rendered.citations[0].details
    assert isinstance(details, RenderedFinancialFactDetails)
    assert details.metric == "revenue" and details.value == "100"
    text = rendered.render_text()
    assert "Type: Reported source fact" in text
    assert "Synthetic ASUS Financial Report — PDF p.10" in text
    assert "Python-calculated" not in text


def test_calculation_renders_exact_value_formula_and_two_input_pages() -> None:
    evidence = unified(calculation())
    claims = (
        claim("E1", FinancialClaimType.CALCULATED_METRIC, "calculated_value", "40.00"),
    )
    rendered = render_competitor_answer(result(evidence, ("E1",), claims=claims), evidence)
    details = rendered.citations[0].details
    assert isinstance(details, RenderedCalculationDetails)
    assert details.value == "40.00"
    assert details.formula == "gross_profit / revenue * 100"
    assert len(details.inputs) == 2
    assert {value.source.page_number for value in details.inputs} == {20, 21}
    text = rendered.render_text()
    assert "Type: Python-calculated metric" in text
    assert "Formula: gross_profit / revenue * 100" in text
    assert "PDF p.20" in text and "PDF p.21" in text


def test_comparison_preserves_ranks_values_direction_and_neutral_language() -> None:
    model = comparison_engine(
        *margin_facts("asus", 2025, "50"),
        *margin_facts("msi", 2025, "40"),
    ).rank_companies("gross_margin", 2025, ("asus", "msi"))
    evidence = unified(model)
    rendered = render_competitor_answer(result(evidence, ("E1",)), evidence)
    details = rendered.citations[0].details
    assert isinstance(details, RenderedComparisonDetails)
    assert details.requested_companies == ("asus", "msi")
    assert [(value.rank, value.company_id, value.value) for value in details.ranked_entries] == [
        (1, "asus", "50.00"),
        (2, "msi", "40.00"),
    ]
    text = rendered.render_text()
    assert "Direction: higher_value_first" in text
    assert "Requested companies: asus, msi" in text
    assert all(word not in text.casefold() for word in ("best", "winner", "outperformed", "superior"))


def test_comparison_preserves_tie_ranks() -> None:
    model = comparison_engine(
        *margin_facts("asus", 2025, "40"),
        *margin_facts("msi", 2025, "40"),
    ).rank_companies("gross_margin", 2025, ("asus", "msi"))
    details = render_competitor_answer(result(unified(model), ("E1",)), unified(model)).citations[0].details
    assert isinstance(details, RenderedComparisonDetails)
    assert tuple(value.rank for value in details.ranked_entries) == (1, 1)


def test_requested_company_order_is_independent_of_ranking_order() -> None:
    model = comparison_engine(
        *margin_facts("asus", 2025, "50"),
        *margin_facts("gigabyte", 2025, "45"),
        *margin_facts("msi", 2025, "40"),
    ).rank_companies("gross_margin", 2025, ("msi", "asus", "gigabyte"))
    evidence = unified(model)
    details = render_competitor_answer(
        result(evidence, ("E1",)), evidence
    ).citations[0].details
    assert isinstance(details, RenderedComparisonDetails)
    assert details.requested_companies == ("msi", "asus", "gigabyte")
    assert tuple(value.company_id for value in details.ranked_entries) == (
        "asus",
        "gigabyte",
        "msi",
    )


def test_partial_comparison_renders_missing_company_and_status() -> None:
    model = comparison_engine(*margin_facts("asus", 2025, "40")).rank_companies(
        "gross_margin", 2025, ("asus", "msi")
    )
    evidence = unified(model)
    rendered = render_competitor_answer(result(evidence, ("E1",)), evidence)
    details = rendered.citations[0].details
    assert isinstance(details, RenderedComparisonDetails)
    assert details.requested_companies == ("asus", "msi")
    assert tuple(value.company_id for value in details.ranked_entries) == ("asus",)
    assert details.status == "partial" and details.missing_companies == ("msi",)
    assert "Status: partial" in rendered.render_text()
    assert "Missing: msi" in rendered.render_text()
    assert "Requested companies: asus, msi" in rendered.render_text()


def test_insufficient_comparison_renders_no_fabricated_entries() -> None:
    model = comparison_engine().rank_companies(
        "gross_margin", 2025, ("asus", "msi")
    )
    evidence = unified(model)
    rendered = render_competitor_answer(
        result(evidence, ("E1",), status=GroundedSynthesisStatus.INSUFFICIENT),
        evidence,
    )
    details = rendered.citations[0].details
    assert isinstance(details, RenderedComparisonDetails)
    assert details.requested_companies == ("asus", "msi")
    assert details.ranked_entries == ()
    assert "Requested companies: asus, msi" in rendered.render_text()
    assert "No comparison provenance available." in rendered.render_text()


def test_comparison_requested_scope_and_input_provenance_are_immutable() -> None:
    model = comparison_engine(
        *margin_facts("asus", 2025, "50"),
        *margin_facts("msi", 2025, "40"),
    ).rank_companies("gross_margin", 2025, ("msi", "asus"))
    evidence = unified(model)
    rendered = render_competitor_answer(result(evidence, ("E1",)), evidence)
    details = rendered.citations[0].details
    assert isinstance(details, RenderedComparisonDetails)
    assert isinstance(details.requested_companies, tuple)
    assert [
        tuple(value.source.page_number for value in entry.calculation.inputs)
        for entry in details.ranked_entries
    ] == [(21, 20), (21, 20)]
    assert [
        tuple(value.source.source_metric for value in entry.calculation.inputs)
        for entry in details.ranked_entries
    ] == [("gross_profit", "revenue"), ("gross_profit", "revenue")]
    assert all(
        value.source.document_type == "consolidated_financial_report"
        for entry in details.ranked_entries
        for value in entry.calculation.inputs
    )
    with pytest.raises(FrozenInstanceError):
        details.requested_companies = ("changed",)  # type: ignore[misc]


def test_change_preserves_values_percentage_points_direction_and_both_years() -> None:
    model = comparison_engine(
        *margin_facts("asus", 2024, "30"),
        *margin_facts("asus", 2025, "40"),
    ).compare_company_years("asus", "gross_margin", 2024, 2025)
    evidence = unified(model)
    claims = (
        claim("E1", FinancialClaimType.FINANCIAL_CHANGE_VALUE, "earlier_value", "30.00"),
        claim("E1", FinancialClaimType.FINANCIAL_CHANGE_VALUE, "later_value", "40.00"),
        claim(
            "E1",
            FinancialClaimType.FINANCIAL_CHANGE_VALUE,
            "percentage_point_change",
            "10.00",
        ),
    )
    rendered = render_competitor_answer(result(evidence, ("E1",), claims=claims), evidence)
    details = rendered.citations[0].details
    assert isinstance(details, RenderedChangeDetails)
    assert (details.earlier_value, details.later_value, details.percentage_point_change) == (
        "30.00",
        "40.00",
        "10.00",
    )
    assert details.direction == "increase"
    assert len(details.earlier_calculation.inputs) == 2
    assert len(details.later_calculation.inputs) == 2
    text = rendered.render_text()
    assert "Change: 10.00 percentage points" in text
    assert "relative" not in text.casefold() and "growth" not in text.casefold()


def test_validated_claims_map_to_citation_without_new_claims_or_raw_dicts() -> None:
    evidence = unified(fact())
    claims = (
        claim("E1", FinancialClaimType.REPORTED_FACT, "reported_value", "100"),
    )
    rendered = render_competitor_answer(result(evidence, ("E1",), claims=claims), evidence)
    assert rendered.financial_claims is claims
    assert rendered.citations[0].financial_claims == claims
    assert not isinstance(rendered.financial_claims[0], dict)


def test_claim_outside_cited_evidence_is_rejected() -> None:
    evidence = unified(qualitative(), fact())
    claims = (
        claim("E2", FinancialClaimType.REPORTED_FACT, "reported_value", "100"),
    )
    with pytest.raises(CitationRenderingError, match="outside validated citations"):
        render_competitor_answer(result(evidence, ("E1",), claims=claims), evidence)


def test_raw_provider_claim_dictionary_is_rejected_at_renderer_boundary() -> None:
    evidence = unified(fact())
    unsafe = GroundedSynthesisResult(
        "Synthetic question?",
        "Synthetic answer [E1].",
        ("E1",),
        GroundedSynthesisStatus.GROUNDED,
        1,
        ({"evidence_id": "E1"},),  # type: ignore[arg-type]
        "fake",
        "fake-model",
    )
    with pytest.raises(CitationRenderingError, match="validated typed claims"):
        render_competitor_answer(unsafe, evidence)


def test_default_text_has_no_private_identity_sha_or_paths() -> None:
    evidence = unified(qualitative(), fact())
    rendered = render_competitor_answer(result(evidence, ("E1", "E2")), evidence)
    text = rendered.render_text()
    assert "data/private" not in text and "data\\private" not in text
    assert "data/vector_store" not in text and "data\\vector_store" not in text
    assert "C:\\" not in text and "/private/" not in text
    assert "b" * 64 not in text
    assert "synthetic:asus:2025:revenue" not in text


def test_rendered_models_and_input_collections_are_immutable() -> None:
    evidence = unified(calculation())
    rendered = render_competitor_answer(result(evidence, ("E1",)), evidence)
    details = rendered.citations[0].details
    assert isinstance(details, RenderedCalculationDetails)
    assert isinstance(rendered.citations, tuple) and isinstance(details.inputs, tuple)
    with pytest.raises(FrozenInstanceError):
        rendered.answer_text = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        details.value = "changed"  # type: ignore[misc]


def test_render_text_is_deterministic_and_machine_readable_data_remains_available() -> None:
    evidence = unified(qualitative(), calculation())
    rendered = render_competitor_answer(result(evidence, ("E1", "E2")), evidence)
    assert rendered.render_text() == rendered.render_text()
    assert rendered.answer_text.startswith("Synthetic validated answer")
    assert len(rendered.citations) == 2
    assert rendered.status is GroundedSynthesisStatus.GROUNDED
    assert rendered.generation_provider == "fake"
