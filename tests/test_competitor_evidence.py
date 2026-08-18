"""Offline tests for unified qualitative and financial evidence adaptation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from enterprise_rag.competitor_analysis import CitationReadyEvidence
from enterprise_rag.competitor_evidence import (
    EvidenceNotFoundError,
    EvidenceType,
    EvidenceValidationError,
    FinancialCalculationEvidenceData,
    FinancialChangeEvidenceData,
    FinancialComparisonEvidenceData,
    FinancialFactEvidenceData,
    QualitativeEvidenceData,
    SourceReference,
    UnifiedEvidence,
    UnifiedEvidenceBuilder,
    UnifiedEvidenceSet,
)
from enterprise_rag.financial_calculations import (
    FinancialCalculationEngine,
    FinancialFactProvenance,
)
from enterprise_rag.financial_comparisons import FinancialComparisonEngine
from enterprise_rag.financial_facts import FinancialFact, FinancialFactCollection


COMPANIES = {
    "gigabyte": ("Gigabyte", "2376"),
    "asus": ("ASUS", "2357"),
    "msi": ("MSI", "2377"),
}


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
        source_title=f"Synthetic {company_name} {year} report",
        page_number=page,
        source_label="Synthetic fixture",
        source_sha256="a" * 64,
        notes="Fictional test value",
    )


def qualitative(**changes: object) -> CitationReadyEvidence:
    values: dict[str, object] = {
        "evidence_id": "E99",
        "company_id": "asus",
        "company_name": "ASUS",
        "ticker": "2357",
        "fiscal_year": 2025,
        "document_type": "annual_report",
        "source_document_id": "synthetic-asus-2025",
        "source_title": "Synthetic ASUS 2025 Annual Report",
        "page_number": 42,
        "chunk_id": "synthetic-asus:page-0042:chunk-000001",
        "text": "ASUS describes a fictional AI infrastructure strategy.",
        "retrieval_score": 0.25,
        "quality_score": 0.9,
        "original_candidate_rank": 3,
        "final_company_rank": 1,
    }
    values.update(changes)
    return CitationReadyEvidence(**values)


def calculation(
    company_id: str = "asus", year: int = 2025, numerator: str = "40"
):
    return FinancialCalculationEngine(
        FinancialFactCollection(
            (
                fact(company_id, year, "revenue", "100", page=11),
                fact(company_id, year, "gross_profit", numerator, page=12),
            )
        )
    ).gross_margin(company_id, year)


def comparison_engine(*facts: FinancialFact) -> FinancialComparisonEngine:
    return FinancialComparisonEngine(
        FinancialCalculationEngine(FinancialFactCollection(facts))
    )


def margin_facts(company_id: str, year: int, numerator: str):
    return (
        fact(company_id, year, "revenue", "100", page=20),
        fact(company_id, year, "gross_profit", numerator, page=21),
    )


def test_qualitative_evidence_preserves_text_identity_and_provenance() -> None:
    item = UnifiedEvidenceBuilder().build((qualitative(),))[0]
    assert item.evidence_type is EvidenceType.QUALITATIVE
    assert item.company_id == "asus" and item.fiscal_year == 2025
    assert item.display_text == "ASUS describes a fictional AI infrastructure strategy."
    assert isinstance(item.data, QualitativeEvidenceData)
    assert item.data.chunk_id == "synthetic-asus:page-0042:chunk-000001"
    assert item.data.retrieval_score == 0.25 and item.data.quality_score == 0.9
    assert item.source_references[0].page_number == 42
    assert item.source_references[0].source_document_id == "synthetic-asus-2025"


def test_financial_fact_preserves_source_and_canonical_decimals() -> None:
    source = fact(value="123.45", page=17)
    item = UnifiedEvidenceBuilder().build((source,))[0]
    assert item.evidence_type is EvidenceType.FINANCIAL_FACT
    assert isinstance(item.data, FinancialFactEvidenceData)
    assert item.data.metric == "revenue"
    assert item.data.source_value == Decimal("123.45")
    assert item.data.source_unit == "million_TWD"
    assert item.data.canonical_value_twd == Decimal("123450000.00")
    assert isinstance(item.data.canonical_value_twd, Decimal)
    assert item.source_references[0].page_number == 17
    assert item.source_references[0].source_metric == "revenue"


def test_calculation_preserves_trusted_result_formula_inputs_and_two_sources() -> None:
    trusted = replace(
        calculation(), value=Decimal("99.99"), formula="trusted synthetic formula"
    )
    item = UnifiedEvidenceBuilder().build((trusted,))[0]
    assert item.evidence_type is EvidenceType.FINANCIAL_CALCULATION
    assert isinstance(item.data, FinancialCalculationEvidenceData)
    assert item.data.value == Decimal("99.99")
    assert item.data.formula == "trusted synthetic formula"
    assert tuple(value.metric for value in item.data.input_facts) == (
        "gross_profit",
        "revenue",
    )
    assert tuple(ref.page_number for ref in item.source_references) == (12, 11)
    assert all(isinstance(value.canonical_value_twd, Decimal) for value in item.data.input_facts)


def test_comparison_preserves_ties_order_status_and_calculation_provenance() -> None:
    engine = comparison_engine(
        *margin_facts("gigabyte", 2025, "50"),
        *margin_facts("asus", 2025, "40"),
        *margin_facts("msi", 2025, "40"),
    )
    result = engine.rank_companies(
        "gross_margin", 2025, ("gigabyte", "msi", "asus")
    )
    item = UnifiedEvidenceBuilder().build((result,))[0]
    assert item.evidence_type is EvidenceType.FINANCIAL_COMPARISON
    assert isinstance(item.data, FinancialComparisonEvidenceData)
    assert item.data.requested_companies == ("gigabyte", "msi", "asus")
    assert item.data.ranking_direction == "higher_value_first"
    assert item.data.status == "complete" and item.data.missing_companies == ()
    assert tuple(entry.rank for entry in item.data.ranked_entries) == (1, 2, 2)
    assert tuple(entry.company_id for entry in item.data.ranked_entries) == (
        "gigabyte",
        "msi",
        "asus",
    )
    assert len(item.source_references) == 6
    assert "best" not in item.display_text and "winner" not in item.display_text


def test_partial_comparison_preserves_missing_companies() -> None:
    engine = comparison_engine(*margin_facts("asus", 2025, "40"))
    result = engine.rank_companies("gross_margin", 2025, ("asus", "msi"))
    data = UnifiedEvidenceBuilder().build((result,))[0].data
    assert isinstance(data, FinancialComparisonEvidenceData)
    assert data.status == "partial"
    assert data.missing_companies == ("msi",)


def test_company_filter_includes_multi_company_comparison() -> None:
    engine = comparison_engine(
        *margin_facts("asus", 2025, "40"),
        *margin_facts("msi", 2025, "30"),
    )
    result = engine.rank_companies("gross_margin", 2025, ("asus", "msi"))
    evidence = UnifiedEvidenceBuilder().build((result,))
    assert evidence.filter_by_company("asus") == evidence.evidence
    assert evidence.filter_by_company("msi") == evidence.evidence
    assert evidence.filter_by_company("gigabyte") == ()


@pytest.mark.parametrize(
    "later_numerator,direction,expected_change",
    (("50", "increase", "10.00"), ("30", "decrease", "-10.00"), ("40", "unchanged", "0.00")),
)
def test_change_preserves_both_years_percentage_points_and_provenance(
    later_numerator: str, direction: str, expected_change: str
) -> None:
    engine = comparison_engine(
        *margin_facts("asus", 2024, "40"),
        *margin_facts("asus", 2025, later_numerator),
    )
    result = engine.compare_company_years("asus", "gross_margin", 2024, 2025)
    item = UnifiedEvidenceBuilder().build((result,))[0]
    assert item.evidence_type is EvidenceType.FINANCIAL_CHANGE
    assert isinstance(item.data, FinancialChangeEvidenceData)
    assert item.data.earlier_year == 2024 and item.data.later_year == 2025
    assert item.data.earlier_value == Decimal("40.00")
    assert item.data.percentage_point_change == Decimal(expected_change)
    assert item.data.direction == direction and item.data.unit == "percentage_points"
    assert len(item.source_references) == 4
    assert {ref.fiscal_year for ref in item.source_references} == {2024, 2025}


def test_mixed_input_order_and_ids_are_deterministic() -> None:
    inputs = (qualitative(), fact(), calculation())
    first = UnifiedEvidenceBuilder().build(inputs)
    second = UnifiedEvidenceBuilder().build(inputs)
    assert first.evidence == second.evidence
    assert tuple(item.evidence_id for item in first) == ("E1", "E2", "E3")
    assert tuple(item.evidence_type for item in first) == (
        EvidenceType.QUALITATIVE,
        EvidenceType.FINANCIAL_FACT,
        EvidenceType.FINANCIAL_CALCULATION,
    )


def test_collection_lookup_indexing_and_filters() -> None:
    evidence = UnifiedEvidenceBuilder().build((qualitative(), fact(), calculation()))
    assert len(evidence) == 3 and evidence[0].evidence_id == "E1"
    assert evidence[:2] == evidence.evidence[:2]
    assert evidence.get("E2") is evidence[1]
    assert evidence.filter_by_type(EvidenceType.FINANCIAL_FACT) == (evidence[1],)
    assert evidence.filter_by_company("ASUS") == evidence.evidence
    with pytest.raises(EvidenceNotFoundError, match="Unknown"):
        evidence.get("E99")


def test_duplicate_evidence_ids_are_rejected() -> None:
    item = UnifiedEvidenceBuilder().build((qualitative(),))[0]
    with pytest.raises(EvidenceValidationError, match="Duplicate"):
        UnifiedEvidenceSet((item, item))


def test_models_and_collection_are_immutable() -> None:
    item = UnifiedEvidenceBuilder().build((qualitative(),))[0]
    with pytest.raises(FrozenInstanceError):
        item.display_text = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        item.source_references[0].source_title = "changed"  # type: ignore[misc]
    copied = list(UnifiedEvidenceSet((item,)).evidence)
    copied.clear()
    assert len(UnifiedEvidenceSet((item,))) == 1


@pytest.mark.parametrize(
    "changes",
    (
        {"source_title": r"C:\\private\\annual-report.pdf"},
        {"source_document_id": r"C:\\private\\annual-report.pdf"},
        {"source_title": "data/private/annual-report.pdf"},
        {"text": r"Evidence copied from C:\\private\\annual-report.pdf"},
    ),
)
def test_qualitative_absolute_or_private_path_leakage_is_rejected(
    changes: dict[str, object]
) -> None:
    with pytest.raises(EvidenceValidationError, match="path"):
        UnifiedEvidenceBuilder().build((qualitative(**changes),))


def test_source_reference_rejects_path_and_is_frozen() -> None:
    with pytest.raises(EvidenceValidationError, match="path"):
        SourceReference("doc", "/home/private/report.pdf", 1)


def test_financial_source_label_path_leakage_is_rejected() -> None:
    with pytest.raises(EvidenceValidationError, match="path"):
        UnifiedEvidenceBuilder().build(
            (replace(fact(), source_label=r"C:\\private\\facts.csv"),)
        )


def test_evidence_type_and_payload_must_match() -> None:
    source = UnifiedEvidenceBuilder().build((qualitative(),))[0]
    with pytest.raises(EvidenceValidationError, match="must match"):
        UnifiedEvidence(
            source.evidence_id,
            EvidenceType.FINANCIAL_FACT,
            source.company_id,
            source.company_name,
            source.ticker,
            source.fiscal_year,
            source.display_text,
            source.source_references,
            source.data,
        )


@pytest.mark.parametrize(
    "text",
    (
        "Evidence copied from /home/private/report.pdf",
        "Leaked token sk-abcdefghijklmnopqrstuvwxyz123456",
    ),
)
def test_display_text_rejects_posix_paths_and_secret_values(text: str) -> None:
    with pytest.raises(EvidenceValidationError):
        UnifiedEvidenceBuilder().build((qualitative(text=text),))


@pytest.mark.parametrize(
    "unsafe",
    (
        r"C:\private\secret.pdf",
        r"C:/private/secret.pdf",
        r"D:\folder\file.pdf",
        r"Source (C:\private\secret.pdf)",
        r"Source \\server\share\secret.pdf",
        "Source //server/share/secret.pdf",
        "Source /home/user/file.pdf",
        "Source /Users/name/file.pdf",
        "Source /opt/company/file.pdf",
        "Source /var/private/file.pdf",
        "Source /tmp/private/file.pdf",
        "Source data/private/source_manifest.json",
        r"Source data\vector_store\index.faiss",
    ),
)
def test_all_required_absolute_and_private_display_paths_are_rejected(
    unsafe: str,
) -> None:
    with pytest.raises(EvidenceValidationError) as raised:
        UnifiedEvidenceBuilder().build((qualitative(text=unsafe),))
    assert unsafe not in str(raised.value)


@pytest.mark.parametrize(
    "field,unsafe",
    (
        ("source_document_id", r"C:\private\document"),
        ("source_title", r"Source (C:\private\secret.pdf)"),
        ("chunk_id", r"C:\private\chunk-1"),
        ("chunk_id", r"\\server\share\chunk-1"),
        ("chunk_id", "//server/share/chunk-1"),
        ("chunk_id", "/opt/company/chunk-1"),
    ),
)
def test_source_metadata_paths_are_rejected_without_value_leakage(
    field: str, unsafe: str
) -> None:
    with pytest.raises(EvidenceValidationError) as raised:
        UnifiedEvidenceBuilder().build((qualitative(**{field: unsafe}),))
    assert unsafe not in str(raised.value)
    assert field in str(raised.value)


@pytest.mark.parametrize(
    "unsafe",
    (r"C:\private\facts.csv", r"\\server\share\facts.csv", "/opt/company/facts.csv"),
)
def test_optional_source_label_paths_are_rejected_without_value_leakage(
    unsafe: str,
) -> None:
    with pytest.raises(EvidenceValidationError) as raised:
        UnifiedEvidenceBuilder().build((replace(fact(), source_label=unsafe),))
    assert unsafe not in str(raised.value)
    assert "source_label" in str(raised.value)


@pytest.mark.parametrize("page", (0, -1, "12", "abc", 12.5, True))
def test_source_reference_rejects_invalid_page_number(page: object) -> None:
    with pytest.raises(EvidenceValidationError, match="positive integer"):
        SourceReference("doc", "Synthetic report", page)  # type: ignore[arg-type]


def test_source_reference_accepts_positive_integer_page() -> None:
    assert SourceReference("doc", "Synthetic report", 12).page_number == 12


def test_raw_string_evidence_type_is_rejected_and_enum_is_retained() -> None:
    source = UnifiedEvidenceBuilder().build((qualitative(),))[0]
    assert isinstance(source.evidence_type, EvidenceType)
    with pytest.raises(EvidenceValidationError, match="EvidenceType"):
        UnifiedEvidence(
            source.evidence_id,
            "qualitative",  # type: ignore[arg-type]
            source.company_id,
            source.company_name,
            source.ticker,
            source.fiscal_year,
            source.display_text,
            source.source_references,
            source.data,
        )


def test_unknown_string_evidence_type_is_rejected() -> None:
    source = UnifiedEvidenceBuilder().build((qualitative(),))[0]
    with pytest.raises(EvidenceValidationError, match="EvidenceType"):
        UnifiedEvidence(
            source.evidence_id,
            "unknown",  # type: ignore[arg-type]
            source.company_id,
            source.company_name,
            source.ticker,
            source.fiscal_year,
            source.display_text,
            source.source_references,
            source.data,
        )


@pytest.mark.parametrize(
    "text",
    (
        "gross_profit / revenue * 100",
        "The reporting period covers 12 / 24 months.",
        "Gross margin was 40.00% in this fictional example.",
        "Revenue and operating income are ordinary financial metrics.",
        "Fiscal years 2024/2025 are compared deterministically.",
    ),
)
def test_normal_financial_slash_and_narrative_text_is_accepted(text: str) -> None:
    item = UnifiedEvidenceBuilder().build((qualitative(text=text),))[0]
    assert item.display_text == text


@pytest.mark.parametrize(
    "document_id,chunk_id",
    (
        ("synthetic-asus-2025", "doc-asus:page-0042:chunk-000001"),
        ("sha256:abcdef", "chunk:000001"),
    ),
)
def test_safe_synthetic_document_and_chunk_ids_are_accepted(
    document_id: str, chunk_id: str
) -> None:
    item = UnifiedEvidenceBuilder().build(
        (qualitative(source_document_id=document_id, chunk_id=chunk_id),)
    )[0]
    assert item.source_references[0].source_document_id == document_id
    assert item.source_references[0].chunk_id == chunk_id


def test_unknown_input_type_fails_explicitly() -> None:
    with pytest.raises(TypeError, match="Unsupported evidence input type"):
        UnifiedEvidenceBuilder().build((object(),))  # type: ignore[arg-type]


def test_calculated_model_retains_original_provenance_objects() -> None:
    result = calculation()
    item = UnifiedEvidenceBuilder().build((result,))[0]
    assert tuple(ref.source_document_id for ref in item.source_references) == tuple(
        provenance.source_document_id for provenance in result.provenance
    )
    assert all(isinstance(value, FinancialFactProvenance) for value in result.provenance)
