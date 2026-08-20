"""Offline tests for guarded synthesis over unified competitor evidence."""

from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from enterprise_rag.competitor_analysis import CitationReadyEvidence
from enterprise_rag.competitor_citations import render_competitor_answer
from enterprise_rag.competitor_evidence import UnifiedEvidenceBuilder, UnifiedEvidenceSet
from enterprise_rag.competitor_grounded_synthesis import (
    FinancialGroundingError,
    FinancialClaimType,
    GroundedCompetitorSynthesizer,
    GroundedGenerationError,
    GroundedResponseFormatError,
    GroundedSynthesisError,
    GroundedSynthesisStatus,
    FINANCIAL_RESPONSE_SCHEMA,
    QualitativeCoverage,
    ResponseLanguage,
    UnknownEvidenceCitationError,
    build_grounded_synthesis_prompt,
)
from enterprise_rag.competitor_planning import DeterministicQuestionRouter
from enterprise_rag.financial_calculations import FinancialCalculationEngine
from enterprise_rag.financial_comparisons import FinancialComparisonEngine
from enterprise_rag.financial_facts import FinancialFact, FinancialFactCollection


COMPANIES = {
    "gigabyte": ("Gigabyte", "2376"),
    "asus": ("ASUS", "2357"),
    "msi": ("MSI", "2377"),
}


class FakeGenerationClient:
    provider = "fake"
    model = "fake-grounded-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeStructuredGenerationClient(FakeGenerationClient):
    def __init__(self, response: str) -> None:
        super().__init__(response)
        self.schemas: list[dict[str, object]] = []

    def generate_structured(self, prompt: str, schema: dict[str, object]) -> str:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return self.response


class SequenceStructuredGenerationClient(FakeStructuredGenerationClient):
    def __init__(self, responses: list[str]) -> None:
        super().__init__("")
        self.responses = iter(responses)

    def generate_structured(self, prompt: str, schema: dict[str, object]) -> str:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return next(self.responses)


class FailingGenerationClient(FakeGenerationClient):
    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        raise RuntimeError("synthetic provider failure")


def response(
    answer_text: str,
    cited: tuple[str, ...] = ("E1",),
    *,
    insufficient: bool = False,
    financial_claims: list[dict[str, object]] | None = None,
) -> str:
    claims = (
        _claims_for_markers(answer_text)
        if financial_claims is None
        else financial_claims
    )
    block_text = re.sub(
        r"\s*\[((?:E\d+\s*(?:,\s*E\d+\s*)*))\]",
        "",
        answer_text,
    ).strip()
    block_text = re.sub(r"\[\[[^\[\]\r\n]+\]\]", "", block_text).strip()
    del cited, insufficient
    return json.dumps({"claims": claims} if claims else {"text": block_text})


def blocks_response(
    blocks: list[dict[str, object]],
    *,
    insufficient: bool = False,
    financial_claims: list[dict[str, object]] | None = None,
) -> str:
    del insufficient
    if financial_claims:
        return json.dumps({"claims": financial_claims})
    text = "\n\n".join(str(block.get("text", "")) for block in blocks)
    return json.dumps({"text": text})


def _claims_for_markers(answer_text: str) -> list[dict[str, object]]:
    claims: list[dict[str, object]] = []
    role_types = {
        "reported_value": "reported_fact",
        "calculated_value": "calculated_metric",
        "ranked_entry": "comparison_entry",
        "earlier_value": "financial_change_value",
        "later_value": "financial_change_value",
        "percentage_point_change": "financial_change_value",
    }
    for marker in re.findall(r"\[\[([^\[\]\r\n]+)\]\]", answer_text):
        fields = marker.split(":")
        if len(fields) not in (3, 5) or fields[1] not in role_types:
            continue
        evidence_id, role, value = fields[0], fields[1], fields[-1]
        claim: dict[str, object] = {
            "evidence_id": evidence_id,
            "claim_type": role_types[role],
            "role": role,
            "value": value,
        }
        if role == "ranked_entry":
            if len(fields) != 5 or not fields[3].isascii() or not fields[3].isdecimal():
                continue
            claim["company_id"] = fields[2]
            claim["rank"] = int(fields[3])
        claims.append(claim)
    return claims


def qualitative(company_id: str = "asus") -> CitationReadyEvidence:
    company_name, ticker = COMPANIES[company_id]
    return CitationReadyEvidence(
        evidence_id="old-evidence-id",
        company_id=company_id,
        company_name=company_name,
        ticker=ticker,
        fiscal_year=2025,
        document_type="annual_report",
        source_document_id=f"synthetic-{company_id}-2025",
        source_title=f"Synthetic {company_name} Annual Report",
        page_number=42,
        chunk_id=f"synthetic-{company_id}:page-0042:chunk-000001",
        text=f"{company_name} describes a fictional AI infrastructure strategy.",
        retrieval_score=0.2,
        quality_score=0.9,
        original_candidate_rank=2,
        final_company_rank=1,
    )


def comparison_response(*companies: str) -> str:
    profiles = {}
    observations = {}
    for index, company in enumerate(companies, start=1):
        evidence_id = f"E{index}"
        profiles[company] = {"summary": f"{company} strategy.", "evidence_ids": [evidence_id]}
        observations[company] = {"text": f"{company} positioning.", "evidence_ids": [evidence_id]}
    return json.dumps({
        "text": "Supported comparison.",
        "company_profiles": profiles,
        "comparison_dimensions": [{"label": "Positioning", "observations": observations}],
    })


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
        source_sha256="a" * 64,
        notes="Fictional test value",
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


def test_qualitative_only_grounded_synthesis() -> None:
    client = FakeGenerationClient(response("Supported qualitative claim [E1]."))
    result = GroundedCompetitorSynthesizer(client).synthesize(
        "Describe the strategy.", unified(qualitative())
    )
    assert result.status is GroundedSynthesisStatus.GROUNDED
    assert result.answer_text == "Supported qualitative claim. [E1]"
    assert result.cited_evidence_ids == ("E1",)
    assert result.financial_claims == ()
    assert result.generation_provider == "fake" and result.generation_model == "fake-grounded-model"


def test_python_owns_complete_qualitative_coverage_and_scope_citations() -> None:
    plan = DeterministicQuestionRouter().plan("Compare ASUS and MSI AI strategies.")
    client = FakeStructuredGenerationClient(comparison_response("asus", "msi"))
    result = GroundedCompetitorSynthesizer(client).synthesize(
        plan.question, unified(qualitative("asus"), qualitative("msi")), plan
    )
    assert result.qualitative_coverage is QualitativeCoverage.COMPLETE
    assert result.missing_qualitative_companies == ()
    assert result.cited_evidence_ids == ("E1", "E2")
    assert len(client.prompts) == 1
    assert result.comparison is not None
    assert result.comparison.covered_companies == ("asus", "msi")


def test_three_company_comparison_and_response_languages_are_deterministic() -> None:
    plan = DeterministicQuestionRouter().plan("Compare ASUS, Gigabyte, and MSI AI strategies.")
    client = FakeStructuredGenerationClient(comparison_response("asus", "gigabyte", "msi"))
    result = GroundedCompetitorSynthesizer(client).synthesize(
        plan.question,
        unified(qualitative("asus"), qualitative("gigabyte"), qualitative("msi")),
        plan,
    )
    assert result.comparison is not None and len(result.comparison.company_profiles) == 3
    assert result.response_language is ResponseLanguage.EN
    assert "Write all generated prose in English." in client.prompts[0]
    assert GroundedCompetitorSynthesizer(FakeGenerationClient(json.dumps({"text": "摘要"}))).synthesize(
        "請說明 ASUS 的 AI 策略。", unified(qualitative("asus"))
    ).response_language is ResponseLanguage.ZH_TW


def test_comparison_schema_uses_exact_company_slots_and_company_evidence_enums() -> None:
    plan = DeterministicQuestionRouter().plan("Compare ASUS, Gigabyte, and MSI AI strategies.")
    client = FakeStructuredGenerationClient(comparison_response("asus", "gigabyte", "msi"))
    result = GroundedCompetitorSynthesizer(client).synthesize(
        plan.question,
        unified(qualitative("asus"), qualitative("gigabyte"), qualitative("msi")),
        plan,
    )

    schema = client.schemas[0]
    profiles = schema["properties"]["company_profiles"]
    assert profiles["required"] == list(plan.requested_companies)
    assert profiles["additionalProperties"] is False
    assert profiles["properties"]["asus"]["properties"]["evidence_ids"]["items"]["enum"] == ["E1"]
    assert profiles["properties"]["gigabyte"]["properties"]["evidence_ids"]["items"]["enum"] == ["E2"]
    dimensions = schema["properties"]["comparison_dimensions"]
    assert dimensions["minItems"] == dimensions["maxItems"] == 1
    observations = dimensions["items"]["properties"]["observations"]
    assert observations["required"] == list(plan.requested_companies)
    assert observations["additionalProperties"] is False
    assert result.comparison is not None
    assert tuple(item.company_id for item in result.comparison.company_profiles) == plan.requested_companies


def test_partial_comparison_schema_omits_missing_company_slots() -> None:
    plan = DeterministicQuestionRouter().plan("Compare ASUS, Gigabyte, and MSI AI strategies.")
    client = FakeStructuredGenerationClient(comparison_response("asus", "gigabyte"))
    GroundedCompetitorSynthesizer(client).synthesize(
        plan.question, unified(qualitative("asus"), qualitative("gigabyte")), plan
    )
    schema = client.schemas[0]
    profiles = schema["properties"]["company_profiles"]
    observations = schema["properties"]["comparison_dimensions"]["items"]["properties"]["observations"]
    expected_covered = [company for company in plan.requested_companies if company != "msi"]
    assert profiles["required"] == expected_covered
    assert set(profiles["properties"]) == {"asus", "gigabyte"}
    assert observations["required"] == expected_covered
    assert "msi" not in observations["properties"]


def test_comparison_rejects_cross_company_and_unknown_evidence_bindings() -> None:
    plan = DeterministicQuestionRouter().plan("Compare ASUS and MSI AI strategies.")
    payload = json.loads(comparison_response("asus", "msi"))
    payload["company_profiles"]["asus"]["evidence_ids"] = ["E2"]
    with pytest.raises(UnknownEvidenceCitationError, match="another company"):
        GroundedCompetitorSynthesizer(FakeStructuredGenerationClient(json.dumps(payload))).synthesize(
            plan.question, unified(qualitative("asus"), qualitative("msi")), plan
        )


def test_comparison_rejects_financial_evidence_id() -> None:
    plan = DeterministicQuestionRouter().plan("Compare ASUS and MSI 2025 revenue and AI strategies.")
    evidence = unified(qualitative("asus"), qualitative("msi"), fact("asus"))
    financial_id = next(
        item.evidence_id for item in evidence if item.evidence_type.value == "financial_fact"
    )
    payload = json.loads(comparison_response("asus", "msi"))
    payload["company_profiles"]["asus"]["evidence_ids"] = [financial_id]
    client = SequenceStructuredGenerationClient([
        json.dumps(payload),
        json.dumps({"claims": []}),
    ])
    with pytest.raises(UnknownEvidenceCitationError, match="unknown evidence"):
        GroundedCompetitorSynthesizer(client).synthesize(
            plan.question,
            evidence,
            plan,
        )
    payload = json.loads(comparison_response("asus", "msi"))
    payload["company_profiles"]["asus"]["evidence_ids"] = ["E99"]
    with pytest.raises(UnknownEvidenceCitationError, match="unknown evidence"):
        GroundedCompetitorSynthesizer(FakeStructuredGenerationClient(json.dumps(payload))).synthesize(
            plan.question, unified(qualitative("asus"), qualitative("msi")), plan
        )


def test_partial_coverage_is_python_owned_and_takeaway_is_omitted() -> None:
    plan = DeterministicQuestionRouter().plan("Compare ASUS, Gigabyte, and MSI AI strategies.")
    result = GroundedCompetitorSynthesizer(FakeStructuredGenerationClient(
        comparison_response("asus", "gigabyte")
    )).synthesize(
        plan.question, unified(qualitative("asus"), qualitative("gigabyte")), plan
    )
    assert result.status is GroundedSynthesisStatus.PARTIAL
    assert result.comparison is not None
    assert result.comparison.missing_companies == ("msi",)
    assert result.comparison.key_takeaway is None


def test_python_owns_partial_qualitative_coverage() -> None:
    plan = DeterministicQuestionRouter().plan("Compare ASUS and MSI AI strategies.")
    client = FakeStructuredGenerationClient(json.dumps({"text": "Available evidence only."}))
    result = GroundedCompetitorSynthesizer(client).synthesize(
        plan.question, unified(qualitative("asus")), plan
    )
    assert result.status is GroundedSynthesisStatus.PARTIAL
    assert result.qualitative_coverage is QualitativeCoverage.PARTIAL
    assert result.missing_qualitative_companies == ("msi",)
    assert result.cited_evidence_ids == ("E1",)
    assert "MSI describes" not in client.prompts[0]


def test_zero_qualitative_evidence_skips_generation() -> None:
    plan = DeterministicQuestionRouter().plan("Compare ASUS and MSI AI strategies.")
    client = FakeStructuredGenerationClient(json.dumps({"text": "must not be used"}))
    result = GroundedCompetitorSynthesizer(client).synthesize(
        plan.question, unified(), plan
    )
    assert result.status is GroundedSynthesisStatus.INSUFFICIENT
    assert result.qualitative_coverage is QualitativeCoverage.INSUFFICIENT
    assert result.missing_qualitative_companies == ("asus", "msi")
    assert client.prompts == []


def test_grounded_synthesis_uses_optional_structured_generation_capability() -> None:
    client = FakeStructuredGenerationClient(
        response("Supported qualitative claim [E1].")
    )
    result = GroundedCompetitorSynthesizer(client).synthesize(
        "Describe the strategy.", unified(qualitative())
    )
    assert result.status is GroundedSynthesisStatus.GROUNDED
    assert len(client.schemas) == 1
    assert client.schemas[0]["additionalProperties"] is False
    assert set(client.schemas[0]["required"]) == {"text"}
    assert client.schemas[0]["properties"]["text"] == {"type": "string"}


def test_financial_fact_synthesis_preserves_exact_reported_value() -> None:
    client = FakeGenerationClient(
        response("Reported revenue was [[E1:reported_value:100]] million TWD [E1].")
    )
    result = GroundedCompetitorSynthesizer(client).synthesize("Revenue?", unified(fact()))
    assert result.answer_text == "ASUS 2025 revenue reported value: 100 million_TWD [E1]"
    assert result.financial_claims[0].claim_type is FinancialClaimType.REPORTED_FACT
    assert "financial_fact" in client.prompts[0]
    assert '"reported_source_value":"100"' in client.prompts[0]


@pytest.mark.parametrize(
    "marker",
    (
        "[[E1:calculated_value:100]]",
        "[[E1:reported_value:101]]",
    ),
)
def test_financial_fact_rejects_wrong_role_or_value(marker: str) -> None:
    client = FakeGenerationClient(response(f"Reported revenue was {marker} [E1]."))
    with pytest.raises(FinancialGroundingError, match="role or value"):
        GroundedCompetitorSynthesizer(client).synthesize("Revenue?", unified(fact()))


def test_structured_reported_fact_remains_authoritative_regardless_of_prose() -> None:
    client = FakeGenerationClient(
        response("Python calculated revenue as [[E1:reported_value:100]] [E1].")
    )
    result = GroundedCompetitorSynthesizer(client).synthesize(
        "Revenue?", unified(fact())
    )
    assert result.financial_claims[0].claim_type is FinancialClaimType.REPORTED_FACT


def test_financial_calculation_is_labeled_and_not_recomputed() -> None:
    client = FakeGenerationClient(
        response("Python calculated gross margin as [[E1:calculated_value:40.00]] percent [E1].")
    )
    result = GroundedCompetitorSynthesizer(client).synthesize(
        "Gross margin?", unified(calculation())
    )
    assert "40.00 percent" in result.answer_text
    assert result.financial_claims[0].claim_type is FinancialClaimType.CALCULATED_METRIC
    prompt = client.prompts[0]
    assert "financial_calculation" in prompt
    assert '"calculated_value":"40.00"' in prompt
    assert "gross_profit / revenue * 100" in prompt
    assert "Never calculate, recalculate" in prompt


@pytest.mark.parametrize(
    "marker",
    (
        "[[E1:reported_value:40.00]]",
        "[[E1:calculated_value:100]]",
        "[[E1:calculated_value:40.0]]",
        "[[E1:calculated_value:40]]",
    ),
)
def test_financial_calculation_rejects_wrong_role_input_or_decimal(marker: str) -> None:
    client = FakeGenerationClient(response(f"Calculated margin was {marker} [E1]."))
    with pytest.raises(FinancialGroundingError, match="role or value"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Margin?", unified(calculation())
        )


def test_structured_calculated_metric_remains_authoritative_regardless_of_prose() -> None:
    client = FakeGenerationClient(
        response("The report reported margin as [[E1:calculated_value:40.00]] [E1].")
    )
    result = GroundedCompetitorSynthesizer(client).synthesize(
        "Margin?", unified(calculation())
    )
    assert result.financial_claims[0].claim_type is FinancialClaimType.CALCULATED_METRIC


@pytest.mark.parametrize(
    ("answer,evidence_factory,expected_type"),
    (
        (
            "Python computed revenue [[E1:reported_value:100]] [E1].",
            fact,
            FinancialClaimType.REPORTED_FACT,
        ),
        (
            "The source stated margin [[E1:calculated_value:40.00]] [E1].",
            calculation,
            FinancialClaimType.CALCULATED_METRIC,
        ),
    ),
)
def test_prose_wording_does_not_override_structured_claim_type(
    answer: str, evidence_factory, expected_type: FinancialClaimType
) -> None:
    result = GroundedCompetitorSynthesizer(
        FakeGenerationClient(response(answer))
    ).synthesize("Question?", unified(evidence_factory()))
    assert result.financial_claims[0].claim_type is expected_type


def test_financial_comparison_preserves_supplied_ranking() -> None:
    engine = comparison_engine(
        *margin_facts("asus", 2025, "50"),
        *margin_facts("msi", 2025, "40"),
    )
    evidence = unified(engine.rank_companies("gross_margin", 2025, ("asus", "msi")))
    client = FakeGenerationClient(
        response("The supplied ranking places ASUS higher at [[E1:ranked_entry:asus:1:50.00]] percent [E1].")
    )
    result = GroundedCompetitorSynthesizer(client).synthesize("Compare margins.", evidence)
    assert result.financial_claims[0].company_id == "asus"
    assert result.financial_claims[0].rank == 1
    prompt = client.prompts[0]
    assert "financial_comparison" in prompt
    assert '"rank":1' in prompt and '"rank":2' in prompt
    assert '"ranking_direction":"higher_value_first"' in prompt
    assert "Never calculate" in prompt


def comparison_evidence() -> UnifiedEvidenceSet:
    engine = comparison_engine(
        *margin_facts("asus", 2025, "50"),
        *margin_facts("msi", 2025, "40"),
    )
    return unified(engine.rank_companies("gross_margin", 2025, ("asus", "msi")))


@pytest.mark.parametrize(
    "marker",
    (
        "[[E1:ranked_entry:msi:1:50.00]]",
        "[[E1:ranked_entry:asus:2:50.00]]",
        "[[E1:ranked_entry:asus:1:40.00]]",
        "[[E1:ranked_entry:msi:2:50.00]]",
        "[[E1:ranked_entry:gigabyte:1:50.00]]",
    ),
)
def test_comparison_rejects_cross_entry_substitution(marker: str) -> None:
    client = FakeGenerationClient(response(f"Supplied ranking: {marker} [E1]."))
    with pytest.raises(FinancialGroundingError, match="role or value"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Compare.", comparison_evidence()
        )


def test_comparison_accepts_tied_entries_with_supplied_rank() -> None:
    engine = comparison_engine(
        *margin_facts("asus", 2025, "40"),
        *margin_facts("msi", 2025, "40"),
    )
    evidence = unified(engine.rank_companies("gross_margin", 2025, ("asus", "msi")))
    client = FakeGenerationClient(
        response(
            "The supplied tie contains [[E1:ranked_entry:asus:1:40.00]] and "
            "[[E1:ranked_entry:msi:1:40.00]] [E1]."
        )
    )
    result = GroundedCompetitorSynthesizer(client).synthesize("Compare.", evidence)
    assert "Rank 1" in result.answer_text
    assert result.answer_text.count("40.00") == 2


def test_partial_comparison_missing_company_cannot_be_invented() -> None:
    model = comparison_engine(*margin_facts("asus", 2025, "40")).rank_companies(
        "gross_margin", 2025, ("asus", "msi")
    )
    client = FakeGenerationClient(
        response("Invented entry [[E1:ranked_entry:msi:2:40.00]] [E1].")
    )
    with pytest.raises(FinancialGroundingError, match="role or value"):
        GroundedCompetitorSynthesizer(client).synthesize("Compare.", unified(model))


def test_financial_change_preserves_supplied_percentage_points() -> None:
    engine = comparison_engine(
        *margin_facts("asus", 2024, "30"),
        *margin_facts("asus", 2025, "40"),
    )
    evidence = unified(engine.compare_company_years("asus", "gross_margin", 2024, 2025))
    client = FakeGenerationClient(
        response("The supplied increase was [[E1:percentage_point_change:10.00]] percentage points [E1].")
    )
    result = GroundedCompetitorSynthesizer(client).synthesize("What changed?", evidence)
    assert "10.00 percentage points" in result.answer_text
    assert result.financial_claims[0].role == "percentage_point_change"
    assert '"direction":"increase"' in client.prompts[0]
    assert '"earlier_year":2024' in client.prompts[0]
    assert '"later_year":2025' in client.prompts[0]


@pytest.mark.parametrize(
    "role,value",
    (
        ("earlier_value", "30.00"),
        ("later_value", "40.00"),
        ("percentage_point_change", "10.00"),
    ),
)
def test_change_accepts_each_role_correct_value(role: str, value: str) -> None:
    engine = comparison_engine(
        *margin_facts("asus", 2024, "30"),
        *margin_facts("asus", 2025, "40"),
    )
    evidence = unified(engine.compare_company_years("asus", "gross_margin", 2024, 2025))
    client = FakeGenerationClient(response(f"Supplied value [[E1:{role}:{value}]] [E1]."))
    result = GroundedCompetitorSynthesizer(client).synthesize("Change?", evidence)
    assert value in result.answer_text


@pytest.mark.parametrize(
    "marker",
    (
        "[[E1:percentage_point_change:40.00]]",
        "[[E1:later_value:10.00]]",
        "[[E1:earlier_value:40.00]]",
    ),
)
def test_change_rejects_role_substitution(marker: str) -> None:
    engine = comparison_engine(
        *margin_facts("asus", 2024, "30"),
        *margin_facts("asus", 2025, "40"),
    )
    evidence = unified(engine.compare_company_years("asus", "gross_margin", 2024, 2025))
    client = FakeGenerationClient(response(f"Supplied change {marker} [E1]."))
    with pytest.raises(FinancialGroundingError, match="role or value"):
        GroundedCompetitorSynthesizer(client).synthesize("Change?", evidence)


def test_combined_qualitative_and_financial_synthesis() -> None:
    evidence = unified(qualitative(), calculation())
    claims = _claims_for_markers("[[E2:calculated_value:40.00]]")
    client = SequenceStructuredGenerationClient(
        [json.dumps({"text": "The strategy is described in the report."}),
         json.dumps({"claims": claims})]
    )
    result = GroundedCompetitorSynthesizer(client).synthesize("Combine them.", evidence)
    assert result.cited_evidence_ids == ("E1", "E2")
    assert "40.00 percent" in result.answer_text
    assert len(client.prompts) == 2
    assert client.schemas[0]["required"] == ["text"]
    assert client.schemas[1]["required"] == ["claims"]


def test_prompt_and_repeated_synthesis_are_deterministic_and_ordered() -> None:
    evidence = unified(qualitative(), fact())
    first_prompt = build_grounded_synthesis_prompt("Question?", evidence)
    second_prompt = build_grounded_synthesis_prompt("Question?", evidence)
    assert first_prompt == second_prompt
    assert first_prompt.index('"evidence_id":"E1"') < first_prompt.index('"evidence_id":"E2"')
    payloads = [response("Qualitative claim [E1]."), response("[[E2:reported_value:100]]")]
    first = GroundedCompetitorSynthesizer(SequenceStructuredGenerationClient(payloads)).synthesize(
        "Question?", evidence
    )
    second = GroundedCompetitorSynthesizer(SequenceStructuredGenerationClient(payloads)).synthesize(
        "Question?", evidence
    )
    assert first == second


def test_qualitative_scope_citations_follow_python_evidence_order() -> None:
    evidence = unified(qualitative(), qualitative())
    client = FakeStructuredGenerationClient(json.dumps({"text": "Jointly supported claim."}))
    result = GroundedCompetitorSynthesizer(client).synthesize("Question?", evidence)
    assert result.answer_text == "Jointly supported claim. [E1, E2]"
    assert result.cited_evidence_ids == ("E1", "E2")
    assert len(client.prompts) == 1


def test_provider_block_text_must_not_contain_citation_syntax() -> None:
    client = FakeGenerationClient(json.dumps({"text": "Claim [E1]."}))
    with pytest.raises(GroundedResponseFormatError, match="citation syntax"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(qualitative())
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"text": ""},
        {"text": "Claim.", "extra": True},
        {"text": 1},
        {},
    ),
)
def test_qualitative_contract_rejects_invalid_values(payload: object) -> None:
    client = FakeGenerationClient(json.dumps(payload))
    with pytest.raises(GroundedResponseFormatError):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(qualitative())
        )


def obsolete_test_financial_marker_must_be_supported_by_its_own_block() -> None:
    evidence = unified(qualitative(), calculation())
    marker = "[[E2:calculated_value:40.00]]"
    client = FakeGenerationClient(
        blocks_response(
            [{"text": f"Unsupported margin {marker}.", "evidence_ids": ["E1"]}],
            financial_claims=_claims_for_markers(marker),
        )
    )
    with pytest.raises(FinancialGroundingError, match="cited evidence"):
        GroundedCompetitorSynthesizer(client).synthesize("Combine.", evidence)


def test_empty_evidence_returns_insufficient_without_provider_call() -> None:
    client = FakeGenerationClient("not used")
    result = GroundedCompetitorSynthesizer(client).synthesize(
        "Question?", UnifiedEvidenceSet(())
    )
    assert result.status is GroundedSynthesisStatus.INSUFFICIENT
    assert result.cited_evidence_ids == () and result.generation_model is None
    assert not client.prompts


def test_non_ready_analysis_plan_returns_insufficient_without_generation() -> None:
    plan = DeterministicQuestionRouter().plan("What about ASUS?")
    client = FakeGenerationClient("not used")
    result = GroundedCompetitorSynthesizer(client).synthesize(
        "What about ASUS?", unified(qualitative()), plan
    )
    assert result.status is GroundedSynthesisStatus.INSUFFICIENT
    assert not client.prompts


def test_insufficient_comparison_returns_without_generation() -> None:
    result_model = comparison_engine().rank_companies(
        "gross_margin", 2025, ("asus", "msi")
    )
    client = FakeGenerationClient("not used")
    result = GroundedCompetitorSynthesizer(client).synthesize(
        "Compare.", unified(result_model)
    )
    assert result.status is GroundedSynthesisStatus.INSUFFICIENT
    assert not client.prompts


def test_partial_comparison_is_serialized_without_filling_missing_company() -> None:
    model = comparison_engine(*margin_facts("asus", 2025, "40")).rank_companies(
        "gross_margin", 2025, ("asus", "msi")
    )
    client = FakeGenerationClient(
        response("Available margin is [[E1:ranked_entry:asus:1:40.00]] percent; MSI is missing [E1].")
    )
    GroundedCompetitorSynthesizer(client).synthesize("Compare.", unified(model))
    assert '"comparison_status":"partial"' in client.prompts[0]
    assert '"missing_companies":["msi"]' in client.prompts[0]


def test_mutated_financial_value_is_rejected() -> None:
    client = FakeGenerationClient(
        response("Calculated margin was [[E1:calculated_value:41.00]] percent [E1].")
    )
    with pytest.raises(FinancialGroundingError, match="changed or invented"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Margin?", unified(calculation())
        )


def obsolete_test_unmarked_financial_number_is_rejected() -> None:
    client = FakeGenerationClient(response("Calculated margin was 41.00 percent [E1]."))
    with pytest.raises(FinancialGroundingError, match="bind every numeric"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Margin?", unified(calculation())
        )


def obsolete_test_insufficient_response_cannot_fill_an_uncited_financial_value() -> None:
    client = FakeGenerationClient(
        response("Evidence is insufficient, so the missing value is 999.", (), insufficient=True)
    )
    with pytest.raises(FinancialGroundingError, match="bind every numeric"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Missing value?", unified(fact())
        )


def obsolete_test_qualitative_citation_cannot_hide_bare_number_when_financial_evidence_exists() -> None:
    client = FakeGenerationClient(response("Unsupported value is 999 [E1]."))
    with pytest.raises(FinancialGroundingError, match="bind every numeric"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Combined?", unified(qualitative(), fact())
        )


def obsolete_test_financial_marker_must_use_cited_financial_evidence() -> None:
    evidence = unified(qualitative(), calculation())
    client = FakeGenerationClient(
        response("Claim [[E2:calculated_value:40.00]] [E1].", ("E1",))
    )
    with pytest.raises(FinancialGroundingError, match="cited evidence"):
        GroundedCompetitorSynthesizer(client).synthesize("Question?", evidence)


def obsolete_test_qualitative_evidence_cannot_authorize_financial_marker() -> None:
    client = FakeGenerationClient(response("Claim [[E1:calculated_value:40.00]] [E1]."))
    with pytest.raises(FinancialGroundingError, match="Qualitative"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(qualitative())
        )


@pytest.mark.parametrize(
    ("marker", "error"),
    (
        ("[[E1:calculated_value]]", "field count"),
        ("[[E1:unknown_role:40.00]]", "role or value"),
        ("[[E1:calculated_value:40.00:extra]]", "field count"),
        ("[[E1:ranked_entry:asus:not-an-int:50.00]]", "rank must be an integer"),
        ("[[E1:calculated_value:not-a-decimal]]", "invalid decimal"),
        ("[[E1:calculated_value:40.00", "malformed marker"),
    ),
)
def obsolete_test_malformed_role_aware_markers_are_rejected(marker: str, error: str) -> None:
    client = FakeGenerationClient(response(f"Claim {marker} [E1]."))
    with pytest.raises(FinancialGroundingError, match=error):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(calculation())
        )


def obsolete_test_role_aware_marker_with_unknown_evidence_is_rejected() -> None:
    client = FakeGenerationClient(
        response("Claim [[E99:calculated_value:40.00]] [E99].", ("E99",))
    )
    with pytest.raises(UnknownEvidenceCitationError):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(calculation())
        )


def obsolete_test_marker_without_structured_claim_is_rejected() -> None:
    client = FakeGenerationClient(
        response(
            "Value [[E1:calculated_value:40.00]] [E1].",
            financial_claims=[],
        )
    )
    with pytest.raises(FinancialGroundingError, match="match answer markers"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(calculation())
        )


def obsolete_test_structured_claim_without_marker_is_rejected() -> None:
    claims = _claims_for_markers("[[E1:calculated_value:40.00]]")
    client = FakeGenerationClient(
        response("Supported claim [E1].", financial_claims=claims)
    )
    with pytest.raises(FinancialGroundingError, match="match answer markers"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(calculation())
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("claim_type", "reported_fact"),
        ("role", "reported_value"),
        ("value", "41.00"),
        ("evidence_id", "E2"),
    ),
)
def obsolete_test_marker_and_structured_claim_must_match(field: str, value: object) -> None:
    claims = _claims_for_markers("[[E1:calculated_value:40.00]]")
    claims[0][field] = value
    client = FakeGenerationClient(
        response(
            "Value [[E1:calculated_value:40.00]] [E1].",
            financial_claims=claims,
        )
    )
    expected = UnknownEvidenceCitationError if field == "evidence_id" else FinancialGroundingError
    with pytest.raises(expected):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(calculation())
        )


def obsolete_test_duplicate_structured_claim_is_rejected() -> None:
    claims = _claims_for_markers("[[E1:calculated_value:40.00]]")
    client = FakeGenerationClient(
        response(
            "Value [[E1:calculated_value:40.00]] [E1].",
            financial_claims=claims + claims,
        )
    )
    with pytest.raises(FinancialGroundingError, match="Duplicate financial claims"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(calculation())
        )


def obsolete_test_structured_claim_order_must_match_marker_order() -> None:
    evidence = unified(calculation("asus"), calculation("msi"))
    answer = (
        "Values [[E1:calculated_value:40.00]] and "
        "[[E2:calculated_value:40.00]] [E1, E2]."
    )
    claims = list(reversed(_claims_for_markers(answer)))
    client = FakeGenerationClient(
        response(answer, ("E1", "E2"), financial_claims=claims)
    )
    with pytest.raises(FinancialGroundingError, match="exactly and in order"):
        GroundedCompetitorSynthesizer(client).synthesize("Question?", evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    (("company_id", "msi"), ("rank", 2), ("value", "40.00")),
)
def obsolete_test_comparison_claim_fields_must_match_marker_fields(
    field: str, value: object
) -> None:
    claims = _claims_for_markers("[[E1:ranked_entry:asus:1:50.00]]")
    claims[0][field] = value
    client = FakeGenerationClient(
        response(
            "ASUS [[E1:ranked_entry:asus:1:50.00]] [E1].",
            financial_claims=claims,
        )
    )
    with pytest.raises(FinancialGroundingError):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Compare.", comparison_evidence()
        )


def obsolete_test_change_claim_role_must_match_marker_role() -> None:
    engine = comparison_engine(
        *margin_facts("asus", 2024, "30"),
        *margin_facts("asus", 2025, "40"),
    )
    evidence = unified(engine.compare_company_years("asus", "gross_margin", 2024, 2025))
    claims = _claims_for_markers("[[E1:later_value:40.00]]")
    claims[0]["role"] = "percentage_point_change"
    client = FakeGenerationClient(
        response(
            "Later [[E1:later_value:40.00]] [E1].",
            financial_claims=claims,
        )
    )
    with pytest.raises(FinancialGroundingError):
        GroundedCompetitorSynthesizer(client).synthesize("Change?", evidence)


@pytest.mark.parametrize(
    "text",
    (
        "An arbitrary 7 appears.",
        "The fiscal year is 2025.",
        "Rank 1 leads.",
        "Margin is 40%.",
        "The value is 40.00.",
        "1. Numbered item.",
    ),
)
def test_qualitative_digits_do_not_fail_or_create_financial_claims(text: str) -> None:
    client = FakeGenerationClient(json.dumps({"text": text}))
    result = GroundedCompetitorSynthesizer(client).synthesize(
        "Question?", unified(qualitative())
    )
    assert result.answer_text == f"{text} [E1]"
    assert result.financial_claims == ()


def test_provider_text_without_digits_is_accepted() -> None:
    client = FakeGenerationClient(json.dumps({"text": "Evidence supports the strategy."}))
    result = GroundedCompetitorSynthesizer(client).synthesize(
        "Question?", unified(qualitative())
    )
    assert result.answer_text == "Evidence supports the strategy. [E1]"


@pytest.mark.parametrize(
    "claims",
    (
        "not-a-list",
        ["not-an-object"],
        [{"evidence_id": "E1", "claim_type": "unknown", "role": "calculated_value", "value": "40.00"}],
        [{"evidence_id": "E1", "claim_type": "calculated_metric", "role": "calculated_value", "value": "40.00", "extra": True}],
        [{"evidence_id": "E1", "claim_type": "calculated_metric", "role": "calculated_value"}],
        [{"evidence_id": "E1", "claim_type": "calculated_metric", "role": "calculated_value", "value": 40.00}],
        [{"evidence_id": "E1", "claim_type": "comparison_entry", "role": "ranked_entry", "company_id": "asus", "rank": "1", "value": "50.00"}],
    ),
)
def test_strict_financial_claim_schema_rejects_invalid_values(claims: object) -> None:
    client = FakeGenerationClient(
        response("Supported claim [E1].", financial_claims=claims)  # type: ignore[arg-type]
    )
    with pytest.raises(GroundedResponseFormatError):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(calculation())
        )


def test_obsolete_combined_contract_is_rejected() -> None:
    client = FakeGenerationClient(
        json.dumps(
            {
                "answer_text": "Claim [E1].",
                "cited_evidence_ids": ["E1"],
                "insufficient": False,
            }
        )
    )
    with pytest.raises(GroundedResponseFormatError, match="schema"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(qualitative())
        )


def test_strict_json_response_contract() -> None:
    client = FakeGenerationClient("not json")
    with pytest.raises(GroundedResponseFormatError, match="JSON"):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(qualitative())
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"text": "Claim.", "unexpected": True},
        {"text": 1},
        {},
        {"claims": []},
    ),
)
def test_strict_json_rejects_schema_or_type_errors(payload: dict[str, object]) -> None:
    client = FakeGenerationClient(json.dumps(payload))
    with pytest.raises(GroundedResponseFormatError):
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(qualitative())
        )


def test_provider_exception_is_wrapped_without_payload_leakage() -> None:
    client = FailingGenerationClient("unused")
    with pytest.raises(GroundedGenerationError, match="failed safely") as caught:
        GroundedCompetitorSynthesizer(client).synthesize(
            "Question?", unified(qualitative())
        )
    assert "synthetic provider failure" not in str(caught.value)


def test_prompt_distinguishes_reported_and_calculated_evidence() -> None:
    prompt = build_grounded_synthesis_prompt(
        "Compare evidence.", unified(fact(), calculation())
    )
    assert "financial_fact" in prompt and "financial_calculation" in prompt
    assert "reported_source_value" in prompt
    assert "calculated_value" in prompt


def test_prompt_serialization_has_no_private_paths_or_provider_objects() -> None:
    prompt = build_grounded_synthesis_prompt("Question?", unified(qualitative(), fact()))
    assert "data/private" not in prompt
    assert "data/vector_store" not in prompt
    assert "C:\\" not in prompt
    assert "FakeGenerationClient" not in prompt
    assert "source_sha256" not in prompt


def test_result_is_immutable_and_provider_is_only_an_abstraction() -> None:
    result = GroundedCompetitorSynthesizer(
        FakeGenerationClient(response("Grounded claim [E1]."))
    ).synthesize("Question?", unified(qualitative()))
    with pytest.raises(FrozenInstanceError):
        result.answer_text = "changed"  # type: ignore[misc]
    assert result.generation_provider == "fake"


@pytest.mark.parametrize(
    "payload",
    (
        {"text": "Valid", "extra": True},
        {"text": ""},
        {"text": "   "},
        {"text": None},
        {"wrong": "Valid"},
    ),
)
def test_production_qualitative_schema_is_exact(payload: object) -> None:
    with pytest.raises(GroundedResponseFormatError):
        GroundedCompetitorSynthesizer(
            FakeGenerationClient(json.dumps(payload))
        ).synthesize("Question?", unified(qualitative()))


@pytest.mark.parametrize(
    "text",
    (
        "Claim [E1].",
        "Claim [E1, E2].",
        "Claim [E99].",
        "Claim [E2, E1].",
    ),
)
def test_provider_cannot_control_qualitative_citations(text: str) -> None:
    with pytest.raises(GroundedResponseFormatError, match="citation syntax"):
        GroundedCompetitorSynthesizer(
            FakeGenerationClient(json.dumps({"text": text}))
        ).synthesize("Question?", unified(qualitative(), qualitative("msi")))


@pytest.mark.parametrize(
    "payload",
    (
        {"claims": [], "extra": True},
        {"claims": "bad"},
        {"claims": {}},
        {},
        [],
    ),
)
def test_production_financial_root_contract_is_strict(payload: object) -> None:
    with pytest.raises(GroundedSynthesisError):
        GroundedCompetitorSynthesizer(
            FakeGenerationClient(json.dumps(payload))
        ).synthesize("Margin?", unified(calculation()))


@pytest.mark.parametrize("count", (1, 2, 3))
def test_scope_citation_order_is_stable_for_selected_evidence(count: int) -> None:
    items = (qualitative("asus"), qualitative("msi"), qualitative("gigabyte"))[:count]
    result = GroundedCompetitorSynthesizer(
        FakeGenerationClient(json.dumps({"text": "Scoped narrative."}))
    ).synthesize("Question?", unified(*items))
    expected = tuple(f"E{index}" for index in range(1, count + 1))
    assert result.cited_evidence_ids == expected
    assert result.answer_text.endswith("[" + ", ".join(expected) + "]")


@pytest.mark.parametrize("reverse", (False, True))
def test_financial_claim_output_preserves_validated_provider_order(reverse: bool) -> None:
    claims = _claims_for_markers(
        "[[E1:calculated_value:40.00]][[E2:calculated_value:40.00]]"
    )
    if reverse:
        claims.reverse()
    result = GroundedCompetitorSynthesizer(
        FakeGenerationClient(json.dumps({"claims": claims}))
    ).synthesize("Question?", unified(calculation("asus"), calculation("msi")))
    assert result.cited_evidence_ids == tuple(
        dict.fromkeys(claim["evidence_id"] for claim in claims)
    )


def test_result_exposes_scope_level_qualitative_provenance_state() -> None:
    result = GroundedCompetitorSynthesizer(
        FakeGenerationClient(json.dumps({"text": "Scoped narrative."}))
    ).synthesize("Question?", unified(qualitative()))
    assert result.qualitative_coverage is QualitativeCoverage.COMPLETE
    assert result.missing_qualitative_companies == ()


@pytest.mark.parametrize(
    ("items", "responses", "required_keys"),
    (
        ((qualitative(),), ({"text": "Narrative."},), (("text",),)),
        ((calculation(),), ({"claims": _claims_for_markers("[[E1:calculated_value:40.00]]")},), (("claims",),)),
        (
            (qualitative(), calculation()),
            ({"text": "Narrative."}, {"claims": _claims_for_markers("[[E2:calculated_value:40.00]]")}),
            (("text",), ("claims",)),
        ),
    ),
)
def test_split_provider_calls_are_routed_by_evidence_type(
    items: tuple[object, ...],
    responses: tuple[dict[str, object], ...],
    required_keys: tuple[tuple[str, ...], ...],
) -> None:
    client = SequenceStructuredGenerationClient(
        [json.dumps(response) for response in responses]
    )
    GroundedCompetitorSynthesizer(client).synthesize("Question?", unified(*items))
    assert tuple(tuple(schema["required"]) for schema in client.schemas) == required_keys


def comparison():
    engine = comparison_engine(
        *margin_facts("asus", 2025, "50"),
        *margin_facts("msi", 2025, "40"),
    )
    return engine.rank_companies("gross_margin", 2025, ("asus", "msi"))


def change():
    engine = comparison_engine(
        *margin_facts("asus", 2024, "30"),
        *margin_facts("asus", 2025, "40"),
    )
    return engine.compare_company_years("asus", "gross_margin", 2024, 2025)


def claim(evidence_id: str, claim_type: str, role: str, value: str, **extra):
    return {
        "evidence_id": evidence_id,
        "claim_type": claim_type,
        "role": role,
        "value": value,
        **extra,
    }


def test_financial_schema_is_a_strict_four_family_union() -> None:
    variants = FINANCIAL_RESPONSE_SCHEMA["properties"]["claims"]["items"]["oneOf"]
    assert len(variants) == 4
    assert all(item["additionalProperties"] is False for item in variants)
    assert all("pattern" not in json.dumps(item) for item in variants)
    assert all("uniqueItems" not in json.dumps(item) for item in variants)


def test_mixed_fact_and_comparison_use_one_financial_call_and_render() -> None:
    evidence = unified(fact(), comparison())
    claims = [
        claim("E1", "reported_fact", "reported_value", "100"),
        claim("E2", "comparison_entry", "ranked_entry", "50.00", company_id="asus", rank=1),
    ]
    client = FakeStructuredGenerationClient(json.dumps({"claims": claims}))
    result = GroundedCompetitorSynthesizer(client).synthesize("Compare.", evidence)
    rendered = render_competitor_answer(result, evidence)
    assert len(client.prompts) == 1
    assert result.cited_evidence_ids == ("E1", "E2")
    assert tuple(item.claim_type.value for item in result.financial_claims) == (
        "reported_fact", "comparison_entry"
    )
    assert len(rendered.citations) == 2


def test_mixed_calculated_and_comparison_use_one_financial_call() -> None:
    evidence = unified(calculation(), comparison())
    claims = [
        claim("E1", "calculated_metric", "calculated_value", "40.00"),
        claim("E2", "comparison_entry", "ranked_entry", "50.00", company_id="asus", rank=1),
    ]
    client = FakeStructuredGenerationClient(json.dumps({"claims": claims}))
    result = GroundedCompetitorSynthesizer(client).synthesize("Compare.", evidence)
    assert len(client.prompts) == 1
    assert [item.evidence_id for item in result.financial_claims] == ["E1", "E2"]


def test_all_financial_claim_families_coexist_in_provider_order() -> None:
    evidence = unified(fact(), calculation(), comparison(), change())
    claims = [
        claim("E1", "reported_fact", "reported_value", "100"),
        claim("E2", "calculated_metric", "calculated_value", "40.00"),
        claim("E3", "comparison_entry", "ranked_entry", "50.00", company_id="asus", rank=1),
        claim("E4", "financial_change_value", "percentage_point_change", "10.00"),
    ]
    client = FakeStructuredGenerationClient(json.dumps({"claims": claims}))
    result = GroundedCompetitorSynthesizer(client).synthesize("Analyze.", evidence)
    assert len(client.prompts) == 1
    assert tuple(item.claim_type.value for item in result.financial_claims) == (
        "reported_fact", "calculated_metric", "comparison_entry",
        "financial_change_value",
    )
    assert result.cited_evidence_ids == ("E1", "E2", "E3", "E4")
    assert len(render_competitor_answer(result, evidence).citations) == 4


@pytest.mark.parametrize(
    ("evidence", "bad_claim"),
    (
        (lambda: unified(fact()), claim("E1", "reported_fact", "reported_value", "100", company_id="asus", rank=1)),
        (lambda: unified(calculation()), claim("E1", "calculated_metric", "calculated_value", "40.00", company_id="asus", rank=1)),
        (lambda: unified(comparison()), claim("E1", "comparison_entry", "ranked_entry", "50.00", company_id="asus")),
        (lambda: unified(comparison()), claim("E1", "comparison_entry", "ranked_entry", "50.00", rank=1)),
        (lambda: unified(change()), claim("E1", "financial_change_value", "percentage_point_change", "10.00", company_id="asus", rank=1)),
    ),
)
def test_family_specific_extra_and_required_fields_remain_strict(evidence, bad_claim) -> None:
    with pytest.raises(GroundedResponseFormatError):
        GroundedCompetitorSynthesizer(
            FakeGenerationClient(json.dumps({"claims": [bad_claim]}))
        ).synthesize("Question?", evidence())


def test_partial_qualitative_and_complete_financial_render_with_real_synthesizer() -> None:
    plan = DeterministicQuestionRouter().plan(
        "Compare ASUS and MSI AI strategies and gross margins in 2025."
    )
    evidence = unified(qualitative("asus"), comparison())
    responses = [
        json.dumps({"text": "Available strategy evidence."}),
        json.dumps({"claims": [
            claim("E2", "comparison_entry", "ranked_entry", "50.00", company_id="asus", rank=1)
        ]}),
    ]
    client = SequenceStructuredGenerationClient(responses)
    result = GroundedCompetitorSynthesizer(client).synthesize(plan.question, evidence, plan)
    rendered = render_competitor_answer(result, evidence)
    assert result.status is GroundedSynthesisStatus.PARTIAL
    assert result.qualitative_coverage is QualitativeCoverage.PARTIAL
    assert result.missing_qualitative_companies == ("msi",)
    assert len(client.prompts) == 2
    assert len(rendered.citations) == 2
