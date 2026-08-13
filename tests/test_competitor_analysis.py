from __future__ import annotations

from io import StringIO

import pytest

from enterprise_rag.cli import main
from enterprise_rag.competitor_analysis import (
    CitationValidationError,
    CompetitorAnalysisPipeline,
    build_citation_ready_evidence,
    build_guarded_prompt,
    render_citation,
    validate_and_build_citations,
)
from enterprise_rag.competitor_retrieval import (
    BalancedRetrievalResponse,
    CompanyEvidenceSet,
    CompetitorRetrievalResult,
)
from enterprise_rag.models import DocumentChunk, EmbeddedChunk, RetrievalResult


class FakeRetriever:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def retrieve(self, question, companies, top_k):
        self.calls.append((question, tuple(companies), top_k))
        return self.response


class FakeGenerator:
    provider = "ollama"
    model = "qwen3:8b"

    def __init__(self, answer="Supported claim [E1]."):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def selected(company="gigabyte", rank=1, page=87, text="AI infrastructure expanded to support enterprise demand."):
    metadata = {
        "company_id": company,
        "company_name": company.title(),
        "ticker": "2376",
        "fiscal_year": 2025,
        "document_type": "annual_report",
        "source_document_id": f"doc-{company}",
        "title": f"{company.title()} 2025 Annual Report",
        "page_number": page,
        "private_path": r"C:\private\secret.pdf",
    }
    chunk = DocumentChunk(
        text, f"{company}/2025/report.pdf", "report.pdf", ".pdf",
        f"doc-{company}:page-{page:04d}", rank - 1,
        f"doc-{company}:page-{page:04d}:chunk-{rank - 1:06d}", metadata,
    )
    retrieval = RetrievalResult(0.8, EmbeddedChunk(chunk, (0.1, 0.2), "nomic-embed-text"), metadata)
    return CompetitorRetrievalResult(company, company.title(), rank, rank + 1, retrieval, 0.9, ("narrative-content",))


def response(groups):
    return BalancedRetrievalResponse(tuple(groups))


def group(company, items=(), requested=2):
    return CompanyEvidenceSet(company, company.title(), requested, len(items), tuple(items), ())


def test_evidence_ids_and_provenance_are_deterministic_and_safe():
    value = response((group("gigabyte", (selected(),)), group("asus", (selected("asus", page=42),))))
    first = build_citation_ready_evidence(value)
    second = build_citation_ready_evidence(value)
    assert first == second
    assert [item.evidence_id for item in first] == ["E1", "E2"]
    assert first[0].page_number == 87 and first[0].chunk_id.endswith("000000")
    assert first[0].ticker == "2376" and first[0].company_name == "Gigabyte"
    assert "C:\\private" not in repr(first)


def test_prompt_contains_only_selected_evidence_and_insufficiency():
    evidence = build_citation_ready_evidence(response((group("gigabyte", (selected(),)),)))
    prompt = build_guarded_prompt("Compare", evidence, ("MSI",))
    assert "[E1]" in prompt and evidence[0].text in prompt
    assert "MSI" in prompt and "Use only the supplied evidence" in prompt
    assert "C:\\private" not in prompt


@pytest.mark.parametrize("answer", ["Claim [E1].", "Claim [E1] again [E1]."])
def test_valid_and_duplicate_references_create_one_metadata_citation(answer):
    evidence = build_citation_ready_evidence(response((group("gigabyte", (selected(),)),)))
    citations = validate_and_build_citations(answer, evidence)
    assert [item.evidence_id for item in citations] == ["E1"]
    assert citations[0].page_number == 87


def test_citation_order_follows_first_reference_order():
    evidence = build_citation_ready_evidence(response((group("gigabyte", (selected(), selected(rank=2, page=88))),)))
    assert [c.evidence_id for c in validate_and_build_citations("[E2] then [E1]", evidence)] == ["E2", "E1"]


def test_grouped_citation_references_are_all_validated_and_rendered():
    evidence = build_citation_ready_evidence(response((group("gigabyte", (selected(), selected(rank=2, page=88))),)))
    citations = validate_and_build_citations("Combined support [E1, E2].", evidence)
    assert [item.evidence_id for item in citations] == ["E1", "E2"]


@pytest.mark.parametrize("answer", ["Invented [E99].", "No citation at all."])
def test_invented_or_missing_citations_are_rejected(answer):
    evidence = build_citation_ready_evidence(response((group("gigabyte", (selected(),)),)))
    with pytest.raises(CitationValidationError):
        validate_and_build_citations(answer, evidence)


def test_single_company_sufficient_generation_and_contract():
    retriever = FakeRetriever(response((group("gigabyte", (selected(),)),)))
    generator = FakeGenerator()
    answer = CompetitorAnalysisPipeline(retriever, generator).answer("AI?", ("gigabyte",), 2)
    assert answer.grounding_status == "grounded"
    assert answer.answered_companies == ("Gigabyte",) and not answer.insufficient_companies
    assert answer.generation_model == "qwen3:8b" and answer.citations[0].evidence_id == "E1"


def test_partial_evidence_calls_generator_and_marks_insufficient_company():
    value = response((group("gigabyte", (selected(),)), group("msi", ())))
    generator = FakeGenerator("Gigabyte evidence is available [E1]; MSI evidence is insufficient.")
    answer = CompetitorAnalysisPipeline(FakeRetriever(value), generator).answer("Compare", ("gigabyte", "msi"))
    assert answer.grounding_status == "partial"
    assert answer.answered_companies == ("Gigabyte",)
    assert answer.insufficient_companies == ("Msi",)
    assert "INSUFFICIENT EVIDENCE:\nMsi" in generator.prompts[0]


def test_all_insufficient_never_calls_generator():
    generator = FakeGenerator()
    value = response((group("asus", ()), group("msi", ())))
    answer = CompetitorAnalysisPipeline(FakeRetriever(value), generator).answer("Compare", ("asus", "msi"))
    assert answer.grounding_status == "insufficient" and answer.citations == ()
    assert answer.insufficient_companies == ("Asus", "Msi")
    assert generator.prompts == []


def test_three_companies_all_sufficient():
    value = response(tuple(group(c, (selected(c),), requested=1) for c in ("gigabyte", "asus", "msi")))
    generator = FakeGenerator("All are supported [E1] [E2] [E3].")
    answer = CompetitorAnalysisPipeline(FakeRetriever(value), generator).answer("Compare", ("gigabyte", "asus", "msi"), 1)
    assert answer.grounding_status == "grounded" and len(answer.citations) == 3


def test_empty_generation_and_generator_error_are_safe():
    value = response((group("gigabyte", (selected(),)),))
    from enterprise_rag.competitor_analysis import CompetitorGenerationError
    for failure in (" ", RuntimeError("private evidence text")):
        with pytest.raises(CompetitorGenerationError, match="failed safely") as error:
            CompetitorAnalysisPipeline(FakeRetriever(value), FakeGenerator(failure)).answer("AI?", ("gigabyte",))
        assert "private evidence" not in str(error.value)


def test_citation_renderer_uses_metadata_not_paths():
    evidence = build_citation_ready_evidence(response((group("gigabyte", (selected(),)),)))
    citation = validate_and_build_citations("Claim [E1].", evidence)[0]
    rendered = render_citation(citation)
    assert rendered == "[E1] Gigabyte — 2025 Gigabyte 2025 Annual Report — PDF p. 87"
    assert "private" not in rendered.casefold()


def test_competitor_ask_cli_sections_and_no_chunk_dump():
    value = response((group("gigabyte", (selected(),)), group("msi", ())))
    pipeline = CompetitorAnalysisPipeline(FakeRetriever(value), FakeGenerator("Supported [E1]."))
    output = StringIO()
    assert main([
        "competitor-ask", "--companies", "gigabyte", "msi",
        "--top-k-per-company", "2", "Compare",
    ], competitor_pipeline=pipeline, output=output) == 0
    rendered = output.getvalue()
    assert "ANSWER" in rendered and "EVIDENCE STATUS" in rendered and "CITATIONS" in rendered
    assert "Gigabyte: sufficient" in rendered and "Msi: insufficient" in rendered
    assert selected().retrieval_result.embedded_chunk.chunk.content not in rendered


def test_competitor_retrieve_parser_remains_separate():
    from enterprise_rag.cli import build_parser
    args = build_parser().parse_args(["competitor-retrieve", "--companies", "asus", "question"])
    assert args.command == "competitor-retrieve"
