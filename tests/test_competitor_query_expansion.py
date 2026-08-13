from __future__ import annotations

import pytest

from enterprise_rag.competitor_query_expansion import (
    MAX_QUERIES_PER_COMPANY,
    CompetitorQueryExpander,
)
from enterprise_rag.competitor_retrieval import (
    CANDIDATES_PER_QUERY,
    BalancedCompetitorRetriever,
)
from enterprise_rag.models import DocumentChunk, EmbeddedChunk, RetrievalResult


@pytest.mark.parametrize(
    ("question", "company", "expected_fragment"),
    [
        ("What does MSI say about AI servers?", "msi", "AI 伺服器"),
        ("What does ASUS say about AI PCs?", "asus", "人工智慧個人電腦"),
        ("Compare enterprise data center strategies", "asus", "資料中心"),
        ("What products or business areas are emphasized?", "msi", "業務內容"),
        ("What is the major growth driver?", "asus", "成長動能"),
        ("Compare AI strategies", "msi", "人工智慧 AI 策略"),
        ("What does Gigabyte say about AI infrastructure?", "gigabyte", "AI server infrastructure"),
    ],
)
def test_controlled_intents_return_original_then_one_source_aware_variant(
    question, company, expected_fragment
):
    variants = CompetitorQueryExpander().expand(question, company)
    assert variants[0] == question
    assert expected_fragment in variants[1]
    assert len(variants) <= MAX_QUERIES_PER_COMPANY == 2
    assert variants == CompetitorQueryExpander().expand(question, company)
    assert len({item.casefold() for item in variants}) == len(variants)


def test_unknown_general_query_returns_original_only():
    assert CompetitorQueryExpander().expand("Who is the chairperson?", "asus") == (
        "Who is the chairperson?",
    )


def test_controlled_terms_do_not_encode_diagnosed_company_answers():
    variants = " ".join(
        CompetitorQueryExpander().expand("What does MSI say about AI servers?", "msi")
    ).casefold()
    for leaked_answer in ("focuses on si", "invests in si", "network security", "hpc"):
        assert leaked_answer not in variants


def result(company: str, index: int, text: str, *, score: float = 0.8) -> RetrievalResult:
    metadata = {
        "company_id": company,
        "company_name": company.title(),
        "fiscal_year": 2025,
        "page_number": index + 1,
        "source_document_id": f"doc-{company}",
        "title": f"{company.title()} Annual Report",
    }
    chunk = DocumentChunk(
        text,
        f"{company}/report.pdf",
        "report.pdf",
        ".pdf",
        f"doc-{company}:page-{index + 1:04d}",
        index,
        f"doc-{company}:page-{index + 1:04d}:chunk-{index:06d}",
        metadata,
    )
    return RetrievalResult(score, EmbeddedChunk(chunk, (0.1, 0.2), "fake"), metadata)


NARRATIVE_A = "Enterprise customers are adopting AI infrastructure to support growing data center demand."
NARRATIVE_B = "The product portfolio is expanding through sustained research and customer-focused development."
TABLE = "100% 0 100% 0\n33,315,472 100% 0 0%\n16,737,013 100% 10 0%"


class MappingRetriever:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        return tuple(self.values.get(query, ()))[:top_k]


class FixedExpander:
    def __init__(self, variants):
        self.variants = variants

    def expand(self, question, company):
        return self.variants


def test_one_query_preserves_legacy_candidate_order():
    items = (result("asus", 0, NARRATIVE_A), result("asus", 1, NARRATIVE_B))
    backend = MappingRetriever({"general": items})
    service = BalancedCompetitorRetriever(
        {"asus": backend}, query_expander=FixedExpander(("general",))
    )
    selected = service.retrieve("general", ("asus",), 2).company_evidence[0].evidence
    assert [item.retrieval_result.embedded_chunk.chunk.chunk_id for item in selected] == [
        item.embedded_chunk.chunk.chunk_id for item in items
    ]
    assert backend.calls == [("general", CANDIDATES_PER_QUERY)]


def test_two_queries_merge_by_rank_original_first_and_retrieve_both_budgets():
    original = (result("asus", 0, NARRATIVE_A), result("asus", 2, NARRATIVE_B))
    expanded = (result("asus", 1, NARRATIVE_B),)
    backend = MappingRetriever({"original": original, "expanded": expanded})
    service = BalancedCompetitorRetriever(
        {"asus": backend}, query_expander=FixedExpander(("original", "expanded"))
    )
    group = service.retrieve("question", ("asus",), 3).company_evidence[0]
    assert [item.retrieval_result.embedded_chunk.chunk.chunk_index for item in group.evidence] == [0, 1, 2]
    assert backend.calls == [
        ("original", CANDIDATES_PER_QUERY),
        ("expanded", CANDIDATES_PER_QUERY),
    ]


def test_duplicate_across_variants_appears_once_and_provenance_is_preserved():
    shared = result("msi", 0, NARRATIVE_A)
    backend = MappingRetriever({"original": (shared,), "expanded": (shared,)})
    service = BalancedCompetitorRetriever(
        {"msi": backend}, query_expander=FixedExpander(("original", "expanded"))
    )
    group = service.retrieve("question", ("msi",), 2).company_evidence[0]
    assert group.candidate_count == 1 and group.returned_count == 1
    assert group.insufficient_evidence
    assert group.evidence[0].retrieval_result.metadata["source_document_id"] == "doc-msi"


def test_expansion_only_relevant_candidate_can_enter_after_quality_gate():
    original = (result("msi", 0, TABLE),)
    expanded = (result("msi", 1, NARRATIVE_A),)
    backend = MappingRetriever({"original": original, "expanded": expanded})
    service = BalancedCompetitorRetriever(
        {"msi": backend}, query_expander=FixedExpander(("original", "expanded"))
    )
    group = service.retrieve("question", ("msi",), 1).company_evidence[0]
    assert group.evidence[0].retrieval_result.embedded_chunk.chunk.chunk_index == 1
    assert any("numeric" in " ".join(reasons) for _, reasons in group.rejected_reasons)


def test_company_order_and_company_local_scores_remain_independent():
    asus = MappingRetriever({"q": (result("asus", 0, NARRATIVE_A, score=0.2),)})
    msi = MappingRetriever({"q": (result("msi", 0, NARRATIVE_B, score=0.9),)})
    service = BalancedCompetitorRetriever(
        {"asus": asus, "msi": msi}, query_expander=FixedExpander(("q",))
    )
    response = service.retrieve("question", ("asus", "msi"), 1)
    assert [item.company_id for item in response] == ["asus", "msi"]
    assert [item.retrieval_result.score for item in response] == [0.2, 0.9]
