from __future__ import annotations

import pytest

from enterprise_rag.competitor_retrieval import BalancedCompetitorRetriever
from enterprise_rag.competitor_semantic_reranking import (
    LightweightSemanticReranker,
    assess_semantic_relevance,
)
from enterprise_rag.models import DocumentChunk, EmbeddedChunk, RetrievalResult


@pytest.mark.parametrize(
    ("question", "direct", "generic"),
    [
        ("What does MSI say about AI servers?", "公司開發 AI 伺服器並服務資料中心客戶。", "公司開發地端 AI 應用軟體。"),
        ("What does ASUS say about AI PCs?", "The company develops AI PC and notebook products.", "The company invests in new technology."),
        ("Compare enterprise server strategies", "資料中心採用伺服器基礎設施。", "Enterprise risk controls were strengthened."),
        ("What products or business areas are emphasized?", "主要產品與業務範圍涵蓋電腦產品線。", "Corporate governance remained stable."),
        ("What is the major growth driver?", "市場需求是業務成長動能。", "The organization hired more people."),
        ("Compare AI strategies", "人工智慧策略涵蓋產品與基礎設施應用。", "The company published an annual plan."),
    ],
)
def test_direct_controlled_concepts_outscore_generic_narrative(question, direct, generic):
    assert assess_semantic_relevance(question, direct).relevance_score > assess_semantic_relevance(question, generic).relevance_score


def test_english_question_matches_zh_tw_and_zh_tw_text_can_match_english_terms():
    english_question = assess_semantic_relevance("What does MSI say about AI servers?", "AI伺服器用於資料中心基礎設施。")
    assert {"ai_server", "data_center", "infrastructure"} <= set(english_question.matched_concepts)
    chinese_text = assess_semantic_relevance("What does ASUS say about AI PCs?", "AI PC notebook products support mobile work.")
    assert {"ai_pc", "notebook"} <= set(chinese_text.matched_concepts)


def test_keyword_repetition_does_not_increase_concept_score():
    once = assess_semantic_relevance("What does MSI say about AI servers?", "AI server supports a data center.")
    stuffed = assess_semantic_relevance("What does MSI say about AI servers?", "AI server AI server AI server supports a data center.")
    assert once.relevance_score == stuffed.relevance_score
    assert once.matched_concepts == stuffed.matched_concepts


def test_unknown_intent_has_stable_zero_score_fallback():
    result = assess_semantic_relevance("Who chairs the board?", "Any excellent narrative text.")
    assert result.relevance_score == 0.0
    assert result.intent is None and result.reasons == ("unrecognized-intent",)


class Candidate:
    def __init__(self, text, rank, quality):
        self._text = text
        self.original_candidate_rank = rank
        self.quality_score = quality

    @property
    def text(self):
        return self._text


def test_semantic_score_precedes_quality_then_original_rank():
    unrelated = Candidate("A coherent unrelated governance sentence.", 1, 1.0)
    direct = Candidate("AI 伺服器服務資料中心。", 9, 0.5)
    ranked = LightweightSemanticReranker().rerank("What does MSI say about AI servers?", (unrelated, direct))
    assert ranked[0][0] is direct
    assert ranked[0][1].relevance_score > ranked[1][1].relevance_score
    assert direct.quality_score == 0.5


def test_unknown_intent_preserves_input_order_even_with_quality_difference():
    first = Candidate("first", 1, 0.2)
    second = Candidate("second", 2, 1.0)
    assert [item for item, _ in LightweightSemanticReranker().rerank("Who is chair?", (first, second))] == [first, second]


NARRATIVE = "The company develops technology for customers and continues research investment."
DIRECT = "The company develops AI server products for data center infrastructure customers."
TABLE = "100% 0 100% 0\n33,315,472 100% 0 0%\n16,737,013 100% 10 0%"


def retrieval(index, text, *, company="msi", page=None, score=0.8):
    page = page or index + 1
    metadata = {
        "company_id": company,
        "company_name": company.title(),
        "fiscal_year": 2025,
        "page_number": page,
        "source_document_id": f"doc-{company}",
        "title": f"{company.title()} Annual Report",
    }
    chunk = DocumentChunk(
        text, f"{company}/report.pdf", "report.pdf", ".pdf",
        f"doc-{company}:page-{page:04d}", index,
        f"doc-{company}:page-{page:04d}:chunk-{index:06d}", metadata,
    )
    return RetrievalResult(score, EmbeddedChunk(chunk, (0.1, 0.2), "fake-model"), metadata)


class MappingRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    def retrieve(self, query, top_k):
        return tuple(self.mapping.get(query, ()))[:top_k]


class TwoQueries:
    def expand(self, question, company):
        return ("original", "expanded")


def service(original, expanded, *, company="msi"):
    return BalancedCompetitorRetriever(
        {company: MappingRetriever({"original": original, "expanded": expanded})},
        query_expander=TwoQueries(),
    )


def test_expansion_only_direct_candidate_moves_above_earlier_weak_candidate():
    weak = retrieval(0, NARRATIVE)
    direct = retrieval(1, DIRECT)
    group = service((weak,), (direct,)).retrieve("What does MSI say about AI servers?", ("msi",), 2).company_evidence[0]
    assert group.evidence[0].retrieval_result.embedded_chunk.chunk.chunk_id == direct.embedded_chunk.chunk.chunk_id
    assert group.evidence[0].semantic_relevance_score > group.evidence[1].semantic_relevance_score
    assert group.evidence[0].original_candidate_rank == 2


def test_rejected_table_cannot_reenter_through_semantic_overlap():
    table = retrieval(0, "AI server data center\n" + TABLE)
    direct = retrieval(1, DIRECT)
    group = service((table,), (direct,)).retrieve("What does MSI say about AI servers?", ("msi",), 1).company_evidence[0]
    assert group.evidence[0].retrieval_result.embedded_chunk.chunk.chunk_id == direct.embedded_chunk.chunk.chunk_id
    assert group.rejected_reasons


def test_duplicate_is_removed_before_final_reranking():
    direct = retrieval(0, DIRECT)
    group = service((direct,), (direct,)).retrieve("What does MSI say about AI servers?", ("msi",), 2).company_evidence[0]
    assert group.candidate_count == 1 and group.returned_count == 1


def test_top_k_provenance_scores_and_embedding_metadata_are_preserved():
    direct = retrieval(4, DIRECT, page=71, score=0.42)
    group = service((retrieval(0, NARRATIVE),), (direct,)).retrieve("What does MSI say about AI servers?", ("msi",), 1).company_evidence[0]
    item = group.evidence[0]
    assert group.returned_count == 1
    assert item.retrieval_result.score == 0.42
    assert item.retrieval_result.metadata["page_number"] == 71
    assert item.retrieval_result.metadata["embedding_model"] == "fake-model"
    assert item.retrieval_result.embedded_chunk.chunk.chunk_id.endswith("000004")


def test_zero_usable_candidates_remain_insufficient():
    group = service((retrieval(0, TABLE),), ()).retrieve("What products are emphasized?", ("msi",), 2).company_evidence[0]
    assert group.returned_count == 0 and group.insufficient_evidence
