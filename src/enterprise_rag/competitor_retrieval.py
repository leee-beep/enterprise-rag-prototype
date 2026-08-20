"""Quality-aware balanced retrieval across per-company FAISS indexes."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.config import Settings
from enterprise_rag.competitor_query_expansion import CompetitorQueryExpander
from enterprise_rag.competitor_semantic_reranking import (
    LightweightSemanticReranker,
    assess_topic_relevance_gate,
)
from enterprise_rag.indexing import IndexingError, validate_index_compatibility
from enterprise_rag.models import RetrievalResult
from enterprise_rag.retrieval import QueryEmbeddingClient, RetrievalError, Retriever
from enterprise_rag.vector_store import FaissVectorStore, VectorStoreError


COMPANY_NAMES = {"gigabyte": "Gigabyte", "asus": "ASUS", "msi": "MSI"}
CANDIDATES_PER_QUERY = 12
MIN_QUALITY_SCORE = 0.45
ADJACENT_OVERLAP_THRESHOLD = 0.45
_SENTENCE_END = re.compile(r"(?:[a-z][.!?]|[。？！])(?:\s|$)")
_TOKEN = re.compile(r"[^\W_]+(?:[.,:/-][^\W_]+)*|[%％]", re.UNICODE)


@dataclass(frozen=True)
class EvidenceQualityAssessment:
    is_usable: bool
    quality_score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CompetitorRetrievalResult:
    company_id: str
    company_name: str
    company_rank: int
    original_candidate_rank: int
    retrieval_result: RetrievalResult
    quality_score: float
    quality_reasons: tuple[str, ...]
    semantic_relevance_score: float = 0.0
    semantic_relevance_reasons: tuple[str, ...] = ()
    topic_relevance_passed: bool = True

    @property
    def text(self) -> str:
        return self.retrieval_result.embedded_chunk.chunk.content


@dataclass(frozen=True)
class CompanyEvidenceSet:
    company_id: str
    company_name: str
    requested_count: int
    candidate_count: int
    evidence: tuple[CompetitorRetrievalResult, ...]
    rejected_reasons: tuple[tuple[int, tuple[str, ...]], ...]

    @property
    def returned_count(self) -> int:
        return len(self.evidence)

    @property
    def insufficient_evidence(self) -> bool:
        return self.returned_count < self.requested_count


@dataclass(frozen=True)
class BalancedRetrievalResponse(Sequence[CompetitorRetrievalResult]):
    """Company evidence summaries with a flat sequence-compatible result view."""

    company_evidence: tuple[CompanyEvidenceSet, ...]

    @property
    def results(self) -> tuple[CompetitorRetrievalResult, ...]:
        return tuple(item for group in self.company_evidence for item in group.evidence)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]

    def __iter__(self) -> Iterator[CompetitorRetrievalResult]:
        return iter(self.results)


def assess_evidence_quality(text: str) -> EvidenceQualityAssessment:
    """Score narrative usability using transparent language-neutral signals."""
    normalized = text.strip()
    if not normalized:
        return EvidenceQualityAssessment(False, 0.0, ("empty",))
    visible = [char for char in normalized if not char.isspace()]
    language_count = sum(char.isalpha() for char in visible)
    digit_count = sum(char.isdigit() for char in visible)
    language_ratio = language_count / max(1, len(visible))
    numeric_ratio = digit_count / max(1, len(visible))
    tokens = _TOKEN.findall(normalized)
    numeric_tokens = sum(
        any(char.isdigit() for char in token) and not any(char.isalpha() for char in token)
        for token in tokens
    )
    numeric_token_ratio = numeric_tokens / max(1, len(tokens))
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    numeric_lines = sum(
        sum(char.isdigit() for char in line)
        > sum(char.isalpha() for char in line)
        for line in lines
    )
    table_line_ratio = numeric_lines / max(1, len(lines))
    short_line_ratio = sum(len(line) < 45 for line in lines) / max(1, len(lines))
    value_row_ratio = sum(bool(re.search(r"(?:\d|[%％])\s*$", line)) for line in lines) / max(1, len(lines))
    sentence_count = len(_SENTENCE_END.findall(normalized))

    score = 1.0
    reasons: list[str] = []
    if language_ratio < 0.35:
        score -= 0.50
        reasons.append("low-language-density")
    elif language_ratio < 0.50:
        score -= 0.15
        reasons.append("moderate-language-density")
    if numeric_ratio > 0.35:
        score -= 0.45
        reasons.append("high-numeric-density")
    elif numeric_ratio > 0.22:
        score -= 0.20
        reasons.append("moderate-numeric-density")
    if numeric_token_ratio > 0.48:
        score -= 0.35
        reasons.append("numeric-token-table-pattern")
    if len(lines) >= 4 and table_line_ratio > 0.50:
        score -= 0.35
        reasons.append("number-dominated-lines")
    if len(lines) >= 6 and short_line_ratio > 0.80:
        score -= 0.15
        reasons.append("short-line-table-pattern")
    if len(lines) >= 3 and value_row_ratio > 0.70:
        score -= 0.65
        reasons.append("repeated-value-rows")
    if sentence_count == 0:
        score -= 0.25
        reasons.append("no-narrative-sentence")
        if len(lines) >= 3:
            score -= 0.40
            reasons.append("multi-line-fragment-pattern")
    elif sentence_count >= 2:
        score += 0.10
        reasons.append("narrative-sentences")
    if len(normalized) < 40:
        score -= 0.20
        reasons.append("very-short")
    score = min(1.0, max(0.0, score))
    return EvidenceQualityAssessment(
        score >= MIN_QUALITY_SCORE,
        score,
        tuple(reasons) if reasons else ("narrative-content",),
    )


def _text_overlap(left: str, right: str, *, width: int = 5) -> float:
    def shingles(value: str) -> set[str]:
        compact = "".join(value.casefold().split())
        if len(compact) <= width:
            return {compact} if compact else set()
        return {compact[index:index + width] for index in range(len(compact) - width + 1)}
    a, b = shingles(left), shingles(right)
    return len(a & b) / max(1, min(len(a), len(b)))


def _is_duplicate(candidate: RetrievalResult, selected: Sequence[CompetitorRetrievalResult]) -> bool:
    chunk = candidate.embedded_chunk.chunk
    for existing in selected:
        other = existing.retrieval_result.embedded_chunk.chunk
        same_document = candidate.metadata.get("source_document_id") == existing.retrieval_result.metadata.get("source_document_id")
        same_page = candidate.metadata.get("page_number") == existing.retrieval_result.metadata.get("page_number")
        if not (same_document and same_page):
            continue
        if chunk.content.strip() == other.content.strip():
            return True
        adjacent = abs(chunk.chunk_index - other.chunk_index) <= 1
        if adjacent and _text_overlap(chunk.content, other.content) >= ADJACENT_OVERLAP_THRESHOLD:
            return True
    return False


class BalancedCompetitorRetriever:
    """Select usable evidence independently from each requested company."""

    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        *,
        query_expander: CompetitorQueryExpander | None = None,
        semantic_reranker: LightweightSemanticReranker | None = None,
    ) -> None:
        self._retrievers = {key.casefold(): value for key, value in retrievers.items()}
        self._query_expander = query_expander or CompetitorQueryExpander()
        self._semantic_reranker = semantic_reranker or LightweightSemanticReranker()

    @classmethod
    def from_index_root(cls, index_root: str | Path, settings: Settings, embedding_client: QueryEmbeddingClient) -> "BalancedCompetitorRetriever":
        root = Path(index_root)
        retrievers: dict[str, Retriever] = {}
        for company in COMPANY_NAMES:
            directory = root / company
            if not directory.is_dir():
                continue
            try:
                validate_index_compatibility(directory, settings)
                retrievers[company] = Retriever(embedding_client, FaissVectorStore.load(directory))
            except (IndexingError, VectorStoreError) as exc:
                raise RetrievalError(f"Company index is invalid or incompatible for {company}.") from exc
        return cls(retrievers)

    def retrieve(self, question: str, company_ids: Sequence[str], top_k_per_company: int) -> BalancedRetrievalResponse:
        normalized = self._validate_request(question, company_ids, top_k_per_company)
        groups: list[CompanyEvidenceSet] = []
        for company in normalized:
            queries = self._query_expander.expand(question.strip(), company)
            candidate_sets = tuple(
                self._retrievers[company].retrieve(query, CANDIDATES_PER_QUERY)
                for query in queries
            )
            candidates = _merge_candidate_sets(candidate_sets)
            usable: list[CompetitorRetrievalResult] = []
            rejected: list[tuple[int, tuple[str, ...]]] = []
            for candidate_rank, result in enumerate(candidates, start=1):
                assessment = assess_evidence_quality(result.embedded_chunk.chunk.content)
                if not assessment.is_usable:
                    rejected.append((candidate_rank, assessment.reasons))
                    continue
                if _is_duplicate(result, usable):
                    rejected.append((candidate_rank, ("duplicate-or-overlapping",)))
                    continue
                usable.append(CompetitorRetrievalResult(
                    company, COMPANY_NAMES[company], 0,
                    candidate_rank, result, assessment.quality_score, assessment.reasons,
                ))
            reranked = self._semantic_reranker.rerank(question.strip(), tuple(usable))
            selected_items: list[CompetitorRetrievalResult] = []
            for item, relevance in reranked:
                gate = assess_topic_relevance_gate(relevance)
                if not gate.passed:
                    rejected.append((item.original_candidate_rank, (gate.reason,)))
                    continue
                selected_items.append(CompetitorRetrievalResult(
                    item.company_id, item.company_name, 0,
                    item.original_candidate_rank, item.retrieval_result,
                    item.quality_score, item.quality_reasons,
                    relevance.relevance_score,
                    relevance.reasons + (gate.reason,),
                    gate.passed,
                ))
                if len(selected_items) == top_k_per_company:
                    break
            selected = tuple(
                CompetitorRetrievalResult(
                    item.company_id, item.company_name, final_rank,
                    item.original_candidate_rank, item.retrieval_result,
                    item.quality_score, item.quality_reasons,
                    item.semantic_relevance_score, item.semantic_relevance_reasons,
                    item.topic_relevance_passed,
                )
                for final_rank, item in enumerate(selected_items, start=1)
            )
            if not candidates:
                rejected.append((0, ("no-candidates",)))
            groups.append(CompanyEvidenceSet(
                company, COMPANY_NAMES[company], top_k_per_company, len(candidates),
                selected, tuple(rejected),
            ))
        return BalancedRetrievalResponse(tuple(groups))

    def _validate_request(self, question: str, company_ids: Sequence[str], top_k: int) -> list[str]:
        if not isinstance(question, str) or not question.strip():
            raise RetrievalError("question must be a non-empty string.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise RetrievalError("top_k_per_company must be a positive integer.")
        if not company_ids:
            raise RetrievalError("At least one company must be selected.")
        normalized = [item.strip().casefold() if isinstance(item, str) else "" for item in company_ids]
        if len(set(normalized)) != len(normalized):
            raise RetrievalError("Duplicate company selection is not allowed.")
        unknown = [item for item in normalized if item not in COMPANY_NAMES]
        if unknown:
            raise RetrievalError(f"Unknown competitor company: {unknown[0]!r}.")
        missing = [item for item in normalized if item not in self._retrievers]
        if missing:
            raise RetrievalError(f"Company index is unavailable for {missing[0]}.")
        return normalized


def _merge_candidate_sets(
    candidate_sets: Sequence[Sequence[RetrievalResult]],
) -> tuple[RetrievalResult, ...]:
    """Merge company-local results by rank, with original-query priority.

    Each query has a fixed ``CANDIDATES_PER_QUERY`` budget. Rank one from every
    query precedes rank two, while the original query wins same-rank ties.
    Duplicate chunk IDs are retained only at their earliest merged position.
    Scores are never combined or compared across queries or companies.
    """
    merged: list[RetrievalResult] = []
    seen: set[str] = set()
    maximum = max((len(items) for items in candidate_sets), default=0)
    for rank in range(maximum):
        for items in candidate_sets:
            if rank >= len(items):
                continue
            candidate = items[rank]
            chunk_id = candidate.embedded_chunk.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(candidate)
    return tuple(merged)
