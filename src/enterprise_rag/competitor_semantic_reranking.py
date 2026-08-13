"""Explainable terminology-aware reranking of usable competitor evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, TypeVar

from enterprise_rag.competitor_query_expansion import (
    CONTROLLED_CONCEPTS,
    detect_competitor_intent,
)


@dataclass(frozen=True)
class SemanticRelevanceAssessment:
    relevance_score: float
    intent: str | None
    matched_concepts: tuple[str, ...]
    direct_phrase_match: bool
    reasons: tuple[str, ...]


class SemanticCandidate(Protocol):
    original_candidate_rank: int
    quality_score: float

    @property
    def text(self) -> str: ...


T = TypeVar("T", bound=SemanticCandidate)


def assess_semantic_relevance(question: str, text: str) -> SemanticRelevanceAssessment:
    """Score bounded concept coverage without model inference or term frequency."""
    intent = detect_competitor_intent(question)
    if intent is None:
        return SemanticRelevanceAssessment(0.0, None, (), False, ("unrecognized-intent",))
    normalized = _normalize(text)
    concepts = CONTROLLED_CONCEPTS[intent]
    matched = tuple(
        name for name, terms in concepts if any(_contains(normalized, term) for term in terms)
    )
    coverage = len(matched) / len(concepts)
    primary_name, primary_terms = concepts[0]
    direct = primary_name in matched and any(
        _contains(normalized, term) and (" " in term or any(ord(ch) > 127 for ch in term))
        for term in primary_terms
    )
    # Concept presence is binary, preventing repetition from increasing score.
    # Direct intent phrases receive a modest fixed boost. Structural quality is
    # deliberately excluded and used only as a later ranking tie-breaker.
    score = min(1.0, 0.8 * coverage + (0.2 if direct else 0.0))
    reasons = tuple([f"concept:{name}" for name in matched] + (["direct-phrase"] if direct else []))
    return SemanticRelevanceAssessment(
        round(score, 6), intent, matched, direct, reasons or ("no-controlled-concept-match",)
    )


class LightweightSemanticReranker:
    """Rank company-local usable candidates using transparent CPU-only signals."""

    def rerank(self, question: str, candidates: tuple[T, ...]) -> tuple[tuple[T, SemanticRelevanceAssessment], ...]:
        assessed = tuple((item, assess_semantic_relevance(question, item.text)) for item in candidates)
        if not assessed or assessed[0][1].intent is None:
            return assessed
        return tuple(sorted(
            assessed,
            key=lambda pair: (
                -pair[1].relevance_score,
                -pair[0].quality_score,
                pair[0].original_candidate_rank,
            ),
        ))


def _normalize(value: str) -> str:
    return " ".join(value.casefold().replace("－", "-").split())


def _contains(text: str, term: str) -> bool:
    normalized = _normalize(term)
    if normalized.isascii() and normalized.replace("-", "").replace(" ", "").isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", text) is not None
    return normalized in text
