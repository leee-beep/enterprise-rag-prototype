"""Balanced retrieval across separate per-company FAISS indexes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.config import Settings
from enterprise_rag.indexing import IndexingError, validate_index_compatibility
from enterprise_rag.models import RetrievalResult
from enterprise_rag.retrieval import QueryEmbeddingClient, RetrievalError, Retriever
from enterprise_rag.vector_store import FaissVectorStore, VectorStoreError


COMPANY_NAMES = {"gigabyte": "Gigabyte", "asus": "ASUS", "msi": "MSI"}


@dataclass(frozen=True)
class CompetitorRetrievalResult:
    company_id: str
    company_name: str
    company_rank: int
    retrieval_result: RetrievalResult


class BalancedCompetitorRetriever:
    """Allocate an equal top-k independently to each selected company."""

    def __init__(self, retrievers: Mapping[str, Retriever]) -> None:
        self._retrievers = {key.casefold(): value for key, value in retrievers.items()}

    @classmethod
    def from_index_root(
        cls, index_root: str | Path, settings: Settings, embedding_client: QueryEmbeddingClient
    ) -> "BalancedCompetitorRetriever":
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

    def retrieve(
        self, question: str, company_ids: Sequence[str], top_k_per_company: int
    ) -> tuple[CompetitorRetrievalResult, ...]:
        if not isinstance(question, str) or not question.strip():
            raise RetrievalError("question must be a non-empty string.")
        if isinstance(top_k_per_company, bool) or not isinstance(top_k_per_company, int) or top_k_per_company < 1:
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
        output: list[CompetitorRetrievalResult] = []
        for company in normalized:
            results = self._retrievers[company].retrieve(question.strip(), top_k_per_company)
            if not results:
                raise RetrievalError(f"Company index returned no retrieval results for {company}.")
            output.extend(
                CompetitorRetrievalResult(company, COMPANY_NAMES[company], rank, result)
                for rank, result in enumerate(results, start=1)
            )
        return tuple(output)
