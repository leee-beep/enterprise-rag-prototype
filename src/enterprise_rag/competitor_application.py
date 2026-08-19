"""Production composition for the local competitor-intelligence backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.competitor_grounded_synthesis import (
    GroundedCompetitorSynthesizer,
)
from enterprise_rag.competitor_indexing import SUPPORTED_COMPANIES
from enterprise_rag.competitor_orchestration import (
    BalancedQualitativeEvidenceProvider,
    CompetitorIntelligencePipeline,
    CompetitorIntelligenceResult,
)
from enterprise_rag.competitor_retrieval import (
    BalancedCompetitorRetriever,
    RetrievalError,
)
from enterprise_rag.config import Settings, load_settings
from enterprise_rag.factory import create_embedding_client, create_generation_client
from enterprise_rag.financial_calculations import FinancialCalculationEngine
from enterprise_rag.financial_comparisons import FinancialComparisonEngine
from enterprise_rag.financial_facts import (
    FinancialFactValidationError,
    load_financial_facts_csv,
)
from enterprise_rag.generation import StructuredGenerationClient
from enterprise_rag.indexing import INDEX_MANIFEST_FILE_NAME
from enterprise_rag.vector_store import INDEX_FILE_NAME, METADATA_FILE_NAME


class CompetitorApplicationConfigurationError(RuntimeError):
    """Raised when the local application cannot be composed safely."""


@dataclass(frozen=True, slots=True)
class CompetitorApplicationReadiness:
    """Non-sensitive structural metadata for future readiness endpoints."""

    company_ids: tuple[str, ...]
    financial_facts_loaded: bool
    embedding_provider: str
    embedding_model: str
    generation_provider: str
    generation_model: str


@dataclass(frozen=True, slots=True)
class CompetitorIntelligenceApplication:
    """Small application boundary around the existing orchestration pipeline."""

    pipeline: CompetitorIntelligencePipeline
    readiness: CompetitorApplicationReadiness

    def run(self, question: str) -> CompetitorIntelligenceResult:
        """Delegate one question unchanged to the production pipeline."""
        return self.pipeline.run(question)


def create_local_competitor_intelligence_application(
    settings: Settings | None = None,
) -> CompetitorIntelligenceApplication:
    """Load local resources and compose the complete competitor backend.

    Construction reads persisted indexes and private financial facts, but does not
    contact Ollama, rebuild indexes, or execute generation.
    """
    resolved = settings or load_settings()
    _validate_local_settings(resolved)
    _validate_index_artifacts(resolved.competitor_index_root)
    _validate_financial_facts_path(resolved.financial_facts_path)

    try:
        embedding_client = create_embedding_client(resolved)
        retriever = BalancedCompetitorRetriever.from_index_root(
            resolved.competitor_index_root,
            resolved,
            embedding_client,
        )
    except RetrievalError as exc:
        raise CompetitorApplicationConfigurationError(
            "A required competitor index is invalid or incompatible."
        ) from exc
    except Exception as exc:
        raise CompetitorApplicationConfigurationError(
            "The local embedding provider could not be configured."
        ) from exc

    try:
        facts = load_financial_facts_csv(resolved.financial_facts_path)
    except FinancialFactValidationError as exc:
        raise CompetitorApplicationConfigurationError(
            "Private financial facts are invalid."
        ) from exc

    calculations = FinancialCalculationEngine(facts)
    comparisons = FinancialComparisonEngine(calculations)
    try:
        generation_client = create_generation_client(resolved)
    except Exception as exc:
        raise CompetitorApplicationConfigurationError(
            "The local generation provider could not be configured."
        ) from exc
    if not isinstance(generation_client, StructuredGenerationClient):
        raise CompetitorApplicationConfigurationError(
            "The configured generation provider does not support structured generation."
        )

    pipeline = CompetitorIntelligencePipeline(
        BalancedQualitativeEvidenceProvider(retriever),
        facts,
        calculations,
        comparisons,
        GroundedCompetitorSynthesizer(generation_client),
    )
    readiness = CompetitorApplicationReadiness(
        tuple(SUPPORTED_COMPANIES),
        True,
        resolved.embedding_provider,
        resolved.selected_embedding_model,
        resolved.generation_provider,
        resolved.selected_generation_model,
    )
    return CompetitorIntelligenceApplication(pipeline, readiness)


def _validate_local_settings(settings: Settings) -> None:
    if not isinstance(settings, Settings):
        raise CompetitorApplicationConfigurationError(
            "settings must be a validated Settings instance."
        )
    if settings.embedding_provider != "ollama":
        raise CompetitorApplicationConfigurationError(
            "Local competitor embedding provider must be Ollama."
        )
    if settings.generation_provider != "ollama":
        raise CompetitorApplicationConfigurationError(
            "Local competitor generation provider must be Ollama."
        )
    required = (
        settings.ollama_base_url,
        settings.ollama_embedding_model,
        settings.ollama_chat_model,
    )
    if any(not isinstance(value, str) or not value.strip() for value in required):
        raise CompetitorApplicationConfigurationError(
            "Required local Ollama configuration is unavailable."
        )
    if not isinstance(settings.competitor_index_root, Path):
        raise CompetitorApplicationConfigurationError(
            "Competitor index configuration is invalid."
        )
    if not isinstance(settings.financial_facts_path, Path):
        raise CompetitorApplicationConfigurationError(
            "Financial facts configuration is invalid."
        )


def _validate_index_artifacts(index_root: Path) -> None:
    required_files = (INDEX_FILE_NAME, METADATA_FILE_NAME, INDEX_MANIFEST_FILE_NAME)
    try:
        if not index_root.is_dir():
            raise CompetitorApplicationConfigurationError(
                "Competitor index root is unavailable."
            )
        for company_id in SUPPORTED_COMPANIES:
            company_directory = index_root / company_id
            if not company_directory.is_dir():
                raise CompetitorApplicationConfigurationError(
                    f"Required competitor index is missing for {company_id}."
                )
            if any(not (company_directory / name).is_file() for name in required_files):
                raise CompetitorApplicationConfigurationError(
                    f"Required competitor index artifacts are missing for {company_id}."
                )
    except OSError as exc:
        raise CompetitorApplicationConfigurationError(
            "Competitor index artifacts could not be inspected."
        ) from exc


def _validate_financial_facts_path(path: Path) -> None:
    try:
        available = path.is_file()
    except OSError as exc:
        raise CompetitorApplicationConfigurationError(
            "Private financial facts could not be inspected."
        ) from exc
    if not available:
        raise CompetitorApplicationConfigurationError(
            "Private financial facts are missing."
        )
