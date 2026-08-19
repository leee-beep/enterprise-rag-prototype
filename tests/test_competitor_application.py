"""Offline tests for local competitor-application composition."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import pytest

import enterprise_rag.competitor_application as application_module
from enterprise_rag.competitor_application import (
    CompetitorApplicationConfigurationError,
    CompetitorApplicationReadiness,
    CompetitorIntelligenceApplication,
    create_local_competitor_intelligence_application,
)
from enterprise_rag.competitor_indexing import SUPPORTED_COMPANIES
from enterprise_rag.config import Settings
from enterprise_rag.financial_facts import (
    FinancialFactCollection,
    FinancialFactValidationError,
)


class FakePipeline:
    def __init__(self, result: object = None, *dependencies: object) -> None:
        self.result = result
        self.dependencies = dependencies
        self.questions: list[str] = []

    def run(self, question: str) -> object:
        self.questions.append(question)
        return self.result


class FakeEmbeddingClient:
    def embed_query(self, text: str) -> tuple[float, ...]:
        raise AssertionError("Factory construction must not embed a query.")


class FakeStructuredGenerationClient:
    provider = "ollama"
    model = "local-generation"

    def generate(self, prompt: str) -> str:
        raise AssertionError("Factory construction must not generate text.")

    def generate_structured(self, prompt: str, schema: object) -> str:
        raise AssertionError("Factory construction must not generate structured text.")


def settings(tmp_path: Path, **changes: object) -> Settings:
    value = Settings(
        gemini_api_key=None,
        generation_model="unused-gemini-generation",
        embedding_model="unused-gemini-embedding",
        documents_dir=tmp_path / "documents",
        vector_store_dir=tmp_path / "vector-store",
        chunk_size=200,
        chunk_overlap=50,
        top_k=4,
        embedding_provider="ollama",
        generation_provider="ollama",
        ollama_base_url="http://local-ollama",
        ollama_embedding_model="local-embedding",
        ollama_chat_model="local-generation",
        competitor_index_root=tmp_path / "indexes",
        financial_facts_path=tmp_path / "private" / "financial_facts.csv",
    )
    return replace(value, **changes)


def create_index_artifacts(root: Path, *, omit: tuple[str, str] | None = None) -> None:
    for company_id in SUPPORTED_COMPANIES:
        directory = root / company_id
        directory.mkdir(parents=True)
        for name in ("index.faiss", "metadata.json", "index_manifest.json"):
            if omit != (company_id, name):
                (directory / name).write_bytes(b"synthetic")


def test_application_run_is_an_unchanged_pipeline_delegation() -> None:
    result = object()
    pipeline = FakePipeline(result)
    readiness = CompetitorApplicationReadiness(
        tuple(SUPPORTED_COMPANIES), True, "ollama", "embed", "ollama", "chat"
    )
    subject = CompetitorIntelligenceApplication(pipeline, readiness)  # type: ignore[arg-type]

    assert subject.run(" Question? ") is result
    assert pipeline.questions == [" Question? "]
    with pytest.raises(AttributeError):
        subject.readiness = readiness  # type: ignore[misc]


def test_empty_question_behavior_remains_pipeline_owned() -> None:
    class RejectingPipeline(FakePipeline):
        def run(self, question: str) -> object:
            raise ValueError("pipeline-owned empty question")

    subject = CompetitorIntelligenceApplication(
        RejectingPipeline(),  # type: ignore[arg-type]
        CompetitorApplicationReadiness((), False, "ollama", "e", "ollama", "g"),
    )
    with pytest.raises(ValueError, match="pipeline-owned"):
        subject.run("")


def test_factory_wires_existing_components_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = settings(tmp_path)
    create_index_artifacts(config.competitor_index_root)
    config.financial_facts_path.parent.mkdir(parents=True)
    config.financial_facts_path.write_text("synthetic", encoding="utf-8")
    calls: dict[str, list[object]] = {
        "embedding": [], "retrieval": [], "qualitative": [], "facts": [],
        "calculation": [], "comparison": [], "generation": [],
        "synthesis": [], "pipeline": [],
    }
    embedded = FakeEmbeddingClient()
    balanced = object()
    qualitative = object()
    facts = FinancialFactCollection(())
    calculation = object()
    comparison = object()
    generation = FakeStructuredGenerationClient()
    synthesizer = object()

    monkeypatch.setattr(application_module, "create_embedding_client", lambda value: calls["embedding"].append(value) or embedded)
    monkeypatch.setattr(application_module.BalancedCompetitorRetriever, "from_index_root", lambda root, value, client: calls["retrieval"].append((root, value, client)) or balanced)
    monkeypatch.setattr(application_module, "BalancedQualitativeEvidenceProvider", lambda value: calls["qualitative"].append(value) or qualitative)
    monkeypatch.setattr(application_module, "load_financial_facts_csv", lambda path: calls["facts"].append(path) or facts)
    monkeypatch.setattr(application_module, "FinancialCalculationEngine", lambda value: calls["calculation"].append(value) or calculation)
    monkeypatch.setattr(application_module, "FinancialComparisonEngine", lambda value: calls["comparison"].append(value) or comparison)
    monkeypatch.setattr(application_module, "create_generation_client", lambda value: calls["generation"].append(value) or generation)
    monkeypatch.setattr(application_module, "GroundedCompetitorSynthesizer", lambda value: calls["synthesis"].append(value) or synthesizer)

    def pipeline_factory(*dependencies: object) -> FakePipeline:
        calls["pipeline"].append(dependencies)
        return FakePipeline("result", *dependencies)

    monkeypatch.setattr(application_module, "CompetitorIntelligencePipeline", pipeline_factory)

    result = create_local_competitor_intelligence_application(config)

    assert calls["embedding"] == [config]
    assert calls["retrieval"] == [(config.competitor_index_root, config, embedded)]
    assert calls["qualitative"] == [balanced]
    assert calls["facts"] == [config.financial_facts_path]
    assert calls["calculation"] == [facts]
    assert calls["comparison"] == [calculation]
    assert calls["generation"] == [config]
    assert calls["synthesis"] == [generation]
    assert calls["pipeline"] == [
        (qualitative, facts, calculation, comparison, synthesizer)
    ]
    assert result.readiness.company_ids == tuple(SUPPORTED_COMPANIES)
    assert result.readiness.financial_facts_loaded is True
    assert result.readiness.embedding_model == "local-embedding"
    assert result.readiness.generation_model == "local-generation"


@pytest.mark.parametrize("provider_field", ("embedding_provider", "generation_provider"))
def test_factory_rejects_nonlocal_provider(
    tmp_path: Path, provider_field: str
) -> None:
    with pytest.raises(CompetitorApplicationConfigurationError, match="must be Ollama"):
        create_local_competitor_intelligence_application(
            settings(tmp_path, **{provider_field: "gemini"})
        )


def test_factory_rejects_malformed_settings_without_private_path_leak(
    tmp_path: Path,
) -> None:
    private_marker = "do-not-expose-private-location"
    config = settings(
        tmp_path,
        ollama_base_url="",
        financial_facts_path=tmp_path / private_marker / "financial_facts.csv",
    )
    with pytest.raises(CompetitorApplicationConfigurationError) as error:
        create_local_competitor_intelligence_application(config)
    assert private_marker not in str(error.value)


def test_factory_rejects_malformed_resource_path_type(tmp_path: Path) -> None:
    config = settings(tmp_path, competitor_index_root="private-index")
    with pytest.raises(
        CompetitorApplicationConfigurationError,
        match="index configuration is invalid",
    ):
        create_local_competitor_intelligence_application(config)


def test_factory_rejects_missing_company_index_safely(tmp_path: Path) -> None:
    config = settings(tmp_path)
    create_index_artifacts(config.competitor_index_root)
    missing = config.competitor_index_root / "msi"
    for item in missing.iterdir():
        item.unlink()
    missing.rmdir()
    with pytest.raises(CompetitorApplicationConfigurationError, match="missing for msi") as error:
        create_local_competitor_intelligence_application(config)
    assert str(tmp_path) not in str(error.value)


def test_factory_rejects_missing_index_artifact_safely(tmp_path: Path) -> None:
    config = settings(tmp_path)
    create_index_artifacts(config.competitor_index_root, omit=("asus", "metadata.json"))
    with pytest.raises(CompetitorApplicationConfigurationError, match="artifacts.*asus") as error:
        create_local_competitor_intelligence_application(config)
    assert str(tmp_path) not in str(error.value)


def test_factory_rejects_missing_financial_facts_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = settings(tmp_path)
    create_index_artifacts(config.competitor_index_root)
    monkeypatch.setattr(application_module, "create_embedding_client", lambda value: FakeEmbeddingClient())
    monkeypatch.setattr(application_module.BalancedCompetitorRetriever, "from_index_root", lambda *args: object())
    with pytest.raises(CompetitorApplicationConfigurationError, match="facts are missing") as error:
        create_local_competitor_intelligence_application(config)
    assert str(tmp_path) not in str(error.value)


def test_factory_sanitizes_invalid_financial_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = settings(tmp_path)
    create_index_artifacts(config.competitor_index_root)
    config.financial_facts_path.parent.mkdir(parents=True)
    config.financial_facts_path.write_text("private-invalid-value", encoding="utf-8")
    monkeypatch.setattr(application_module, "create_embedding_client", lambda value: FakeEmbeddingClient())
    monkeypatch.setattr(application_module.BalancedCompetitorRetriever, "from_index_root", lambda *args: object())
    monkeypatch.setattr(application_module, "load_financial_facts_csv", lambda path: (_ for _ in ()).throw(FinancialFactValidationError("private-invalid-value")))
    with pytest.raises(CompetitorApplicationConfigurationError, match="facts are invalid") as error:
        create_local_competitor_intelligence_application(config)
    assert "private-invalid-value" not in str(error.value)
    assert error.value.__cause__ is not None


def test_factory_sanitizes_provider_construction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = settings(tmp_path)
    create_index_artifacts(config.competitor_index_root)
    config.financial_facts_path.parent.mkdir(parents=True)
    config.financial_facts_path.write_text("synthetic", encoding="utf-8")
    monkeypatch.setattr(application_module, "create_embedding_client", lambda value: (_ for _ in ()).throw(RuntimeError("private provider detail")))
    with pytest.raises(CompetitorApplicationConfigurationError, match="embedding provider") as error:
        create_local_competitor_intelligence_application(config)
    assert "private provider detail" not in str(error.value)


def test_import_has_no_network_index_or_private_data_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import enterprise_rag.competitor_retrieval as retrieval_module
    import enterprise_rag.factory as factory_module
    import enterprise_rag.financial_facts as facts_module

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("import performed resource I/O")

    monkeypatch.setattr(factory_module, "create_embedding_client", forbidden)
    monkeypatch.setattr(factory_module, "create_generation_client", forbidden)
    monkeypatch.setattr(facts_module, "load_financial_facts_csv", forbidden)
    monkeypatch.setattr(
        retrieval_module.BalancedCompetitorRetriever, "from_index_root", forbidden
    )
    importlib.reload(application_module)
