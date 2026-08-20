"""Offline tests for deterministic multi-document competitor indexes."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from enterprise_rag.competitor_analysis import build_citation_ready_evidence
from enterprise_rag.competitor_indexing import (
    CompetitorIndexingError,
    CompetitorIndexingService,
)
from enterprise_rag.competitor_metadata import make_source_document_id
from enterprise_rag.competitor_retrieval import BalancedCompetitorRetriever
from enterprise_rag.config import Settings
from enterprise_rag.indexing import (
    IndexingError,
    corpus_fingerprint,
    load_index_manifest,
)
from enterprise_rag.retrieval import RetrievalError
from enterprise_rag.vector_store import FaissVectorStore


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def embed(self, *, model: str, contents: Sequence[str]):
        self.calls.append((model, tuple(contents)))
        return [(float(index), float(len(text))) for index, text in enumerate(contents)]


class FixedQueryClient:
    def __init__(self, vector: tuple[float, ...]) -> None:
        self.vector = vector

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self.vector


def make_pdf(path: Path, *pages: str) -> None:
    from tests.test_pdf_loader import write_pdf

    path.parent.mkdir(parents=True, exist_ok=True)
    content = pages or (
        "AI server strategy and enterprise infrastructure growth. " * 4,
        "Second page discusses products and market opportunities. " * 4,
    )
    write_pdf(path, list(content))


def settings(tmp_path: Path) -> Settings:
    return Settings(
        None,
        "unused",
        "unused",
        tmp_path,
        tmp_path / "index",
        80,
        20,
        2,
        embedding_provider="ollama",
        ollama_embedding_model="fake-model",
        ollama_embedding_batch_size=32,
        competitor_index_root=tmp_path / "indexes",
    )


def entry(
    relative: str,
    *,
    title: str,
    kind: str = "annual_report",
    year: int = 2025,
) -> dict[str, object]:
    return {
        "file": relative,
        "company_id": "asus",
        "company_name": "ASUS",
        "ticker": "2357",
        "fiscal_year": year,
        "period": "FY",
        "document_type": kind,
        "title": title,
        "language": "en",
        "source_url": f"https://www.asus.com/{kind}.pdf",
        "source_relative_path": relative,
    }


def write_manifest(path: Path, entries: Sequence[dict[str, object]]) -> None:
    path.write_text(json.dumps(list(entries)), encoding="utf-8")


def service(config: Settings) -> CompetitorIndexingService:
    return CompetitorIndexingService(
        config,
        embedding_client=FakeEmbeddingClient(),
        clock=lambda: datetime(2026, 8, 20, tzinfo=timezone.utc),
        monotonic=iter((1.0, 1.5)).__next__,
    )


def build_corpus(
    tmp_path: Path,
    entries: Sequence[dict[str, object]],
    *,
    output_name: str,
) -> tuple[Settings, Path, object]:
    config = settings(tmp_path)
    manifest = tmp_path / f"{output_name}.json"
    write_manifest(manifest, entries)
    output = config.competitor_index_root / output_name
    result = service(config).build(
        source_root=tmp_path / "private",
        manifest_path=manifest,
        company_id="asus",
        output_directory=output,
    )
    return config, output, result


def build_single_source_corpus(tmp_path: Path) -> tuple[Settings, Path]:
    relative = "asus/2025/report.pdf"
    make_pdf(tmp_path / "private" / relative)
    config, output, _ = build_corpus(
        tmp_path,
        (entry(relative, title="ASUS report"),),
        output_name="asus",
    )
    return config, output


def rewrite_corpus_manifest(output: Path, mutate) -> None:
    path = output / "index_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest, manifest["source_documents"][0])
    manifest["source_documents"].sort(
        key=lambda item: item["source_document_id"]
    )
    manifest["fiscal_years"] = sorted(
        {item["fiscal_year"] for item in manifest["source_documents"]}
    )
    manifest["document_types"] = sorted(
        {item["document_type"] for item in manifest["source_documents"]}
    )
    fingerprint = corpus_fingerprint(manifest["source_documents"])
    manifest["corpus_fingerprint"] = fingerprint
    manifest["source_identifier"] = (
        f"competitor-corpus:{manifest['company_id']}:{fingerprint[:16]}"
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")


def rewrite_first_chunk(output: Path, mutate) -> None:
    path = output / "metadata.json"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    mutate(persisted["items"][0]["chunk"]["metadata"])
    path.write_text(json.dumps(persisted), encoding="utf-8")


def test_multiple_documents_build_one_company_index_and_preserve_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    annual = "asus/2025/annual.pdf"
    presentation = "asus/2025/presentation.pdf"
    make_pdf(root / annual)
    make_pdf(
        root / presentation,
        "Investor presentation describes GPU server platforms and AI computing. " * 4,
    )
    entries = (
        entry(presentation, title="ASUS Investor Presentation", kind="investor_presentation"),
        entry(annual, title="ASUS 2025 Annual Report"),
    )

    _, output, result = build_corpus(tmp_path, entries, output_name="asus")
    store = FaissVectorStore.load(output)
    manifest = load_index_manifest(output)
    assert manifest is not None

    source_ids = {item.chunk.metadata["source_document_id"] for item in store.items}
    assert result.loaded_document_count == 3
    assert result.chunk_count == store.size
    assert len(source_ids) == manifest["source_count"] == 2
    assert manifest["schema_version"] == 2
    assert manifest["document_types"] == ["annual_report", "investor_presentation"]
    assert manifest["source_documents"] == sorted(
        manifest["source_documents"], key=lambda item: item["source_document_id"]
    )
    assert all(item.chunk.metadata["company_id"] == "asus" for item in store.items)
    assert all(item.chunk.metadata["page_number"] >= 1 for item in store.items)
    assert {item.chunk.metadata["title"] for item in store.items} == {
        "ASUS 2025 Annual Report",
        "ASUS Investor Presentation",
    }
    assert str(tmp_path) not in json.dumps(manifest)


def test_manifest_order_does_not_change_identity_or_chunk_relationships(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    first = entry("asus/2025/a.pdf", title="Source A")
    second = entry(
        "asus/2025/b.pdf",
        title="Source B",
        kind="investor_presentation",
    )
    make_pdf(root / str(first["file"]), "Source A discusses AI server products. " * 5)
    make_pdf(root / str(second["file"]), "Source B discusses data center systems. " * 5)

    _, left_path, _ = build_corpus(tmp_path, (first, second), output_name="left")
    _, right_path, _ = build_corpus(tmp_path, (second, first), output_name="right")
    left_manifest = load_index_manifest(left_path)
    right_manifest = load_index_manifest(right_path)
    left_store = FaissVectorStore.load(left_path)
    right_store = FaissVectorStore.load(right_path)

    assert left_manifest["corpus_fingerprint"] == right_manifest["corpus_fingerprint"]
    assert left_manifest["source_documents"] == right_manifest["source_documents"]
    assert [item.chunk.chunk_id for item in left_store.items] == [
        item.chunk.chunk_id for item in right_store.items
    ]
    assert [item.chunk.metadata["source_document_id"] for item in left_store.items] == [
        item.chunk.metadata["source_document_id"] for item in right_store.items
    ]


def test_duplicate_content_is_rejected_even_under_different_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    first = root / "asus/2025/annual.pdf"
    duplicate = root / "asus/2025/copy.pdf"
    make_pdf(first)
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(first, duplicate)
    entries = (
        entry("asus/2025/annual.pdf", title="Original"),
        entry(
            "asus/2025/copy.pdf",
            title="Duplicate copy",
            kind="investor_presentation",
        ),
    )
    write_manifest(tmp_path / "manifest.json", entries)

    with pytest.raises(CompetitorIndexingError, match="Duplicate competitor source"):
        service(settings(tmp_path)).build(
            source_root=root,
            manifest_path=tmp_path / "manifest.json",
            company_id="asus",
            output_directory=tmp_path / "out",
        )


def test_inconsistent_company_names_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "private"
    first = entry("asus/2025/annual.pdf", title="Annual")
    second = entry(
        "asus/2025/presentation.pdf",
        title="Presentation",
        kind="investor_presentation",
    )
    second["company_name"] = "Inconsistent ASUS name"
    make_pdf(root / str(first["file"]))
    make_pdf(root / str(second["file"]), "Different source content. " * 6)
    write_manifest(tmp_path / "manifest.json", (first, second))

    with pytest.raises(CompetitorIndexingError, match="metadata is invalid"):
        service(settings(tmp_path)).build(
            source_root=root,
            manifest_path=tmp_path / "manifest.json",
            company_id="asus",
            output_directory=tmp_path / "out",
        )


def test_same_filename_with_different_content_remains_distinguishable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    first = "asus/2025/annual/report.pdf"
    second = "asus/2025/presentation/report.pdf"
    make_pdf(root / first, "Annual report AI server strategy. " * 5)
    make_pdf(root / second, "Presentation data center roadmap. " * 5)
    entries = (
        entry(first, title="Annual report"),
        entry(second, title="Presentation", kind="investor_presentation"),
    )

    _, output, _ = build_corpus(tmp_path, entries, output_name="asus")
    store = FaissVectorStore.load(output)
    assert {item.chunk.file_name for item in store.items} == {"report.pdf"}
    assert len({item.chunk.metadata["source_document_id"] for item in store.items}) == 2


def test_full_rebuild_adds_and_removes_source_chunks(tmp_path: Path) -> None:
    root = tmp_path / "private"
    entries = [
        entry("asus/2025/a.pdf", title="Source A"),
        entry("asus/2025/b.pdf", title="Source B", kind="investor_presentation"),
        entry("asus/2025/c.pdf", title="Source C", kind="earnings_release"),
    ]
    for index, item in enumerate(entries):
        make_pdf(root / str(item["file"]), f"Synthetic source {index} discusses AI products. " * 5)

    _, base_path, _ = build_corpus(tmp_path, entries[:2], output_name="base")
    _, added_path, _ = build_corpus(tmp_path, entries, output_name="added")
    _, removed_path, _ = build_corpus(tmp_path, entries[1:], output_name="removed")
    base_ids = {item.chunk.metadata["source_document_id"] for item in FaissVectorStore.load(base_path).items}
    added_ids = {item.chunk.metadata["source_document_id"] for item in FaissVectorStore.load(added_path).items}
    removed_ids = {item.chunk.metadata["source_document_id"] for item in FaissVectorStore.load(removed_path).items}

    assert base_ids < added_ids
    assert removed_ids < added_ids
    assert base_ids - removed_ids
    assert base_ids & removed_ids


def test_stale_manifest_and_chunk_provenance_fail_safely(tmp_path: Path) -> None:
    root = tmp_path / "private"
    relative = "asus/2025/report.pdf"
    make_pdf(root / relative)
    config, output, _ = build_corpus(
        tmp_path,
        (entry(relative, title="ASUS report"),),
        output_name="asus",
    )
    manifest_path = output / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IndexingError, match="fingerprint is stale"):
        load_index_manifest(output)

    # Restore the valid manifest, then corrupt only persisted provenance.
    manifest = dict(manifest)
    from enterprise_rag.indexing import corpus_fingerprint

    manifest["corpus_fingerprint"] = corpus_fingerprint(manifest["source_documents"])
    manifest["source_identifier"] = (
        f"competitor-corpus:asus:{manifest['corpus_fingerprint'][:16]}"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metadata_path = output / "metadata.json"
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    persisted["items"][0]["chunk"]["metadata"]["title"] = "Stale title"
    metadata_path.write_text(json.dumps(persisted), encoding="utf-8")
    with pytest.raises(RetrievalError, match="invalid or incompatible") as error:
        BalancedCompetitorRetriever.from_index_root(
            config.competitor_index_root,
            config,
            FixedQueryClient((0.0, 1.0)),
        )
    assert str(tmp_path) not in str(error.value)


def test_retrieval_and_citations_span_documents_and_keep_relevance_gate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    relevant = "asus/2025/server.pdf"
    governance = "asus/2025/governance.pdf"
    make_pdf(
        root / relevant,
        "AI server products support data center infrastructure and GPU computing. " * 5,
    )
    make_pdf(
        root / governance,
        "Employee welfare and corporate governance training remained stable. " * 5,
    )
    entries = (
        entry(relevant, title="Server strategy"),
        entry(governance, title="Governance report", kind="sustainability_report"),
    )
    config, output, _ = build_corpus(tmp_path, entries, output_name="asus")
    store = FaissVectorStore.load(output)
    source_vectors = {
        item.chunk.metadata["title"]: item.vector for item in store.items
    }
    retriever = BalancedCompetitorRetriever.from_index_root(
        config.competitor_index_root,
        config,
        FixedQueryClient(source_vectors["Server strategy"]),
    )
    response = retriever.retrieve(
        "What is ASUS AI server positioning?", ("asus",), 2
    )
    assert response.company_evidence[0].evidence
    assert {
        item.retrieval_result.metadata["title"] for item in response
    } == {"Server strategy"}
    citation_ready = build_citation_ready_evidence(response)
    assert {item.source_title for item in citation_ready} == {"Server strategy"}
    assert all(item.source_document_id for item in citation_ready)


def test_queries_can_select_different_documents_in_one_company_index(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    products = "asus/2025/products.pdf"
    sustainability = "asus/2025/sustainability.pdf"
    make_pdf(root / products, "Synthetic product roadmap topic alpha. " * 5)
    make_pdf(root / sustainability, "Synthetic sustainability topic beta. " * 5)
    entries = (
        entry(products, title="Product roadmap", kind="official_product_document"),
        entry(
            sustainability,
            title="Sustainability report",
            kind="sustainability_report",
        ),
    )
    config, output, _ = build_corpus(tmp_path, entries, output_name="asus")
    store = FaissVectorStore.load(output)
    source_vectors = {
        item.chunk.metadata["title"]: item.vector for item in store.items
    }

    for question, expected_title in (
        ("Explain topic alpha.", "Product roadmap"),
        ("Explain topic beta.", "Sustainability report"),
    ):
        retriever = BalancedCompetitorRetriever.from_index_root(
            config.competitor_index_root,
            config,
            FixedQueryClient(source_vectors[expected_title]),
        )
        response = retriever.retrieve(question, ("asus",), 1)
        assert response.company_evidence[0].evidence[0].retrieval_result.metadata[
            "title"
        ] == expected_title


def test_optional_filters_keep_controlled_subset_rebuild(tmp_path: Path) -> None:
    root = tmp_path / "private"
    annual = "asus/2025/annual.pdf"
    other = "asus/2024/presentation.pdf"
    make_pdf(root / annual)
    make_pdf(root / other, "Investor presentation AI products. " * 5)
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        (
            entry(annual, title="Annual"),
            entry(other, title="Presentation", kind="investor_presentation", year=2024),
        ),
    )
    output = tmp_path / "subset"
    result = service(settings(tmp_path)).build(
        source_root=root,
        manifest_path=manifest,
        company_id="asus",
        output_directory=output,
        fiscal_year=2025,
        document_type="annual_report",
    )
    assert result.manifest["source_count"] == 1
    assert result.manifest["fiscal_years"] == [2025]


def test_missing_scope_unknown_company_and_private_path_errors_are_safe(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(CompetitorIndexingError, match="No approved competitor sources"):
        service(settings(tmp_path)).build(
            source_root=root,
            manifest_path=manifest,
            company_id="asus",
            output_directory=tmp_path / "out",
        )
    with pytest.raises(CompetitorIndexingError, match="Unknown") as error:
        service(settings(tmp_path)).build(
            source_root=tmp_path / "private-secret",
            manifest_path=manifest,
            company_id="unknown",
            output_directory=tmp_path / "out",
        )
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("source_id", "source identity"),
        ("document_type", "document_type"),
        ("fiscal_year", "fiscal_year"),
        ("period", "period"),
    ),
)
def test_schema_v2_rejects_noncanonical_source_semantics_after_refingerprint(
    tmp_path: Path, case: str, message: str
) -> None:
    _, output = build_single_source_corpus(tmp_path)

    def mutate(_manifest, source) -> None:
        if case == "source_id":
            source["source_document_id"] = (
                "competitor:asus:annual_report:2025:ffffffffffffffff"
            )
            return
        if case == "document_type":
            source["document_type"] = "provider_invented_type"
        elif case == "fiscal_year":
            source["fiscal_year"] = 2099
        else:
            source["period"] = "Q4"
        source["source_document_id"] = make_source_document_id(
            "asus",
            source["document_type"],
            source["fiscal_year"],
            source["source_sha256"],
        )

    rewrite_corpus_manifest(output, mutate)

    with pytest.raises(IndexingError, match=message):
        load_index_manifest(output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("company_name", "MSI", "company_name"),
        ("ticker", "2377", "ticker"),
    ),
)
def test_schema_v2_rejects_wrong_corpus_company_identity(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    _, output = build_single_source_corpus(tmp_path)
    rewrite_corpus_manifest(
        output, lambda manifest, _source: manifest.__setitem__(field, value)
    )

    with pytest.raises(IndexingError, match=message):
        load_index_manifest(output)


def test_schema_v2_rejects_coordinated_invalid_source_metadata(
    tmp_path: Path,
) -> None:
    _, output = build_single_source_corpus(tmp_path)

    def mutate(_manifest, source) -> None:
        source["document_type"] = "provider_invented_type"
        source["fiscal_year"] = 2099
        source["period"] = "UNKNOWN"
        source["source_document_id"] = make_source_document_id(
            "asus",
            source["document_type"],
            source["fiscal_year"],
            source["source_sha256"],
        )

    rewrite_corpus_manifest(output, mutate)

    with pytest.raises(IndexingError, match="unsupported"):
        load_index_manifest(output)


@pytest.mark.parametrize("case", ("zero", "beyond", "mismatched_count"))
def test_schema_v2_rejects_invalid_chunk_page_provenance(
    tmp_path: Path, case: str
) -> None:
    config, output = build_single_source_corpus(tmp_path)
    manifest = load_index_manifest(output)
    source_page_count = manifest["source_documents"][0]["page_count"]

    def mutate(metadata) -> None:
        if case == "zero":
            metadata["page_number"] = 0
        elif case == "beyond":
            metadata["page_number"] = source_page_count + 1
        else:
            metadata["page_count"] = source_page_count + 1

    rewrite_first_chunk(output, mutate)

    with pytest.raises(RetrievalError, match="invalid or incompatible"):
        BalancedCompetitorRetriever.from_index_root(
            config.competitor_index_root,
            config,
            FixedQueryClient((0.0, 1.0)),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("company_name", "MSI"), ("ticker", "2377")),
)
def test_schema_v2_rejects_chunk_company_identity_mismatch(
    tmp_path: Path, field: str, value: str
) -> None:
    config, output = build_single_source_corpus(tmp_path)
    rewrite_first_chunk(output, lambda metadata: metadata.__setitem__(field, value))

    with pytest.raises(RetrievalError, match="invalid or incompatible"):
        BalancedCompetitorRetriever.from_index_root(
            config.competitor_index_root,
            config,
            FixedQueryClient((0.0, 1.0)),
        )


def test_schema_v2_accepts_chunk_on_final_source_page(tmp_path: Path) -> None:
    config, output = build_single_source_corpus(tmp_path)
    manifest = load_index_manifest(output)
    source_page_count = manifest["source_documents"][0]["page_count"]

    def mutate(metadata) -> None:
        metadata["page_number"] = source_page_count
        metadata["page_count"] = source_page_count

    rewrite_first_chunk(output, mutate)

    retriever = BalancedCompetitorRetriever.from_index_root(
        config.competitor_index_root,
        config,
        FixedQueryClient((0.0, 1.0)),
    )
    assert isinstance(retriever, BalancedCompetitorRetriever)


def test_legacy_manifest_remains_valid(tmp_path: Path) -> None:
    directory = tmp_path / "legacy"
    directory.mkdir()
    data = {
        "schema_version": 1,
        "built_at": "2026-01-01T00:00:00Z",
        "source_identifier": "docs",
        "embedding_provider": "ollama",
        "embedding_model": "fake",
        "embedding_dimension": 2,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "document_count": 1,
        "chunk_count": 1,
    }
    (directory / "index_manifest.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    assert load_index_manifest(directory) == data
