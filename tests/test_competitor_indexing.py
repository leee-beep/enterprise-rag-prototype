from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence

import pytest
from pypdf import PdfWriter

from enterprise_rag.competitor_indexing import CompetitorIndexingError, CompetitorIndexingService
from enterprise_rag.config import Settings
from enterprise_rag.indexing import load_index_manifest
from enterprise_rag.vector_store import FaissVectorStore


class FakeEmbeddingClient:
    def __init__(self): self.calls=[]
    def embed(self, *, model: str, contents: Sequence[str]):
        self.calls.append((model, tuple(contents)))
        return [(float(i), float(len(text))) for i, text in enumerate(contents)]


def make_pdf(path: Path) -> None:
    from tests.test_pdf_loader import write_pdf
    write_pdf(path, ["AI server strategy and enterprise infrastructure growth. " * 4, "Second page discusses products and market opportunities. " * 4])


def settings(tmp_path: Path) -> Settings:
    return Settings(None, "unused", "unused", tmp_path, tmp_path / "index", 80, 20, 2,
        embedding_provider="ollama", ollama_embedding_model="fake-model", ollama_embedding_batch_size=32)


def manifest(path: Path, pdf: Path, *, year=2025, kind="annual_report") -> None:
    value=[{"file":"asus/2025/report.pdf","company_id":"asus","company_name":"ASUS","ticker":"2357","fiscal_year":year,"period":"FY","document_type":kind,"title":"ASUS 2025 Annual Report","language":"en","source_url":"https://www.asus.com/report.pdf","source_relative_path":"asus/2025/report.pdf"}]
    path.write_text(json.dumps(value),encoding="utf-8")


def test_build_selects_2025_annual_report_and_preserves_metadata(tmp_path: Path):
    root=tmp_path/"private"; pdf=root/"asus/2025/report.pdf"; pdf.parent.mkdir(parents=True); make_pdf(pdf)
    mf=tmp_path/"manifest.json"; manifest(mf,pdf)
    client=FakeEmbeddingClient(); output=tmp_path/"indexes/asus"
    service=CompetitorIndexingService(settings(tmp_path), embedding_client=client,
        clock=lambda: datetime(2026,8,12,tzinfo=timezone.utc), monotonic=iter((1.0,1.5)).__next__)
    result=service.build(source_root=root,manifest_path=mf,company_id="asus",output_directory=output)
    store=FaissVectorStore.load(output); item=store.search((0.0,1.0),1)[0]
    assert result.loaded_document_count==2 and result.chunk_count==store.size
    assert item.chunk.metadata["company_id"]=="asus" and item.chunk.metadata["page_number"] in (1,2)
    assert item.chunk.metadata["source_document_id"].startswith("competitor:asus:annual_report:2025:")
    saved=load_index_manifest(output)
    for key in ("company_id","company_name","ticker","fiscal_year","document_type","source_document_id","source_sha256","page_count"):
        assert key in saved
    assert saved["chunk_size"]==80 and saved["chunk_overlap"]==20
    assert str(tmp_path) not in json.dumps(saved)


def test_only_exact_manifest_selection_is_allowed(tmp_path: Path):
    root=tmp_path/"private"; root.mkdir(); mf=tmp_path/"manifest.json"; mf.write_text("[]",encoding="utf-8")
    with pytest.raises(CompetitorIndexingError,match="exactly one"):
        CompetitorIndexingService(settings(tmp_path),embedding_client=FakeEmbeddingClient()).build(source_root=root,manifest_path=mf,company_id="asus",output_directory=tmp_path/"out")


def test_unknown_company_and_private_path_errors_are_safe(tmp_path: Path):
    service=CompetitorIndexingService(settings(tmp_path),embedding_client=FakeEmbeddingClient())
    with pytest.raises(CompetitorIndexingError,match="Unknown") as err:
        service.build(source_root=tmp_path/"private-secret",manifest_path=tmp_path/"manifest.json",company_id="unknown",output_directory=tmp_path/"out")
    assert str(tmp_path) not in str(err.value)


def test_legacy_manifest_remains_valid(tmp_path: Path):
    directory=tmp_path/"legacy"; directory.mkdir()
    data={"schema_version":1,"built_at":"2026-01-01T00:00:00Z","source_identifier":"docs","embedding_provider":"ollama","embedding_model":"fake","embedding_dimension":2,"chunk_size":500,"chunk_overlap":100,"document_count":1,"chunk_count":1}
    (directory/"index_manifest.json").write_text(json.dumps(data),encoding="utf-8")
    assert load_index_manifest(directory)==data
