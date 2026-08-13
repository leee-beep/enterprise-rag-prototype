from __future__ import annotations

from io import StringIO
import pytest

from enterprise_rag.cli import main
from enterprise_rag.competitor_retrieval import (
    BalancedCompetitorRetriever,
    assess_evidence_quality,
)
from enterprise_rag.models import DocumentChunk, EmbeddedChunk
from enterprise_rag.retrieval import RetrievalError, Retriever


ENGLISH = "AI server demand increased 28% as enterprises expanded infrastructure. Management expects continued growth in this strategic market."
CHINESE = "人工智慧伺服器需求持續成長，公司將擴大研發投資並提供企業客戶完整的基礎設施解決方案。管理階層預期市場將持續成長。"
REVENUE = "Revenue grew from 12.4 billion to 15.8 billion, supported by stronger server shipments and expanding enterprise demand."
TABLE = "100% 0 100% 0\n33,315,472 100% 0 0%\n16,737,013 100% 10 0%\n80,000 100% 0 0%"
OWNERSHIP = "Company Name Shares Ownership Book Value\nABC LTD 10,000,000 100% 500,000\nXYZ INC 20,000,000 83.5% 900,000\nDEF CO 9,000,000 100% 700,000"
AFFILIATES = "ASUS EUROPE B.V. 100%\nASUS JAPAN INC 100%\nASUS INDIA PRIVATE LIMITED 100%\nASUS AUSTRALIA PTY 100%"


class Client:
    def __init__(self): self.queries=[]
    def embed_query(self,text): self.queries.append(text); return (0.0,0.0)


class Store:
    def __init__(self,company,contents=None): self.company=company; self.contents=contents or [ENGLISH, CHINESE, REVENUE]; self.calls=[]
    @property
    def size(self): return len(self.contents)
    def search_with_scores(self,vector,top_k):
        self.calls.append(top_k)
        return [(embedded(self.company,i,text),float(i)) for i,text in enumerate(self.contents[:top_k])]


def embedded(company,index,content,page=1):
    meta={"company_id":company,"company_name":company.title(),"fiscal_year":2025,"page_number":page,"source_document_id":f"doc-{company}","title":f"{company.title()} 2025 Annual Report"}
    chunk=DocumentChunk(content,f"{company}/2025/report.pdf","report.pdf",".pdf",f"doc-{company}:page-{page:04d}",index,f"doc-{company}:page-{page:04d}:chunk-{index:06d}",meta)
    return EmbeddedChunk(chunk,(0.0,0.0),"fake")


def balanced(contents=None,companies=("gigabyte","asus","msi")):
    clients={c:Client() for c in companies}; stores={c:Store(c,contents) for c in companies}
    return BalancedCompetitorRetriever({c:Retriever(clients[c],stores[c]) for c in companies}),clients,stores


@pytest.mark.parametrize("text",[
    ENGLISH, CHINESE, "## AI Strategy\n\nWe are expanding enterprise AI products.",
    "Server revenue increased 17.5% because enterprise customers accelerated AI deployments.", REVENUE,
])
def test_quality_gate_accepts_meaningful_narrative(text):
    assessment=assess_evidence_quality(text)
    assert assessment.is_usable and assessment.quality_score >= 0.45


@pytest.mark.parametrize("text",[TABLE,OWNERSHIP,AFFILIATES,"1 2 3 4 5 6 7 8 9 10"])
def test_quality_gate_rejects_table_and_low_language_density(text):
    assessment=assess_evidence_quality(text)
    assert not assessment.is_usable and assessment.reasons


def test_candidate_budget_is_twelve_per_query_and_noise_is_skipped():
    service,_,stores=balanced([TABLE,ENGLISH,OWNERSHIP,CHINESE,REVENUE,AFFILIATES],("asus",))
    response=service.retrieve("AI strategy",("asus",),2); group=response.company_evidence[0]
    assert stores["asus"].calls==[12,12]
    assert [item.original_candidate_rank for item in group.evidence]==[2,4]
    assert [item.company_rank for item in group.evidence]==[1,2]
    assert not group.insufficient_evidence


def test_exact_and_high_overlap_adjacent_duplicates_are_suppressed():
    overlapping=ENGLISH + " Additional detail."
    service,_,_=balanced([ENGLISH,overlapping,ENGLISH,CHINESE],("asus",))
    response=service.retrieve("AI",("asus",),3); group=response.company_evidence[0]
    assert len(group.evidence)==2
    assert any("duplicate-or-overlapping" in reasons for _,reasons in group.rejected_reasons)


def test_distinct_adjacent_chunks_and_different_pages_are_retained():
    distinct="The company launched new notebooks and gaming monitors for professional creators."
    service,_,_=balanced([ENGLISH,distinct,CHINESE],("asus",))
    assert len(service.retrieve("products",("asus",),3))==3


def test_insufficient_evidence_is_explicit_and_not_backfilled():
    service,_,_=balanced([TABLE,ENGLISH,OWNERSHIP],("asus",))
    group=service.retrieve("strategy",("asus",),2).company_evidence[0]
    assert group.returned_count==1 and group.insufficient_evidence


@pytest.mark.parametrize("companies",[("asus",),("asus","msi"),("gigabyte","asus","msi")])
def test_company_allocation_order_local_ranks_and_queries(companies):
    service,clients,_=balanced(companies=companies)
    response=service.retrieve(" compare AI ",companies,2)
    assert [(r.company_id,r.company_rank) for r in response]==[(c,k) for c in companies for k in (1,2)]
    assert all(clients[c].queries==["compare AI"] for c in companies)


def test_company_deduplication_never_crosses_company_boundary():
    service,_,_=balanced([ENGLISH],("asus","msi"))
    response=service.retrieve("AI",("asus","msi"),1)
    assert [item.company_id for item in response]==["asus","msi"]


@pytest.mark.parametrize(("companies","top_k","message"),[((),2,"At least one"),(("asus","asus"),2,"Duplicate"),(("unknown",),2,"Unknown"),(("asus",),0,"positive integer")])
def test_selection_errors(companies,top_k,message):
    service,_,_=balanced(companies=("asus",))
    with pytest.raises(RetrievalError,match=message): service.retrieve("q",companies,top_k)


def test_empty_query_and_missing_index_are_safe(tmp_path):
    service,_,_=balanced(companies=("asus",))
    with pytest.raises(RetrievalError,match="question"): service.retrieve(" ",( "asus",),1)
    with pytest.raises(RetrievalError,match="unavailable") as err: service.retrieve("q",("msi",),1)
    assert str(tmp_path) not in str(err.value)


def test_scores_are_company_local_not_globally_sorted():
    service,_,_=balanced(companies=("msi","gigabyte","asus"))
    results=service.retrieve("q",("msi","gigabyte","asus"),2)
    assert [r.company_id for r in results]==["msi","msi","gigabyte","gigabyte","asus","asus"]
    assert [r.retrieval_result.score for r in results]==[1.0,0.5]*3


def test_cli_shows_quality_candidate_rank_and_insufficiency(monkeypatch):
    service,_,_=balanced([TABLE,ENGLISH],("asus",))
    monkeypatch.setattr(BalancedCompetitorRetriever,"from_index_root",classmethod(lambda cls,*args: service))
    output=StringIO()
    assert main(["competitor-retrieve","--companies","asus","--top-k-per-company","2","AI strategy"],query_embedding_client=Client(),output=output)==0
    rendered=output.getvalue()
    assert "Requested: 2" in rendered and "Returned: 1" in rendered
    assert "insufficient usable evidence" in rendered and "Original candidate rank: 2" in rendered
    assert "Quality: usable" in rendered and "Page: 1" in rendered
