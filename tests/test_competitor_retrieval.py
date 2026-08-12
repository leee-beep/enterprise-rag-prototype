from __future__ import annotations

from collections.abc import Sequence
from io import StringIO
import pytest

from enterprise_rag.competitor_retrieval import BalancedCompetitorRetriever
from enterprise_rag.models import DocumentChunk, EmbeddedChunk
from enterprise_rag.retrieval import RetrievalError, Retriever
from enterprise_rag.cli import main


class Client:
    def __init__(self): self.queries=[]
    def embed_query(self,text): self.queries.append(text); return (0.0,0.0)

class Store:
    size=3
    def __init__(self,company): self.company=company
    def search_with_scores(self,vector,top_k):
        return [(embedded(self.company,i),float(i)) for i in range(top_k)]

def embedded(company,index):
    meta={"company_id":company,"company_name":company.title(),"fiscal_year":2025,"page_number":index+1,"source_document_id":f"doc-{company}"}
    chunk=DocumentChunk(f"{company} chunk {index}",f"{company}/2025/report.pdf","report.pdf",".pdf",f"doc-{company}:page-{index+1:04d}",index,f"doc-{company}:page-{index+1:04d}:chunk-{index:06d}",meta)
    return EmbeddedChunk(chunk,(0.0,0.0),"fake")

def balanced(companies=("gigabyte","asus","msi")):
    clients={c:Client() for c in companies}
    return BalancedCompetitorRetriever({c:Retriever(clients[c],Store(c)) for c in companies}),clients

@pytest.mark.parametrize("companies",[("asus",),("asus","msi"),("gigabyte","asus","msi")])
def test_equal_allocation_order_ranks_and_metadata(companies):
    service,clients=balanced(); results=service.retrieve(" compare AI ",companies,2)
    assert [(r.company_id,r.company_rank) for r in results]==[(c,k) for c in companies for k in (1,2)]
    assert all(r.retrieval_result.metadata["company_id"]==r.company_id for r in results)
    assert all(clients[c].queries==["compare AI"] for c in companies)

@pytest.mark.parametrize(("companies","top_k","message"),[
    ((),2,"At least one"),(("asus","asus"),2,"Duplicate"),(("unknown",),2,"Unknown"),(("asus",),0,"positive integer")])
def test_selection_errors(companies,top_k,message):
    service,_=balanced(("asus",))
    with pytest.raises(RetrievalError,match=message): service.retrieve("q",companies,top_k)

def test_empty_query_missing_index_and_no_private_path_leakage(tmp_path):
    service,_=balanced(("asus",))
    with pytest.raises(RetrievalError,match="question"): service.retrieve(" ",( "asus",),1)
    with pytest.raises(RetrievalError,match="unavailable") as err: service.retrieve("q",("msi",),1)
    assert str(tmp_path) not in str(err.value)

def test_scores_remain_company_local_and_are_not_globally_sorted():
    service,_=balanced()
    results=service.retrieve("q",("msi","gigabyte","asus"),2)
    assert [r.company_id for r in results]==["msi","msi","gigabyte","gigabyte","asus","asus"]
    assert [r.retrieval_result.score for r in results]==[1.0,0.5]*3

def test_competitor_cli_outputs_company_page_and_chunk(monkeypatch):
    service,_=balanced(("asus",))
    monkeypatch.setattr(BalancedCompetitorRetriever,"from_index_root",classmethod(lambda cls,*args: service))
    output=StringIO()
    assert main(["competitor-retrieve","--companies","asus","--top-k-per-company","1","AI strategy"],query_embedding_client=Client(),output=output)==0
    rendered=output.getvalue()
    assert "ASUS Rank 1" in rendered and "Page: 1" in rendered and "Chunk ID:" in rendered
