from __future__ import annotations

import json
import socket
from urllib import error
import pytest

from enterprise_rag.embeddings import EmbeddingValidationError
from enterprise_rag.generation import GenerationValidationError
from enterprise_rag.providers.ollama import (
    OllamaConnectionError, OllamaEmbeddingClient, OllamaGenerationClient,
    OllamaHTTPError, OllamaResponseError, OllamaTimeoutError, UrllibJsonTransport,
)

class FakeTransport:
    def __init__(self, result=None, exception=None):
        self.result=result; self.exception=exception; self.calls=[]
    def post_json(self, url, payload, timeout):
        self.calls.append((url,payload,timeout))
        if self.exception: raise self.exception
        return self.result

def embedding_client(result):
    transport=FakeTransport(result)
    return OllamaEmbeddingClient(base_url="http://localhost:11434/", model="embed-model", timeout=2, transport=transport), transport

def generation_client(result):
    transport=FakeTransport(result)
    return OllamaGenerationClient(base_url="http://localhost:11434/", model="chat-model", timeout=2, transport=transport), transport

def test_ollama_embedding_batch_success():
    client,transport=embedding_client({"embeddings":[[1,2],[3,4]]})
    assert client.embed_documents(["a","b"]) == ((1.0,2.0),(3.0,4.0))
    assert transport.calls == [("http://localhost:11434/api/embed", {"model":"embed-model","input":["a","b"]}, 2)]

def test_ollama_embedding_query_success():
    client,_=embedding_client({"embeddings":[[0.1,0.2]]})
    assert client.embed_query("query") == (0.1,0.2)

@pytest.mark.parametrize("payload,match", [
    ({"embeddings":[[]]}, "empty"),
    ({"embeddings":[[1,2],[3,4,5]]}, "inconsistent"),
])
def test_ollama_embedding_vector_validation(payload, match):
    client,_=embedding_client(payload)
    texts=["a"] if len(payload["embeddings"])==1 else ["a","b"]
    with pytest.raises(EmbeddingValidationError, match=match): client.embed_documents(texts)

@pytest.mark.parametrize("payload", [{}, {"embeddings":"bad"}, {"vectors":[[1.0]]}])
def test_ollama_embedding_rejects_unexpected_schema(payload):
    client,_=embedding_client(payload)
    with pytest.raises(OllamaResponseError, match="embeddings array"): client.embed_documents(["a"])

def test_ollama_generation_success_and_disables_streaming():
    client,transport=generation_client({"response":" local answer "})
    assert client.generate("prompt") == "local answer"
    assert transport.calls == [("http://localhost:11434/api/generate", {"model":"chat-model","prompt":"prompt","stream":False}, 2)]

@pytest.mark.parametrize("payload", [{"response":""}, {"response":None}])
def test_ollama_generation_rejects_empty_response(payload):
    client,_=generation_client(payload)
    with pytest.raises(GenerationValidationError): client.generate("prompt")

def test_ollama_generation_rejects_unexpected_schema():
    client,_=generation_client({"message":{"content":"wrong endpoint schema"}})
    with pytest.raises(OllamaResponseError, match="response field"): client.generate("prompt")

class FakeResponse:
    def __init__(self, body, status=200): self.body=body; self.status=status
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self): return self.body
    def getcode(self): return self.status

def test_transport_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr("enterprise_rag.providers.ollama.request.urlopen", lambda *a,**k: FakeResponse(b"not-json"))
    with pytest.raises(OllamaResponseError, match="invalid JSON"):
        UrllibJsonTransport().post_json("http://local/api/embed", {}, 1)

def test_transport_rejects_non_2xx_http(monkeypatch):
    def fail(*args,**kwargs): raise error.HTTPError("url",404,"missing",None,None)
    monkeypatch.setattr("enterprise_rag.providers.ollama.request.urlopen", fail)
    with pytest.raises(OllamaHTTPError, match="404"):
        UrllibJsonTransport().post_json("http://local/api/embed", {}, 1)

def test_transport_maps_timeout(monkeypatch):
    monkeypatch.setattr("enterprise_rag.providers.ollama.request.urlopen", lambda *a,**k: (_ for _ in ()).throw(socket.timeout()))
    with pytest.raises(OllamaTimeoutError):
        UrllibJsonTransport().post_json("http://local/api/embed", {}, 1)

def test_transport_maps_connection_error(monkeypatch):
    monkeypatch.setattr("enterprise_rag.providers.ollama.request.urlopen", lambda *a,**k: (_ for _ in ()).throw(error.URLError("refused")))
    with pytest.raises(OllamaConnectionError):
        UrllibJsonTransport().post_json("http://local/api/embed", {}, 1)

def test_fake_transport_can_surface_http_timeout_and_connection_errors():
    for exception in (OllamaHTTPError("500"), OllamaTimeoutError("timeout"), OllamaConnectionError("refused")):
        client=OllamaGenerationClient(base_url="http://local", model="m", timeout=1, transport=FakeTransport(exception=exception))
        with pytest.raises(type(exception)): client.generate("prompt")
def test_ollama_embedding_rejects_nan_and_infinity():
    for value in (float("nan"), float("inf")):
        client,_=embedding_client({"embeddings":[[value]]})
        with pytest.raises(EmbeddingValidationError, match="NaN or Infinity"):
            client.embed_query("q")

def test_ollama_generation_surfaces_invalid_json_transport_error():
    client=OllamaGenerationClient(
        base_url="http://local", model="m", timeout=1,
        transport=FakeTransport(exception=OllamaResponseError("invalid JSON")),
    )
    with pytest.raises(OllamaResponseError, match="invalid JSON"):
        client.generate("prompt")