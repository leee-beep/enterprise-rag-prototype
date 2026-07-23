from pathlib import Path
from types import SimpleNamespace
import pytest

from enterprise_rag.config import ConfigurationError, Settings
from enterprise_rag.generation import GenerationValidationError
from enterprise_rag.providers.gemini import GeminiEmbeddingClient, GeminiGenerationClient

def settings(key="fake-key"):
    return Settings(
        gemini_api_key=key, generation_model="chat-model", embedding_model="embed-model",
        documents_dir=Path("docs"), vector_store_dir=Path("vectors"),
        chunk_size=100, chunk_overlap=10, top_k=4,
    )

class FakeModels:
    def __init__(self, response): self.response=response; self.calls=[]
    def generate_content(self, **kwargs): self.calls.append(kwargs); return self.response

class FakeClient:
    def __init__(self, response): self.models=FakeModels(response)

def test_gemini_generation_returns_text_with_mock_sdk():
    sdk=FakeClient(SimpleNamespace(text=" answer "))
    client=GeminiGenerationClient(settings(), sdk_client=sdk)
    assert client.generate("prompt") == "answer"
    assert sdk.models.calls == [{"model":"chat-model", "contents":"prompt"}]

@pytest.mark.parametrize("response", [SimpleNamespace(text=""), SimpleNamespace()])
def test_gemini_generation_rejects_missing_or_empty_text(response):
    with pytest.raises(GenerationValidationError):
        GeminiGenerationClient(settings(), sdk_client=FakeClient(response)).generate("prompt")

def test_gemini_generation_validates_key_only_on_generate():
    client=GeminiGenerationClient(settings(None), sdk_client=FakeClient(SimpleNamespace(text="unused")))
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        client.generate("prompt")
class FakeEmbeddingModels:
    def __init__(self, vectors): self.vectors=vectors; self.calls=[]
    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in self.vectors])

class FakeEmbeddingSdk:
    def __init__(self, vectors): self.models=FakeEmbeddingModels(vectors)

def test_gemini_embedding_batch_and_query_with_mock_sdk():
    sdk=FakeEmbeddingSdk([[1,2],[3,4]])
    client=GeminiEmbeddingClient(settings(), sdk_client=sdk)
    assert client.embed_documents(["a","b"]) == ((1.0,2.0),(3.0,4.0))
    query_sdk=FakeEmbeddingSdk([[5,6]])
    assert GeminiEmbeddingClient(settings(), sdk_client=query_sdk).embed_query("q") == (5.0,6.0)

def test_gemini_embedding_validates_key_only_on_embed():
    client=GeminiEmbeddingClient(settings(None), sdk_client=FakeEmbeddingSdk([[1,2]]))
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        client.embed_query("q")