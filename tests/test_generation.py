"""Offline tests for generation models, prompt building, providers, and CLI."""
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from enterprise_rag.cli import main, run_generate_command
from enterprise_rag.config import Settings
from enterprise_rag.generation import (
    GenerationValidationError,
    PromptBuilder,
    generate_prompt,
)
from enterprise_rag.models import GenerationResult
from enterprise_rag.providers.gemini import GeminiGenerationClient
from enterprise_rag.providers.ollama import OllamaGenerationClient


class FakeGenerationClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self, answer: str = "offline answer") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class FakeGeminiModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="Gemini offline answer")


class FakeGeminiSdk:
    def __init__(self) -> None:
        self.models = FakeGeminiModels()


class FakeOllamaTransport:
    def __init__(self) -> None:
        self.calls = []

    def post_json(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        return {"response": "Ollama offline answer"}


def settings() -> Settings:
    return Settings(
        gemini_api_key="fake-key",
        generation_model="gemini-test-model",
        embedding_model="embedding-test-model",
        documents_dir=Path("docs"),
        vector_store_dir=Path("vectors"),
        chunk_size=100,
        chunk_overlap=10,
        top_k=4,
    )


def test_generation_result_normalizes_fields():
    result = GenerationResult(answer=" answer ", provider=" gemini ", model=" model ")
    assert (result.answer, result.provider, result.model) == ("answer", "gemini", "model")


def test_prompt_builder_supports_system_context_and_question():
    prompt = PromptBuilder().build(
        system_prompt="Be concise.",
        context="RAG uses retrieved context.",
        question="What is RAG?",
    )
    assert prompt == (
        "System:\nBe concise.\n\n"
        "Context:\nRAG uses retrieved context.\n\n"
        "Question:\nWhat is RAG?"
    )


def test_prompt_builder_omits_optional_empty_sections():
    assert PromptBuilder().build(system_prompt="Rules", context=" ", question=None) == "System:\nRules"


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_is_rejected_before_client_call(prompt):
    client = FakeGenerationClient()
    with pytest.raises(GenerationValidationError, match="prompt"):
        generate_prompt(client, prompt)
    assert client.prompts == []


def test_generate_prompt_records_provider_and_model():
    client = FakeGenerationClient()
    result = generate_prompt(client, " hello ")
    assert result == GenerationResult(
        answer="offline answer", provider="fake", model="fake-model"
    )
    assert client.prompts == ["hello"]


def test_mock_gemini_generation_is_offline():
    sdk = FakeGeminiSdk()
    client = GeminiGenerationClient(settings(), sdk_client=sdk)
    result = generate_prompt(client, "prompt")
    assert result.provider == "gemini"
    assert result.model == "gemini-test-model"
    assert result.answer == "Gemini offline answer"
    assert sdk.models.calls == [
        {"model": "gemini-test-model", "contents": "prompt"}
    ]


def test_mock_ollama_generation_is_offline():
    transport = FakeOllamaTransport()
    client = OllamaGenerationClient(
        base_url="http://offline.invalid",
        model="ollama-test-model",
        timeout=1,
        transport=transport,
    )
    result = generate_prompt(client, "prompt")
    assert (result.provider, result.model, result.answer) == (
        "ollama",
        "ollama-test-model",
        "Ollama offline answer",
    )
    assert transport.calls == [
        (
            "http://offline.invalid/api/generate",
            {"model": "ollama-test-model", "prompt": "prompt", "stream": False},
            1,
        )
    ]


def test_generate_cli_prints_answer_without_retrieval():
    client = FakeGenerationClient("CLI answer")
    output = StringIO()
    assert run_generate_command(client, prompt="prompt", output=output) == 0
    assert output.getvalue() == "CLI answer\n"
    assert client.prompts == ["prompt"]


def test_generate_cli_prompts_and_supports_injected_client():
    client = FakeGenerationClient()
    output = StringIO()
    prompts = []
    run_generate_command(
        client,
        prompt=None,
        input_fn=lambda label: prompts.append(label) or "interactive",
        output=output,
    )
    assert prompts == ["Prompt: "]
    assert client.prompts == ["interactive"]
    assert main(["generate", "direct"], generation_client=client) == 0
    assert client.prompts[-1] == "direct"
