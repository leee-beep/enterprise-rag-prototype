"""Offline tests for one-shot RAG orchestration and its CLI boundary."""
from __future__ import annotations

from io import StringIO

import pytest

from enterprise_rag.cli import CLIConfigurationError, main, run_ask_command, run_cli
from enterprise_rag.generation import PromptBuilder
from enterprise_rag.models import (
    DocumentChunk,
    EmbeddedChunk,
    GenerationResult,
    RetrievalResult,
)
from enterprise_rag.pipeline import ContextFormatter, RAGPipeline


def retrieval_result(index: int, *, source: str, content: str, score: float):
    chunk = DocumentChunk(
        content=content,
        source=source,
        file_name=source.split("/")[-1],
        file_type=".md",
        document_id=f"doc-{index}",
        chunk_index=index,
        chunk_id=f"doc-{index}:chunk-{index:06d}",
    )
    embedded = EmbeddedChunk(
        chunk=chunk,
        vector=(float(index), 0.0),
        embedding_model="fake-embedding",
    )
    return RetrievalResult(score=score, embedded_chunk=embedded)


class FakeRetriever:
    def __init__(self, results=(), events=None):
        self.results = tuple(results)
        self.calls = []
        self.events = events

    def retrieve(self, question, top_k):
        self.calls.append((question, top_k))
        if self.events is not None:
            self.events.append("retrieve")
        return self.results


class FakeGenerationClient:
    provider = "fake"
    model = "fake-generation"

    def __init__(self, answer="grounded answer", events=None):
        self.answer = answer
        self.prompts = []
        self.events = events

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.events is not None:
            self.events.append("generate")
        return self.answer


class RecordingFormatter(ContextFormatter):
    def __init__(self, events):
        self.events = events

    def format(self, results):
        self.events.append("format")
        return super().format(results)


class RecordingPromptBuilder(PromptBuilder):
    def __init__(self, events):
        self.events = events

    def build(self, **kwargs):
        self.events.append("build")
        return super().build(**kwargs)


def test_context_formatter_keeps_rank_source_score_and_content():
    results = (
        retrieval_result(0, source="guide.md", content="RAG basics", score=0.9),
        retrieval_result(1, source="advanced/search.md", content="Search notes", score=0.5),
    )
    assert ContextFormatter().format(results) == (
        "[Context 1]\nSource: guide.md\nScore: 0.900000\nContent:\nRAG basics\n\n"
        "[Context 2]\nSource: advanced/search.md\nScore: 0.500000\n"
        "Content:\nSearch notes"
    )


def test_pipeline_retrieves_formats_builds_and_generates_in_order():
    events = []
    result = retrieval_result(0, source="guide.md", content="RAG context", score=1.0)
    retriever = FakeRetriever([result], events)
    generator = FakeGenerationClient(events=events)
    pipeline = RAGPipeline(
        retriever,
        generator,
        top_k=2,
        system_prompt="Use context.",
        context_formatter=RecordingFormatter(events),
        prompt_builder=RecordingPromptBuilder(events),
    )

    answer = pipeline.answer("What is RAG?")

    assert events == ["retrieve", "format", "build", "generate"]
    assert retriever.calls == [("What is RAG?", 2)]
    assert "Context:\n[Context 1]" in generator.prompts[0]
    assert "Question:\nWhat is RAG?" in generator.prompts[0]
    assert answer == GenerationResult(
        answer="grounded answer", provider="fake", model="fake-generation"
    )


def test_empty_retrieval_omits_context_and_still_generates():
    retriever = FakeRetriever()
    generator = FakeGenerationClient("no context answer")
    result = RAGPipeline(
        retriever, generator, system_prompt="Be honest."
    ).answer("Unknown?")
    assert result.answer == "no context answer"
    assert generator.prompts == ["System:\nBe honest.\n\nQuestion:\nUnknown?"]


def test_pipeline_validates_top_k():
    with pytest.raises(ValueError, match="top_k"):
        RAGPipeline(FakeRetriever(), FakeGenerationClient(), top_k=0)


def test_ask_cli_outputs_answer_and_supports_interactive_question():
    retriever = FakeRetriever()
    generator = FakeGenerationClient("CLI answer")
    pipeline = RAGPipeline(retriever, generator)
    output = StringIO()
    prompts = []
    assert run_ask_command(
        pipeline,
        question=None,
        input_fn=lambda label: prompts.append(label) or "Question",
        output=output,
    ) == 0
    assert prompts == ["Question: "]
    assert output.getvalue() == "CLI answer\n"
    assert retriever.calls == [("Question", 4)]


def test_ask_cli_requires_injected_pipeline_with_retriever():
    with pytest.raises(CLIConfigurationError, match="existing Retriever"):
        main(["ask", "Question"])
    error_output = StringIO()
    assert run_cli(["ask", "Question"], error_output=error_output) == 2
    assert "existing Retriever" in error_output.getvalue()
    assert "Traceback" not in error_output.getvalue()


def test_main_ask_uses_injected_pipeline_without_provider_network():
    pipeline = RAGPipeline(FakeRetriever(), FakeGenerationClient("offline"))
    assert main(["ask", "Question"], pipeline=pipeline) == 0
