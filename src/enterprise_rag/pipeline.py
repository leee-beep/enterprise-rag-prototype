"""High-level orchestration interfaces for indexing and question answering."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from enterprise_rag.generation import GenerationClient, PromptBuilder, generate_prompt
from enterprise_rag.models import GenerationResult, RetrievalResult

DEFAULT_SYSTEM_PROMPT = (
    "Answer the question using the supplied context. "
    "If the context is insufficient, say that you do not have enough information."
)


class ContextFormatter:
    """Convert ranked retrieval results into provider-neutral prompt context."""

    def format(self, results: Sequence[RetrievalResult]) -> str:
        sections: list[str] = []
        for rank, result in enumerate(results, start=1):
            chunk = result.embedded_chunk.chunk
            sections.append(
                f"[Context {rank}]\n"
                f"Source: {chunk.source}\n"
                f"Score: {result.score:.6f}\n"
                f"Content:\n{chunk.content}"
            )
        return "\n\n".join(sections)


class RetrievalClient(Protocol):
    def retrieve(
        self, question: str, top_k: int
    ) -> Sequence[RetrievalResult]: ...


class RAGPipeline:
    """Orchestrate retrieval, context formatting, prompting, and generation."""

    def __init__(
        self,
        retriever: RetrievalClient,
        generation_client: GenerationClient,
        *,
        top_k: int = 4,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        context_formatter: ContextFormatter | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        self._retriever = retriever
        self._generation_client = generation_client
        self._top_k = top_k
        self._system_prompt = system_prompt
        self._context_formatter = context_formatter or ContextFormatter()
        self._prompt_builder = prompt_builder or PromptBuilder()

    def answer(self, question: str) -> GenerationResult:
        results = self._retriever.retrieve(question, self._top_k)
        context = self._context_formatter.format(results)
        prompt = self._prompt_builder.build(
            system_prompt=self._system_prompt,
            context=context or None,
            question=question,
        )
        return generate_prompt(self._generation_client, prompt)


def build_index() -> None:
    """Run the future document ingestion and indexing workflow."""
    # TODO: Orchestrate loading, chunking, embedding, and FAISS persistence.
    raise NotImplementedError("Index construction is not implemented yet.")


def answer_question(query: str) -> str:
    """Legacy placeholder; use an explicitly constructed ``RAGPipeline``."""
    raise NotImplementedError(
        "Construct RAGPipeline with an existing Retriever and GenerationClient."
    )
