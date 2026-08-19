"""Provider-neutral text generation interfaces."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from enterprise_rag.config import Settings
from enterprise_rag.models import GenerationResult

class GenerationError(RuntimeError):
    """Base error for generation operations."""

class GenerationValidationError(GenerationError):
    """Raised when a provider returns no usable text."""

class GenerationClient(Protocol):
    @property
    def provider(self) -> str:
        """Stable provider identifier used in generation results."""
        ...

    @property
    def model(self) -> str:
        """Model selected for this client."""
        ...

    def generate(self, prompt: str) -> str:
        """Generate one non-streaming text response for a prompt."""
        ...


@runtime_checkable
class StructuredGenerationClient(GenerationClient, Protocol):
    """Optional provider capability for schema-constrained JSON generation."""

    def generate_structured(
        self, prompt: str, schema: Mapping[str, Any]
    ) -> str:
        """Generate one response constrained by a JSON schema."""
        ...

def validate_generation_prompt(prompt: object) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise GenerationValidationError("Generation prompt must not be empty.")
    return prompt.strip()

def validate_generated_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GenerationValidationError("Generation response text is missing or empty.")
    return value.strip()

class PromptBuilder:
    """Build a deterministic plain-text prompt without retrieval orchestration."""

    def build(
        self,
        *,
        system_prompt: str,
        context: str | None = None,
        question: str | None = None,
    ) -> str:
        system = validate_generation_prompt(system_prompt)
        sections = [f"System:\n{system}"]
        if context is not None and context.strip():
            sections.append(f"Context:\n{context.strip()}")
        if question is not None and question.strip():
            sections.append(f"Question:\n{question.strip()}")
        return "\n\n".join(sections)

def generate_prompt(client: GenerationClient, prompt: str) -> GenerationResult:
    """Generate provider-level text and attach the selected provider and model."""
    normalized_prompt = validate_generation_prompt(prompt)
    answer = client.generate(normalized_prompt)
    return GenerationResult(
        answer=validate_generated_text(answer),
        provider=client.provider,
        model=client.model,
    )

def generate_answer(query: str, context: list[Any], settings: Settings) -> str:
    """Future RAG orchestration placeholder; provider generation uses GenerationClient."""
    raise NotImplementedError("Answer generation is not implemented yet.")
