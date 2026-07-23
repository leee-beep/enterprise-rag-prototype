"""Provider-neutral text generation interfaces."""
from __future__ import annotations

from typing import Any, Protocol

from enterprise_rag.config import Settings

class GenerationError(RuntimeError):
    """Base error for generation operations."""

class GenerationValidationError(GenerationError):
    """Raised when a provider returns no usable text."""

class GenerationClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate one non-streaming text response for a prompt."""
        ...

def validate_generated_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GenerationValidationError("Generation response text is missing or empty.")
    return value.strip()

def generate_answer(query: str, context: list[Any], settings: Settings) -> str:
    """Future RAG orchestration placeholder; provider generation uses GenerationClient."""
    raise NotImplementedError("Answer generation is not implemented yet.")