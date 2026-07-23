"""Grounded response generation interfaces."""

from typing import Any

from enterprise_rag.config import Settings


def generate_answer(query: str, context: list[Any], settings: Settings) -> str:
    """Generate an answer grounded in retrieved context with citations."""
    # TODO: Build a guarded prompt, call Gemini, and format source/page citations.
    raise NotImplementedError("Answer generation is not implemented yet.")