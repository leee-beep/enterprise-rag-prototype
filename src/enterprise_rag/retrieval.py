"""Semantic retrieval interfaces."""

from typing import Any


def retrieve_context(vector_store: Any, query: str, *, top_k: int = 4) -> list[Any]:
    """Retrieve the most relevant chunks for a query."""
    # TODO: Add explicit top-k, score thresholds, and empty-result handling.
    raise NotImplementedError("Semantic retrieval is not implemented yet.")