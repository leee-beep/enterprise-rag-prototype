"""High-level orchestration interfaces for indexing and question answering."""


def build_index() -> None:
    """Run the future document ingestion and indexing workflow."""
    # TODO: Orchestrate loading, chunking, embedding, and FAISS persistence.
    raise NotImplementedError("Index construction is not implemented yet.")


def answer_question(query: str) -> str:
    """Run the future retrieval and grounded-generation workflow."""
    # TODO: Load settings/index, retrieve context, and generate a cited answer.
    raise NotImplementedError("Question answering is not implemented yet.")