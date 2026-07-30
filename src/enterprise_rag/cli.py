"""Small command-line presentation layer for retrieval results."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO
import sys

from enterprise_rag.config import ConfigurationError, load_settings
from enterprise_rag.embeddings import EmbeddingError
from enterprise_rag.factory import create_embedding_client, create_generation_client
from enterprise_rag.indexing import (
    IndexingError,
    IndexingService,
    validate_index_compatibility,
)
from enterprise_rag.retrieval import QueryEmbeddingClient, Retriever
from enterprise_rag.generation import GenerationClient, generate_prompt
from enterprise_rag.pipeline import RAGPipeline
from enterprise_rag.vector_store import FaissVectorStore, VectorStoreError

class CLIConfigurationError(RuntimeError):
    """Raised when a command requires an application dependency not yet available."""

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="enterprise-rag")
    subcommands = parser.add_subparsers(dest="command", required=True)
    retrieve = subcommands.add_parser("retrieve", help="Retrieve relevant chunks without generating an answer.")
    retrieve.add_argument("question", nargs="?", help="Question to embed and search.")
    retrieve.add_argument("--top-k", type=int, default=4, help="Maximum number of chunks to return.")
    generate = subcommands.add_parser("generate", help="Generate text from a prompt without retrieval.")
    generate.add_argument("prompt", nargs="?", help="Prompt to send to the configured generation client.")
    ask = subcommands.add_parser("ask", help="Answer a question using an injected RAG pipeline.")
    ask.add_argument("question", nargs="?", help="Question to retrieve and answer.")
    ask.add_argument("--index-path", type=Path, help="Load an existing persisted FAISS index.")
    ask.add_argument("--top-k", type=int, default=4, help="Maximum number of chunks to retrieve.")
    build_index = subcommands.add_parser(
        "build-index",
        help="Load documents, create embeddings, and persist a FAISS index.",
    )
    build_index.add_argument(
        "--input",
        type=Path,
        help="Document directory (default: DOCUMENTS_DIR).",
    )
    build_index.add_argument(
        "--output",
        type=Path,
        help="Index directory (default: VECTOR_STORE_DIR).",
    )
    build_index.add_argument(
        "--index-path",
        dest="legacy_index_path",
        type=Path,
        help="Legacy injected-builder destination; use --output for standalone indexing.",
    )
    build_index.add_argument(
        "--overwrite",
        action="store_true",
        help="Safely replace an existing output directory.",
    )
    build_index.add_argument(
        "--verbose",
        action="store_true",
        help="Show indexing progress.",
    )
    return parser

def run_retrieve_command(
    retriever: Retriever,
    *,
    question: str | None,
    top_k: int,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> int:
    question = question if question is not None else input_fn("Question: ")
    results = retriever.retrieve(question, top_k)
    if not results:
        print("No retrieval results.", file=output)
        return 0
    print(f"Top {len(results)} retrieval results", file=output)
    for rank, result in enumerate(results, start=1):
        chunk = result.embedded_chunk.chunk
        print(f"\n[{rank}] Score: {result.score:.6f}", file=output)
        print(f"Source: {chunk.source}", file=output)
        print(f"Chunk: {chunk.content}", file=output)
    return 0

def run_generate_command(
    client: GenerationClient,
    *,
    prompt: str | None,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> int:
    prompt = prompt if prompt is not None else input_fn("Prompt: ")
    result = generate_prompt(client, prompt)
    print(result.answer, file=output)
    return 0

def run_ask_command(
    pipeline: RAGPipeline,
    *,
    question: str | None,
    input_fn: Callable[[str], str] = input,
    output: TextIO | None = None,
) -> int:
    output = sys.stdout if output is None else output
    question = question if question is not None else input_fn("Question: ")
    result = pipeline.answer(question)
    print(result.answer, file=output)
    return 0

def run_build_index_command(
    index_builder: Callable[[], FaissVectorStore],
    *,
    index_path: Path,
    output: TextIO = sys.stdout,
) -> int:
    vector_store = index_builder()
    vector_store.save(index_path)
    print(
        f"Saved {vector_store.size} chunks to {index_path}.",
        file=output,
    )
    return 0


def run_indexing_command(
    service: IndexingService,
    *,
    input_directory: Path | None,
    output_directory: Path | None,
    overwrite: bool,
    verbose: bool,
    output: TextIO = sys.stdout,
) -> int:
    """Run the standalone indexing workflow and print its result summary."""
    progress = (lambda message: print(message, file=output)) if verbose else None
    result = service.build(
        input_directory=input_directory,
        output_directory=output_directory,
        overwrite=overwrite,
        progress=progress,
    )
    print(f"Documents loaded: {result.loaded_document_count}", file=output)
    print(f"Chunks created: {result.chunk_count}", file=output)
    print(f"Vectors embedded: {result.embedded_vector_count}", file=output)
    print(f"Embedding dimension: {result.embedding_dimension}", file=output)
    print(f"Index saved to: {result.output_path}", file=output)
    return 0

def main(
    argv: Sequence[str] | None = None,
    *,
    retriever: Retriever | None = None,
    generation_client: GenerationClient | None = None,
    pipeline: RAGPipeline | None = None,
    query_embedding_client: QueryEmbeddingClient | None = None,
    index_builder: Callable[[], FaissVectorStore] | None = None,
    indexing_service: IndexingService | None = None,
    output: TextIO | None = None,
) -> int:
    output = sys.stdout if output is None else output
    args = build_parser().parse_args(argv)
    if args.command == "retrieve":
        if retriever is None:
            raise CLIConfigurationError(
                "The retrieve command requires an in-memory Retriever. "
                "Index persistence and pipeline construction are not implemented yet."
            )
        return run_retrieve_command(
            retriever,
            question=args.question,
            top_k=args.top_k,
            output=output,
        )
    if args.command == "generate":
        if generation_client is None:
            generation_client = create_generation_client(load_settings())
        return run_generate_command(
            generation_client,
            prompt=args.prompt,
            output=output,
        )
    if args.command == "ask":
        if pipeline is None:
            if args.index_path is None:
                raise CLIConfigurationError(
                    "The ask command requires an injected RAGPipeline with an existing "
                    "Retriever, or --index-path for a saved index. The CLI does not "
                    "automatically rebuild an index."
                )
            settings = load_settings()
            query_embedding_client = (
                query_embedding_client or create_embedding_client(settings)
            )
            generation_client = (
                generation_client or create_generation_client(settings)
            )
            validate_index_compatibility(args.index_path, settings)
            vector_store = FaissVectorStore.load(args.index_path)
            pipeline = RAGPipeline(
                Retriever(query_embedding_client, vector_store),
                generation_client,
                top_k=args.top_k,
            )
        return run_ask_command(
            pipeline,
            question=args.question,
            output=output,
        )
    if args.command == "build-index":
        if index_builder is not None:
            index_path = args.output or args.legacy_index_path
            if index_path is None:
                index_path = load_settings().vector_store_dir
            return run_build_index_command(
                index_builder,
                index_path=index_path,
                output=output,
            )
        if args.legacy_index_path is not None:
            raise CLIConfigurationError(
                "The legacy build-index --index-path form requires an injected "
                "index builder. Use --output for the standalone indexing workflow."
            )
        service = indexing_service
        if service is None:
            service = IndexingService(load_settings())
        return run_indexing_command(
            service,
            input_directory=args.input,
            output_directory=args.output,
            overwrite=args.overwrite,
            verbose=args.verbose,
            output=output,
        )
    raise CLIConfigurationError(f"Unsupported command: {args.command!r}.")

def run_cli(
    argv: Sequence[str] | None = None,
    *,
    retriever: Retriever | None = None,
    generation_client: GenerationClient | None = None,
    pipeline: RAGPipeline | None = None,
    query_embedding_client: QueryEmbeddingClient | None = None,
    index_builder: Callable[[], FaissVectorStore] | None = None,
    indexing_service: IndexingService | None = None,
    output: TextIO | None = None,
    error_output: TextIO = sys.stderr,
) -> int:
    """Run the user-facing CLI without exposing internal tracebacks."""
    try:
        return main(
            argv,
            retriever=retriever,
            generation_client=generation_client,
            pipeline=pipeline,
            query_embedding_client=query_embedding_client,
            index_builder=index_builder,
            indexing_service=indexing_service,
            output=output,
        )
    except (
        CLIConfigurationError,
        ConfigurationError,
        IndexingError,
        EmbeddingError,
        VectorStoreError,
    ) as exc:
        print(f"Error: {exc}", file=error_output)
        return 2
