"""Small command-line presentation layer for retrieval results."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import TextIO
import sys

from enterprise_rag.retrieval import Retriever

class CLIConfigurationError(RuntimeError):
    """Raised when a command requires an application dependency not yet available."""

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="enterprise-rag")
    subcommands = parser.add_subparsers(dest="command", required=True)
    retrieve = subcommands.add_parser("retrieve", help="Retrieve relevant chunks without generating an answer.")
    retrieve.add_argument("question", nargs="?", help="Question to embed and search.")
    retrieve.add_argument("--top-k", type=int, default=4, help="Maximum number of chunks to return.")
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

def main(argv: Sequence[str] | None = None, *, retriever: Retriever | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "retrieve":
        if retriever is None:
            raise CLIConfigurationError(
                "The retrieve command requires an in-memory Retriever. "
                "Index persistence and pipeline construction are not implemented yet."
            )
        return run_retrieve_command(
            retriever, question=args.question, top_k=args.top_k
        )
    raise CLIConfigurationError(f"Unsupported command: {args.command!r}.")

def run_cli(
    argv: Sequence[str] | None = None,
    *,
    retriever: Retriever | None = None,
    error_output: TextIO = sys.stderr,
) -> int:
    """Run the user-facing CLI without exposing internal tracebacks."""
    try:
        return main(argv, retriever=retriever)
    except CLIConfigurationError as exc:
        print(f"Error: {exc}", file=error_output)
        return 2