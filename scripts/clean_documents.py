"""CLI for conservative raw Markdown/MDX cleaning."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from enterprise_rag.document_cleaning import (
    CleaningConfig,
    DocumentCleaningError,
    MarkdownDocumentCleaner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean raw Markdown/MDX into ingestion-ready Markdown."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw_documents/langchain"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/documents/langchain"),
    )
    parser.add_argument("--source-name", default="langchain")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = MarkdownDocumentCleaner(
            CleaningConfig(
                input_dir=args.input,
                output_dir=args.output,
                source_name=args.source_name,
                dry_run=args.dry_run,
            )
        ).run()
    except DocumentCleaningError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if args.verbose or args.dry_run:
        for item in result.files:
            print(
                f"{item.status}: {item.source_relative_path} "
                f"-> {item.destination_relative_path}"
            )
        for warning in result.warnings:
            print(f"WARNING: {warning}")
    print(f"Scanned: {result.scanned_count}")
    print(f"Cleaned: {result.cleaned_count}")
    print(f"Skipped: {result.skipped_count}")
    print(f"Warnings: {result.warning_count}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
