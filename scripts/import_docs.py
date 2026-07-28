"""Command-line entry point for local LangChain documentation import."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from enterprise_rag.document_import import (
    DocumentImportError,
    ImportConfig,
    LangChainDocumentImporter,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import selected local LangChain Markdown/MDX documentation."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("../langchain-docs-source/src"),
        help="Source documentation directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw_documents/langchain"),
        help="Destination directory for selected documents.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report matches without writing files or a manifest.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="List each selected destination path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = LangChainDocumentImporter(
            ImportConfig(
                source=args.source,
                output=args.output,
                dry_run=args.dry_run,
            )
        ).run()
    except DocumentImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.verbose or args.dry_run:
        label = "Would import" if args.dry_run else "Imported"
        for path in result.output_paths:
            print(f"{label}: {path}")
    print(f"Scanned: {result.scanned_count}")
    print(f"Matched: {result.matched_count}")
    print(f"Copied: {result.copied_count}")
    print(f"Skipped: {result.skipped_count}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
