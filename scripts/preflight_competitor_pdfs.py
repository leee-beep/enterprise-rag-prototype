"""Read-only quality preflight for a manifest of local competitor PDFs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from enterprise_rag.competitor_metadata import (
    CompetitorDocumentMetadata,
    CompetitorMetadataError,
)
from enterprise_rag.pdf_loader import PDFLoadingError, load_competitor_pdf


class PreflightError(RuntimeError):
    """Raised when a preflight manifest or source entry is invalid."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local competitor PDFs without writing documents or indexes."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        entries = _read_manifest(args.manifest)
        source_root = _validate_source_root(args.source_root)
        passed = 0
        failed = 0
        for position, entry in enumerate(entries, start=1):
            try:
                source_file = _safe_manifest_file(entry, position)
                path = (source_root / PurePosixPath(source_file)).resolve()
                if not path.is_relative_to(source_root):
                    raise PreflightError(
                        f"Manifest entry {position} resolves outside the source root."
                    )
                metadata = _metadata_from_entry(path, entry, position)
                result = load_competitor_pdf(path, metadata)
                quality = result.quality
                print(
                    f"PASS {source_file}: pages={quality.page_count}, "
                    f"pages_with_text={quality.pages_with_text}, "
                    f"ratio={quality.extractable_page_ratio:.1%}, "
                    f"characters={quality.total_characters}, "
                    f"replacement_ratio={quality.replacement_character_ratio:.2%}"
                )
                for warning in quality.warnings:
                    print(f"  WARNING: {warning}")
                passed += 1
            except (CompetitorMetadataError, PDFLoadingError, PreflightError) as exc:
                print(f"FAIL entry {position}: {exc}")
                failed += 1
        print(f"Preflight summary: total={len(entries)}, passed={passed}, failed={failed}")
        return 0 if failed == 0 else 2
    except PreflightError as exc:
        print(f"Preflight error: {exc}")
        return 2


def _read_manifest(path: Path) -> list[dict[str, object]]:
    safe_name = path.name or "manifest"
    if not path.is_file():
        raise PreflightError(f"Manifest file does not exist: {safe_name}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise PreflightError(f"Manifest is not valid UTF-8: {safe_name}.") from exc
    except json.JSONDecodeError as exc:
        raise PreflightError(
            f"Manifest is invalid JSON at line {exc.lineno}, column {exc.colno}: "
            f"{safe_name}."
        ) from exc
    if not isinstance(value, list) or not value:
        raise PreflightError("Manifest root must be a non-empty JSON array.")
    if not all(isinstance(item, dict) for item in value):
        raise PreflightError("Every manifest entry must be a JSON object.")
    return value


def _validate_source_root(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise PreflightError(
            f"Source root does not exist or is not a directory: {path.name or 'source root'}."
        )
    return path.resolve()


def _safe_manifest_file(entry: dict[str, object], position: int) -> str:
    value = entry.get("file")
    if not isinstance(value, str) or not value.strip():
        raise PreflightError(f"Manifest entry {position} requires a non-empty 'file'.")
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or path.anchor or ".." in path.parts or ":" in path.parts[0]:
        raise PreflightError(f"Manifest entry {position} file must be a safe relative path.")
    if path.suffix.casefold() != ".pdf":
        raise PreflightError(f"Manifest entry {position} file must end with .pdf.")
    return path.as_posix()


def _metadata_from_entry(
    path: Path, entry: dict[str, object], position: int
) -> CompetitorDocumentMetadata:
    required = (
        "company_id",
        "company_name",
        "ticker",
        "fiscal_year",
        "period",
        "document_type",
        "title",
        "language",
        "source_url",
        "source_relative_path",
    )
    missing = [name for name in required if name not in entry]
    if missing:
        raise PreflightError(
            f"Manifest entry {position} is missing: {', '.join(missing)}."
        )
    return CompetitorDocumentMetadata.from_pdf(
        path,
        company_id=entry["company_id"],
        company_name=entry["company_name"],
        ticker=entry["ticker"],
        fiscal_year=entry["fiscal_year"],
        period=entry["period"],
        document_type=entry["document_type"],
        title=entry["title"],
        language=entry["language"],
        source_url=entry["source_url"],
        source_relative_path=entry["source_relative_path"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
