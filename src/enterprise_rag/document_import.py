"""Deterministic local import of selected LangChain Markdown documentation."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile

SUPPORTED_EXTENSIONS = frozenset({".md", ".mdx"})
INCLUDE_KEYWORDS = (
    "rag",
    "retrieval",
    "retriever",
    "embedding",
    "embeddings",
    "vector store",
    "vectorstore",
    "vector_store",
    "text splitter",
    "text splitting",
    "splitter",
    "semantic search",
    "knowledge base",
    "document loader",
    "document loading",
)
EXCLUDE_PATH_KEYWORDS = (
    "javascript",
    "js",
    "migration",
    "migrate",
    "release",
    "releases",
    "changelog",
    "deprecated",
    "deprecation",
    "legacy",
    "image",
    "images",
    "font",
    "fonts",
    "code-samples",
    "node_modules",
    ".git",
)
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILE_NAME = "manifest.json"
CONTENT_SCAN_CHARS = 16_384


class DocumentImportError(RuntimeError):
    """Base error for document import operations."""


class ImportPathError(DocumentImportError):
    """Raised when source or output paths are missing, invalid, or unsafe."""


class DocumentImportIOError(DocumentImportError):
    """Raised when an import file cannot be read or written."""


class NoMatchingDocumentsError(DocumentImportError):
    """Raised when the source contains no documentation matching the rules."""


@dataclass(frozen=True)
class ImportConfig:
    """Paths and behavior for one local document import."""

    source: Path
    output: Path
    source_name: str = "langchain"
    dry_run: bool = False
    include_keywords: tuple[str, ...] = INCLUDE_KEYWORDS
    exclude_path_keywords: tuple[str, ...] = EXCLUDE_PATH_KEYWORDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "output", Path(self.output))
        if not self.source_name.strip():
            raise ImportPathError("source_name must not be empty.")
        if not self.include_keywords:
            raise ImportPathError("At least one include keyword is required.")


@dataclass(frozen=True)
class ImportResult:
    """Summary of scanned, selected, copied, and unchanged documents."""

    scanned_count: int
    matched_count: int
    copied_count: int
    skipped_count: int
    output_paths: tuple[Path, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _MatchedFile:
    source_path: Path
    source_relative_path: Path
    destination_path: Path
    sha256: str
    size_bytes: int


class LangChainDocumentImporter:
    """Select and copy RAG-related local Markdown/MDX files."""

    def __init__(
        self,
        config: ImportConfig,
        *,
        now_utc: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))

    def run(self) -> ImportResult:
        source, output = validate_import_paths(
            self.config.source, self.config.output
        )
        candidates = scan_markdown_files(source)
        matched = tuple(
            self._match_file(path, source, output)
            for path in candidates
            if not is_excluded_path(
                path.relative_to(source), self.config.exclude_path_keywords
            )
            and file_matches_include(
                path,
                path.relative_to(source),
                self.config.include_keywords,
            )
        )
        if not matched:
            raise NoMatchingDocumentsError(
                f"No Markdown or MDX documents matched the import rules in '{source}'."
            )

        copied_count = 0
        skipped_count = 0
        if not self.config.dry_run:
            for item in matched:
                if _same_file_content(item.destination_path, item.sha256):
                    skipped_count += 1
                    continue
                _copy_import_file(item.source_path, item.destination_path)
                copied_count += 1
            self._write_manifest(
                source,
                output,
                candidates_count=len(candidates),
                matched=matched,
                copied_count=copied_count,
                skipped_count=skipped_count,
            )

        return ImportResult(
            scanned_count=len(candidates),
            matched_count=len(matched),
            copied_count=copied_count,
            skipped_count=skipped_count,
            output_paths=tuple(item.destination_path for item in matched),
        )

    def _match_file(
        self, path: Path, source: Path, output: Path
    ) -> _MatchedFile:
        relative = path.relative_to(source)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise DocumentImportIOError(
                f"Could not read source document '{path}'."
            ) from exc
        return _MatchedFile(
            source_path=path,
            source_relative_path=relative,
            destination_path=output / relative,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )

    def _write_manifest(
        self,
        source: Path,
        output: Path,
        *,
        candidates_count: int,
        matched: Sequence[_MatchedFile],
        copied_count: int,
        skipped_count: int,
    ) -> None:
        generated_at = self._now_utc()
        if generated_at.tzinfo is None:
            raise DocumentImportError(
                "Manifest timestamp provider must return a timezone-aware datetime."
            )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_name": self.config.source_name.strip(),
            "source_root": describe_source_root(self.config.source),
            "generated_at": generated_at.astimezone(timezone.utc).isoformat(),
            "include_keywords": list(self.config.include_keywords),
            "exclude_path_keywords": list(self.config.exclude_path_keywords),
            "scanned_count": candidates_count,
            "matched_count": len(matched),
            "copied_count": copied_count,
            "skipped_count": skipped_count,
            "imported_files": [
                {
                    "source_relative_path": item.source_relative_path.as_posix(),
                    "destination_relative_path": item.source_relative_path.as_posix(),
                    "extension": item.source_path.suffix.casefold(),
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in matched
            ],
        }
        try:
            output.mkdir(parents=True, exist_ok=True)
            (output / MANIFEST_FILE_NAME).write_text(
                json.dumps(
                    manifest, ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise DocumentImportIOError(
                f"Could not write import manifest to '{output}'."
            ) from exc


def validate_import_paths(source: Path, output: Path) -> tuple[Path, Path]:
    """Resolve and validate source/output paths without creating directories."""
    source = Path(source)
    output = Path(output)
    if not source.exists():
        raise ImportPathError(f"Source directory does not exist: '{source}'.")
    if not source.is_dir():
        raise ImportPathError(f"Source path is not a directory: '{source}'.")
    try:
        resolved_source = source.resolve(strict=True)
        resolved_output = output.resolve(strict=False)
    except OSError as exc:
        raise ImportPathError("Could not resolve source or output path.") from exc
    if resolved_output == resolved_source:
        raise ImportPathError("Output directory must not be the source directory.")
    if resolved_output.is_relative_to(resolved_source):
        raise ImportPathError(
            "Output directory must not be inside the source directory."
        )
    if resolved_source.is_relative_to(resolved_output):
        raise ImportPathError(
            "Output directory must not contain the source directory."
        )
    return resolved_source, resolved_output


def scan_markdown_files(source: Path) -> tuple[Path, ...]:
    """Return supported files in deterministic relative-path order."""
    try:
        paths = tuple(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        )
        return tuple(
            sorted(
                paths,
                key=lambda path: (
                    path.relative_to(source).as_posix().casefold(),
                    path.relative_to(source).as_posix(),
                ),
            )
        )
    except OSError as exc:
        raise DocumentImportIOError(
            f"Could not scan source directory '{source}'."
        ) from exc


def is_excluded_path(
    relative_path: Path,
    exclude_keywords: Sequence[str] = EXCLUDE_PATH_KEYWORDS,
) -> bool:
    """Match exclusions against path segments, with exact matching for ``js``."""
    segments = tuple(part.casefold() for part in relative_path.parts)
    for keyword in exclude_keywords:
        normalized = keyword.casefold()
        if normalized in {"js", ".git", "node_modules"}:
            if normalized in segments:
                return True
        elif any(normalized in segment for segment in segments):
            return True
    return False


def file_matches_include(
    path: Path,
    relative_path: Path,
    include_keywords: Sequence[str] = INCLUDE_KEYWORDS,
) -> bool:
    """Match keywords case-insensitively in the path and initial file content."""
    searchable_path = relative_path.as_posix().casefold()
    normalized_keywords = tuple(keyword.casefold() for keyword in include_keywords)
    if any(keyword in searchable_path for keyword in normalized_keywords):
        return True
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            prefix = stream.read(CONTENT_SCAN_CHARS).casefold()
    except (OSError, UnicodeDecodeError) as exc:
        raise DocumentImportIOError(
            f"Could not read UTF-8 source document '{path}'."
        ) from exc
    return any(keyword in prefix for keyword in normalized_keywords)


def describe_source_root(source: Path) -> str:
    """Return a useful source description without exposing an absolute user path."""
    source = Path(source)
    if source.is_absolute():
        return source.name
    return source.as_posix()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(65_536), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise DocumentImportIOError(f"Could not read file '{path}'.") from exc


def _same_file_content(destination: Path, source_sha256: str) -> bool:
    return destination.is_file() and sha256_file(destination) == source_sha256


def _copy_import_file(source: Path, destination: Path) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)
    except OSError as exc:
        raise DocumentImportIOError(
            f"Could not copy '{source}' to '{destination}'."
        ) from exc
