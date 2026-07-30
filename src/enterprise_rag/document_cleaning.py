"""Conservative, deterministic MDX-to-Markdown document cleaning."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from enterprise_rag.document_import import describe_source_root

CLEANING_SCHEMA_VERSION = 2
CLEANING_MANIFEST_NAME = "cleaning_manifest.json"
SUPPORTED_EXTENSIONS = frozenset({".md", ".mdx"})
FRONT_MATTER_KEYS = frozenset(
    {"title", "description", "source", "slug", "tags"}
)
TAG_PATTERN = re.compile(
    r"<(?P<closing>/)?(?P<name>[A-Za-z][\w.-]*)(?P<attrs>[^<>]*?)(?P<self>/)?>"
)
ATTRIBUTE_PATTERN = re.compile(
    r"([\w:-]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|\{[\"']([^\"']*)[\"']\})"
)
FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
H1_PATTERN = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
MODULE_PATTERN = re.compile(r"^(import\s+|export\s+(?:const|let|var|default)\b)")
ADMONITIONS = frozenset(
    {"note", "tip", "warning", "caution", "info", "danger", "admonition"}
)
OUTER_COMPONENTS = frozenset(
    {
        "tabs",
        "accordiongroup",
        "cardgroup",
        "steps",
        "codegroup",
        "frame",
        "figure",
    }
)


class DocumentCleaningError(RuntimeError):
    """Base error for document-cleaning operations."""


class CleaningPathError(DocumentCleaningError):
    """Raised when input/output paths are invalid or unsafe."""


class CleaningReadError(DocumentCleaningError):
    """Raised when a source file cannot be decoded or read."""


class CleaningWriteError(DocumentCleaningError):
    """Raised when an output or manifest cannot be written atomically."""


class CleaningSyntaxError(DocumentCleaningError):
    """Raised for an unsafe-to-continue source structure."""


class CleaningBatchError(DocumentCleaningError):
    """Raised after strict-mode preflight collects all invalid sources."""


class DestinationCollisionError(DocumentCleaningError):
    """Raised when multiple sources map to the same Markdown output."""


class NoCleanDocumentsError(DocumentCleaningError):
    """Raised when no source produces meaningful Markdown."""


@dataclass(frozen=True)
class CleaningConfig:
    """Input, output, and source identity for one cleaning run."""

    input_dir: Path
    output_dir: Path
    source_name: str = "langchain"
    dry_run: bool = False
    strict: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dir", Path(self.input_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.source_name.strip():
            raise CleaningPathError("source_name must not be empty.")


@dataclass(frozen=True)
class CleanedFileResult:
    """Cleaning and synchronization outcome for one source file."""

    source_relative_path: str
    destination_relative_path: str
    input_extension: str
    source_sha256: str
    output_sha256: str
    source_size_bytes: int
    output_size_bytes: int
    title: str
    front_matter: dict[str, object]
    transformations: tuple[str, ...]
    warnings: tuple[str, ...]
    status: str
    error: str | None = None


@dataclass(frozen=True)
class CleaningResult:
    """Aggregate result for a deterministic cleaning run."""

    scanned_count: int
    cleaned_count: int
    skipped_count: int
    warning_count: int
    output_paths: tuple[Path, ...]
    warnings: tuple[str, ...]
    files: tuple[CleanedFileResult, ...]
    invalid_count: int = 0
    empty_count: int = 0
    error_count: int = 0


@dataclass(frozen=True)
class CleanedMarkdown:
    """In-memory output of the pure cleaning pipeline."""

    text: str
    title: str
    front_matter: dict[str, object]
    transformations: tuple[str, ...]
    warnings: tuple[str, ...]


class MarkdownDocumentCleaner:
    """Clean a local Markdown/MDX tree into ingestion-ready Markdown."""

    def __init__(
        self,
        config: CleaningConfig,
        *,
        now_utc: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))

    def run(self) -> CleaningResult:
        input_root, output_root = validate_cleaning_paths(
            self.config.input_dir, self.config.output_dir
        )
        sources, discovery_warnings = discover_source_documents(input_root)
        destinations = validate_destinations(sources, input_root)
        results: list[CleanedFileResult] = []
        prepared_outputs: list[tuple[CleanedFileResult, bytes]] = []
        output_paths: list[Path] = []
        warnings = list(discovery_warnings)
        cleaned_count = 0
        skipped_count = 0
        invalid_count = 0
        empty_count = 0
        errors: list[tuple[str, str]] = []

        for source, destination_relative in zip(sources, destinations):
            relative = source.relative_to(input_root)
            try:
                source_bytes = _read_source_bytes(source, relative)
                source_sha = hashlib.sha256(source_bytes).hexdigest()
                source_text = source_bytes.decode("utf-8-sig")
                cleaned = clean_markdown_text(
                    source_text,
                    source_relative_path=relative.as_posix(),
                    source_name=self.config.source_name,
                    source_sha256=source_sha,
                    fallback_title=source.stem,
                )
            except UnicodeDecodeError as exc:
                error = (
                    f"CleaningReadError in '{relative.as_posix()}': "
                    "source document is not valid UTF-8."
                )
                invalid_count += 1
                errors.append((relative.as_posix(), error))
                warnings.append(error)
                results.append(
                    _invalid_file_result(
                        relative,
                        destination_relative,
                        source,
                        error,
                    )
                )
                continue
            except DocumentCleaningError as exc:
                error = str(exc)
                if relative.as_posix() not in error:
                    error = (
                        f"{type(exc).__name__} in '{relative.as_posix()}': "
                        f"{error}"
                    )
                invalid_count += 1
                errors.append((relative.as_posix(), error))
                warnings.append(error)
                results.append(
                    _invalid_file_result(
                        relative,
                        destination_relative,
                        source,
                        error,
                    )
                )
                continue
            file_warnings = tuple(
                f"{relative.as_posix()}: {warning}"
                for warning in cleaned.warnings
            )
            warnings.extend(file_warnings)
            destination = output_root / destination_relative

            if not has_meaningful_content(cleaned.text):
                empty_count += 1
                message = "No meaningful content remained."
                results.append(
                    CleanedFileResult(
                        relative.as_posix(),
                        destination_relative.as_posix(),
                        source.suffix.casefold(),
                        source_sha,
                        "",
                        len(source_bytes),
                        0,
                        cleaned.title,
                        cleaned.front_matter,
                        cleaned.transformations,
                        file_warnings + (message,),
                        "EMPTY",
                        message,
                    )
                )
                warnings.append(
                    f"{relative.as_posix()}: {message}"
                )
                continue

            output_bytes = cleaned.text.encode("utf-8")
            output_sha = hashlib.sha256(output_bytes).hexdigest()
            unchanged = (
                destination.is_file()
                and _sha256_path(destination) == output_sha
            )
            if unchanged:
                status = "SKIP"
                skipped_count += 1
            else:
                status = "UPDATE" if destination.exists() else "CLEAN"
                cleaned_count += 1
            output_paths.append(destination)
            file_result = CleanedFileResult(
                relative.as_posix(),
                destination_relative.as_posix(),
                source.suffix.casefold(),
                source_sha,
                output_sha,
                len(source_bytes),
                len(output_bytes),
                cleaned.title,
                cleaned.front_matter,
                cleaned.transformations,
                file_warnings,
                status,
            )
            results.append(file_result)
            prepared_outputs.append((file_result, output_bytes))

        if errors and self.config.strict:
            details = "\n".join(
                f"- {relative}: {message}"
                for relative, message in errors
            )
            raise CleaningBatchError(
                f"Strict cleaning preflight found {len(errors)} invalid "
                f"document(s); no output was published:\n{details}"
            )
        if not output_paths:
            raise NoCleanDocumentsError(
                "No source documents produced meaningful cleaned Markdown."
            )
        result = CleaningResult(
            scanned_count=len(sources),
            cleaned_count=cleaned_count,
            skipped_count=skipped_count,
            warning_count=len(warnings),
            output_paths=tuple(output_paths),
            warnings=tuple(warnings),
            files=tuple(results),
            invalid_count=invalid_count,
            empty_count=empty_count,
            error_count=0,
        )
        if not self.config.dry_run:
            manifest = self._manifest_bytes(result)
            _publish_cleaned_dataset(
                output_root,
                prepared_outputs,
                manifest,
            )
        return result

    def _manifest_bytes(self, result: CleaningResult) -> bytes:
        generated_at = self._now_utc()
        if generated_at.tzinfo is None:
            raise DocumentCleaningError(
                "Manifest timestamp provider must return a timezone-aware datetime."
            )
        manifest = {
            "schema_version": CLEANING_SCHEMA_VERSION,
            "cleaning_schema_version": CLEANING_SCHEMA_VERSION,
            "source_name": self.config.source_name.strip(),
            "input_root": describe_source_root(self.config.input_dir),
            "output_root": describe_source_root(self.config.output_dir),
            "generated_at": generated_at.astimezone(timezone.utc).isoformat(),
            "scanned_count": result.scanned_count,
            "cleaned_count": result.cleaned_count,
            "skipped_count": result.skipped_count,
            "invalid_count": result.invalid_count,
            "empty_count": result.empty_count,
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "extension_mapping": {".md": ".md", ".mdx": ".md"},
            "configuration": {
                "strict": self.config.strict,
                "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
                "cleaner_schema_version": CLEANING_SCHEMA_VERSION,
            },
            "skipped_or_error_files": [
                {
                    "source_relative_path": item.source_relative_path,
                    "status": item.status,
                    "reason": (
                        item.error
                        or "Output content was unchanged."
                        if item.status == "SKIP"
                        else item.error
                    ),
                }
                for item in result.files
                if item.status in {"SKIP", "EMPTY", "INVALID"}
            ],
            "files": [
                {
                    "source_relative_path": item.source_relative_path,
                    "destination_relative_path": item.destination_relative_path,
                    "input_extension": item.input_extension,
                    "source_size_bytes": item.source_size_bytes,
                    "output_size_bytes": item.output_size_bytes,
                    "source_sha256": item.source_sha256,
                    "output_sha256": item.output_sha256,
                    "title": item.title,
                    "front_matter": item.front_matter,
                    "transformations": list(item.transformations),
                    "warnings": list(item.warnings),
                    "status": item.status,
                    "error": item.error,
                }
                for item in result.files
            ],
        }
        return (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")


def _invalid_file_result(
    relative: Path,
    destination_relative: Path,
    source: Path,
    error: str,
) -> CleanedFileResult:
    try:
        source_bytes = source.read_bytes()
    except OSError:
        source_bytes = b""
    return CleanedFileResult(
        source_relative_path=relative.as_posix(),
        destination_relative_path=destination_relative.as_posix(),
        input_extension=source.suffix.casefold(),
        source_sha256=hashlib.sha256(source_bytes).hexdigest()
        if source_bytes
        else "",
        output_sha256="",
        source_size_bytes=len(source_bytes),
        output_size_bytes=0,
        title="",
        front_matter={},
        transformations=(),
        warnings=(error,),
        status="INVALID",
        error=error,
    )


def _source_syntax_error(
    source_relative_path: str,
    error: CleaningSyntaxError,
) -> CleaningSyntaxError:
    return CleaningSyntaxError(
        f"CleaningSyntaxError in '{source_relative_path}': {error}"
    )


def _publish_cleaned_dataset(
    output_root: Path,
    prepared_outputs: Sequence[tuple[CleanedFileResult, bytes]],
    manifest: bytes,
) -> None:
    """Publish a complete dataset through a validated sibling directory."""
    parent = output_root.parent
    temporary: Path | None = None
    backup: Path | None = None
    try:
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.cleaning-tmp-",
                dir=parent,
            )
        ).resolve()
        for result, content in prepared_outputs:
            atomic_write_bytes(
                temporary / result.destination_relative_path,
                content,
            )
        atomic_write_bytes(
            temporary / CLEANING_MANIFEST_NAME,
            manifest,
        )
        _validate_staged_dataset(
            temporary,
            expected_count=len(prepared_outputs),
        )

        if output_root.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_root.name}.cleaning-backup-",
                    dir=parent,
                )
            ).resolve()
            backup.rmdir()
            os.replace(output_root, backup)
        try:
            os.replace(temporary, output_root)
        except OSError:
            if (
                backup is not None
                and backup.exists()
                and not output_root.exists()
            ):
                os.replace(backup, output_root)
                backup = None
            raise
        temporary = None
        if backup is not None:
            _remove_staging_directory(backup, parent)
            backup = None
    except DocumentCleaningError:
        raise
    except OSError as exc:
        raise CleaningWriteError(
            f"Could not safely publish cleaned dataset to '{output_root}'."
        ) from exc
    finally:
        if temporary is not None and temporary.exists():
            _remove_staging_directory(temporary, parent)
        if backup is not None and backup.exists():
            if not output_root.exists():
                os.replace(backup, output_root)
            else:
                _remove_staging_directory(backup, parent)


def _validate_staged_dataset(
    directory: Path,
    *,
    expected_count: int,
) -> None:
    manifest_path = directory / CLEANING_MANIFEST_NAME
    if not manifest_path.is_file():
        raise CleaningWriteError(
            "Staged cleaning dataset is missing its manifest."
        )
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CleaningWriteError(
            "Staged cleaning manifest could not be validated."
        ) from exc
    if metadata.get("schema_version") != CLEANING_SCHEMA_VERSION:
        raise CleaningWriteError(
            "Staged cleaning manifest has an incompatible schema version."
        )
    documents = tuple(
        path
        for path in directory.rglob("*.md")
        if path.name != CLEANING_MANIFEST_NAME
    )
    if len(documents) != expected_count:
        raise CleaningWriteError(
            "Staged cleaning document count does not match the manifest: "
            f"expected {expected_count}, found {len(documents)}."
        )


def _remove_staging_directory(path: Path, expected_parent: Path) -> None:
    resolved = path.resolve()
    parent = expected_parent.resolve()
    if resolved.parent != parent or not (
        ".cleaning-tmp-" in resolved.name
        or ".cleaning-backup-" in resolved.name
    ):
        raise CleaningWriteError(
            f"Refusing to remove unsafe cleaning staging path: '{resolved}'."
        )
    shutil.rmtree(resolved)


def validate_cleaning_paths(
    input_dir: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Validate non-overlapping input and output directories."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        raise CleaningPathError(
            f"Input directory does not exist: '{input_dir}'."
        )
    if not input_dir.is_dir():
        raise CleaningPathError(
            f"Input path is not a directory: '{input_dir}'."
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise CleaningPathError(
            f"Output path is not a directory: '{output_dir}'."
        )
    try:
        source = input_dir.resolve(strict=True)
        output = output_dir.resolve(strict=False)
    except OSError as exc:
        raise CleaningPathError(
            "Could not resolve input or output path."
        ) from exc
    if source == output:
        raise CleaningPathError("Output directory must not equal input.")
    if output.is_relative_to(source):
        raise CleaningPathError("Output directory must not be inside input.")
    if source.is_relative_to(output):
        raise CleaningPathError("Input directory must not be inside output.")
    return source, output


def discover_source_documents(
    input_root: Path,
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Discover supported regular files without following symlinks."""
    paths: list[Path] = []
    warnings: list[str] = []
    try:
        for root, directories, files in os.walk(
            input_root, topdown=True, followlinks=False
        ):
            root_path = Path(root)
            retained: list[str] = []
            for name in directories:
                candidate = root_path / name
                if candidate.is_symlink():
                    warnings.append(
                        f"{candidate.relative_to(input_root).as_posix()}: "
                        "Symlink directory was ignored."
                    )
                else:
                    retained.append(name)
            directories[:] = retained
            for name in files:
                candidate = root_path / name
                relative = candidate.relative_to(input_root)
                if candidate.is_symlink():
                    warnings.append(
                        f"{relative.as_posix()}: Symlink file was ignored."
                    )
                elif candidate.suffix.casefold() in SUPPORTED_EXTENSIONS:
                    paths.append(candidate)
    except OSError as exc:
        raise CleaningReadError(
            f"Could not scan input directory '{input_root.name}'."
        ) from exc
    paths.sort(
        key=lambda path: (
            path.relative_to(input_root).as_posix().casefold(),
            path.relative_to(input_root).as_posix(),
        )
    )
    return tuple(paths), tuple(warnings)


def validate_destinations(
    sources: Sequence[Path], input_root: Path
) -> tuple[Path, ...]:
    """Map inputs to .md outputs and reject deterministic collisions."""
    mapped: dict[str, list[str]] = {}
    destinations: list[Path] = []
    for source in sources:
        relative = source.relative_to(input_root)
        destination = relative.with_suffix(".md")
        key = destination.as_posix().casefold()
        mapped.setdefault(key, []).append(relative.as_posix())
        destinations.append(destination)
    collisions = [items for items in mapped.values() if len(items) > 1]
    if collisions:
        details = "; ".join(", ".join(items) for items in collisions)
        raise DestinationCollisionError(
            f"Multiple sources map to the same Markdown destination: {details}."
        )
    return tuple(destinations)


def clean_markdown_text(
    source_text: str,
    *,
    source_relative_path: str,
    source_name: str,
    source_sha256: str,
    fallback_title: str,
) -> CleanedMarkdown:
    """Run deterministic, conservative cleaning stages in memory."""
    transformations: list[str] = []
    warnings: list[str] = []
    text = normalize_newlines(source_text)
    try:
        body, front_matter, front_warnings = parse_front_matter(text)
    except CleaningSyntaxError as exc:
        raise _source_syntax_error(source_relative_path, exc) from exc
    if front_matter:
        transformations.append("front_matter_extracted")
    warnings.extend(front_warnings)
    body, code_group_count = re.subn(
        r"</?CodeGroup\s*>", "\n", body, flags=re.IGNORECASE
    )
    if code_group_count:
        transformations.append("component:CodeGroup")
    try:
        protected, fenced = protect_fenced_code(body)
    except CleaningSyntaxError as exc:
        raise _source_syntax_error(source_relative_path, exc) from exc
    protected, inline = protect_inline_code(protected)
    protected, removed_modules, module_warnings = remove_mdx_modules(protected)
    if removed_modules:
        transformations.append("mdx_modules_removed")
    warnings.extend(module_warnings)
    protected, component_names, component_warnings = transform_mdx_components(
        protected
    )
    transformations.extend(
        f"component:{name}" for name in sorted(component_names)
    )
    warnings.extend(component_warnings)
    protected = normalize_markdown_whitespace(protected)
    heading_view = restore_protected_regions(protected, inline, {})
    existing_h1 = H1_PATTERN.search(heading_view)
    front_title = front_matter.get("title")
    title = (
        existing_h1.group(1).strip()
        if existing_h1
        else str(front_title).strip()
        if front_title
        else fallback_title.replace("_", " ").replace("-", " ").strip()
    )
    if not existing_h1:
        if not front_title:
            warnings.append("No title or H1; filename stem was used as title.")
            transformations.append("fallback_title_added")
        else:
            transformations.append("front_matter_title_added")
        protected = f"# {title}\n\n{protected.lstrip()}"
    body = restore_protected_regions(protected, inline, fenced)
    header = (
        "<!--\n"
        f"source_name: {source_name.strip()}\n"
        f"source_relative_path: {source_relative_path}\n"
        f"source_sha256: {source_sha256}\n"
        f"cleaning_schema_version: {CLEANING_SCHEMA_VERSION}\n"
        "-->\n\n"
    )
    return CleanedMarkdown(
        text=(header + body.strip() + "\n"),
        title=title,
        front_matter=front_matter,
        transformations=tuple(dict.fromkeys(transformations)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def normalize_newlines(text: str) -> str:
    """Remove a leading BOM and normalize all newlines to LF."""
    return text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def parse_front_matter(
    text: str,
) -> tuple[str, dict[str, object], tuple[str, ...]]:
    """Parse a deliberately small subset of leading YAML front matter."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text, {}, ()
    end = next(
        (index for index, line in enumerate(lines[1:], start=1)
         if line.strip() == "---"),
        None,
    )
    if end is None:
        raise CleaningSyntaxError("Front matter opening marker is not closed.")
    metadata: dict[str, object] = {}
    warnings: list[str] = []
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            warnings.append(f"Unsupported front matter line was ignored: {line!r}.")
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip().casefold()
        value = raw_value.strip()
        if key not in FRONT_MATTER_KEYS:
            continue
        if value in {"|", ">"} or not value:
            warnings.append(
                f"Complex front matter field '{key}' was ignored."
            )
            continue
        value = value.strip("\"'")
        if key == "tags" and value.startswith("[") and value.endswith("]"):
            metadata[key] = [
                item.strip().strip("\"'")
                for item in value[1:-1].split(",")
                if item.strip()
            ]
        else:
            metadata[key] = value
    return "\n".join(lines[end + 1 :]), metadata, tuple(warnings)


def protect_fenced_code(text: str) -> tuple[str, dict[str, str]]:
    """Replace complete fenced blocks with stable placeholders."""
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    protected: dict[str, str] = {}
    index = 0
    while index < len(lines):
        match = FENCE_PATTERN.match(lines[index].rstrip("\n"))
        if not match:
            output.append(lines[index])
            index += 1
            continue
        fence = match.group(1)
        block = [lines[index]]
        index += 1
        closed = False
        while index < len(lines):
            block.append(lines[index])
            candidate = lines[index].rstrip("\n")
            index += 1
            closing_pattern = re.compile(
                rf"^ {{0,3}}{re.escape(fence[0])}"
                rf"{{{len(fence)},}}[ \t]*$"
            )
            if closing_pattern.match(candidate):
                closed = True
                break
        if not closed:
            raise CleaningSyntaxError("Fenced code block is not closed.")
        token = f"@@RAG_FENCED_{len(protected)}@@"
        protected[token] = "".join(block).rstrip("\n")
        output.append(token + "\n")
    return "".join(output), protected


def protect_inline_code(text: str) -> tuple[str, dict[str, str]]:
    """Protect Markdown links and balanced inline backtick spans."""
    protected: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"@@RAG_INLINE_{len(protected)}@@"
        protected[token] = match.group(0)
        return token

    link_pattern = re.compile(r"!?\[[^\]\n]*\]\([^\)\n]*\)")
    text = link_pattern.sub(replace, text)
    code_pattern = re.compile(r"(`+)([^\n]*?)\1")
    return code_pattern.sub(replace, text), protected


def remove_mdx_modules(
    text: str,
) -> tuple[str, int, tuple[str, ...]]:
    """Remove only clear top-level import/export module statements."""
    lines = text.splitlines()
    output: list[str] = []
    warnings: list[str] = []
    removed = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if line != line.lstrip() or not MODULE_PATTERN.match(line):
            output.append(line)
            index += 1
            continue
        if line.startswith("import ") and re.search(
            r"(?:\sfrom\s+|^import\s+)[\"']", line
        ):
            removed += 1
            index += 1
            continue
        if line.startswith("import "):
            output.append(line)
            index += 1
            continue
        buffer = [line]
        balance = line.count("{") - line.count("}")
        index += 1
        while balance > 0 and index < len(lines):
            buffer.append(lines[index])
            balance += lines[index].count("{") - lines[index].count("}")
            index += 1
        if balance == 0:
            removed += 1
        else:
            output.extend(buffer)
            warnings.append(
                "An unbalanced export statement was preserved."
            )
    return "\n".join(output), removed, tuple(warnings)


def transform_mdx_components(
    text: str,
) -> tuple[str, set[str], tuple[str, ...]]:
    """Convert known components and conservatively strip unknown tag syntax."""
    transformed: set[str] = set()
    warnings: list[str] = []

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        if raw.startswith(("<http://", "<https://", "<mailto:")):
            return raw
        name = match.group("name")
        lowered = name.casefold()
        closing = bool(match.group("closing"))
        attributes = parse_attributes(match.group("attrs"))
        transformed.add(name)
        if closing:
            return ""
        if lowered in ADMONITIONS:
            label = attributes.get("title") or name
            return f"\n> **{label}:**\n"
        if lowered in OUTER_COMPONENTS:
            return ""
        if lowered in {"tab", "tabitem"}:
            label = (
                attributes.get("label")
                or attributes.get("title")
                or attributes.get("value")
                or "Tab"
            )
            return f"\n#### {label}\n"
        if lowered in {"accordion", "details"}:
            label = (
                attributes.get("title")
                or attributes.get("summary")
                or "Details"
            )
            return f"\n### {label}\n"
        if lowered == "summary":
            return "\n**Summary:** "
        if lowered == "card":
            title = attributes.get("title") or "Card"
            href = attributes.get("href")
            return f"\n### [{title}]({href})\n" if href else f"\n### {title}\n"
        if lowered == "step":
            title = attributes.get("title")
            return f"\n### Step: {title}\n" if title else "\n### Step\n"
        if lowered in {"image", "img", "figure"}:
            return _media_description("Image", attributes)
        if lowered == "video":
            return _media_description("Video", attributes)
        if lowered == "link":
            href = attributes.get("href")
            label = attributes.get("label") or attributes.get("title")
            return f"[{label or href}]({href}) " if href else ""
        if lowered == "badge":
            label = attributes.get("label") or attributes.get("text")
            return f"**{label}** " if label else ""
        if lowered in {"snippet"}:
            return _readable_self_closing(name, attributes)
        readable = _readable_self_closing(name, attributes)
        warnings.append(f"Unknown component '{name}' was preserved conservatively.")
        return readable if match.group("self") else ""

    return (
        TAG_PATTERN.sub(replace, text),
        transformed,
        tuple(dict.fromkeys(warnings)),
    )


def parse_attributes(raw: str) -> dict[str, str]:
    """Extract only quoted scalar JSX/HTML attributes."""
    values: dict[str, str] = {}
    for match in ATTRIBUTE_PATTERN.finditer(raw):
        value = next(
            group for group in match.groups()[1:] if group is not None
        )
        values[match.group(1).casefold()] = value
    return values


def normalize_markdown_whitespace(text: str) -> str:
    """Trim trailing spaces and limit consecutive blank lines."""
    output: list[str] = []
    blank_count = 0
    for line in text.splitlines():
        line = line.rstrip()
        if line:
            blank_count = 0
            output.append(line)
        else:
            blank_count += 1
            if blank_count <= 2:
                output.append("")
    return "\n".join(output).strip()


def restore_protected_regions(
    text: str, inline: dict[str, str], fenced: dict[str, str]
) -> str:
    """Restore inline spans and fenced blocks exactly after tag cleaning."""
    for token, value in inline.items():
        text = text.replace(token, value)
    for token, value in fenced.items():
        text = text.replace(token, value)
    return text


def has_meaningful_content(text: str) -> bool:
    """Reject output containing only metadata, headings, or navigation labels."""
    without_header = re.sub(r"^<!--.*?-->\s*", "", text, flags=re.DOTALL)
    candidates = []
    for line in without_header.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or set(stripped) <= {"-", "*", "_"}:
            continue
        if stripped.casefold() in {"next", "previous", "back", "home"}:
            continue
        candidates.append(stripped)
    return any(re.search(r"[\w]{3,}", candidate, flags=re.UNICODE) for candidate in candidates)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write bytes through a same-directory temporary file and atomic replace."""
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise CleaningWriteError(
            f"Could not atomically write '{path.name}'."
        ) from exc


def _media_description(kind: str, attributes: dict[str, str]) -> str:
    values = [
        attributes.get("alt"),
        attributes.get("title"),
        attributes.get("caption"),
        attributes.get("src"),
    ]
    readable = [value for value in values if value]
    return f"[{kind}: {' — '.join(readable)}]" if readable else ""


def _readable_self_closing(name: str, attributes: dict[str, str]) -> str:
    values = [
        attributes.get("label"),
        attributes.get("title"),
        attributes.get("alt"),
        attributes.get("href"),
        attributes.get("src"),
    ]
    readable = [value for value in values if value]
    return f"[{name}: {' — '.join(readable)}]" if readable else ""


def _read_source_bytes(path: Path, relative: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CleaningReadError(
            f"Could not read source document '{relative.as_posix()}'."
        ) from exc


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CleaningReadError(
            f"Could not read existing output '{path.name}'."
        ) from exc
