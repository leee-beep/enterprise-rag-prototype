"""Offline tests for deterministic local LangChain document import."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from enterprise_rag.document_import import (
    MANIFEST_FILE_NAME,
    DocumentImportIOError,
    ImportConfig,
    ImportPathError,
    LangChainDocumentImporter,
    NoMatchingDocumentsError,
    file_matches_include,
    is_excluded_path,
    scan_markdown_files,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = PROJECT_ROOT / "scripts" / "import_docs.py"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def importer(
    source: Path,
    output: Path,
    *,
    dry_run: bool = False,
) -> LangChainDocumentImporter:
    return LangChainDocumentImporter(
        ImportConfig(source=source, output=output, dry_run=dry_run),
        now_utc=lambda: datetime(2026, 7, 28, 1, 2, 3, tzinfo=timezone.utc),
    )


def test_scan_recursively_finds_md_mdx_ignores_other_files_and_sorts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    write(source / "z" / "retrieval.MDX", "Retrieval")
    write(source / "A" / "embedding.md", "Embedding")
    write(source / "ignored.txt", "retrieval")

    paths = scan_markdown_files(source)

    assert [path.relative_to(source).as_posix() for path in paths] == [
        "A/embedding.md",
        "z/retrieval.MDX",
    ]


@pytest.mark.parametrize(
    "relative,content",
    [
        ("concepts/Retrieval.mdx", "unrelated"),
        ("concepts/guide.mdx", "# EMBEDDINGS"),
        ("concepts/search.md", "A semantic SEARCH tutorial"),
    ],
)
def test_include_matching_is_case_insensitive(
    tmp_path: Path, relative: str, content: str
) -> None:
    path = write(tmp_path / relative, content)
    assert file_matches_include(path, Path(relative))


def test_include_can_match_document_title_or_initial_content(tmp_path: Path) -> None:
    path = write(tmp_path / "guide.mdx", "# Document Loader\n\nDetails")
    assert file_matches_include(path, Path("guide.mdx"))


def test_exclude_has_priority_over_include(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write(source / "legacy" / "retrieval.mdx", "# Retrieval")
    write(source / "concepts" / "retrieval.mdx", "# Retrieval")

    result = importer(source, output).run()

    assert result.matched_count == 1
    assert (output / "concepts" / "retrieval.mdx").is_file()
    assert not (output / "legacy" / "retrieval.mdx").exists()


def test_js_requires_an_explicit_path_segment() -> None:
    assert is_excluded_path(Path("guides/js/retrieval.mdx"))
    assert not is_excluded_path(Path("objects/retrieval.mdx"))
    assert not is_excluded_path(Path("projects/retrieval.mdx"))


def test_relative_structure_and_same_names_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write(source / "oss" / "guide.mdx", "# Retrieval for OSS")
    write(source / "python" / "guide.mdx", "# Embedding for Python")

    result = importer(source, output).run()

    assert result.copied_count == 2
    assert (output / "oss" / "guide.mdx").read_text(encoding="utf-8").endswith("OSS")
    assert (output / "python" / "guide.mdx").read_text(encoding="utf-8").endswith("Python")


def test_output_paths_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write(source / "z" / "retrieval.md", "retrieval")
    write(source / "a" / "embedding.mdx", "embedding")

    result = importer(source, tmp_path / "output", dry_run=True).run()

    assert [path.relative_to(tmp_path / "output").as_posix() for path in result.output_paths] == [
        "a/embedding.mdx",
        "z/retrieval.md",
    ]


def test_dry_run_does_not_create_output_or_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write(source / "retrieval.mdx", "# Retrieval")

    result = importer(source, output, dry_run=True).run()

    assert result.scanned_count == 1
    assert result.matched_count == 1
    assert result.copied_count == 0
    assert result.skipped_count == 0
    assert not output.exists()


def test_first_run_copies_and_second_identical_run_skips(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write(source / "retrieval.mdx", "# Retrieval")

    first = importer(source, output).run()
    second = importer(source, output).run()

    assert (first.copied_count, first.skipped_count) == (1, 0)
    assert (second.copied_count, second.skipped_count) == (0, 1)


def test_changed_source_overwrites_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source_file = write(source / "retrieval.mdx", "# Retrieval v1")
    importer(source, output).run()
    source_file.write_text("# Retrieval v2", encoding="utf-8")

    result = importer(source, output).run()

    assert result.copied_count == 1
    assert result.skipped_count == 0
    assert (output / "retrieval.mdx").read_text(encoding="utf-8") == "# Retrieval v2"


def test_manifest_schema_hash_utf8_and_private_source_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "private-user-name" / "source"
    output = tmp_path / "output"
    content = "# Retrieval\n\n繁體中文內容"
    source_file = write(source / "概念" / "retrieval.mdx", content)
    source_bytes = source_file.read_bytes()

    importer(source, output).run()
    raw = (output / MANIFEST_FILE_NAME).read_bytes()
    manifest = json.loads(raw.decode("utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["source_name"] == "langchain"
    assert manifest["source_root"] == "source"
    assert "private-user-name" not in manifest["source_root"]
    assert manifest["generated_at"] == "2026-07-28T01:02:03+00:00"
    assert manifest["scanned_count"] == manifest["matched_count"] == 1
    item = manifest["imported_files"][0]
    assert item == {
        "destination_relative_path": "概念/retrieval.mdx",
        "extension": ".mdx",
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "size_bytes": len(source_bytes),
        "source_relative_path": "概念/retrieval.mdx",
    }
    assert "retrieval" in manifest["include_keywords"]
    assert "legacy" in manifest["exclude_path_keywords"]


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_invalid_source_has_clear_error(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    if kind == "file":
        source.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ImportPathError, match="Source"):
        importer(source, tmp_path / "output").run()


@pytest.mark.parametrize("output_kind", ["same", "inside", "parent"])
def test_source_output_conflicts_are_rejected(
    tmp_path: Path, output_kind: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outputs = {
        "same": source,
        "inside": source / "output",
        "parent": tmp_path,
    }
    with pytest.raises(ImportPathError, match="Output"):
        importer(source, outputs[output_kind]).run()


def test_no_matching_documents_does_not_create_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write(source / "intro.md", "# General introduction")
    write(source / "retrieval.txt", "ignored extension")

    with pytest.raises(NoMatchingDocumentsError, match="No Markdown or MDX"):
        importer(source, output).run()
    assert not output.exists()


def test_invalid_utf8_document_has_path_in_domain_error(tmp_path: Path) -> None:
    source = tmp_path / "source"
    path = source / "guide.md"
    path.parent.mkdir()
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(DocumentImportIOError, match="guide.md"):
        importer(source, tmp_path / "output").run()


def test_cli_success_and_dry_run_write_nothing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write(source / "retrieval.mdx", "# Retrieval")
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--source",
            str(source),
            "--output",
            str(output),
            "--dry-run",
            "--verbose",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    assert "Scanned: 1" in completed.stdout
    assert "Matched: 1" in completed.stdout
    assert "Would import:" in completed.stdout
    assert completed.stderr == ""
    assert not output.exists()


def test_cli_normal_import_writes_document_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write(source / "concepts" / "retrieval.mdx", "# Retrieval")
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    assert "Copied: 1" in completed.stdout
    assert completed.stderr == ""
    assert (output / "concepts" / "retrieval.mdx").is_file()
    assert (output / MANIFEST_FILE_NAME).is_file()


def test_cli_expected_error_has_no_traceback(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--source",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "output"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Error: Source directory does not exist" in completed.stderr
    assert "Traceback" not in completed.stderr
