"""Offline tests for conservative MDX-to-Markdown cleaning."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from enterprise_rag.document_cleaning import (
    CLEANING_MANIFEST_NAME,
    CleaningBatchError,
    CleaningConfig,
    CleaningPathError,
    CleaningReadError,
    CleaningSyntaxError,
    CleaningWriteError,
    DestinationCollisionError,
    MarkdownDocumentCleaner,
    NoCleanDocumentsError,
    atomic_write_bytes,
    clean_markdown_text,
    discover_source_documents,
    validate_destinations,
)
from enterprise_rag.documents import load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "clean_documents.py"


def write(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding, newline="")
    return path


def clean(text: str, *, name: str = "guide.mdx"):
    return clean_markdown_text(
        text,
        source_relative_path=f"concepts/{name}",
        source_name="langchain",
        source_sha256="a" * 64,
        fallback_title=Path(name).stem,
    )


def cleaner(
    input_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    strict: bool = True,
):
    return MarkdownDocumentCleaner(
        CleaningConfig(
            input_dir,
            output_dir,
            dry_run=dry_run,
            strict=strict,
        ),
        now_utc=lambda: datetime(2026, 7, 28, 2, 3, 4, tzinfo=timezone.utc),
    )


def test_discovery_recursive_ignores_non_docs_manifest_gitkeep_and_sorts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    write(root / "z" / "b.mdx", "# B\nbody")
    write(root / "A" / "a.md", "# A\nbody")
    write(root / "manifest.json", "{}")
    write(root / ".gitkeep", "")
    write(root / "ignored.txt", "body")
    paths, warnings = discover_source_documents(root)
    assert [path.relative_to(root).as_posix() for path in paths] == [
        "A/a.md",
        "z/b.mdx",
    ]
    assert warnings == ()


def test_mdx_and_md_destination_mapping_and_collision(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    md = write(root / "nested" / "foo.md", "# Foo\nbody")
    mdx = write(root / "other" / "bar.mdx", "# Bar\nbody")
    assert [p.as_posix() for p in validate_destinations((md, mdx), root)] == [
        "nested/foo.md",
        "other/bar.md",
    ]
    collision = write(root / "nested" / "foo.mdx", "# Other\nbody")
    with pytest.raises(DestinationCollisionError, match="foo.md.*foo.mdx"):
        validate_destinations((md, collision), root)


@pytest.mark.parametrize("kind", ["same", "inside", "parent"])
def test_input_output_conflicts(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    output = {"same": source, "inside": source / "out", "parent": tmp_path}[kind]
    with pytest.raises(CleaningPathError):
        cleaner(source, output).run()


def test_symlink_files_are_ignored_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "raw"
    link = write(root / "link.mdx", "# Outside\nsecret")
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == link or original(self),
    )
    paths, warnings = discover_source_documents(root)
    assert paths == ()
    assert "Symlink file was ignored" in warnings[0]


def test_utf8_bom_crlf_and_single_trailing_newline() -> None:
    result = clean("\ufeff# Title\r\n\r\nBody\r\n\r\n\r\n")
    assert result.text.startswith("<!--\n")
    assert "\r" not in result.text
    assert result.text.endswith("Body\n")
    assert not result.text.endswith("\n\n")


def test_invalid_utf8_names_relative_source(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    path = root / "bad.mdx"
    path.parent.mkdir()
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(CleaningBatchError, match="CleaningReadError.*bad.mdx"):
        cleaner(root, tmp_path / "documents").run()


def test_front_matter_extracts_metadata_and_adds_nonduplicate_title() -> None:
    result = clean(
        "---\ntitle: Retrieval\ndescription: Search docs\ntags: [rag, search]\n"
        "layout: docs\n---\nBody text."
    )
    assert result.front_matter == {
        "title": "Retrieval",
        "description": "Search docs",
        "tags": ["rag", "search"],
    }
    assert result.text.count("# Retrieval") == 1
    assert "layout:" not in result.text


def test_existing_h1_wins_without_duplicate() -> None:
    result = clean("---\ntitle: Retrieval\n---\n# Existing\n\nBody")
    assert result.title == "Existing"
    assert result.text.count("# Existing") == 1
    assert "# Retrieval" not in result.text


def test_filename_fallback_title_warns() -> None:
    result = clean("Useful retrieval body.", name="retrieval_guide.mdx")
    assert "# retrieval guide" in result.text
    assert "filename stem" in result.warnings[0]


def test_unclosed_front_matter_and_fence_are_errors() -> None:
    with pytest.raises(CleaningSyntaxError, match="Front matter"):
        clean("---\ntitle: Broken\nBody")
    with pytest.raises(CleaningSyntaxError, match="Fenced"):
        clean("# T\n```python\nprint('x')")


def test_fenced_code_is_preserved_from_module_and_component_cleaning() -> None:
    block = "```javascript\nimport X from 'x'\n<Note>\n\ncode\n````"
    result = clean(f"# Guide\n\n{block}\n\nText")
    assert block in result.text


def test_inline_code_table_and_markdown_link_are_preserved() -> None:
    body = (
        "# Guide\n\nUse `<Note>` and [docs](https://example.com/a?x=<y>).\n\n"
        "| A | B |\n|---|---|\n| `<Tag>` | value |"
    )
    result = clean(body)
    assert "`<Note>`" in result.text
    assert "[docs](https://example.com/a?x=<y>)" in result.text
    assert "| `<Tag>` | value |" in result.text


def test_top_level_modules_removed_but_normal_text_preserved() -> None:
    result = clean(
        "import Tabs from '@theme/Tabs'\n"
        "export const metadata = {\n  title: 'x'\n}\n"
        "# Guide\n\nimport data carefully in normal prose."
    )
    assert "@theme/Tabs" not in result.text
    assert "metadata =" not in result.text
    assert "import data carefully" in result.text


@pytest.mark.parametrize(
    "source,expected",
    [
        ("<Note>Use retrieval.</Note>", "> **Note:**"),
        ("<Warning>Check vectors.</Warning>", "> **Warning:**"),
        (
            '<Tabs><TabItem value="py" label="Python">Code</TabItem>'
            '<TabItem label="JS">Other</TabItem></Tabs>',
            "#### Python",
        ),
        ('<Details title="More">Details body</Details>', "### More"),
        ('<Card title="Retriever" href="/r">Card body</Card>', "[Retriever](/r)"),
        ('<Image alt="Diagram" src="images/r.png" />', "[Image: Diagram — images/r.png]"),
        ("<CodeGroup>```python\nprint('x')\n```</CodeGroup>", "```python"),
        ('<Step title="Install">Do it</Step>', "### Step: Install"),
    ],
)
def test_known_components_preserve_readable_knowledge(
    source: str, expected: str
) -> None:
    result = clean(f"# Guide\n\n{source}")
    assert expected in result.text
    assert "body" not in source or "body" in result.text
    if "TabItem" in source:
        assert "#### JS" in result.text and "Code" in result.text and "Other" in result.text


def test_unknown_tags_keep_inner_text_and_report_warning() -> None:
    result = clean(
        '# Guide\n\n<CustomPanel mode="wide">Keep this knowledge.</CustomPanel>\n'
        '<Widget label="Useful" href="/w" />'
    )
    assert "Keep this knowledge." in result.text
    assert "[Widget: Useful — /w]" in result.text
    assert "mode=\"wide\"" not in result.text
    assert any("CustomPanel" in warning for warning in result.warnings)


def test_unbalanced_unknown_tag_does_not_swallow_following_content() -> None:
    result = clean("# Guide\n\n<Unknown>Keep this.\n\nFollowing paragraph.")
    assert "Keep this." in result.text
    assert "Following paragraph." in result.text
    assert result.warnings


def test_source_header_is_stable_and_private_path_free() -> None:
    result = clean("# Guide\n\nUseful body")
    assert "source_name: langchain" in result.text
    assert "source_relative_path: concepts/guide.mdx" in result.text
    assert "source_sha256: " + "a" * 64 in result.text
    assert "C:\\Users" not in result.text
    assert "generated_at" not in result.text


def test_clean_run_structure_hashes_manifest_and_repeat_skip(tmp_path: Path) -> None:
    root = tmp_path / "private-user" / "raw"
    output = tmp_path / "documents"
    source = write(root / "oss" / "retrieval.mdx", "---\ntitle: Retrieval\n---\nBody")
    first = cleaner(root, output).run()
    second = cleaner(root, output).run()
    destination = output / "oss" / "retrieval.md"
    manifest = json.loads((output / CLEANING_MANIFEST_NAME).read_text(encoding="utf-8"))
    item = manifest["files"][0]
    assert first.cleaned_count == 1 and second.skipped_count == 1
    assert destination.is_file()
    assert item["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert item["output_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert manifest["schema_version"] == 2
    assert manifest["generated_at"] == "2026-07-28T02:03:04+00:00"
    assert manifest["input_root"] == "raw"
    assert "private-user" not in json.dumps(manifest)
    assert item["front_matter"] == {"title": "Retrieval"}


def test_source_change_updates_output(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "documents"
    source = write(root / "guide.mdx", "# Guide\n\nVersion one")
    cleaner(root, output).run()
    source.write_text("# Guide\n\nVersion two", encoding="utf-8")
    result = cleaner(root, output).run()
    assert result.files[0].status == "UPDATE"
    assert "Version two" in (output / "guide.md").read_text(encoding="utf-8")


def test_cleaned_markdown_is_loadable_by_existing_document_loader(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "documents"
    write(root / "nested" / "guide.mdx", "<Note>Useful retrieval body.</Note>")
    cleaner(root, output).run()
    loaded = load_documents(output)
    assert len(loaded.documents) == 1
    assert loaded.documents[0].source == "nested/guide.md"
    assert "Useful retrieval body." in loaded.documents[0].content


def test_empty_document_is_skipped_and_all_empty_batch_errors(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "documents"
    write(root / "empty.mdx", "<Frame><Badge /></Frame>")
    with pytest.raises(NoCleanDocumentsError):
        cleaner(root, output).run()
    assert not output.exists()


def test_dry_run_performs_validation_without_writes(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "documents"
    write(root / "guide.mdx", "<Note>Useful retrieval content.</Note>")
    result = cleaner(root, output, dry_run=True).run()
    assert result.cleaned_count == 1
    assert result.files[0].transformations
    assert not output.exists()


def test_atomic_write_replaces_complete_file(tmp_path: Path) -> None:
    destination = tmp_path / "guide.md"
    destination.write_bytes(b"old")
    atomic_write_bytes(destination, b"new")
    assert destination.read_bytes() == b"new"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_failure_preserves_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "guide.md"
    destination.write_bytes(b"old")
    monkeypatch.setattr(
        "enterprise_rag.document_cleaning.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("blocked")),
    )
    with pytest.raises(CleaningWriteError):
        atomic_write_bytes(destination, b"new")
    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob("*.tmp")) == []


def test_cli_success_verbose_and_dry_run(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "documents"
    write(root / "guide.mdx", "# Guide\n\nUseful retrieval")
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--input",
            str(root),
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
    assert "CLEAN: guide.mdx -> guide.md" in completed.stdout
    assert "Scanned: 1" in completed.stdout
    assert completed.stderr == ""
    assert not output.exists()


def test_cli_expected_error_and_all_empty_exit_two(tmp_path: Path) -> None:
    missing = subprocess.run(
        [sys.executable, str(CLI), "--input", str(tmp_path / "missing")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert missing.returncode == 2
    assert "Error: Input directory does not exist" in missing.stderr
    assert "Traceback" not in missing.stderr

    root = tmp_path / "raw"
    write(root / "empty.mdx", "<Frame />")
    empty = subprocess.run(
        [sys.executable, str(CLI), "--input", str(root), "--output", str(tmp_path / "out")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert empty.returncode == 2
    assert "No source documents produced" in empty.stderr


@pytest.mark.parametrize("indent", ["", " ", "  ", "   "])
def test_fence_closing_allows_zero_to_three_spaces(indent: str) -> None:
    block = f"```python\nprint('ok')\n{indent}```"
    result = clean(f"---\ntitle: Fence\n---\n{block}")
    assert block in result.text


def test_four_space_nested_fence_does_not_close_outer_block() -> None:
    block = (
        "```python\n"
        "def prompt():\n"
        "    \"\"\"\n"
        "    ```<prompt:system>\n"
        "    Keep this nested prompt.\n"
        "    ```\n"
        "    \"\"\"\n"
        "```\n"
    )
    result = clean(f"---\ntitle: Nested\n---\n{block}")
    assert block.rstrip("\n") in result.text
    assert result.text.count("```") == block.count("```")


def test_four_space_fence_alone_is_not_a_markdown_fence() -> None:
    source = "# Guide\n\n    ```python\n    print('literal')\n"
    result = clean(source)
    assert "    ```python" in result.text


def test_unclosed_fence_error_contains_type_and_source_path() -> None:
    with pytest.raises(
        CleaningSyntaxError,
        match=r"CleaningSyntaxError.*concepts/broken\.mdx.*not closed",
    ):
        clean("# Guide\n```python\nprint('broken')", name="broken.mdx")


def test_code_comment_does_not_replace_frontmatter_title() -> None:
    result = clean(
        "---\ntitle: Retrieval\n---\n"
        "Intro.\n\n```python\n# code comment\nprint('ok')\n```\n"
    )
    assert "# Retrieval\n" in result.text
    assert result.title == "Retrieval"


def test_real_body_h1_prevents_duplicate_frontmatter_title() -> None:
    result = clean(
        "---\ntitle: Front title\n---\n"
        "# Body title\n\n```python\n# code comment\n```\n"
    )
    assert result.text.count("# Body title") == 1
    assert "# Front title" not in result.text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            '<Card\n  title="Retriever"\n  href="/retrieval"\n>\n'
            "Readable card body.\n</Card>",
            "### [Retriever](/retrieval)",
        ),
        (
            '<Accordion\n  title="Advanced"\n>\nDetails.\n</Accordion>',
            "### Advanced",
        ),
        (
            '<Tabs\n  defaultValue="python"\n>\n'
            '<TabItem\n  label="Python"\n>\nCode\n</TabItem>\n</Tabs>',
            "#### Python",
        ),
        (
            '<Note\n  title="Remember"\n>\nUseful note.\n</Note>',
            "> **Remember:**",
        ),
        (
            '<Warning\n  title="Careful"\n>\nWarning body.\n</Warning>',
            "> **Careful:**",
        ),
        (
            '<Tip\n  title="Hint"\n>\nTip body.\n</Tip>',
            "> **Hint:**",
        ),
        (
            '<CodeGroup\n  label="Examples"\n>\n'
            "```python\nprint('kept')\n```\n</CodeGroup>",
            "```python",
        ),
    ],
)
def test_multiline_known_components_preserve_content(
    source: str,
    expected: str,
) -> None:
    result = clean(f"---\ntitle: Components\n---\n{source}")
    assert expected in result.text
    assert not re.search(
        r"</?(?:Card|Accordion|Tabs|TabItem|Note|Warning|Tip|CodeGroup)\b",
        result.text,
    )


def test_multiline_jsx_inside_code_block_is_unchanged() -> None:
    block = (
        "```mdx\n"
        "<Card\n"
        '  title=\"Do not transform\"\n'
        ">\n"
        "Code sample\n"
        "</Card>\n"
        "```\n"
    )
    result = clean(f"---\ntitle: Code\n---\n{block}")
    assert block.rstrip("\n") in result.text


def test_fenced_code_preserves_trailing_spaces_and_blank_lines() -> None:
    block = (
        "```text\n"
        "line with spaces   \n"
        "\n"
        "\n"
        "\n"
        "last line\n"
        "```\n"
    )
    result = clean(f"---\ntitle: Exact Code\n---\nBefore.\n\n{block}\nAfter.")
    assert block.rstrip("\n") in result.text


def test_strict_mode_collects_all_errors_without_publishing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "documents"
    write(root / "good.mdx", "# Good\n\nUseful body.")
    write(root / "first.mdx", "```python\nbroken")
    write(root / "nested" / "second.mdx", "---\ntitle: Broken")
    output.mkdir()
    marker = write(output / "existing.md", "stable output")

    with pytest.raises(CleaningBatchError) as caught:
        cleaner(root, output, strict=True).run()

    message = str(caught.value)
    assert "first.mdx" in message
    assert "nested/second.mdx" in message
    assert marker.read_text(encoding="utf-8") == "stable output"
    assert not (output / CLEANING_MANIFEST_NAME).exists()
    assert not list(tmp_path.glob(".documents.cleaning-*"))


def test_non_strict_mode_skips_invalid_and_publishes_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "documents"
    write(root / "good.mdx", "---\ntitle: Good\n---\nUseful body.")
    write(root / "broken.mdx", "```python\nbroken")

    result = cleaner(root, output, strict=False).run()
    manifest = json.loads(
        (output / CLEANING_MANIFEST_NAME).read_text(encoding="utf-8")
    )

    assert result.invalid_count == 1
    assert result.error_count == 0
    assert (output / "good.md").is_file()
    assert not (output / "broken.md").exists()
    assert manifest["invalid_count"] == 1
    assert manifest["error_count"] == 0
    invalid = next(
        item for item in manifest["files"] if item["status"] == "INVALID"
    )
    assert invalid["source_relative_path"] == "broken.mdx"
    assert "CleaningSyntaxError" in invalid["error"]


def test_successful_publish_replaces_stale_dataset_as_a_unit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "documents"
    write(root / "fresh.mdx", "# Fresh\n\nUseful body.")
    output.mkdir()
    write(output / "stale.md", "stale")

    cleaner(root, output).run()

    assert (output / "fresh.md").is_file()
    assert not (output / "stale.md").exists()
    assert (output / CLEANING_MANIFEST_NAME).is_file()
    assert not list(tmp_path.glob(".documents.cleaning-*"))


def test_publish_failure_restores_existing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "documents"
    write(root / "fresh.mdx", "# Fresh\n\nUseful body.")
    output.mkdir()
    marker = write(output / "stable.md", "stable")
    real_replace = os.replace

    def fail_final_publish(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            ".cleaning-tmp-" in source_path.name
            and destination_path == output.resolve()
        ):
            raise OSError("simulated publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        "enterprise_rag.document_cleaning.os.replace",
        fail_final_publish,
    )

    with pytest.raises(CleaningWriteError, match="safely publish"):
        cleaner(root, output).run()

    assert marker.read_text(encoding="utf-8") == "stable"
    assert not (output / "fresh.md").exists()
    assert not list(tmp_path.glob(".documents.cleaning-*"))


def test_manifest_schema_two_counts_and_configuration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "documents"
    write(root / "good.mdx", "# Good\n\nUseful body.")
    write(root / "empty.mdx", "<Frame><Badge /></Frame>")
    write(root / "broken.mdx", "```python\nbroken")

    result = cleaner(root, output, strict=False).run()
    manifest = json.loads(
        (output / CLEANING_MANIFEST_NAME).read_text(encoding="utf-8")
    )

    assert result.scanned_count == 3
    assert result.cleaned_count == 1
    assert result.skipped_count == 0
    assert result.invalid_count == 1
    assert result.empty_count == 1
    assert result.error_count == 0
    assert manifest["schema_version"] == 2
    assert manifest["extension_mapping"] == {".md": ".md", ".mdx": ".md"}
    assert manifest["configuration"] == {
        "cleaner_schema_version": 2,
        "strict": False,
        "supported_extensions": [".md", ".mdx"],
    }
    assert manifest["scanned_count"] == 3
    assert manifest["cleaned_count"] == 1
    assert manifest["skipped_count"] == 0
    assert manifest["invalid_count"] == 1
    assert manifest["empty_count"] == 1
    assert manifest["error_count"] == 0
    assert len(manifest["skipped_or_error_files"]) == 2


def test_cli_skip_invalid_reports_counts_without_traceback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    output = tmp_path / "documents"
    write(root / "good.mdx", "# Good\n\nUseful body.")
    write(root / "broken.mdx", "```python\nbroken")

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--input",
            str(root),
            "--output",
            str(output),
            "--skip-invalid",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0
    assert "Cleaned: 1" in completed.stdout
    assert "Invalid: 1" in completed.stdout
    assert "Errors: 0" in completed.stdout
    assert "Traceback" not in completed.stderr
