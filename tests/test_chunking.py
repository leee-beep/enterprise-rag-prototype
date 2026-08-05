"""Tests for deterministic character-based document splitting."""

from pathlib import Path

import pytest

from enterprise_rag.chunking import split_documents
from enterprise_rag.config import Settings, load_settings
from enterprise_rag.documents import load_documents
from enterprise_rag.models import LoadedDocument

FIXTURES = Path(__file__).parent / "fixtures" / "documents"


def make_settings(*, chunk_size: int, chunk_overlap: int) -> Settings:
    return Settings(
        gemini_api_key=None,
        generation_model="unused-generation-model",
        embedding_model="unused-embedding-model",
        documents_dir=FIXTURES,
        vector_store_dir=Path("unused-vector-store"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=4,
    )


def make_document(content: str) -> LoadedDocument:
    return LoadedDocument(
        content=content,
        source="sample.txt",
        file_name="sample.txt",
        file_type=".txt",
        document_id="doc-stable-test-id",
    )


def test_long_documents_are_split_with_size_limit() -> None:
    documents = load_documents(FIXTURES).documents
    settings = make_settings(chunk_size=80, chunk_overlap=10)

    chunks = split_documents(documents, settings)

    assert len(chunks) > len(documents)
    assert all(0 < len(chunk.content) <= 80 for chunk in chunks)


def test_chunk_metadata_preserves_document_metadata() -> None:
    document = make_document("A" * 25)
    chunks = split_documents([document], make_settings(chunk_size=10, chunk_overlap=2))

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    for chunk in chunks:
        assert chunk.source == document.source
        assert chunk.file_name == document.file_name
        assert chunk.file_type == document.file_type
        assert chunk.document_id == document.document_id
        assert chunk.chunk_id == (
            f"{document.document_id}:chunk-{chunk.chunk_index:06d}"
        )


def test_chunk_ids_are_stable() -> None:
    document = make_document("0123456789" * 4)
    settings = make_settings(chunk_size=12, chunk_overlap=3)

    first = split_documents([document], settings)
    second = split_documents([document], settings)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.content for chunk in first] == [chunk.content for chunk in second]


def test_chunk_overlap_is_exact() -> None:
    document = make_document("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    chunks = split_documents([document], make_settings(chunk_size=10, chunk_overlap=3))

    assert [chunk.content for chunk in chunks] == [
        "ABCDEFGHIJ",
        "HIJKLMNOPQ",
        "OPQRSTUVWX",
        "VWXYZ",
    ]
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.content[-3:] == current.content[:3]


def test_prefers_paragraph_boundary_when_available() -> None:
    document = make_document(
        "First paragraph has text.\n\nSecond paragraph has more text for testing."
    )

    chunks = split_documents([document], make_settings(chunk_size=35, chunk_overlap=5))

    assert chunks[0].content.endswith("\n\n")
    assert len(chunks[0].content) <= 35


def test_high_overlap_still_makes_progress() -> None:
    document = make_document("line one\nline two\nline three\nline four")

    chunks = split_documents([document], make_settings(chunk_size=12, chunk_overlap=10))

    assert chunks
    assert len(chunks) < len(document.content)
    assert all(len(chunk.content) <= 12 for chunk in chunks)


def test_splitter_does_not_require_gemini_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("DOCUMENTS_DIR", str(FIXTURES))
    monkeypatch.setenv("CHUNK_SIZE", "90")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")
    settings = load_settings(load_env_file=False)
    documents = load_documents(settings.documents_dir).documents

    chunks = split_documents(documents, settings)

    assert settings.gemini_api_key is None
    assert chunks

def make_markdown(content: str, *, metadata: dict | None = None) -> LoadedDocument:
    return LoadedDocument(
        content=content,
        source="guide.md",
        file_name="guide.md",
        file_type=".md",
        document_id="doc-markdown-stable",
        metadata=metadata or {},
    )


def test_english_sentence_boundary_precedes_whitespace_fallback() -> None:
    document = make_document(
        "Alpha version 3.14 is stable. Second sentence contains additional details."
    )
    chunks = split_documents([document], make_settings(chunk_size=38, chunk_overlap=5))
    assert chunks[0].content.endswith("stable.")
    assert "3.14" in chunks[0].content


def test_chinese_sentence_boundary_is_supported() -> None:
    document = make_document(
        "\u7b2c\u4e00\u53e5\u8aaa\u660e\u3002\u7b2c\u4e8c\u53e5\u5305\u542b\u66f4\u591a\u6280\u8853\u7d30\u7bc0\u3002"
    )
    chunks = split_documents([document], make_settings(chunk_size=12, chunk_overlap=2))
    assert chunks[0].content.endswith("\u3002")


def test_whitespace_fallback_avoids_midword_cut() -> None:
    document = make_document("retriever returns relevant documents from storage")
    chunks = split_documents([document], make_settings(chunk_size=24, chunk_overlap=4))
    assert all(not (chunk.content[0].isalnum() and index > 0 and chunks[index - 1].content[-1].isalnum())
               for index, chunk in enumerate(chunks))
    assert all(len(chunk.content) <= 24 for chunk in chunks)


def test_unbroken_token_uses_deterministic_hard_cut() -> None:
    document = make_document("x" * 41)
    settings = make_settings(chunk_size=10, chunk_overlap=2)
    first = split_documents([document], settings)
    second = split_documents([document], settings)
    assert [chunk.content for chunk in first] == [chunk.content for chunk in second]
    assert all(len(chunk.content) <= 10 for chunk in first)


def test_markdown_heading_stays_with_first_paragraph() -> None:
    content = (
        "Introductory material before the section boundary.\n\n"
        "## Retriever\n\n"
        "A retriever is an interface that returns documents.\n\n"
        "Additional details follow here."
    )
    chunks = split_documents(
        [make_markdown(content)], make_settings(chunk_size=80, chunk_overlap=10)
    )
    assert any(
        "## Retriever" in chunk.content
        and "A retriever is an interface" in chunk.content
        for chunk in chunks
    )
    assert not any(chunk.content.strip() == "## Retriever" for chunk in chunks)


def test_multi_level_headings_preserve_order_and_are_not_standalone() -> None:
    content = "# Guide\n\nOverview text.\n\n### Details\n\nDetailed paragraph."
    chunks = split_documents(
        [make_markdown(content)], make_settings(chunk_size=35, chunk_overlap=5)
    )
    joined = "\n".join(chunk.content for chunk in chunks)
    assert joined.find("# Guide") < joined.find("### Details")
    assert not any(chunk.content.strip() in {"# Guide", "### Details"} for chunk in chunks)


def test_backtick_fenced_code_is_atomic_even_when_oversized() -> None:
    code = "```python\n# not a heading\n" + "print('value')\n" * 8 + "```\n"
    chunks = split_documents(
        [make_markdown("## Example\n\n" + code + "\nAfter code.")],
        make_settings(chunk_size=60, chunk_overlap=10),
    )
    containing = [chunk for chunk in chunks if "```python" in chunk.content]
    assert len(containing) == 1
    assert containing[0].content.count("```") == 2
    assert "# not a heading" in containing[0].content
    assert len(containing[0].content) > 60


def test_tilde_fence_with_indentation_and_language_is_atomic() -> None:
    code = "   ~~~~javascript\n# heading-like code\nconst value = 1;\n   ~~~~\n"
    chunks = split_documents(
        [make_markdown("Before.\n\n" + code + "\nAfter.")],
        make_settings(chunk_size=35, chunk_overlap=5),
    )
    containing = [chunk for chunk in chunks if "~~~~javascript" in chunk.content]
    assert len(containing) == 1
    assert "const value = 1;" in containing[0].content
    assert "   ~~~~" in containing[0].content


def test_unclosed_fence_is_preserved_as_atomic_to_end() -> None:
    content = "Before.\n\n```python\nprint('still code')\n# not heading\n"
    chunks = split_documents(
        [make_markdown(content)], make_settings(chunk_size=24, chunk_overlap=4)
    )
    code_chunks = [chunk for chunk in chunks if "```python" in chunk.content]
    assert len(code_chunks) == 1
    assert code_chunks[0].content.endswith("# not heading\n")


def test_multiple_fenced_blocks_remain_complete() -> None:
    content = (
        "```python\nprint(1)\n```\n\nText.\n\n"
        "~~~bash\necho ok\n~~~\n"
    )
    chunks = split_documents(
        [make_markdown(content)], make_settings(chunk_size=25, chunk_overlap=4)
    )
    assert sum("```python" in chunk.content for chunk in chunks) == 1
    assert sum("~~~bash" in chunk.content for chunk in chunks) == 1


def test_markdown_link_and_inline_code_are_not_split() -> None:
    content = (
        "Read the [retrieval documentation](https://example.com/retrieval) "
        "and call `vectorstore.as_retriever()` for results."
    )
    chunks = split_documents(
        [make_markdown(content)], make_settings(chunk_size=42, chunk_overlap=8)
    )
    assert any("[retrieval documentation](https://example.com/retrieval)" in chunk.content
               for chunk in chunks)
    assert any("`vectorstore.as_retriever()`" in chunk.content for chunk in chunks)


def test_list_table_and_blockquote_text_are_preserved() -> None:
    content = (
        "## Data\n\n- first\n- second\n\n"
        "| Name | Value |\n| --- | --- |\n| A | 1 |\n\n"
        "> quoted guidance\n"
    )
    chunks = split_documents(
        [make_markdown(content)], make_settings(chunk_size=45, chunk_overlap=5)
    )
    combined = "".join(chunk.content for chunk in chunks)
    for expected in ("- first", "| Name | Value |", "> quoted guidance"):
        assert expected in combined


def test_overlap_start_avoids_midword_and_always_progresses() -> None:
    content = "one two three four five six seven eight nine ten eleven twelve"
    chunks = split_documents(
        [make_document(content)], make_settings(chunk_size=20, chunk_overlap=9)
    )
    assert len(chunks) < len(content)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.content for chunk in chunks)
    for chunk in chunks[1:]:
        start = content.find(chunk.content)
        assert start <= 0 or not (content[start - 1].isalnum() and content[start].isalnum())


def test_metadata_and_json_identity_are_preserved() -> None:
    document = LoadedDocument(
        content="# FAQ\n\nA retriever returns documents. " * 5,
        source="faq.json#$[0]",
        file_name="faq.json",
        file_type=".json",
        document_id="doc-json-stable",
        metadata={"json_path": "$[0]", "title": "FAQ", "nested": {"a": 1}},
    )
    chunks = split_documents([document], make_settings(chunk_size=60, chunk_overlap=8))
    assert chunks
    for index, chunk in enumerate(chunks):
        assert chunk.source == document.source
        assert chunk.file_name == document.file_name
        assert chunk.file_type == ".json"
        assert chunk.document_id == document.document_id
        assert chunk.chunk_index == index
        assert chunk.chunk_id == f"{document.document_id}:chunk-{index:06d}"
        assert chunk.metadata == document.metadata
        assert chunk.metadata is not document.metadata


def test_empty_and_whitespace_documents_produce_no_chunks() -> None:
    documents = [make_document(""), make_document(" \n\t ")]
    assert split_documents(documents, make_settings(chunk_size=20, chunk_overlap=4)) == ()
