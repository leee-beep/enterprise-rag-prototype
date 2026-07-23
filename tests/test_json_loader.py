from __future__ import annotations
import json
from pathlib import Path
import pytest
from enterprise_rag.documents import load_documents
from enterprise_rag.json_config import JsonLoaderSettings
from enterprise_rag.json_loader import JsonDepthError, JsonFileSizeError, JsonParseError, JsonRecordLimitError, load_json_file


def write(path: Path, value, bom=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    path.write_text(text, encoding="utf-8-sig" if bom else "utf-8")
    return path


def load(tmp_path, value, config=None, suffix=".json", bom=False):
    path = write(tmp_path / ("sample" + suffix), value, bom)
    return load_json_file(path, path.name, config)


def test_single_object_and_metadata(tmp_path):
    out = load(tmp_path, {"title":"Intro", "content":"RAG text", "author":"Ada"})
    doc = out.documents[0]
    assert "# Intro" in doc.content and "RAG text" in doc.content
    assert doc.metadata | {"author":"Ada"} == doc.metadata
    assert doc.metadata["json_path"] == "$"
    assert doc.metadata["format"] == "json"


def test_object_array_paths_and_stable_ids(tmp_path):
    value = [{"title":"A", "body":"one"}, {"title":"B", "body":"two"}]
    first = load(tmp_path, value); second = load(tmp_path, value)
    assert [d.metadata["json_path"] for d in first.documents] == ["$[0]", "$[1]"]
    assert [d.document_id for d in first.documents] == [d.document_id for d in second.documents]


def test_nested_object_parent_headings(tmp_path):
    out = load(tmp_path, {"title":"Guide", "sections":{"title":"Search", "content":"Hybrid details"}}, JsonLoaderSettings(extraction_mode="recursive"))
    doc = out.documents[0]
    assert doc.metadata["parent_headings"] == ["Guide"]
    assert doc.metadata["title"] == "Search"
    assert doc.metadata["json_path"] == "$.sections"


def test_nested_array_and_mixed_structure(tmp_path):
    value = {"title":"Root", "chapters":[{"heading":"One", "paragraphs":["alpha", "beta"]}, {"ignored":7, "body":"gamma"}]}
    out = load(tmp_path, value, JsonLoaderSettings(extraction_mode="recursive"))
    assert len(out.documents) == 2
    assert "alpha" in out.documents[0].content and "beta" in out.documents[0].content
    assert "gamma" in out.documents[1].content


def test_root_string_array(tmp_path):
    out = load(tmp_path, ["first paragraph", "second paragraph"], JsonLoaderSettings(extraction_mode="recursive"))
    assert len(out.documents) == 1 and "second paragraph" in out.documents[0].content


def test_faq_question_answer(tmp_path):
    out = load(tmp_path, {"question":"What is RAG?", "answer":"Retrieval plus generation."})
    assert out.documents[0].metadata["title"] == "What is RAG?"
    assert "Retrieval plus generation." in out.documents[0].content


def test_custom_keys_and_metadata(tmp_path):
    cfg = JsonLoaderSettings(content_keys=("article",), title_keys=("label",), metadata_keys=("team",))
    doc = load(tmp_path, {"label":"Custom", "article":"usable", "team":"search"}, cfg).documents[0]
    assert doc.metadata["title"] == "Custom" and doc.metadata["team"] == "search"


def test_excluded_and_unknown_primitives_not_body(tmp_path):
    value = {"title":"Safe", "content":"keep", "embedding":[0.1,0.2], "base64":"AAAA", "secret_note":"do not infer"}
    text = load(tmp_path, value).documents[0].content
    assert "keep" in text and "0.1" not in text and "AAAA" not in text and "do not infer" not in text


def test_jsonl_and_invalid_line_modes(tmp_path):
    text = '{"content":"one"}\nnot-json\n{"content":"two"}\n'
    path = write(tmp_path / "items.jsonl", text)
    with pytest.raises(JsonParseError, match="line 2"):
        load_json_file(path, "items.jsonl", JsonLoaderSettings(strict_mode=True))
    out = load_json_file(path, "items.jsonl", JsonLoaderSettings(strict_mode=False))
    assert len(out.documents) == 2 and out.warnings[0].line_number == 2


def test_utf8_bom(tmp_path):
    assert load(tmp_path, {"content":"BOM works"}, bom=True).documents[0].content == "BOM works"


def test_invalid_json_location(tmp_path):
    path = write(tmp_path / "broken.json", '{"content": }')
    with pytest.raises(JsonParseError, match=r"broken.json.*line 1, column"):
        load_json_file(path, "broken.json")


def test_depth_strict_and_non_strict(tmp_path):
    value = {"outer":{"inner":{"content":"too deep"}}}
    cfg = JsonLoaderSettings(extraction_mode="recursive", max_depth=1)
    with pytest.raises(JsonDepthError): load(tmp_path, value, cfg)
    out = load(tmp_path, value, JsonLoaderSettings(extraction_mode="recursive", max_depth=1, strict_mode=False))
    assert not out.documents and out.warnings


def test_record_limit(tmp_path):
    value = [{"content":"one"}, {"content":"two"}]
    with pytest.raises(JsonRecordLimitError): load(tmp_path, value, JsonLoaderSettings(max_records=1))
    out = load(tmp_path, value, JsonLoaderSettings(max_records=1, strict_mode=False))
    assert len(out.documents) == 1 and out.warnings


def test_file_size_strict_and_non_strict(tmp_path):
    path = write(tmp_path / "large.json", {"content":"x" * 200})
    cfg = JsonLoaderSettings(max_file_size_mb=0.00001)
    with pytest.raises(JsonFileSizeError): load_json_file(path, "large.json", cfg)
    out = load_json_file(path, "large.json", JsonLoaderSettings(max_file_size_mb=0.00001, strict_mode=False))
    assert not out.documents and out.warnings


def test_empty_content_is_reported(tmp_path):
    out = load(tmp_path, {"content":"   "}, JsonLoaderSettings(extraction_mode="record"))
    assert not out.documents


def test_document_dispatch_preserves_txt_md_and_ignores_pdf(tmp_path):
    write(tmp_path / "a.md", "markdown"); write(tmp_path / "b.txt", "text"); write(tmp_path / "c.pdf", "fake")
    write(tmp_path / "d.json", {"content":"json"})
    result = load_documents(tmp_path)
    assert {d.file_type for d in result.documents} == {".md", ".txt", ".json"}