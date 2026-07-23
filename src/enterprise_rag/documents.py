"""Recursive loading of supported local documents."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.json_config import JsonLoaderSettings
from enterprise_rag.json_loader import load_json_file
from enterprise_rag.models import DocumentLoadWarning, LoadedDocument

SUPPORTED_EXTENSIONS = frozenset({".md", ".txt", ".json", ".jsonl"})
JSON_EXTENSIONS = frozenset({".json", ".jsonl"})

class DocumentLoadingError(RuntimeError): pass
class DocumentsDirectoryNotFoundError(DocumentLoadingError): pass
class NoSupportedDocumentsError(DocumentLoadingError): pass
class DocumentDecodeError(DocumentLoadingError): pass

@dataclass(frozen=True)
class DocumentLoadResult:
    documents: tuple[LoadedDocument, ...]
    skipped_empty: tuple[str, ...]
    warnings: tuple[DocumentLoadWarning, ...] = ()

def _document_id(source: str) -> str:
    return f"doc-{hashlib.sha256(source.encode('utf-8')).hexdigest()}"

def load_documents(source_dir: Path, *, json_config: JsonLoaderSettings | None = None) -> DocumentLoadResult:
    """Load supported files recursively; unsupported extensions are ignored."""
    source_dir = Path(source_dir).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise DocumentsDirectoryNotFoundError(f"Documents directory does not exist or is not a directory: {source_dir}")
    files = sorted((p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
                   key=lambda p: p.relative_to(source_dir).as_posix().casefold())
    if not files:
        raise NoSupportedDocumentsError(f"文件目錄中找不到支援的檔案：{source_dir}。目前支援：{', '.join(sorted(SUPPORTED_EXTENSIONS))}。")
    documents: list[LoadedDocument] = []
    skipped: list[str] = []
    warnings: list[DocumentLoadWarning] = []
    for path in files:
        source = path.relative_to(source_dir).as_posix()
        if path.suffix.lower() in JSON_EXTENSIONS:
            output = load_json_file(path, source, json_config)
            documents.extend(output.documents); skipped.extend(output.skipped_empty); warnings.extend(output.warnings)
            continue
        try: content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentDecodeError(f"Document cannot be decoded as UTF-8: {path}") from exc
        if not content.strip():
            skipped.append(source); continue
        documents.append(LoadedDocument(content, source, path.name, path.suffix.lower(), _document_id(source)))
    return DocumentLoadResult(tuple(documents), tuple(skipped), tuple(warnings))