"""Configurable JSON and JSONL document extraction."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from enterprise_rag.json_config import JsonLoaderSettings
from enterprise_rag.models import DocumentLoadWarning, LoadedDocument

class JsonIngestionError(RuntimeError): pass
class JsonFileSizeError(JsonIngestionError): pass
class JsonParseError(JsonIngestionError): pass
class JsonDepthError(JsonIngestionError): pass
class JsonRecordLimitError(JsonIngestionError): pass

@dataclass(frozen=True)
class JsonLoadOutput:
    documents: tuple[LoadedDocument, ...]
    warnings: tuple[DocumentLoadWarning, ...] = ()
    skipped_empty: tuple[str, ...] = ()

@dataclass
class _State:
    path: Path
    source: str
    fmt: str
    requested_mode: str
    resolved_mode: str
    config: JsonLoaderSettings
    documents: list[LoadedDocument] = field(default_factory=list)
    warnings: list[DocumentLoadWarning] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    stopped: bool = False

    def issue(self, message: str, json_path: str, exc_type: type[JsonIngestionError]) -> None:
        full = f"{message} File: {self.path}; JSON path: {json_path}."
        if self.config.strict_mode:
            raise exc_type(full)
        self.warnings.append(DocumentLoadWarning(self.source, full, json_path=json_path))


def _lookup(obj: dict[str, Any], keys: tuple[str, ...]) -> list[tuple[str, Any]]:
    wanted = {key.casefold() for key in keys}
    return [(key, value) for key, value in obj.items() if key.casefold() in wanted]


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _text_parts(value: Any) -> list[str]:
    if _is_text(value):
        return [value.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item.strip() for item in value if item.strip()]
    return []


def _record_like(value: Any, config: JsonLoaderSettings) -> bool:
    return isinstance(value, dict) and any(_text_parts(v) for _, v in _lookup(value, config.content_keys))


def _auto_mode(root: Any, config: JsonLoaderSettings) -> str:
    if _record_like(root, config):
        return "record"
    if isinstance(root, list) and any(_record_like(item, config) for item in root):
        return "record"
    return "recursive"


def _metadata(obj: dict[str, Any], config: JsonLoaderSettings, inherited: dict[str, Any]) -> dict[str, Any]:
    result = dict(inherited)
    for key, value in _lookup(obj, config.metadata_keys):
        if isinstance(value, (str, int, float, bool)) or value is None or (
            isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value)
        ):
            result[key] = value
    return result


def _title(obj: dict[str, Any], config: JsonLoaderSettings) -> str | None:
    for _, value in _lookup(obj, config.title_keys):
        if _is_text(value):
            return value.strip()
    return None


def _make_document(state: _State, parts: list[str], json_path: str, title: str | None,
                   headings: tuple[str, ...], meta: dict[str, Any]) -> None:
    parts = [part.strip() for part in parts if part.strip()]
    if not parts:
        state.skipped.append(f"{state.source}#{json_path}")
        state.warnings.append(DocumentLoadWarning(state.source, "No extractable text; record skipped.", json_path=json_path))
        return
    if len(state.documents) >= state.config.max_records:
        state.issue(f"Record limit exceeded (max_records={state.config.max_records}).", json_path, JsonRecordLimitError)
        state.stopped = True
        return
    prefix = [f"{'#' * min(i + 1, 6)} {heading}" for i, heading in enumerate(headings)]
    if title and (not headings or title != headings[-1]):
        prefix.append(f"{'#' * min(len(headings) + 1, 6)} {title}")
    content = "\n\n".join(prefix + parts)
    index = len(state.documents)
    stable = hashlib.sha256(f"{state.source}#{json_path}".encode()).hexdigest()
    metadata = {
        "source": state.source, "format": state.fmt, "json_path": json_path,
        "record_index": index, "extraction_mode": state.requested_mode,
        "resolved_extraction_mode": state.resolved_mode, **meta,
    }
    if title: metadata["title"] = title
    if headings: metadata["parent_headings"] = list(headings)
    state.documents.append(LoadedDocument(content, state.source, state.path.name, state.path.suffix.lower(), f"doc-{stable}", metadata))


def _check_depth(state: _State, depth: int, json_path: str) -> bool:
    if depth <= state.config.max_depth:
        return True
    state.issue(f"Maximum JSON depth exceeded (max_depth={state.config.max_depth}).", json_path, JsonDepthError)
    return False


def _record_walk(value: Any, state: _State, json_path: str = "$", depth: int = 0,
                 headings: tuple[str, ...] = (), inherited: dict[str, Any] | None = None) -> None:
    if state.stopped or not _check_depth(state, depth, json_path): return
    inherited = inherited or {}
    if isinstance(value, dict):
        excluded = {k.casefold() for k in state.config.exclude_keys}
        title = _title(value, state.config)
        meta = _metadata(value, state.config, inherited)
        content_fields = _lookup(value, state.config.content_keys)
        parts = [part for _, item in content_fields for part in _text_parts(item)]
        if content_fields:
            _make_document(state, parts, json_path, title, headings, meta)
        else:
            next_headings = headings + ((title,) if title else ())
            for key, child in value.items():
                if key.casefold() in excluded: continue
                if isinstance(child, (dict, list)):
                    _record_walk(child, state, f"{json_path}.{key}", depth + 1, next_headings, meta)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _record_walk(child, state, f"{json_path}[{index}]", depth + 1, headings, inherited)


def _recursive_walk(value: Any, state: _State, json_path: str = "$", depth: int = 0,
                    headings: tuple[str, ...] = (), inherited: dict[str, Any] | None = None) -> None:
    if state.stopped or not _check_depth(state, depth, json_path): return
    inherited = inherited or {}
    if isinstance(value, dict):
        excluded = {k.casefold() for k in state.config.exclude_keys}
        title = _title(value, state.config)
        meta = _metadata(value, state.config, inherited)
        content_fields = _lookup(value, state.config.content_keys)
        parts = [part for _, item in content_fields for part in _text_parts(item)]
        current_headings = headings + ((title,) if title else ())
        if content_fields: _make_document(state, parts, json_path, title, headings, meta)
        structural = set(state.config.content_keys + state.config.title_keys + state.config.metadata_keys)
        structural = {key.casefold() for key in structural} | excluded
        for key, child in value.items():
            if key.casefold() in structural: continue
            child_path = f"{json_path}.{key}"
            if isinstance(child, (dict, list)):
                _recursive_walk(child, state, child_path, depth + 1, current_headings, meta)
    elif isinstance(value, list):
        strings = _text_parts(value)
        if strings:
            _make_document(state, strings, json_path, None, headings, inherited)
        else:
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    _recursive_walk(child, state, f"{json_path}[{index}]", depth + 1, headings, inherited)


def _read_roots(path: Path, source: str, config: JsonLoaderSettings) -> tuple[list[tuple[Any, str]], list[DocumentLoadWarning]]:
    size_limit = config.max_file_size_mb * 1024 * 1024
    if path.stat().st_size > size_limit:
        raise JsonFileSizeError(f"JSON file exceeds max_file_size_mb={config.max_file_size_mb}: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise JsonParseError(f"JSON file is not valid UTF-8: {path}") from exc
    warnings: list[DocumentLoadWarning] = []
    if path.suffix.lower() == ".json":
        try: return [(json.loads(text), "$")], warnings
        except json.JSONDecodeError as exc:
            raise JsonParseError(f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    roots: list[tuple[Any, str]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        try: roots.append((json.loads(line), f"$line[{line_no}]"))
        except json.JSONDecodeError as exc:
            message = f"Invalid JSONL in {path} at line {line_no}, column {exc.colno}: {exc.msg}"
            if config.strict_mode: raise JsonParseError(message) from exc
            warnings.append(DocumentLoadWarning(source, message, line_number=line_no))
    return roots, warnings


def load_json_file(path: Path, source: str, config: JsonLoaderSettings | None = None) -> JsonLoadOutput:
    """Read and extract one JSON or JSONL file without external services."""
    config = config or JsonLoaderSettings()
    try:
        roots, warnings = _read_roots(path, source, config)
    except JsonIngestionError as exc:
        if config.strict_mode: raise
        return JsonLoadOutput((), (DocumentLoadWarning(source, str(exc)),), ())
    all_documents: list[LoadedDocument] = []
    all_warnings = list(warnings)
    skipped: list[str] = []
    for root, root_path in roots:
        remaining = config.max_records - len(all_documents)
        if remaining < 1:
            message = f"Record limit exceeded (max_records={config.max_records}). File: {path}."
            if config.strict_mode:
                raise JsonRecordLimitError(message)
            all_warnings.append(DocumentLoadWarning(source, message, json_path=root_path))
            break
        resolved = _auto_mode(root, config) if config.extraction_mode == "auto" else config.extraction_mode
        state_config = replace(config, max_records=remaining)
        state = _State(path, source, path.suffix.lower().lstrip("."), config.extraction_mode, resolved, state_config)
        if resolved == "record": _record_walk(root, state, root_path)
        else: _recursive_walk(root, state, root_path)
        offset = len(all_documents)
        for local_document in state.documents:
            normalized = dict(local_document.metadata)
            normalized["record_index"] = offset + int(normalized["record_index"])
            all_documents.append(replace(local_document, metadata=normalized))
        all_warnings.extend(state.warnings)
        skipped.extend(state.skipped)
    return JsonLoadOutput(tuple(all_documents), tuple(all_warnings), tuple(skipped))