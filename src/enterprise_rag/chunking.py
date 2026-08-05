"""Deterministic structure-aware text splitting."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from enterprise_rag.config import Settings
from enterprise_rag.models import DocumentChunk, LoadedDocument

_MARKDOWN_EXTENSIONS = frozenset({".md", ".mdx"})
_FENCE_LINE = re.compile(r"(?m)^ {0,3}(`{3,}|~{3,})([^\r\n]*)(?:\r?\n|$)")
_HEADING_LINE = re.compile(r"(?m)^ {0,3}#{1,6}[ \t]+[^\r\n]*(?:\r?\n|$)")
_PARAGRAPH_BOUNDARY = re.compile(r"\r?\n[ \t]*\r?\n")
_NEWLINE_BOUNDARY = re.compile(r"\r?\n")
_WHITESPACE_BOUNDARY = re.compile(r"[ \t]+")
_INLINE_CODE = re.compile(r"(`{1,2})([^\r\n]*?)\1")
_MARKDOWN_LINK = re.compile(r"!?\[[^\]\r\n]+\]\([^\r\n)]*\)")
_LATIN_SENTENCE_END = re.compile(r"[.!?](?=[ \t\r\n]|$)")
_CJK_SENTENCE_END = re.compile(r"[\u3002\uff1f\uff01]")


@dataclass(frozen=True)
class _ProtectedRange:
    start: int
    end: int
    kind: str
    atomic: bool = False


def _inside(position: int, ranges: Sequence[_ProtectedRange]) -> _ProtectedRange | None:
    """Return the smallest protected range whose interior contains position."""
    matches = [item for item in ranges if item.start < position < item.end]
    return min(matches, key=lambda item: item.end - item.start) if matches else None


def _outside_fences(position: int, fences: Sequence[_ProtectedRange]) -> bool:
    return not any(item.start <= position < item.end for item in fences)


def _fenced_ranges(text: str) -> tuple[_ProtectedRange, ...]:
    ranges: list[_ProtectedRange] = []
    opening: tuple[int, str, int] | None = None
    for match in _FENCE_LINE.finditer(text):
        token = match.group(1)
        remainder = match.group(2).strip()
        if opening is None:
            opening = (match.start(), token[0], len(token))
        elif token[0] == opening[1] and len(token) >= opening[2] and not remainder:
            ranges.append(_ProtectedRange(opening[0], match.end(), "fence", atomic=True))
            opening = None
    if opening is not None:
        ranges.append(_ProtectedRange(opening[0], len(text), "fence", atomic=True))
    return tuple(ranges)


def _first_block_end(
    text: str,
    content_start: int,
    fences: Sequence[_ProtectedRange],
) -> int:
    for fence in fences:
        if fence.start == content_start:
            return fence.end
    paragraph = _PARAGRAPH_BOUNDARY.search(text, content_start)
    return paragraph.end() if paragraph else len(text)


def _heading_bundles(
    text: str,
    fences: Sequence[_ProtectedRange],
    hard_span: int,
) -> tuple[_ProtectedRange, ...]:
    bundles: list[_ProtectedRange] = []
    for match in _HEADING_LINE.finditer(text):
        if not _outside_fences(match.start(), fences):
            continue
        content_start = match.end()
        while content_start < len(text) and text[content_start] in "\r\n":
            content_start += 1
        if content_start >= len(text):
            continue
        end = _first_block_end(text, content_start, fences)
        if end - match.start() <= hard_span:
            bundles.append(_ProtectedRange(match.start(), end, "heading_bundle"))
    return tuple(bundles)


def _inline_ranges(
    text: str,
    fences: Sequence[_ProtectedRange],
) -> tuple[_ProtectedRange, ...]:
    ranges: list[_ProtectedRange] = []
    for kind, pattern in (("inline_code", _INLINE_CODE), ("markdown_link", _MARKDOWN_LINK)):
        for match in pattern.finditer(text):
            if _outside_fences(match.start(), fences):
                ranges.append(_ProtectedRange(match.start(), match.end(), kind, atomic=True))
    return tuple(ranges)


def _protected_ranges(
    text: str,
    *,
    markdown: bool,
    hard_span: int,
) -> tuple[_ProtectedRange, ...]:
    if not markdown:
        return ()
    fences = _fenced_ranges(text)
    return tuple(sorted(
        (*fences, *_heading_bundles(text, fences, hard_span), *_inline_ranges(text, fences)),
        key=lambda item: (item.start, item.end, item.kind),
    ))


def _sentence_boundaries(text: str) -> tuple[int, ...]:
    positions = [match.end() for match in _LATIN_SENTENCE_END.finditer(text)]
    positions.extend(match.end() for match in _CJK_SENTENCE_END.finditer(text))
    return tuple(sorted(positions))


def _boundary_groups(
    text: str,
    *,
    markdown: bool,
    protected: Sequence[_ProtectedRange],
) -> tuple[tuple[int, ...], ...]:
    fences = tuple(item for item in protected if item.kind == "fence")
    headings = (
        tuple(match.start() for match in _HEADING_LINE.finditer(text)
              if _outside_fences(match.start(), fences))
        if markdown else ()
    )
    fence_boundaries = tuple(position for item in fences for position in (item.start, item.end))
    return (
        headings,
        fence_boundaries,
        tuple(match.end() for match in _PARAGRAPH_BOUNDARY.finditer(text)),
        tuple(match.end() for match in _NEWLINE_BOUNDARY.finditer(text)),
        _sentence_boundaries(text),
        tuple(match.end() for match in _WHITESPACE_BOUNDARY.finditer(text)),
    )


def _choose_candidate(
    groups: Sequence[Sequence[int]],
    *,
    start: int,
    target: int,
    minimum: int,
    hard_end: int,
    protected: Sequence[_ProtectedRange],
) -> int | None:
    for group in groups:
        candidates = [
            position for position in group
            if minimum <= position <= hard_end
            and position > start
            and _inside(position, protected) is None
        ]
        backward = [position for position in candidates if position <= target]
        if backward:
            return max(backward)
        forward = [position for position in candidates if position > target]
        if forward:
            return min(forward)
    return None


def _preferred_end(
    text: str,
    start: int,
    maximum_end: int,
    chunk_overlap: int,
    *,
    markdown: bool = False,
    protected: Sequence[_ProtectedRange] = (),
    boundary_groups: Sequence[Sequence[int]] | None = None,
    forward_tolerance: int = 0,
) -> int:
    """Choose the highest-priority safe boundary near the target end."""
    if maximum_end >= len(text):
        return len(text)
    target = maximum_end
    hard_end = min(len(text), target + max(0, forward_tolerance))
    minimum = start + max(chunk_overlap + 1, (target - start) // 2)

    containing_start = _inside(start, protected)
    if containing_start is not None and containing_start.atomic:
        return containing_start.end

    containing_target = _inside(target, protected)
    if containing_target is not None:
        prefix = containing_target.start - start
        if prefix >= max(chunk_overlap + 1, (target - start) // 2):
            return containing_target.start
        if containing_target.atomic or containing_target.end <= hard_end:
            return containing_target.end

    groups = boundary_groups or _boundary_groups(
        text, markdown=markdown, protected=protected
    )
    candidate = _choose_candidate(
        groups,
        start=start,
        target=target,
        minimum=minimum,
        hard_end=hard_end,
        protected=protected,
    )
    if candidate is not None:
        return candidate

    if containing_target is not None and containing_target.start > start:
        return containing_target.start
    return target


def _safe_overlap_start(
    text: str,
    *,
    current_start: int,
    end: int,
    overlap: int,
    protected: Sequence[_ProtectedRange],
    boundary_groups: Sequence[Sequence[int]],
) -> int:
    if overlap == 0 or end - current_start <= overlap:
        return end
    desired = max(current_start + 1, end - overlap)
    if desired >= end:
        return end

    containing = _inside(desired, protected)
    if containing is not None:
        if containing.atomic and containing.end <= end:
            return containing.end
        if current_start < containing.start < end:
            desired = containing.start
        elif current_start < containing.end < end:
            desired = containing.end
        else:
            return end

    tolerance = max(4, min(64, overlap // 2))
    candidates = {
        position
        for group in boundary_groups
        for position in group
        if current_start < position < end
        and abs(position - desired) <= tolerance
        and _inside(position, protected) is None
    }
    if candidates:
        desired = min(candidates, key=lambda position: (abs(position - desired), position < desired))
    return min(end, max(current_start + 1, desired))


def _chunk_id(document_id: str, chunk_index: int) -> str:
    """Build a stable chunk identifier from stable document information."""
    return f"{document_id}:chunk-{chunk_index:06d}"


def split_documents(
    documents: Iterable[LoadedDocument],
    settings: Settings,
) -> tuple[DocumentChunk, ...]:
    """Split documents at deterministic, structure-aware character boundaries.

    Normal chunks never exceed `chunk_size`. Markdown fenced code blocks are
    atomic: a block larger than that bound is emitted intact as an explicitly
    oversized chunk. Overlap is best-effort and may shrink to avoid starting
    inside a protected Markdown region.
    """
    if settings.chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if settings.chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative.")
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[DocumentChunk] = []
    forward_tolerance = 0
    hard_span = settings.chunk_size + forward_tolerance
    for document in documents:
        if not document.content.strip():
            continue
        text = document.content
        markdown = document.file_type.lower() in _MARKDOWN_EXTENSIONS
        protected = _protected_ranges(text, markdown=markdown, hard_span=hard_span)
        groups = _boundary_groups(text, markdown=markdown, protected=protected)

        start = 0
        chunk_index = 0
        while start < len(text):
            maximum_end = min(start + settings.chunk_size, len(text))
            end = _preferred_end(
                text,
                start,
                maximum_end,
                settings.chunk_overlap,
                markdown=markdown,
                protected=protected,
                boundary_groups=groups,
                forward_tolerance=forward_tolerance,
            )
            if end <= start:
                end = min(len(text), start + settings.chunk_size)
            content = text[start:end]
            if not content:
                break

            chunks.append(
                DocumentChunk(
                    content=content,
                    source=document.source,
                    file_name=document.file_name,
                    file_type=document.file_type,
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    chunk_id=_chunk_id(document.document_id, chunk_index),
                    metadata=dict(document.metadata),
                )
            )

            if end >= len(text):
                break
            next_start = _safe_overlap_start(
                text,
                current_start=start,
                end=end,
                overlap=settings.chunk_overlap,
                protected=protected,
                boundary_groups=groups,
            )
            start = next_start if next_start > start else end
            chunk_index += 1

    return tuple(chunks)
