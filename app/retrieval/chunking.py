"""Normalize source text, split into chunks, and assign stable IDs."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.retrieval.document_store import DocumentChunk

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Trim edges and collapse internal whitespace runs.

    :param text: Raw document text.
    :returns: Normalized text, possibly empty."""
    return _WHITESPACE_RE.sub(" ", text).strip()


class SourceDocument(BaseModel):
    """A raw source document before chunking.

    :ivar source_id: Stable identifier for the source (often the file stem).
    :ivar title: Human-readable title.
    :ivar text: Full source body text.
    :ivar framework: Optional framework filter metadata.
    :ivar language: Optional language filter metadata.
    :ivar metadata: Additional non-secret provenance fields."""

    source_id: str
    title: str
    text: str
    framework: str | None = None
    language: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


def chunk_document(
    source: SourceDocument,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Split a source into overlapping character windows with stable IDs.

    Chunk IDs use ``{source_id}:{ordinal}`` so re-running the same splitter on
    the same normalized text yields the same identifiers. Oversized windows are
    not silently truncated beyond ``chunk_size``; each window is at most
    ``chunk_size`` characters.

    :param source: Source document to chunk.
    :param chunk_size: Maximum characters per chunk; must be >= 1.
    :param chunk_overlap: Characters shared with the previous chunk; must be
        >= 0 and < ``chunk_size``.
    :returns: Ordered chunk list, empty when normalized text is blank.
    :raises ValueError: If size/overlap arguments are invalid."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    text = normalize_text(source.text)
    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[DocumentChunk] = []
    start = 0
    ordinal = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end]
        chunks.append(
            DocumentChunk(
                source_id=source.source_id,
                chunk_id=f"{source.source_id}:{ordinal}",
                text=piece,
                title=source.title,
                framework=source.framework,
                language=source.language,
                metadata=dict(source.metadata),
            )
        )
        ordinal += 1
        if end >= len(text):
            break
        start += step
    return chunks
