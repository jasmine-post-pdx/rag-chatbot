"""Normalize source text, split into chunks, and assign stable IDs."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.document_store import DocumentChunk

_WHITESPACE_RE = re.compile(r"\s+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```")


def normalize_text(text: str) -> str:
    """Trim edges and collapse internal whitespace runs.

    :param text: Raw document text.
    :returns: Normalized text, possibly empty."""
    return _WHITESPACE_RE.sub(" ", text).strip()


class ChunkingStrategy(str, Enum):
    """Supported Step 7 chunking variants."""

    RECURSIVE_CHARACTER = "recursive_character"
    TOKEN_AWARE = "token_aware"
    STRUCTURE_AWARE = "structure_aware"


class ChunkingConfig(BaseModel):
    """Configuration for a chunking strategy run.

    ``chunk_size`` / ``chunk_overlap`` are characters for character and
    structure strategies, and tokens for ``token_aware``.

    :ivar strategy: Which splitter to run.
    :ivar chunk_size: Maximum size per chunk in strategy units.
    :ivar chunk_overlap: Overlap with the previous chunk in strategy units.
    :ivar token_counter: Optional counter for token-aware splitting."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: ChunkingStrategy
    chunk_size: int = 500
    chunk_overlap: int = 50
    token_counter: Any | None = None


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


def _validate_size_overlap(*, chunk_size: int, chunk_overlap: int) -> None:
    """Raise ``ValueError`` when size/overlap arguments are invalid.

    :param chunk_size: Maximum chunk size.
    :param chunk_overlap: Overlap with the previous chunk.
    :raises ValueError: If arguments cannot be used."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")


def _make_chunk(
    source: SourceDocument,
    *,
    ordinal: int,
    text: str,
) -> DocumentChunk:
    """Build a ``DocumentChunk`` with inherited provenance.

    :param source: Parent source document.
    :param ordinal: Zero-based chunk ordinal within the source.
    :param text: Chunk body text.
    :returns: A provenance-preserving chunk."""
    return DocumentChunk(
        source_id=source.source_id,
        chunk_id=f"{source.source_id}:{ordinal}",
        text=text,
        title=source.title,
        framework=source.framework,
        language=source.language,
        metadata=dict(source.metadata),
    )


def _split_character_windows(
    source: SourceDocument,
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Split ``text`` into overlapping character windows with stable IDs.

    :param source: Parent source for provenance and IDs.
    :param text: Already-prepared body text.
    :param chunk_size: Maximum characters per chunk.
    :param chunk_overlap: Characters shared with the previous chunk.
    :returns: Ordered chunk list."""
    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[DocumentChunk] = []
    start = 0
    ordinal = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(_make_chunk(source, ordinal=ordinal, text=text[start:end]))
        ordinal += 1
        if end >= len(text):
            break
        start += step
    return chunks


def _split_token_aware(
    source: SourceDocument,
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    token_counter: Any,
) -> list[DocumentChunk]:
    """Split ``text`` so each chunk stays within a token budget.

    :param source: Parent source for provenance and IDs.
    :param text: Normalized body text.
    :param chunk_size: Maximum tokens per chunk.
    :param chunk_overlap: Tokens of overlap with the previous chunk.
    :param token_counter: Object with a ``count(text) -> int`` method.
    :returns: Ordered chunk list."""
    if not text:
        return []

    words = text.split(" ")
    chunks: list[DocumentChunk] = []
    ordinal = 0
    start = 0

    while start < len(words):
        end = start
        piece = ""
        while end < len(words):
            candidate = " ".join(words[start : end + 1])
            if token_counter.count(candidate) > chunk_size:
                break
            piece = candidate
            end += 1

        if not piece:
            # Single oversized token: emit it alone so progress continues.
            piece = words[start]
            end = start + 1

        chunks.append(_make_chunk(source, ordinal=ordinal, text=piece))
        ordinal += 1

        if end >= len(words):
            break

        # Advance with token overlap measured on the emitted piece.
        if chunk_overlap <= 0:
            start = end
            continue

        overlap_words: list[str] = []
        for word in reversed(piece.split(" ")):
            candidate = " ".join([word, *overlap_words]) if overlap_words else word
            if token_counter.count(candidate) > chunk_overlap:
                break
            overlap_words.insert(0, word)
        start = max(end - len(overlap_words), start + 1)

    return chunks


def _iter_structure_units(text: str) -> list[str]:
    """Split markdown into heading sections and fenced code units.

    :param text: Raw markdown with newlines preserved.
    :returns: Ordered structural units (non-empty)."""
    lines = text.splitlines(keepends=True)
    units: list[str] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal buffer
        if buffer:
            units.append("".join(buffer))
            buffer = []

    for line in lines:
        if _FENCE_RE.match(line.strip()):
            if in_fence:
                buffer.append(line)
                flush()
                in_fence = False
            else:
                # Start a new fence unit; flush prior prose first.
                flush()
                buffer.append(line)
                in_fence = True
            continue

        if in_fence:
            buffer.append(line)
            continue

        if _HEADING_RE.match(line.rstrip("\n")):
            flush()
            buffer.append(line)
            continue

        buffer.append(line)

    flush()
    return [unit for unit in units if unit.strip()]


def _split_structure_aware(
    source: SourceDocument,
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Split on markdown structure, falling back to character windows.

    :param source: Parent source for provenance and IDs.
    :param text: Body text with newlines preserved.
    :param chunk_size: Maximum characters per chunk.
    :param chunk_overlap: Character overlap for oversized fallbacks.
    :returns: Ordered chunk list."""
    stripped = text.strip()
    if not stripped:
        return []

    units = _iter_structure_units(text)
    if not units:
        return []

    pieces: list[str] = []
    for unit in units:
        unit_text = unit.strip("\n")
        if len(unit_text) <= chunk_size:
            pieces.append(unit_text)
            continue
        # Oversized section: character windows without collapsing newlines first.
        step = chunk_size - chunk_overlap
        start = 0
        while start < len(unit_text):
            end = min(start + chunk_size, len(unit_text))
            pieces.append(unit_text[start:end])
            if end >= len(unit_text):
                break
            start += step

    return [
        _make_chunk(source, ordinal=index, text=piece)
        for index, piece in enumerate(pieces)
    ]


def chunk_document(
    source: SourceDocument,
    *,
    config: ChunkingConfig | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Split a source into overlapping windows with stable IDs.

    When ``config`` is omitted, behavior matches the Step 6 character-window
    splitter (normalize whitespace, then fixed windows). When ``config`` is
    provided, dispatch to the selected ``ChunkingStrategy``.

    Chunk IDs use ``{source_id}:{ordinal}`` so re-running the same splitter on
    the same prepared text yields the same identifiers.

    :param source: Source document to chunk.
    :param config: Optional strategy configuration.
    :param chunk_size: Legacy maximum characters per chunk when ``config`` is
        omitted; must be >= 1.
    :param chunk_overlap: Legacy character overlap when ``config`` is omitted;
        must be >= 0 and < ``chunk_size``.
    :returns: Ordered chunk list, empty when prepared text is blank.
    :raises ValueError: If size/overlap arguments are invalid."""
    if config is None:
        _validate_size_overlap(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text = normalize_text(source.text)
        return _split_character_windows(
            source,
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    _validate_size_overlap(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    if config.strategy == ChunkingStrategy.RECURSIVE_CHARACTER:
        text = normalize_text(source.text)
        return _split_character_windows(
            source,
            text,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    if config.strategy == ChunkingStrategy.TOKEN_AWARE:
        from app.services.token_budget import ApproximateTokenCounter

        counter = config.token_counter
        if counter is None:
            counter = ApproximateTokenCounter()
        text = normalize_text(source.text)
        return _split_token_aware(
            source,
            text,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            token_counter=counter,
        )

    if config.strategy == ChunkingStrategy.STRUCTURE_AWARE:
        return _split_structure_aware(
            source,
            source.text,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    raise ValueError(f"Unsupported chunking strategy: {config.strategy}")
