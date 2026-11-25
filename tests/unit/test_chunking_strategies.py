"""Unit tests for Step 7 chunking strategy variants.

These tests lock the ``ChunkingStrategy`` / ``ChunkingConfig`` contract.
They are expected to fail until the Step 7 implementation lands.
"""

from __future__ import annotations

import pytest

from app.retrieval.chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    SourceDocument,
    chunk_document,
)
from app.services.token_budget import ApproximateTokenCounter


def _source(
    *,
    source_id: str = "demo",
    title: str = "Demo",
    text: str,
    framework: str | None = "fastapi",
    language: str | None = "python",
    metadata: dict[str, str] | None = None,
) -> SourceDocument:
    """Build a source document for strategy tests.

    :returns: A ``SourceDocument`` with provenance fields."""
    return SourceDocument(
        source_id=source_id,
        title=title,
        text=text,
        framework=framework,
        language=language,
        metadata=metadata
        if metadata is not None
        else {"path": "data/raw/demo.md", "version": "1"},
    )


@pytest.mark.parametrize(
    "strategy",
    [
        ChunkingStrategy.RECURSIVE_CHARACTER,
        ChunkingStrategy.TOKEN_AWARE,
        ChunkingStrategy.STRUCTURE_AWARE,
    ],
)
def test_each_strategy_assigns_deterministic_ids(strategy: ChunkingStrategy):
    """Re-running the same source and config yields identical chunk IDs."""
    source = _source(text=("alpha beta gamma delta epsilon zeta " * 20).strip())
    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=40 if strategy != ChunkingStrategy.TOKEN_AWARE else 12,
        chunk_overlap=8 if strategy != ChunkingStrategy.TOKEN_AWARE else 2,
        token_counter=ApproximateTokenCounter()
        if strategy == ChunkingStrategy.TOKEN_AWARE
        else None,
    )

    first = chunk_document(source, config=config)
    second = chunk_document(source, config=config)

    assert first
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == "demo:0"
    for index, chunk in enumerate(first):
        assert chunk.chunk_id == f"demo:{index}"


@pytest.mark.parametrize(
    "strategy",
    [
        ChunkingStrategy.RECURSIVE_CHARACTER,
        ChunkingStrategy.TOKEN_AWARE,
        ChunkingStrategy.STRUCTURE_AWARE,
    ],
)
def test_each_strategy_preserves_inherited_metadata(strategy: ChunkingStrategy):
    """Every chunk inherits source title, filters, path, and version metadata."""
    source = _source(
        source_id="rag-notes",
        title="RAG Notes",
        text="Context budgeting reserves tokens for instructions and output.",
        framework=None,
        language="python",
        metadata={"path": "data/raw/rag.md", "version": "2"},
    )
    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=200 if strategy != ChunkingStrategy.TOKEN_AWARE else 64,
        chunk_overlap=20 if strategy != ChunkingStrategy.TOKEN_AWARE else 4,
        token_counter=ApproximateTokenCounter()
        if strategy == ChunkingStrategy.TOKEN_AWARE
        else None,
    )

    chunks = chunk_document(source, config=config)

    assert chunks
    for chunk in chunks:
        assert chunk.source_id == "rag-notes"
        assert chunk.title == "RAG Notes"
        assert chunk.framework is None
        assert chunk.language == "python"
        assert chunk.metadata["path"] == "data/raw/rag.md"
        assert chunk.metadata["version"] == "2"


def test_recursive_character_applies_modest_overlap():
    """Recursive character splitting shares an overlap window between adjacent chunks."""
    source = _source(text="abcdefghijklmnopqrstuvwxyz0123456789")
    config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
        chunk_size=12,
        chunk_overlap=4,
    )

    chunks = chunk_document(source, config=config)

    assert len(chunks) >= 2
    assert chunks[0].text[-4:] == chunks[1].text[:4]
    assert all(len(chunk.text) <= 12 for chunk in chunks)


def test_token_aware_never_exceeds_token_limit():
    """Token-aware chunks stay within the configured max token count."""
    counter = ApproximateTokenCounter()
    source = _source(text=("token word " * 80).strip())
    config = ChunkingConfig(
        strategy=ChunkingStrategy.TOKEN_AWARE,
        chunk_size=10,
        chunk_overlap=2,
        token_counter=counter,
    )

    chunks = chunk_document(source, config=config)

    assert chunks
    for chunk in chunks:
        assert counter.count(chunk.text) <= 10


def test_structure_aware_keeps_heading_sections_separated():
    """Structure-aware splitting does not glue distinct Markdown sections together."""
    source = _source(
        text=(
            "# Injection\n"
            "FastAPI dependency injection wires routers to services.\n\n"
            "# Lifespan\n"
            "Lifespan handlers create shared clients on startup and close them.\n"
        )
    )
    config = ChunkingConfig(
        strategy=ChunkingStrategy.STRUCTURE_AWARE,
        chunk_size=200,
        chunk_overlap=20,
    )

    chunks = chunk_document(source, config=config)

    assert len(chunks) >= 2
    joined_heads = " || ".join(chunk.text for chunk in chunks)
    # Distinct heading titles should appear in separate chunks when both fit.
    injection_chunks = [c for c in chunks if "Injection" in c.text]
    lifespan_chunks = [c for c in chunks if "Lifespan" in c.text]
    assert injection_chunks
    assert lifespan_chunks
    assert not any(
        "Injection" in c.text and "Lifespan" in c.text for c in chunks
    ), joined_heads


def test_structure_aware_does_not_split_inside_fitting_fenced_code_block():
    """A fenced code block that fits in one chunk stays intact."""
    fence = (
        "```python\n"
        "def wire_depends():\n"
        "    return 'ok'\n"
        "```\n"
    )
    source = _source(
        text=f"# Tools\nUse the helper below.\n\n{fence}\n# After\nContinue reading.\n"
    )
    config = ChunkingConfig(
        strategy=ChunkingStrategy.STRUCTURE_AWARE,
        chunk_size=500,
        chunk_overlap=20,
    )

    chunks = chunk_document(source, config=config)

    code_chunks = [c for c in chunks if "def wire_depends" in c.text]
    assert code_chunks
    for chunk in code_chunks:
        assert "```python" in chunk.text
        assert chunk.text.count("```") >= 2


def test_structure_aware_oversized_section_still_yields_bounded_chunks():
    """An oversized heading section is split into size-bounded chunks."""
    body = ("word " * 400).strip()
    source = _source(text=f"# Huge\n{body}\n")
    config = ChunkingConfig(
        strategy=ChunkingStrategy.STRUCTURE_AWARE,
        chunk_size=80,
        chunk_overlap=10,
    )

    chunks = chunk_document(source, config=config)

    assert len(chunks) >= 2
    assert all(len(chunk.text) <= 80 for chunk in chunks)


@pytest.mark.parametrize(
    "strategy",
    [
        ChunkingStrategy.RECURSIVE_CHARACTER,
        ChunkingStrategy.TOKEN_AWARE,
        ChunkingStrategy.STRUCTURE_AWARE,
    ],
)
def test_empty_normalized_text_yields_no_chunks_for_each_strategy(
    strategy: ChunkingStrategy,
):
    """Whitespace-only sources produce an empty chunk list for every strategy."""
    source = _source(text="   \n\t  ")
    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=100 if strategy != ChunkingStrategy.TOKEN_AWARE else 20,
        chunk_overlap=10 if strategy != ChunkingStrategy.TOKEN_AWARE else 2,
        token_counter=ApproximateTokenCounter()
        if strategy == ChunkingStrategy.TOKEN_AWARE
        else None,
    )

    assert chunk_document(source, config=config) == []


@pytest.mark.parametrize(
    "strategy",
    [
        ChunkingStrategy.RECURSIVE_CHARACTER,
        ChunkingStrategy.TOKEN_AWARE,
        ChunkingStrategy.STRUCTURE_AWARE,
    ],
)
@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "match"),
    [
        (0, 0, "chunk_size"),
        (10, -1, "chunk_overlap"),
        (10, 10, "chunk_overlap"),
        (10, 11, "chunk_overlap"),
    ],
)
def test_invalid_config_raises_value_error(
    strategy: ChunkingStrategy,
    chunk_size: int,
    chunk_overlap: int,
    match: str,
):
    """Invalid size/overlap combinations raise ``ValueError`` for every strategy."""
    source = _source(text="enough text to chunk")
    config = ChunkingConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        token_counter=ApproximateTokenCounter()
        if strategy == ChunkingStrategy.TOKEN_AWARE
        else None,
    )

    with pytest.raises(ValueError, match=match):
        chunk_document(source, config=config)


def test_omitting_config_preserves_legacy_character_window_behavior():
    """Calling ``chunk_document`` without ``config`` keeps the Step 6 signature."""
    source = _source(text="abcdefghijklmnopqrstuvwxyz0123456789")

    chunks = chunk_document(source, chunk_size=12, chunk_overlap=4)

    assert len(chunks) >= 2
    assert chunks[0].text[-4:] == chunks[1].text[:4]
