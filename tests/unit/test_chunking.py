"""Unit tests for text normalization, chunking, and stable chunk IDs."""

from app.retrieval.chunking import SourceDocument, chunk_document, normalize_text
from app.retrieval.document_store import DocumentChunk


def test_normalize_text_collapses_whitespace():
    """Normalization trims edges and collapses internal whitespace runs."""
    assert normalize_text("  hello\n\n  world\t!  ") == "hello world !"


def test_chunk_document_assigns_stable_ordinal_ids():
    """Chunk IDs are deterministic ``{source_id}:{ordinal}`` values."""
    source = SourceDocument(
        source_id="fastapi-guide",
        title="FastAPI Guide",
        text=("word " * 40).strip(),
        framework="fastapi",
        language="python",
        metadata={"path": "data/raw/fastapi.md"},
    )

    first = chunk_document(source, chunk_size=40, chunk_overlap=10)
    second = chunk_document(source, chunk_size=40, chunk_overlap=10)

    assert first
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == "fastapi-guide:0"
    if len(first) > 1:
        assert first[1].chunk_id == "fastapi-guide:1"


def test_chunk_document_preserves_source_title_and_metadata():
    """Every chunk inherits source identity and provenance fields."""
    source = SourceDocument(
        source_id="rag-notes",
        title="RAG Notes",
        text="Context budgeting reserves tokens for instructions and output.",
        framework=None,
        language="python",
        metadata={"path": "data/raw/rag.md", "version": "1"},
    )

    chunks = chunk_document(source, chunk_size=200, chunk_overlap=20)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, DocumentChunk)
    assert chunk.source_id == "rag-notes"
    assert chunk.title == "RAG Notes"
    assert chunk.framework is None
    assert chunk.language == "python"
    assert chunk.metadata == {"path": "data/raw/rag.md", "version": "1"}
    assert "budgeting" in chunk.text


def test_chunk_document_applies_overlap_between_adjacent_chunks():
    """Adjacent chunks share an overlap window of characters."""
    source = SourceDocument(
        source_id="overlap-demo",
        title="Overlap",
        text="abcdefghijklmnopqrstuvwxyz0123456789",
    )

    chunks = chunk_document(source, chunk_size=12, chunk_overlap=4)

    assert len(chunks) >= 2
    assert chunks[0].text[-4:] == chunks[1].text[:4]


def test_oversized_chunk_size_keeps_single_chunk():
    """When the source fits in one window, exactly one chunk is produced."""
    source = SourceDocument(
        source_id="short",
        title="Short",
        text="tiny document",
    )

    chunks = chunk_document(source, chunk_size=1000, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "short:0"
    assert chunks[0].text == "tiny document"


def test_empty_normalized_text_yields_no_chunks():
    """Whitespace-only sources produce an empty chunk list."""
    source = SourceDocument(
        source_id="blank",
        title="Blank",
        text="   \n\t  ",
    )

    assert chunk_document(source, chunk_size=100, chunk_overlap=10) == []
