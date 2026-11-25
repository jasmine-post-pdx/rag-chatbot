"""Read-only document store port used by document tools.

Step 5 tools call this interface only. Step 6 can replace the in-memory
adapter with a vector-backed implementation without changing tool code.
"""

from typing import Protocol

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A stored document chunk available for search or direct lookup.

    :ivar source_id: Identifier of the source document.
    :ivar chunk_id: Stable identifier of the chunk within that source.
    :ivar text: Chunk body text returned to tools as untrusted reference.
    :ivar title: Optional human-readable source title.
    :ivar framework: Optional framework metadata used for filtered search.
    :ivar language: Optional language metadata used for filtered search.
    :ivar metadata: Additional non-secret provenance fields (e.g. path)."""

    source_id: str
    chunk_id: str
    text: str
    title: str | None = None
    framework: str | None = None
    language: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SearchHit(BaseModel):
    """A search result compatible with ``RankedChunk`` field names.

    :ivar source_id: Identifier of the source document.
    :ivar chunk_id: Identifier of the matched chunk.
    :ivar text: Matched chunk text.
    :ivar title: Optional human-readable source title.
    :ivar metadata: Additional non-secret provenance fields.
    :ivar rank: One-based rank among returned hits; lower is better.
    :ivar score: Optional relevance score (cosine similarity for vector search);
        not a calibrated confidence. Unused by greedy budgeting."""

    source_id: str
    chunk_id: str
    text: str
    title: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    rank: int | None = None
    score: float | None = None


class DocumentStore(Protocol):
    """Read-only operations required by document tools."""

    def search_documents(
        self,
        query: str,
        *,
        framework: str | None = None,
        language: str | None = None,
        limit: int = 5,
    ) -> list[SearchHit]:
        """Search for chunks matching the query and optional filters.

        :param query: Free-text query.
        :param framework: Optional framework filter (case-insensitive).
        :param language: Optional language filter (case-insensitive).
        :param limit: Maximum hits to return; implementations enforce a hard max.
        :returns: Matching hits in relevance order, possibly empty."""
        ...

    def get_document_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Load a single chunk by identifier.

        :param chunk_id: Stable chunk identifier.
        :returns: The chunk, or ``None`` when it does not exist."""
        ...
