"""Public request and response models for retrieval-only search."""

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """Body for ``POST /v1/search``.

    :ivar query: Free-text retrieval query.
    :ivar framework: Optional framework metadata filter.
    :ivar language: Optional language metadata filter.
    :ivar top_k: Optional requested hit count; service enforces the hard max."""

    query: str = Field(min_length=1)
    framework: str | None = None
    language: str | None = None
    top_k: int | None = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        """Reject whitespace-only queries after stripping.

        :param value: Raw query string.
        :returns: Stripped query.
        :raises ValueError: If the query is blank."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class SearchHitResponse(BaseModel):
    """One retrieval hit returned to API clients.

    :ivar chunk_id: Stable chunk identifier.
    :ivar score: Cosine similarity score (not calibrated confidence).
    :ivar source: Source document identifier.
    :ivar title: Optional source title.
    :ivar metadata: Non-secret provenance fields.
    :ivar text: Chunk text."""

    chunk_id: str
    score: float
    source: str
    title: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    text: str


class SearchResponse(BaseModel):
    """Retrieval-only search response with index provenance.

    :ivar hits: Ranked hits, possibly empty.
    :ivar embedding_model: Embedding model used for the query.
    :ivar similarity_metric: Similarity metric label (``cosine``)."""

    hits: list[SearchHitResponse]
    embedding_model: str
    similarity_metric: str
