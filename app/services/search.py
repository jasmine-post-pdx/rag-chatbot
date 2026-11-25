"""Retrieval-only search orchestration over a DocumentStore."""

from app.retrieval.document_store import DocumentStore
from app.schemas.search import SearchHitResponse, SearchResponse


class SearchTopKOutOfRangeError(ValueError):
    """Raised when a requested ``top_k`` exceeds the configured maximum."""


class SearchService:
    """Run top-k search against a vector-backed document store.

    :ivar _store: Document store used for search.
    :ivar embedding_model: Model identity reported in responses.
    :ivar similarity_metric: Metric label reported in responses.
    :ivar default_top_k: Default hit count when the client omits ``top_k``.
    :ivar max_top_k: Hard maximum accepted ``top_k``."""

    def __init__(
        self,
        store: DocumentStore,
        *,
        embedding_model: str,
        similarity_metric: str = "cosine",
        default_top_k: int = 5,
        max_top_k: int = 20,
    ) -> None:
        """Configure the search service.

        :param store: Document store used for retrieval.
        :param embedding_model: Embedding model identity for responses.
        :param similarity_metric: Similarity metric label.
        :param default_top_k: Default hit count.
        :param max_top_k: Hard maximum hit count.
        :raises ValueError: If top-k bounds are invalid."""
        if default_top_k < 1 or max_top_k < 1:
            raise ValueError("top_k bounds must be >= 1")
        if default_top_k > max_top_k:
            raise ValueError("default_top_k must be <= max_top_k")
        self._store = store
        self.embedding_model = embedding_model
        self.similarity_metric = similarity_metric
        self.default_top_k = default_top_k
        self.max_top_k = max_top_k

    def search(
        self,
        query: str,
        *,
        framework: str | None = None,
        language: str | None = None,
        top_k: int | None = None,
    ) -> SearchResponse:
        """Search the store and map hits to the public response shape.

        :param query: Non-blank query text.
        :param framework: Optional framework filter.
        :param language: Optional language filter.
        :param top_k: Optional requested hit count.
        :returns: Retrieval-only response with provenance metadata.
        :raises SearchTopKOutOfRangeError: If ``top_k`` exceeds ``max_top_k``."""
        limit = self.default_top_k if top_k is None else top_k
        if limit < 1 or limit > self.max_top_k:
            raise SearchTopKOutOfRangeError(
                f"top_k must be between 1 and {self.max_top_k}"
            )

        hits = self._store.search_documents(
            query,
            framework=framework,
            language=language,
            limit=limit,
        )
        return SearchResponse(
            hits=[
                SearchHitResponse(
                    chunk_id=hit.chunk_id,
                    score=float(hit.score or 0.0),
                    source=hit.source_id,
                    title=hit.title,
                    metadata=dict(hit.metadata),
                    text=hit.text,
                )
                for hit in hits
            ],
            embedding_model=self.embedding_model,
            similarity_metric=self.similarity_metric,
        )
