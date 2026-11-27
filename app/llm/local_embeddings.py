"""Optional sentence-transformers embedding adapter."""

from __future__ import annotations

import asyncio


class LocalEmbeddingClient:
    """Embed texts with a local sentence-transformers model.

    Requires the optional ``[embeddings]`` extra
    (``sentence-transformers``).

    :ivar _model_name: Hugging Face / local model identifier.
    :ivar _model: Loaded SentenceTransformer instance."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Load the configured sentence-transformers model.

        :param model: Model name or path.
        :raises RuntimeError: If sentence-transformers is not installed."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Local embeddings require the optional 'embeddings' extra "
                "(pip install '.[embeddings]')"
            ) from exc
        self._model_name = model
        self._model = SentenceTransformer(model)

    @property
    def model(self) -> str:
        """Return the embedding model identifier."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Return the embedding vector width reported by the model."""
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed one or more document texts.

        :param texts: Non-empty list of document strings.
        :returns: One embedding per input text.
        :raises ValueError: If ``texts`` is empty or any item is blank."""
        if not texts:
            raise ValueError("Cannot embed empty document batch")
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty or blank text")
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        :param text: Non-blank query string.
        :returns: Query embedding.
        :raises ValueError: If ``text`` is blank."""
        if not text.strip():
            raise ValueError("Cannot embed empty or blank query")
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async document embedding; offloads CPU encode off the event loop.

        :param texts: Non-empty list of document strings.
        :returns: One embedding per input text.
        :raises ValueError: If ``texts`` is empty or any item is blank."""
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Async query embedding; offloads CPU encode off the event loop.

        :param text: Non-blank query string.
        :returns: Query embedding.
        :raises ValueError: If ``text`` is blank."""
        return await asyncio.to_thread(self.embed_query, text)

    async def aclose(self) -> None:
        """No-op close for the local model (idempotent)."""
        return None
