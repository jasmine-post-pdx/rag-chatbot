"""Provider-neutral embedding client interface and deterministic fake."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens.

    :param text: Raw input text.
    :returns: Token list, possibly empty."""
    return re.findall(r"[a-z0-9]+", text.casefold())


def _hash_embed(text: str, dimensions: int) -> list[float]:
    """Build a deterministic bag-of-words embedding for ``text``.

    Overlapping tokens push vectors closer under cosine similarity so retrieval
    tests stay meaningful without a live embedding model.

    :param text: Text to embed.
    :param dimensions: Output vector width.
    :returns: L2-normalized embedding vector.
    :raises ValueError: If ``text`` has no tokens after normalization."""
    tokens = _tokenize(text)
    if not tokens:
        raise ValueError("Cannot embed empty or blank text")

    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class EmbeddingClient(Protocol):
    """Application-owned embedding boundary used by ingestion and search."""

    @property
    def model(self) -> str:
        """Return the embedding model identifier."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the embedding vector width."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed one or more document texts.

        :param texts: Non-empty list of document strings.
        :returns: One embedding per input text.
        :raises ValueError: If ``texts`` is empty or any item is blank."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        :param text: Non-blank query string.
        :returns: Query embedding in the same space as documents.
        :raises ValueError: If ``text`` is blank."""
        ...


class FakeEmbeddingClient:
    """Deterministic local embedder for unit tests and default CI.

    :ivar _dimensions: Fixed vector width.
    :ivar _model: Reported model name."""

    def __init__(self, *, dimensions: int = 8, model: str = "fake") -> None:
        """Configure the fake embedder.

        :param dimensions: Output vector width; must be >= 1.
        :param model: Model identity reported to callers.
        :raises ValueError: If ``dimensions`` is less than 1."""
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        self._dimensions = dimensions
        self._model = model

    @property
    def model(self) -> str:
        """Return the embedding model identifier."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Return the embedding vector width."""
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each document text with the deterministic hash embedder.

        :param texts: Non-empty list of document strings.
        :returns: One embedding per input text.
        :raises ValueError: If ``texts`` is empty or any item is blank."""
        if not texts:
            raise ValueError("Cannot embed empty document batch")
        return [_hash_embed(text, self._dimensions) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query in the same space as documents.

        :param text: Non-blank query string.
        :returns: Query embedding.
        :raises ValueError: If ``text`` is blank."""
        if not text.strip():
            raise ValueError("Cannot embed empty or blank query")
        return _hash_embed(text, self._dimensions)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async document embedding (Step 9 stub).

        :param texts: Non-empty list of document strings.
        :returns: One embedding per input text.
        :raises NotImplementedError: Until the async path is implemented."""
        raise NotImplementedError("FakeEmbeddingClient.aembed_documents is not implemented")

    async def aembed_query(self, text: str) -> list[float]:
        """Async query embedding (Step 9 stub).

        :param text: Non-blank query string.
        :returns: Query embedding.
        :raises NotImplementedError: Until the async path is implemented."""
        raise NotImplementedError("FakeEmbeddingClient.aembed_query is not implemented")

    async def aclose(self) -> None:
        """Release async resources (Step 9 stub).

        :raises NotImplementedError: Until the async path is implemented."""
        raise NotImplementedError("FakeEmbeddingClient.aclose is not implemented")
