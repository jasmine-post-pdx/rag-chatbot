"""Unit tests for the embedding client boundary."""

import pytest

from app.llm.embeddings import FakeEmbeddingClient


def test_fake_embedding_client_reports_model_and_dimensions():
    """Fake client exposes model identity and fixed vector width."""
    client = FakeEmbeddingClient(dimensions=8, model="fake-embed")

    assert client.model == "fake-embed"
    assert client.dimensions == 8


def test_embed_documents_returns_one_vector_per_text():
    """Batch embedding returns a vector for each non-empty document text."""
    client = FakeEmbeddingClient(dimensions=8)

    vectors = client.embed_documents(["alpha beta", "gamma delta", "epsilon"])

    assert len(vectors) == 3
    assert all(len(vector) == 8 for vector in vectors)
    assert all(isinstance(value, float) for vector in vectors for value in vector)


def test_embed_query_uses_same_space_as_documents():
    """Query and document embeddings share dimension and are comparable."""
    client = FakeEmbeddingClient(dimensions=8)
    text = "dependency injection in FastAPI"

    query_vector = client.embed_query(text)
    document_vectors = client.embed_documents([text])

    assert len(query_vector) == client.dimensions
    assert query_vector == document_vectors[0]


def test_identical_texts_produce_identical_vectors():
    """Deterministic fake embeddings are stable across calls."""
    client = FakeEmbeddingClient(dimensions=16)

    first = client.embed_documents(["stable text"])
    second = client.embed_documents(["stable text"])

    assert first == second


def test_different_texts_produce_different_vectors():
    """Distinct texts should not collide in the fake embedding space."""
    client = FakeEmbeddingClient(dimensions=16)

    left = client.embed_query("FastAPI lifespan handlers")
    right = client.embed_query("Express middleware ordering")

    assert left != right


def test_embed_documents_rejects_empty_batch():
    """An empty document batch is a client error, not a silent no-op."""
    client = FakeEmbeddingClient(dimensions=8)

    with pytest.raises(ValueError, match="empty"):
        client.embed_documents([])


def test_embed_query_rejects_blank_text():
    """Blank queries are rejected before any vector is produced."""
    client = FakeEmbeddingClient(dimensions=8)

    with pytest.raises(ValueError, match="empty|blank"):
        client.embed_query("   ")


def test_similar_texts_are_closer_than_unrelated_texts():
    """Bag-of-words fake vectors rank related phrases above unrelated ones."""
    client = FakeEmbeddingClient(dimensions=32)
    query = client.embed_query("FastAPI dependency injection")
    related = client.embed_documents(["FastAPI Depends injects shared resources"])[0]
    unrelated = client.embed_documents(["Express middleware attaches request IDs"])[0]

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b)

    assert cosine(query, related) > cosine(query, unrelated)
