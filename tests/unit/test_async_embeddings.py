"""Async embedding client contract tests for Step 9."""

from __future__ import annotations

import pytest

from app.llm.embeddings import FakeEmbeddingClient


@pytest.mark.asyncio
async def test_aembed_documents_matches_sync_dimensions_and_batch_size():
    """``aembed_documents`` returns one vector per input with the configured width."""
    client = FakeEmbeddingClient(dimensions=16, model="fake-async")
    texts = [
        "FastAPI Depends injects shared resources.",
        "Context budgeting reserves output tokens.",
    ]

    vectors = await client.aembed_documents(texts)

    assert len(vectors) == 2
    assert all(len(vector) == 16 for vector in vectors)
    sync = client.embed_documents(texts)
    assert vectors == sync


@pytest.mark.asyncio
async def test_aembed_query_matches_document_space():
    """``aembed_query`` lives in the same space as document embeddings."""
    client = FakeEmbeddingClient(dimensions=12, model="fake-async")
    text = "FastAPI routing and dependency injection"

    query = await client.aembed_query(text)
    docs = await client.aembed_documents([text])

    assert len(query) == 12
    assert query == docs[0]
    assert query == client.embed_query(text)


@pytest.mark.asyncio
async def test_aembed_documents_rejects_empty_batch():
    """Async document embedding rejects an empty batch."""
    client = FakeEmbeddingClient(dimensions=8)

    with pytest.raises(ValueError, match="empty"):
        await client.aembed_documents([])


@pytest.mark.asyncio
async def test_aembed_documents_rejects_blank_text():
    """Async document embedding rejects blank items."""
    client = FakeEmbeddingClient(dimensions=8)

    with pytest.raises(ValueError, match="empty|blank"):
        await client.aembed_documents(["valid tokens here", "   "])


@pytest.mark.asyncio
async def test_aembed_query_rejects_blank_text():
    """Async query embedding rejects blank input."""
    client = FakeEmbeddingClient(dimensions=8)

    with pytest.raises(ValueError, match="empty|blank"):
        await client.aembed_query("\n\t")


@pytest.mark.asyncio
async def test_aclose_is_awaitable_and_idempotent():
    """``aclose`` can be awaited more than once without error."""
    client = FakeEmbeddingClient(dimensions=8)

    await client.aclose()
    await client.aclose()


def test_sync_embed_methods_remain_unchanged():
    """Sync fake embedders still work for search and existing ingestion."""
    client = FakeEmbeddingClient(dimensions=10, model="fake-sync")
    docs = client.embed_documents(["FastAPI Depends", "Express middleware"])
    query = client.embed_query("FastAPI Depends")

    assert len(docs) == 2
    assert len(query) == 10
    assert docs[0] == query
