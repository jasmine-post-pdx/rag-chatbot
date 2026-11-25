"""Write-side vector store types shared by memory and pgvector adapters."""

from typing import Protocol

from pydantic import BaseModel, Field

from app.retrieval.document_store import DocumentChunk, DocumentStore


class EmbeddedChunk(BaseModel):
    """A document chunk paired with its embedding vector.

    :ivar chunk: Chunk payload stored for retrieval and lookup.
    :ivar embedding: Embedding in the active model space."""

    chunk: DocumentChunk
    embedding: list[float] = Field(min_length=1)


class VectorDocumentStore(DocumentStore, Protocol):
    """Document store with upsert and source-level delete operations."""

    def upsert_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """Insert or replace chunks by ``chunk_id``.

        :param chunks: Embedded chunks to store."""
        ...

    def delete_source(self, source_id: str) -> int:
        """Remove every chunk belonging to ``source_id``.

        :param source_id: Source identifier to clear.
        :returns: Number of chunks removed."""
        ...
