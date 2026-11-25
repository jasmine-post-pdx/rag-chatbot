"""Indexing, search, ranking, and context assembly."""

from app.retrieval.chunking import SourceDocument, chunk_document, normalize_text
from app.retrieval.document_store import DocumentChunk, DocumentStore, SearchHit
from app.retrieval.ingestion import IngestionPipeline, IngestResult
from app.retrieval.memory import InMemoryDocumentStore, sample_corpus
from app.retrieval.memory_vector import InMemoryVectorDocumentStore
from app.retrieval.vector_store import EmbeddedChunk

__all__ = [
    "DocumentChunk",
    "DocumentStore",
    "EmbeddedChunk",
    "InMemoryDocumentStore",
    "InMemoryVectorDocumentStore",
    "IngestResult",
    "IngestionPipeline",
    "SearchHit",
    "SourceDocument",
    "chunk_document",
    "normalize_text",
    "sample_corpus",
]
