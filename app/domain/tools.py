"""Domain models for allow-listed read-only tool calls."""

from pydantic import BaseModel, Field


class SearchDocumentsArgs(BaseModel):
    """Validated arguments for ``search_documents``.

    :ivar query: Free-text search query.
    :ivar framework: Optional framework filter.
    :ivar language: Optional language filter.
    :ivar limit: Maximum hits requested by the model."""

    query: str = Field(min_length=1)
    framework: str | None = None
    language: str | None = None
    limit: int = Field(default=5, ge=1)


class GetDocumentChunkArgs(BaseModel):
    """Validated arguments for ``get_document_chunk``.

    :ivar chunk_id: Stable chunk identifier to load."""

    chunk_id: str = Field(min_length=1)


class ToolCallProposal(BaseModel):
    """A model-proposed tool call before local validation and execution.

    :ivar call_id: Provider call identifier used to correlate observations.
    :ivar name: Proposed tool name; must match the allow-list to run.
    :ivar arguments: Raw argument mapping from the model."""

    call_id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class ToolObservation(BaseModel):
    """Safe observation returned to the model after a tool attempt.

    Content is capped and must not be treated as application instructions.
    Error values are stable categories, not exception messages with secrets.

    :ivar call_id: Matching provider call identifier.
    :ivar name: Tool name that was proposed.
    :ivar success: Whether execution produced usable content.
    :ivar content: JSON or text payload for the model; may be truncated.
    :ivar error: Stable error category when ``success`` is false.
    :ivar truncated: Whether ``content`` was shortened to the size cap."""

    call_id: str
    name: str
    success: bool
    content: str = ""
    error: str | None = None
    truncated: bool = False
