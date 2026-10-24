"""Public API request and response schemas.

These Pydantic models define the external HTTP contract. They intentionally
mirror the service-layer DTOs they wrap without inheriting from them, so the
API contract can stay stable while internal domain models evolve.
"""

from typing import Literal

from pydantic import BaseModel

from app.services.questions import QuestionResult


class QuestionJobResponse(BaseModel):
    """Public response shape for polling a submitted question.

    Attributes:
        request_id: Unique identifier of the question job.
        status: Whether processing is pending or completed.
        result: Completed answer and metadata, or ``None`` while pending.
    """

    request_id: str
    status: Literal["pending", "completed"]
    result: QuestionResult | None = None
