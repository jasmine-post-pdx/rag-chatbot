"""Public HTTP error schemas."""

from pydantic import BaseModel, Field


class ValidationErrorResponse(BaseModel):
    """Public error response for 422 validation failures."""

    detail: str = Field(description="Concise explanation of the validation problem.")


class ServerErrorResponse(BaseModel):
    """Public error response for 500 internal server errors."""

    detail: str = Field(
        description="Generic server-error message without implementation details."
    )
