"""Public HTTP error schemas."""

from pydantic import BaseModel


class ValidationErrorResponse(BaseModel):
    """Public error response for 422 validation failures."""

    detail: str


class ServerErrorResponse(BaseModel):
    """Public error response for 500 internal server errors."""

    detail: str
