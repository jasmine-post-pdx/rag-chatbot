"""Public health and readiness response schemas."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Public response for the liveness probe."""

    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    """Public response for the readiness probe."""

    status: Literal["ready"]
