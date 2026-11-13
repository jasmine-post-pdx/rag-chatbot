"""Public health and readiness response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Public response for the liveness probe."""

    status: Literal["ok"] = Field(
        description='Always "ok" when the process is running.'
    )


class ReadinessResponse(BaseModel):
    """Public response for the readiness probe."""

    status: Literal["ready"] = Field(
        description='Always "ready" when dependencies are usable.'
    )
