"""Health and readiness routes."""

import asyncio

from fastapi import APIRouter, HTTPException, status

from app.dependencies.question_jobs import QuestionJobStoreDep
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report that the API process is running.

    :returns: A status message indicating the API is running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
async def ready(question_job_store: QuestionJobStoreDep) -> ReadinessResponse:
    """Report whether the question-job store is ready.

    The readiness probe offloads the synchronous SQLite check with
    ``asyncio.to_thread`` so a blocked database call does not freeze the event
    loop.

    :param question_job_store: The store created during application startup.
    :returns: A status message indicating the API is ready.
    :raises HTTPException: If the backing store is not usable."""
    if not await asyncio.to_thread(question_job_store.is_ready):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        )
    return ReadinessResponse(status="ready")
