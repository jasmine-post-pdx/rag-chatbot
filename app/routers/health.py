"""Health and readiness routes."""

import asyncio

from fastapi import APIRouter, HTTPException, Request, status

from app.dependencies.question_jobs import QuestionJobStoreDep
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report that the API process is running.

    :returns: A status message indicating the API is running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
async def ready(
    request: Request,
    question_job_store: QuestionJobStoreDep,
) -> ReadinessResponse:
    """Report whether required backing stores are ready.

    The readiness probe offloads synchronous checks with ``asyncio.to_thread``
    so a blocked database call does not freeze the event loop. When the
    configured vector backend exposes ``is_ready`` (pgvector), that probe runs
    after the question-job store check.

    :param request: Current request (provides ``app.state.vector_store``).
    :param question_job_store: The store created during application startup.
    :returns: A status message indicating the API is ready.
    :raises HTTPException: If a required backing store is not usable."""
    if not await asyncio.to_thread(question_job_store.is_ready):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        )
    vector_store = getattr(request.app.state, "vector_store", None)
    is_ready = getattr(vector_store, "is_ready", None)
    if callable(is_ready) and not await asyncio.to_thread(is_ready):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        )
    return ReadinessResponse(status="ready")
