"""Health and readiness routes."""

from fastapi import APIRouter, HTTPException, status

from app.dependencies.question_jobs import QuestionJobStoreDep
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the API process is running.

    :returns: A status message indicating the API is running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
def ready(question_job_store: QuestionJobStoreDep) -> ReadinessResponse:
    """Report whether the question-job store is ready.

    :param question_job_store: The store created during application startup.
    :returns: A status message indicating the API is ready.
    :raises HTTPException: If the backing store is not usable."""
    if not question_job_store.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        )
    return ReadinessResponse(status="ready")
