"""Health and readiness routes."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies.questions import get_question_workflow
from app.services.question_workflow import QuestionWorkflow

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running.

    Returns:
        A status message indicating the API is running.
    """
    return {"status": "ok"}


@router.get("/ready")
def ready(
    question_workflow: QuestionWorkflow = Depends(get_question_workflow),  # noqa: B008
) -> dict[str, str]:
    """Report whether the API and its SQLite backing store are ready.

    Args:
        question_workflow: The workflow that checks its backing store.

    Returns:
        A status message indicating the API is ready.

    Raises:
        HTTPException: If the backing store is not usable.
    """
    if not question_workflow.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        )
    return {"status": "ready"}