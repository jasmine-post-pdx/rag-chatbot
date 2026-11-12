"""Dependencies for question-job persistence."""

from typing import Annotated

from fastapi import Depends, Request

from app.stores.question_job_store import QuestionJobStore


def get_question_job_store(request: Request) -> QuestionJobStore:
    """Return the shared question-job store from application state.

    Args:
        request: The current HTTP request.

    Returns:
        The store created during application startup.
    """
    return request.app.state.question_job_store


QuestionJobStoreDep = Annotated[QuestionJobStore, Depends(get_question_job_store)]
