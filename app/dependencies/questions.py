"""Dependencies for question workflows."""

from fastapi import Request

from app.services.question_workflow import QuestionWorkflow


def get_question_workflow(request: Request) -> QuestionWorkflow:
    """Return the shared question workflow from application state.

    Args:
        request: The current HTTP request.

    Returns:
        The configured question service.
    """
    return request.app.state.question_workflow