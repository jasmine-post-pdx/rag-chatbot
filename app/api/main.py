"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.middleware import RequestLogMiddleware
from app.core.config import settings
from app.core.logging import configure_logging
from app.services.questions import (
    QuestionJob,
    QuestionResult,
    QuestionService,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Manage application startup and shutdown state.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control is returned to the application while it runs.
    """
    configure_logging()
    app.state.question_service = QuestionService(db_path=settings.questions_db_path)
    try:
        yield
    finally:
        app.state.question_service.close()
        app.state.question_service = None


class ValidationErrorResponse(BaseModel):
    """Public error response for 422 validation failures."""

    detail: str


class ServerErrorResponse(BaseModel):
    """Public error response for 500 internal server errors."""

    detail: str


app = FastAPI(
    title="RAG Chatbot API",
    version="0.1.0",
    lifespan=lifespan,
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ValidationErrorResponse,
            "description": "Validation error",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ServerErrorResponse,
            "description": "Internal server error",
        },
    },
)

app.add_middleware(RequestLogMiddleware)


def get_question_service(request: Request) -> QuestionService:
    """Return the shared question service instance from application state.

    Args:
        request: The current HTTP request.

    Returns:
        The configured question service.
    """
    return request.app.state.question_service


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a concise explanation for invalid request data.

    Args:
        request: The request that triggered the validation error.
        exc: The validation error raised by FastAPI.

    Returns:
        A JSON response with a human-readable detail message.
    """
    request.state.error_category = "validation"
    issues = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        message = error.get("msg", "Invalid value")
        issues.append(f"{location}: {message}" if location else message)
    detail = "Invalid request: " + "; ".join(issues)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=ValidationErrorResponse(detail=detail).model_dump(),
    )


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic server error without exposing implementation details.

    Args:
        request: The request that triggered the error.
        exc: The unhandled exception raised during processing.

    Returns:
        A JSON response with a generic internal server error detail.
    """
    request.state.error_category = "server"
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ServerErrorResponse(detail="Internal server error").model_dump(),
    )


class QuestionRequest(BaseModel):
    """Request body for submitting a question.

    Attributes:
        question: The question to process.
    """

    question: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running.

    Returns:
        A status message indicating the API is running.
    """
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Report that this dependency-free API is ready to accept requests.

    Returns:
        A status message indicating the API is ready.
    """
    return {"status": "ready"}


@app.post(
    "/v1/questions",
    response_model=QuestionResult,
)
def submit_question(
    request: QuestionRequest,
    question_service: QuestionService = Depends(get_question_service),  # noqa: B008
) -> QuestionResult:
    """Process a question and return its answer and metadata.

    Args:
        request: The question payload from the client.
        question_service: The service used to create and complete the job.

    Returns:
        The answer and its retrieval and model metadata.
    """
    request_id = str(uuid4())
    question_service.create_job(request_id)
    question_service.complete_job(request_id, request.question)
    job = question_service.get_job(request_id)
    assert job is not None and job.result is not None
    return job.result


@app.get("/v1/questions/{request_id}", response_model=QuestionJob)
def question_status(
    request_id: str,
    question_service: QuestionService = Depends(get_question_service),  # noqa: B008
) -> QuestionJob:
    """Return the current state of a question job.

    Args:
        request_id: Unique identifier of the requested job.
        question_service: The service used to look up the job.

    Returns:
        The current job state and its result when complete.

    Raises:
        HTTPException: If no job exists for the given identifier.
    """
    job = question_service.get_job(request_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    return job
