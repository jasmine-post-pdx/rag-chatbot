"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from app.services.questions import (
    QuestionJob,
    QuestionResult,
    QuestionService,
)

@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Initialize application state at startup and clean up on shutdown."""
    app.state.question_service = QuestionService()
    yield
    app.state.question_service = None


app = FastAPI(
    title="RAG Chatbot API",
    version="0.1.0",
    lifespan=lifespan,
)


def get_question_service(request: Request) -> QuestionService:
    """Provide the shared question service instance."""
    return request.app.state.question_service


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a concise explanation for invalid request data."""
    issues = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        message = error.get("msg", "Invalid value")
        issues.append(f"{location}: {message}" if location else message)
    detail = "Invalid request: " + "; ".join(issues)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": detail},
    )


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic server error without exposing implementation details."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


class QuestionRequest(BaseModel):
    """Request body for submitting a question.

    Attributes:
        question: The question to process.
    """

    question: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Report that this dependency-free API is ready to accept requests."""
    return {"status": "ready"}


@app.post(
    "/v1/questions",
    response_model=QuestionResult,
)
def submit_question(
    request: QuestionRequest,
    question_service: QuestionService = Depends(get_question_service),
) -> QuestionResult:
    """Process a question and return its answer and metadata.

    Args:
        request: The question payload from the client.

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
    question_service: QuestionService = Depends(get_question_service),
) -> QuestionJob:
    """Return the current state of a question job.

    Args:
        request_id: Unique identifier of the requested job.

    Returns:
        The current job state and its result when complete.

    Raises:
        HTTPException: If no job exists for the given identifier.
    """
    job = question_service.get_job(request_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    return job
