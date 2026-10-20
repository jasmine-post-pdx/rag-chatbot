"""FastAPI application entry point."""

from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from app.services.questions import (
    QuestionJob,
    QuestionResult,
    complete_job,
    create_job,
    get_job,
)

app = FastAPI(title="RAG Chatbot API", version="0.1.0")


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
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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


class QuestionResponse(BaseModel):
    """Status envelope returned when submitting or polling a question.

    Attributes:
        request_id: Unique identifier for the question job.
        status: Current job state.
        result: Completed result, or ``None`` while the job is pending.
    """

    request_id: str
    status: str
    result: QuestionResult | None = None


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
) -> QuestionResult:
    """Process a question and return its answer and metadata.

    Args:
        request: The question payload from the client.

    Returns:
        The answer and its retrieval and model metadata.
    """
    request_id = str(uuid4())
    create_job(request_id)
    complete_job(request_id, request.question)
    job = get_job(request_id)
    assert job is not None and job.result is not None
    return job.result


@app.get("/v1/questions/{request_id}", response_model=QuestionResponse)
def question_status(request_id: str) -> QuestionJob:
    """Return the current state of a question job.

    Args:
        request_id: Unique identifier of the requested job.

    Returns:
        The current job state and its result when complete.

    Raises:
        HTTPException: If no job exists for the given identifier.
    """
    job = get_job(request_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    return job
