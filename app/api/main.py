"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints

from app.api.middleware import REQUEST_ID_HEADER, RequestLogMiddleware
from app.api.schemas import QuestionJobResponse
from app.core.config import settings
from app.core.logging import (
    configure_logging,
    log_exception_summary,
    log_validation_failure,
)
from app.models.questions import QuestionResult
from app.services.processor_factory import build_question_processor
from app.services.question_workflow import QuestionProcessingError, QuestionWorkflow
from app.stores.sqlite_question_jobs import SQLiteQuestionJobStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Manage application startup and shutdown state.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control is returned to the application while it runs.
    """
    configure_logging()
    processor = build_question_processor(
        settings.generation_provider,
        settings.generation_model,
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key is not None
        else None,
    )
    app.state.question_workflow = QuestionWorkflow(
        store=SQLiteQuestionJobStore(db_path=str(settings.questions_db_path)),
        processor=processor,
    )
    try:
        yield
    finally:
        app.state.question_workflow.close()
        app.state.question_workflow = None


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


def get_question_workflow(request: Request) -> QuestionWorkflow:
    """Return the shared question workflow from application state.

    Args:
        request: The current HTTP request.

    Returns:
        The configured question service.
    """
    return request.app.state.question_workflow


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
    log_validation_failure(logger, request.state.request_id)
    response = JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=ValidationErrorResponse(detail=detail).model_dump(),
    )
    response.headers[REQUEST_ID_HEADER] = request.state.request_id
    return response


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
    if isinstance(exc, QuestionProcessingError):
        logger.error(
            "Question processing failed",
            extra={
                "request_id": request.state.request_id,
                "question_id": exc.request_id,
                "error_category": "processing_error",
                "exception_type": type(exc.__cause__).__name__ if exc.__cause__ else "Exception",
            },
        )
    else:
        log_exception_summary(
            logger,
            exc,
            message="Unhandled exception",
            request_id=request.state.request_id,
            error_category="server",
        )
    response = JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ServerErrorResponse(detail="Internal server error").model_dump(),
    )
    response.headers[REQUEST_ID_HEADER] = request.state.request_id
    if isinstance(exc, QuestionProcessingError):
        response.headers["x-question-id"] = exc.request_id
    return response


class QuestionRequest(BaseModel):
    """Request body for submitting a question.

    Attributes:
        question: The question to process.
    """

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
    ]


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running.

    Returns:
        A status message indicating the API is running.
    """
    return {"status": "ok"}


@app.get("/ready")
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


@app.post(
    "/v1/questions",
    response_model=QuestionResult,
)
def submit_question(
    request: QuestionRequest,
    http_request: Request,
    question_workflow: QuestionWorkflow = Depends(get_question_workflow),  # noqa: B008
) -> QuestionResult:
    """Process a question and return its answer and metadata.

    Args:
        request: The question payload from the client.
        http_request: The HTTP request containing the middleware correlation ID.
        question_workflow: The workflow used to process and persist the job.

    Returns:
        The answer and its retrieval and model metadata.
    """
    request_id = str(uuid4())
    return question_workflow.submit(
        request_id,
        request.question,
        correlation_id=http_request.state.request_id,
    )


@app.get("/v1/questions/{request_id}", response_model=QuestionJobResponse)
def question_status(
    request_id: str,
    question_workflow: QuestionWorkflow = Depends(get_question_workflow),  # noqa: B008
) -> QuestionJobResponse:
    """Return the current state of a question job.

    Args:
        request_id: Unique identifier of the requested job.
        question_workflow: The workflow used to look up the job.

    Returns:
        The current job state, its result on success, or a safe error category.

    Raises:
        HTTPException: If no job exists for the given identifier.
    """
    job = question_workflow.get_job(request_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    return QuestionJobResponse(
        request_id=job.request_id,
        status=job.status,
        result=job.result,
        error_category=job.error_category,
    )
