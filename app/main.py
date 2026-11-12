"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config.logging import (
    configure_logging,
    log_exception_summary,
    log_validation_failure,
)
from app.config.settings import settings
from app.middleware import REQUEST_ID_HEADER, RequestLogMiddleware
from app.routers import health, questions
from app.schemas.errors import ServerErrorResponse, ValidationErrorResponse
from app.services.question_processor import build_question_processor
from app.services.question_workflow import (
    QuestionProcessingError,
    QuestionStatePersistenceError,
    QuestionWorkflow,
)
from app.stores.sqlite import SQLiteQuestionJobStore

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
    store = SQLiteQuestionJobStore(db_path=str(settings.questions_db_path))
    try:
        store.verify_integrity()
        app.state.question_job_store = store
        app.state.question_workflow = QuestionWorkflow(
            store=store,
            processor=processor,
        )
        yield
    finally:
        store.close()
        app.state.question_job_store = None
        app.state.question_workflow = None


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
app.include_router(health.router)
app.include_router(questions.router)


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
                "question_id": exc.question_id,
                "error_category": "processing_error",
                "exception_type": exc.exception_type,
            },
        )
    elif isinstance(exc, QuestionStatePersistenceError):
        logger.error(
            "Question failure state persistence failed",
            extra={
                "request_id": request.state.request_id,
                "question_id": exc.question_id,
                "error_category": "server",
                "exception_type": exc.exception_type,
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
        response.headers["x-question-id"] = exc.question_id
    return response
