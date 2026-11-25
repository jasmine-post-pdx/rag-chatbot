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
from app.retrieval.factory import (
    build_embedding_client,
    build_vector_store,
    seed_corpus_if_present,
)
from app.routers import health, questions, search
from app.schemas.errors import ServerErrorResponse, ValidationErrorResponse
from app.services.question_processor import build_question_processor
from app.services.question_workflow import (
    QuestionProcessingError,
    QuestionStatePersistenceError,
    QuestionWorkflow,
)
from app.services.search import SearchService
from app.stores.sqlite import SQLiteQuestionJobStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Manage application startup and shutdown state.

    Owns the SQLite job store, vector search stack, and when configured a shared
    async OpenAI SDK client. Resources are closed on shutdown or failed startup
    after creation.

    :param app: The FastAPI application instance.
    :yields: Control is returned to the application while it runs."""
    configure_logging()
    openai_api_key = (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key is not None
        else None
    )
    generation_client = None
    if settings.generation_provider == "openai" and openai_api_key is not None:
        from openai import AsyncOpenAI

        generation_client = AsyncOpenAI(
            api_key=openai_api_key,
            timeout=settings.generation_timeout_seconds,
            max_retries=0,
        )

    embedding_client = build_embedding_client(settings)
    vector_store = build_vector_store(settings, embedding_client)
    seed_corpus_if_present(settings, vector_store, embedding_client)
    search_service = SearchService(
        vector_store,
        embedding_model=embedding_client.model,
        similarity_metric=settings.similarity_metric,
        default_top_k=settings.search_default_top_k,
        max_top_k=settings.search_max_top_k,
    )

    # Step 6: tools use the same vector-backed store as HTTP search.
    document_store = None
    if settings.generation_provider == "openai":
        document_store = vector_store

    store = None
    try:
        processor = build_question_processor(
            settings.generation_provider,
            settings.generation_model,
            openai_api_key,
            timeout_seconds=settings.generation_timeout_seconds,
            max_output_tokens=settings.generation_max_output_tokens,
            instructions=settings.generation_instructions,
            max_retries=settings.generation_max_retries,
            openai_client=generation_client,
            context_window_tokens=settings.generation_context_window_tokens,
            safety_margin_tokens=settings.generation_budget_safety_margin_tokens,
            budget_debug=settings.generation_budget_debug,
            document_store=document_store,
            tool_max_result_chars=settings.tool_max_result_chars,
            tool_call_timeout_seconds=settings.tool_call_timeout_seconds,
            tool_search_max_limit=settings.tool_search_max_limit,
            tool_max_turns=settings.tool_max_turns,
            tool_max_calls=settings.tool_max_calls,
        )
        store = SQLiteQuestionJobStore(db_path=str(settings.questions_db_path))
        store.verify_integrity()
        app.state.question_job_store = store
        app.state.generation_client = generation_client
        app.state.embedding_client = embedding_client
        app.state.vector_store = vector_store
        app.state.search_service = search_service
        app.state.question_workflow = QuestionWorkflow(
            store=store,
            processor=processor,
        )
        yield
    finally:
        if store is not None:
            store.close()
        if generation_client is not None:
            await generation_client.close()
        app.state.question_job_store = None
        app.state.generation_client = None
        app.state.embedding_client = None
        app.state.vector_store = None
        app.state.search_service = None
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
app.include_router(search.router)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a concise explanation for invalid request data.

    :param request: The request that triggered the validation error.
    :param exc: The validation error raised by FastAPI.
    :returns: A JSON response with a human-readable detail message."""
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

    :param request: The request that triggered the error.
    :param exc: The unhandled exception raised during processing.
    :returns: A JSON response with a generic internal server error detail."""
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
            "Question job state persistence failed",
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
