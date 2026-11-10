"""Question submission and status routes."""

from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from app.dependencies.questions import get_question_workflow
from app.schemas.questions import (
    Citation,
    ModelMetrics,
    QuestionJobResponse,
    QuestionRequest,
    QuestionResult,
    RetrievalMetrics,
)
from app.services.models import QuestionResult as QuestionResultDTO
from app.services.question_workflow import QuestionWorkflow
from app.stores.question_jobs import IdempotencyKeyReuseError

router = APIRouter()


def _result_response(result: QuestionResultDTO) -> QuestionResult:
    """Map the service result and its nested values to public response models."""
    def field(value: Any, name: str) -> Any:
        return value[name] if isinstance(value, Mapping) else getattr(value, name)

    retrieval = field(result, "retrieval")
    model = field(result, "model")
    return QuestionResult(
        request_id=field(result, "request_id"),
        answer=field(result, "answer"),
        citations=[
            Citation(
                source_id=field(citation, "source_id"),
                chunk_id=field(citation, "chunk_id"),
                text=field(citation, "text"),
            )
            for citation in field(result, "citations")
        ],
        retrieval=RetrievalMetrics(
            documents_considered=field(retrieval, "documents_considered"),
            chunks_returned=field(retrieval, "chunks_returned"),
        ),
        model=ModelMetrics(
            provider=field(model, "provider"),
            name=field(model, "name"),
            input_tokens=field(model, "input_tokens"),
            output_tokens=field(model, "output_tokens"),
        ),
    )


@router.post(
    "/v1/questions",
    response_model=QuestionResult,
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "Idempotency-Key was reused with a different question"
        }
    },
)
def submit_question(
    request: QuestionRequest,
    http_request: Request,
    question_workflow: QuestionWorkflow = Depends(get_question_workflow),  # noqa: B008
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ] = None,
) -> QuestionResult:
    """Process a question and return its answer and metadata.

    Args:
        request: The question payload from the client.
        http_request: The HTTP request containing the middleware correlation ID.
        question_workflow: The workflow used to process and persist the job.
        idempotency_key: Optional client key for reusing a prior terminal result.

    Returns:
        The answer and its retrieval and model metadata.
    """
    if idempotency_key is not None:
        idempotency_key = idempotency_key.strip()
        if not idempotency_key:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Idempotency-Key must not be blank",
            )

    try:
        result = question_workflow.submit_question(
            request.question,
            correlation_id=http_request.state.request_id,
            idempotency_key=idempotency_key,
        )
        return _result_response(result)
    except IdempotencyKeyReuseError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Idempotency-Key was already used for a different question",
        ) from None


@router.get("/v1/questions/{request_id}", response_model=QuestionJobResponse)
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
        result=_result_response(job.result) if job.result is not None else None,
        error_category=job.error_category,
    )