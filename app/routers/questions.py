"""Question submission and status routes."""

from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.dependencies.questions import QuestionWorkflowDep
from app.domain.errors import (
    IdempotencyKeyReuseError,
    IncompleteIdempotencyRecordError,
)
from app.domain.questions import QuestionResult
from app.schemas.questions import (
    CitationResponse,
    ModelMetricsResponse,
    QuestionCreate,
    QuestionJobResponse,
    QuestionResultResponse,
    RetrievalMetricsResponse,
)

router = APIRouter(prefix="/v1/questions", tags=["questions"])


def to_question_result_response(result: QuestionResult) -> QuestionResultResponse:
    """Map the service result and its nested values to public response models."""

    def field(value: Any, name: str) -> Any:
        return value[name] if isinstance(value, Mapping) else getattr(value, name)

    retrieval = field(result, "retrieval")
    model = field(result, "model")
    return QuestionResultResponse(
        question_id=field(result, "question_id"),
        answer=field(result, "answer"),
        citations=[
            CitationResponse(
                source_id=field(citation, "source_id"),
                chunk_id=field(citation, "chunk_id"),
                text=field(citation, "text"),
            )
            for citation in field(result, "citations")
        ],
        retrieval=RetrievalMetricsResponse(
            documents_considered=field(retrieval, "documents_considered"),
            chunks_returned=field(retrieval, "chunks_returned"),
        ),
        model=ModelMetricsResponse(
            provider=field(model, "provider"),
            name=field(model, "name"),
            input_tokens=field(model, "input_tokens"),
            output_tokens=field(model, "output_tokens"),
        ),
    )


@router.post(
    "",
    response_model=QuestionResultResponse,
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "Idempotency-Key was reused with a different question"
        }
    },
)
def create_question(
    body: QuestionCreate,
    http_request: Request,
    question_workflow: QuestionWorkflowDep,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=255),
    ] = None,
) -> QuestionResultResponse:
    """Process a question and return its answer and metadata.

    Args:
        body: The question payload from the client.
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
            body.question,
            request_id=http_request.state.request_id,
            idempotency_key=idempotency_key,
        )
        return to_question_result_response(result)
    except IdempotencyKeyReuseError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Idempotency-Key was already used for a different question",
        ) from None
    except IncompleteIdempotencyRecordError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Idempotency-Key record is incomplete",
        ) from None


@router.get("/{question_id}", response_model=QuestionJobResponse)
def get_question(
    question_id: str,
    question_workflow: QuestionWorkflowDep,
) -> QuestionJobResponse:
    """Return the current state of a question job.

    Args:
        question_id: Unique identifier of the requested job.
        question_workflow: The workflow used to look up the job.

    Returns:
        The current job state, its result on success, or a safe error category.

    Raises:
        HTTPException: If no job exists for the given identifier.
    """
    job = question_workflow.get_job(question_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found")
    return QuestionJobResponse(
        question_id=job.question_id,
        status=job.status,
        result=to_question_result_response(job.result) if job.result is not None else None,
        error_category=job.error_category,
    )
