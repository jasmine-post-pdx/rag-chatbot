"""In-process tests for the public HTTP API."""

import logging
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.services.questions import QuestionService


@pytest.fixture
def question_service():
    """Return an in-memory question service for tests.

    Returns:
        A fresh in-memory question service instance.
    """
    return QuestionService()


@pytest.fixture
def client(question_service):
    """Yield a TestClient with the question service dependency overridden.

    Args:
        question_service: The question service to inject into the API.

    Yields:
        A test client configured to use the injected service.
    """
    main.app.dependency_overrides[main.get_question_service] = lambda: question_service
    try:
        with TestClient(main.app, raise_server_exceptions=False) as test_client:
            yield test_client
    finally:
        main.app.dependency_overrides.pop(main.get_question_service, None)


# /health


def test_health_returns_ok(client):
    """Verify the health endpoint returns an ok status."""
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}


# /ready


def test_readiness_returns_ready(client):
    """Verify the readiness endpoint returns a ready status."""
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


def test_ready_returns_503_when_database_unavailable(client):
    """Verify /ready returns 503 when the SQLite backing store is unusable."""

    class UnreadyQuestionService(QuestionService):
        """Stub service that reports itself as not ready."""

        def is_ready(self) -> bool:
            """Return False to simulate a failed dependency."""
            return False

    unready_service = UnreadyQuestionService()
    main.app.dependency_overrides[main.get_question_service] = lambda: unready_service
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Service unavailable"}


# /v1/questions


def test_submit_and_poll_question_returns_answer_and_metadata(client):
    """Verify submitting a question returns a completed result and can be polled."""
    response = client.post("/v1/questions", json={"question": "What is RAG?"})

    assert response.status_code == 200
    submitted = response.json()
    request_id = submitted["request_id"]
    assert isinstance(request_id, str)
    assert request_id
    assert submitted == {
        "request_id": request_id,
        "answer": "This is a placeholder answer for: What is RAG?",
        "citations": [],
        "retrieval": {"documents_considered": 0, "chunks_returned": 0},
        "model": {
            "provider": "local-or-api",
            "name": "stub",
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }

    polled = client.get(f"/v1/questions/{request_id}")
    assert polled.status_code == 200
    assert polled.json() == {
        "request_id": request_id,
        "status": "completed",
        "result": {
            "request_id": request_id,
            "answer": "This is a placeholder answer for: What is RAG?",
            "citations": [],
            "retrieval": {"documents_considered": 0, "chunks_returned": 0},
            "model": {
                "provider": "local-or-api",
                "name": "stub",
                "input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }


def test_submit_trims_whitespace_from_question(client):
    """Verify leading and trailing whitespace is stripped from the question."""
    response = client.post("/v1/questions", json={"question": "  What is RAG?  "})

    assert response.status_code == 200
    assert response.json()["answer"] == "This is a placeholder answer for: What is RAG?"


def test_submit_rejects_missing_question_field(client):
    """Verify a missing question field returns a field-required error."""
    response = client.post("/v1/questions", json={})

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request: question: Field required"}


def test_submit_rejects_empty_question(client):
    """Verify an empty question string returns a minimum-length error."""
    response = client.post("/v1/questions", json={"question": ""})

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Invalid request: question: String should have at least 1 character"
    }


def test_submit_rejects_whitespace_question(client):
    """Verify a whitespace-only question returns a minimum-length error."""
    response = client.post("/v1/questions", json={"question": " "})

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Invalid request: question: String should have at least 1 character"
    }


def test_submit_rejects_question_with_only_tabs_and_newlines(client):
    """Verify a question of only tabs/newlines returns a minimum-length error."""
    response = client.post("/v1/questions", json={"question": "\t\n"})

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Invalid request: question: String should have at least 1 character"
    }


def test_submit_rejects_question_that_is_too_long(client):
    """Verify a question over the max length returns a maximum-length error."""
    response = client.post("/v1/questions", json={"question": "x" * 4001})

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Invalid request: question: String should have at most 4000 characters"
    }


def test_submit_rejects_non_string_question(client):
    """Verify a non-string question returns a type error."""
    response = client.post("/v1/questions", json={"question": 42})

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Invalid request: question: Input should be a valid string"
    }


def test_submit_rejects_null_question(client):
    """Verify a null question returns a type error."""
    response = client.post("/v1/questions", json={"question": None})

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Invalid request: question: Input should be a valid string"
    }


def test_submit_rejects_malformed_json(client):
    """Verify malformed JSON bodies are rejected with a structured 422 response."""
    response = client.post(
        "/v1/questions", content='{"question":', headers={"content-type": "application/json"}
    )

    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"detail"}
    assert body["detail"] == "Invalid request: 12: JSON decode error"


@pytest.mark.parametrize(
    ("body", "expected_detail"),
    [
        ([], "Invalid request: Input should be a valid dictionary or object to extract fields from"),
        ("What is RAG?", "Invalid request: Input should be a valid dictionary or object to extract fields from"),
        (42, "Invalid request: Input should be a valid dictionary or object to extract fields from"),
        (None, "Invalid request: Field required"),
    ],
)
def test_submit_rejects_non_object_json_bodies(client, body, expected_detail):
    """Verify valid JSON with a non-object root is rejected with a structured 422 response."""
    response = client.post("/v1/questions", json=body)

    assert response.status_code == 422
    response_body = response.json()
    assert set(response_body) == {"detail"}
    assert response_body["detail"] == expected_detail


def test_poll_unknown_question_is_404(client):
    """Verify polling an unknown question returns 404."""
    response = client.get("/v1/questions/missing")
    assert response.status_code == 404
    assert response.json() == {"detail": "Question not found"}


def test_invalid_server_response_data_returns_500(client):
    """Verify response-validation failures for server-generated data return 500, not 4xx."""

    class BadResponseQuestionService(QuestionService):
        """Stub service that returns malformed response data."""

        def get_job(self, request_id):
            """Return a fake job whose result violates the response model."""

            class FakeJob:
                result: ClassVar[dict] = {
                    "request_id": request_id,
                    "answer": 123,
                    "citations": "not a list",
                    "retrieval": {
                        "documents_considered": "x",
                        "chunks_returned": "y",
                    },
                    "model": {
                        "provider": "stub",
                        "name": "stub",
                        "input_tokens": 0,
                        "output_tokens": 0,
                    },
                }

            return FakeJob()

    bad_service = BadResponseQuestionService()
    main.app.dependency_overrides[main.get_question_service] = lambda: bad_service
    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_submit_returns_500_when_result_is_missing(client):
    """Verify a job without a result becomes a generic 500 instead of an assert."""

    class MissingResultQuestionService(QuestionService):
        """Stub service that returns a pending job without a result."""

        def get_job(self, request_id):
            """Return a pending job to simulate an incomplete pipeline."""
            from app.services.questions import QuestionJob

            return QuestionJob(request_id=request_id, status="pending")

    missing_service = MissingResultQuestionService()
    main.app.dependency_overrides[main.get_question_service] = lambda: missing_service
    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


@pytest.mark.parametrize(
    ("failure_stage", "failure_type", "error_message"),
    [
        pytest.param(
            "retrieval",
            ConnectionError,
            "retriever endpoint and credential details",
            id="retrieval-connection-error",
        ),
        pytest.param(
            "model provider",
            TimeoutError,
            "provider endpoint and API key details",
            id="model-provider-timeout",
        ),
    ],
)
def test_submit_pipeline_failure_returns_generic_500(
    client, failure_stage, failure_type, error_message
):
    """Verify retrieval and model-provider failures are hidden behind a generic 500."""

    class FailingQuestionService(QuestionService):
        """Stub service that fails while processing a question."""

        def complete_job(self, request_id, question):
            """Raise an error at the simulated pipeline stage."""
            raise failure_type(f"{failure_stage} failed: {error_message}")

    failing_service = FailingQuestionService()
    main.app.dependency_overrides[main.get_question_service] = lambda: failing_service
    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert error_message not in response.text
    assert "Traceback" not in response.text


def test_submit_sqlite_failure_returns_generic_500(client, question_service):
    """Verify a SQLite operational error becomes a generic 500 without leaking details."""
    question_service._conn.execute("DROP TABLE question_jobs")
    question_service._conn.commit()

    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "no such table" not in response.text
    assert "Traceback" not in response.text


# Request logging middleware (exercised via /v1/questions)


def _find_request_records(caplog):
    """Return records emitted by the request logging middleware."""
    return [r for r in caplog.records if r.name == "app.api.middleware"]


def test_request_log_includes_context(client, caplog):
    """Verify successful requests log method, route, status, timing, and ID."""
    caplog.set_level(logging.INFO)
    response = client.post("/v1/questions", json={"question": "What is RAG?"})

    assert response.status_code == 200
    records = _find_request_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record.method == "POST"
    assert record.route == "/v1/questions"
    assert record.status == 200
    assert record.elapsed_ms >= 0
    assert record.request_id


def test_request_id_header_is_returned(client):
    """Verify each response carries a generated correlation ID."""
    response = client.post("/v1/questions", json={"question": "What is RAG?"})

    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert response.headers["x-request-id"]


def test_request_id_header_is_preserved(client, caplog):
    """Verify an incoming correlation ID is kept and logged."""
    caplog.set_level(logging.INFO)
    response = client.post(
        "/v1/questions",
        json={"question": "What is RAG?"},
        headers={"x-request-id": "existing-request-id"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "existing-request-id"
    records = _find_request_records(caplog)
    assert records[0].request_id == "existing-request-id"


def test_request_id_header_is_returned_on_server_error(client):
    """Verify a 500 response still carries the correlation ID header."""

    class FailingQuestionService(QuestionService):
        """Stub service that raises an error when creating a job."""

        def create_job(self, request_id):
            """Raise a runtime error to simulate a storage failure."""
            raise RuntimeError("simulated failure")

    main.app.dependency_overrides[main.get_question_service] = (
        lambda: FailingQuestionService()
    )
    response = client.post(
        "/v1/questions",
        json={"question": "What is RAG?"},
        headers={"x-request-id": "error-request-id"},
    )

    assert response.status_code == 500
    assert response.headers["x-request-id"] == "error-request-id"


def test_request_log_does_not_contain_question_text(client, caplog):
    """Verify raw request bodies are not written to logs."""
    caplog.set_level(logging.INFO)
    response = client.post("/v1/questions", json={"question": "secret question text"})

    assert response.status_code == 200
    assert "secret question text" not in caplog.text


def test_validation_error_logs_category(client, caplog):
    """Verify validation failures include the validation error category."""
    caplog.set_level(logging.INFO)
    response = client.post("/v1/questions", json={})

    assert response.status_code == 422
    records = _find_request_records(caplog)
    assert records[0].status == 422
    assert records[0].error_category == "validation"


def test_server_error_logs_category(client, caplog):
    """Verify unhandled service errors include the server error category."""
    caplog.set_level(logging.INFO)

    class FailingQuestionService(QuestionService):
        """Stub service that raises an error when creating a job."""

        def create_job(self, request_id):
            """Raise a runtime error to simulate a storage failure.

            Args:
                request_id: Unique identifier for the job (unused).
            """
            raise RuntimeError("database password and stack details")

    failing_service = FailingQuestionService()
    main.app.dependency_overrides[main.get_question_service] = lambda: failing_service
    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    records = _find_request_records(caplog)
    assert records[0].status == 500
    assert records[0].error_category == "server"


def test_validation_error_handler_logs_detail(client, caplog):
    """Verify the validation exception handler logs the structured detail message."""
    caplog.set_level(logging.WARNING)
    response = client.post("/v1/questions", json={"question": ""})

    assert response.status_code == 422
    handler_records = [
        r for r in caplog.records if r.name == "app.api.main" and r.levelno == logging.WARNING
    ]
    assert len(handler_records) == 1
    assert "Validation error" in handler_records[0].message
    assert "String should have at least 1 character" in handler_records[0].message


def test_internal_error_handler_logs_exception(client, caplog):
    """Verify the generic exception handler logs the original exception."""
    caplog.set_level(logging.ERROR)

    class FailingQuestionService(QuestionService):
        """Stub service that raises an error when creating a job."""

        def create_job(self, request_id):
            """Raise a runtime error with sensitive details."""
            raise RuntimeError("database password and stack details")

    main.app.dependency_overrides[main.get_question_service] = lambda: FailingQuestionService()
    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    handler_records = [
        r for r in caplog.records if r.name == "app.api.main" and r.levelno == logging.ERROR
    ]
    assert len(handler_records) == 1
    assert "Unhandled exception" in handler_records[0].message
