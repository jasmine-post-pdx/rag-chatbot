"""In-process tests for the public HTTP API."""

import logging

import pytest

from app.api import main
from app.services.question_workflow import QuestionWorkflow
from app.stores.sqlite_question_jobs import SQLiteQuestionJobStore

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


def test_ready_returns_503_when_database_unavailable(client, store_factory):
    """Verify /ready returns 503 when the SQLite backing store is unusable."""

    class UnreadyQuestionWorkflow(QuestionWorkflow):
        """Stub service that reports itself as not ready."""

        def is_ready(self) -> bool:
            """Return False to simulate a failed dependency."""
            return False

    unready_workflow = UnreadyQuestionWorkflow(store_factory())
    main.app.dependency_overrides[main.get_question_workflow] = lambda: unready_workflow
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
        "error_category": None,
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


def test_invalid_server_response_data_returns_500(client, store_factory):
    """Verify response-validation failures for server-generated data return 500, not 4xx."""

    class BadResponseQuestionWorkflow(QuestionWorkflow):
        """Stub service that returns malformed response data."""

        def get_job(self, request_id):
            """Return a fake job whose result violates the response model."""

            class FakeJob:
                def __init__(self, job_id):
                    self.request_id = job_id
                    self.status = "completed"
                    self.error_category = None
                    self.result = {
                        "request_id": job_id,
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

            return FakeJob(request_id)

    bad_service = BadResponseQuestionWorkflow(store_factory())
    main.app.dependency_overrides[main.get_question_workflow] = lambda: bad_service
    response = client.get("/v1/questions/bad-result")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


@pytest.mark.parametrize(
    ("failure_stage", "failure_type", "error_message"),
    [
        pytest.param(
            "retrieval",
            ConnectionError,
            "retriever endpoint api_key=topsecret",
            id="retrieval-connection-error",
        ),
        pytest.param(
            "model provider",
            TimeoutError,
            "provider endpoint password=topsecret",
            id="model-provider-timeout",
        ),
    ],
)
def test_submit_pipeline_failure_returns_generic_500(
    client, caplog, store_factory, failure_stage, failure_type, error_message
):
    """Verify retrieval and model-provider failures are hidden behind a generic 500."""
    caplog.set_level(logging.INFO)

    class FailingProcessor:
        """Processor that fails at the simulated pipeline stage."""

        def process(self, request_id, question, *, correlation_id=None):
            raise failure_type(f"{failure_stage} failed: {error_message}")

    store = store_factory()
    failing_workflow = QuestionWorkflow(store, FailingProcessor())
    main.app.dependency_overrides[main.get_question_workflow] = lambda: failing_workflow
    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert error_message not in response.text
    assert "Traceback" not in response.text
    assert "topsecret" not in caplog.text
    error_records = [
        record
        for record in caplog.records
        if record.name == "app.api.main" and record.levelno == logging.ERROR
    ]
    assert len(error_records) == 1
    assert error_records[0].exception_type == failure_type.__name__
    request_id = response.headers["x-question-id"]
    assert error_records[0].question_id == request_id
    request_records = [
        record for record in caplog.records if record.name == "app.api.middleware"
    ]
    assert len(request_records) == 1
    assert request_records[0].status == 500
    poll = client.get(f"/v1/questions/{request_id}")
    assert poll.status_code == 200
    assert poll.json() == {
        "request_id": request_id,
        "status": "failed",
        "result": None,
        "error_category": "processing_error",
    }


def test_submit_failure_state_persistence_error_returns_generic_500(client, store_factory):
    """Verify failure-state persistence errors produce a generic server response."""
    class FailingProcessor:
        def process(self, request_id, question, *, correlation_id=None):
            raise TimeoutError("provider token must not reach the client")

    class FailingFailureStore(SQLiteQuestionJobStore):
        def __init__(self):
            super().__init__()
            self.created_request_id = None

        def create_job(self, request_id):
            self.created_request_id = request_id
            return super().create_job(request_id)

        def fail_job(self, request_id, error_category):
            raise RuntimeError("database connection detail must not reach the client")

    store = store_factory(store_type=FailingFailureStore)
    workflow = QuestionWorkflow(store, FailingProcessor())
    main.app.dependency_overrides[main.get_question_workflow] = lambda: workflow

    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "provider token" not in response.text
    assert "database connection detail" not in response.text
    pending = store.get_job(store.created_request_id)
    assert pending is not None
    assert pending.status == "pending"


def test_submit_sqlite_failure_returns_generic_500(client, question_store):
    """Verify a SQLite operational error becomes a generic 500 without leaking details."""
    question_store._conn.execute("DROP TABLE question_jobs")
    question_store._conn.commit()

    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "no such table" not in response.text
    assert "Traceback" not in response.text
