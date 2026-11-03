"""Integration tests for request logging and correlation IDs."""

import logging

from app.api import main
from app.services.question_workflow import QuestionWorkflow
from app.stores.sqlite_question_jobs import SQLiteQuestionJobStore


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


def test_invalid_request_id_is_replaced_in_response_and_log(client, caplog):
    """Verify unsafe caller IDs are replaced before use in logs and responses."""
    caplog.set_level(logging.INFO)
    response = client.post(
        "/v1/questions",
        json={"question": "What is RAG?"},
        headers={"x-request-id": "unsafe id"},
    )

    assert response.status_code == 200
    normalized_id = response.headers["x-request-id"]
    assert normalized_id != "unsafe id"
    records = _find_request_records(caplog)
    assert len(records) == 1
    assert records[0].request_id == normalized_id


def test_request_id_header_is_returned_on_server_error(client, store_factory):
    """Verify a 500 response still carries the correlation ID header."""

    class FailingStore(SQLiteQuestionJobStore):
        """Store that raises an error when creating a completed job."""

        def create_completed_job(self, request_id, result, **kwargs):
            """Raise a runtime error to simulate a storage failure."""
            raise RuntimeError("simulated failure")

    failing_workflow = QuestionWorkflow(store_factory(store_type=FailingStore))
    main.app.dependency_overrides[main.get_question_workflow] = lambda: failing_workflow
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


def test_server_error_logs_category(client, caplog, store_factory):
    """Verify unhandled service errors include the server error category."""
    caplog.set_level(logging.INFO)

    class FailingStore(SQLiteQuestionJobStore):
        """Store that raises an error when creating a completed job."""

        def create_completed_job(self, request_id, result, **kwargs):
            """Raise a runtime error to simulate a storage failure.

            Args:
                request_id: Unique identifier for the job (unused).
                result: The completed question result (unused).
            """
            raise RuntimeError("database password and stack details")

    failing_workflow = QuestionWorkflow(store_factory(store_type=FailingStore))
    main.app.dependency_overrides[main.get_question_workflow] = lambda: failing_workflow
    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    records = _find_request_records(caplog)
    assert records[0].status == 500
    assert records[0].error_category == "server"


def test_validation_error_log_omits_validation_details(client, caplog):
    """Verify validation logs carry a category without field or input details."""
    caplog.set_level(logging.WARNING)
    response = client.post("/v1/questions", json={"question": ""})

    assert response.status_code == 422
    handler_records = [
        r for r in caplog.records if r.name == "app.api.main" and r.levelno == logging.WARNING
    ]
    assert len(handler_records) == 1
    assert "Request validation failed" in handler_records[0].message
    assert handler_records[0].error_category == "validation"
    assert "String should have at least 1 character" not in handler_records[0].message


def test_internal_error_handler_logs_exception(client, caplog, store_factory):
    """Verify the generic exception handler logs the original exception."""
    caplog.set_level(logging.ERROR)

    class FailingStore(SQLiteQuestionJobStore):
        """Store that raises an error when creating a completed job."""

        def create_completed_job(self, request_id, result, **kwargs):
            """Raise a runtime error with sensitive details."""
            raise RuntimeError("database password=hidden and token=hidden-too")

    failing_workflow = QuestionWorkflow(store_factory(store_type=FailingStore))
    main.app.dependency_overrides[main.get_question_workflow] = lambda: failing_workflow
    response = client.post("/v1/questions", json={"question": "Help"})

    assert response.status_code == 500
    handler_records = [
        r for r in caplog.records if r.name == "app.api.main" and r.levelno == logging.ERROR
    ]
    assert len(handler_records) == 1
    assert "Unhandled exception" in handler_records[0].message
    assert handler_records[0].exception_type == "RuntimeError"
    assert "password=<redacted>" in handler_records[0].message
    assert "token=<redacted>" in handler_records[0].message
    assert "hidden" not in handler_records[0].message
