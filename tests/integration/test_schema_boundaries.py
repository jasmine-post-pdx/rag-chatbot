"""Tests for the boundary between question DTOs and HTTP schemas."""

import sqlite3

import pytest

from app import main
from app.dependencies.questions import get_question_workflow
from app.services.question_workflow import QuestionWorkflow


def test_public_models_are_independent_from_service_models():
    """The public result and every nested model have separate definitions."""
    from app.domain import questions as internal
    from app.schemas import questions as public

    public_names = {
        "Citation": "CitationResponse",
        "RetrievalMetrics": "RetrievalMetricsResponse",
        "ModelMetrics": "ModelMetricsResponse",
        "QuestionResult": "QuestionResultResponse",
    }
    for internal_name, public_name in public_names.items():
        assert getattr(public, public_name) is not getattr(internal, internal_name)
        assert not issubclass(getattr(public, public_name), getattr(internal, internal_name))
    assert public.QuestionJobResponse.model_fields["result"].annotation == (
        public.QuestionResultResponse | None
    )


def test_submit_and_poll_map_nested_results(client, store_factory):
    """A result with citations survives both response mappings unchanged."""
    from app.domain.questions import Citation, ModelMetrics, QuestionResult, RetrievalMetrics

    class Processor:
        def process(self, request_id, question, *, correlation_id=None):
            return QuestionResult(
                request_id=request_id,
                answer="Answer",
                citations=[Citation(source_id="doc", chunk_id="part", text="excerpt")],
                retrieval=RetrievalMetrics(documents_considered=2, chunks_returned=1),
                model=ModelMetrics(provider="test", name="model", input_tokens=3, output_tokens=4),
            )

    workflow = QuestionWorkflow(store_factory(), Processor())
    main.app.dependency_overrides[get_question_workflow] = lambda: workflow
    submitted = client.post("/v1/questions", json={"question": "Help"})
    assert submitted.status_code == 200
    result = submitted.json()
    assert result == {
        "request_id": result["request_id"],
        "answer": "Answer",
        "citations": [{"source_id": "doc", "chunk_id": "part", "text": "excerpt"}],
        "retrieval": {"documents_considered": 2, "chunks_returned": 1},
        "model": {"provider": "test", "name": "model", "input_tokens": 3, "output_tokens": 4},
    }
    polled = client.get(f"/v1/questions/{result['request_id']}")
    assert polled.status_code == 200
    assert polled.json() == {
        "request_id": result["request_id"], "status": "completed",
        "result": result, "error_category": None,
    }


def test_poll_legacy_job_with_nested_result(client, tmp_path, store_factory):
    """A pre-migration serialized job still maps to the public HTTP contract."""
    path = tmp_path / "legacy.sqlite"
    job_json = (
        '{"request_id":"legacy","status":"completed","result":'
        '{"request_id":"legacy","answer":"Old answer",'
        '"citations":[{"source_id":"doc","chunk_id":"chunk","text":"excerpt"}],'
        '"retrieval":{"documents_considered":1,"chunks_returned":1},'
        '"model":{"provider":"local","name":"stub","input_tokens":0,"output_tokens":0}}}'
    )
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE question_jobs (request_id TEXT PRIMARY KEY, job_json TEXT)")
        connection.execute("INSERT INTO question_jobs VALUES (?, ?)", ("legacy", job_json))
    store = store_factory(str(path))
    workflow = QuestionWorkflow(store)
    main.app.dependency_overrides[get_question_workflow] = lambda: workflow

    response = client.get("/v1/questions/legacy")
    assert response.status_code == 200
    assert response.json() == {
        "request_id": "legacy", "status": "completed", "error_category": None,
        "result": {
            "request_id": "legacy", "answer": "Old answer",
            "citations": [{"source_id": "doc", "chunk_id": "chunk", "text": "excerpt"}],
            "retrieval": {"documents_considered": 1, "chunks_returned": 1},
            "model": {"provider": "local", "name": "stub", "input_tokens": 0, "output_tokens": 0},
        },
    }
    assert store._conn.execute("SELECT job_json FROM question_jobs").fetchone()[0] == job_json


@pytest.mark.parametrize("endpoint", ["submit", "poll"])
def test_malformed_nested_dto_returns_generic_500(client, store_factory, endpoint):
    """Unvalidated internal objects must be validated at the public boundary."""
    from app.domain.questions import ModelMetrics, QuestionJob, QuestionResult, RetrievalMetrics

    class BadWorkflow(QuestionWorkflow):
        def result(self):
            return QuestionResult.model_construct(
                request_id="bad", answer="Answer", citations=[],
                retrieval=RetrievalMetrics.model_construct(documents_considered="bad", chunks_returned=1),
                model=ModelMetrics(provider="test", name="model", input_tokens=0, output_tokens=0),
            )

        def submit_question(self, question, *, correlation_id=None, idempotency_key=None):
            return self.result()

        def get_job(self, request_id):
            return QuestionJob(request_id=request_id, status="completed", result=self.result())

    workflow = BadWorkflow(store_factory())
    main.app.dependency_overrides[get_question_workflow] = lambda: workflow
    response = (
        client.post("/v1/questions", json={"question": "Help"})
        if endpoint == "submit"
        else client.get("/v1/questions/bad")
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_mapping_shaped_workflow_result_is_validated(client, store_factory):
    """Keep accepting mapping-shaped server results as FastAPI previously did."""
    result = {
        "request_id": "mapped", "answer": "Answer",
        "citations": [{"source_id": "doc", "chunk_id": "part", "text": "excerpt"}],
        "retrieval": {"documents_considered": 2, "chunks_returned": 1},
        "model": {"provider": "test", "name": "model", "input_tokens": 3, "output_tokens": 4},
    }

    class MappingWorkflow(QuestionWorkflow):
        def submit_question(self, question, *, correlation_id=None, idempotency_key=None):
            return result

    workflow = MappingWorkflow(store_factory())
    main.app.dependency_overrides[get_question_workflow] = lambda: workflow
    response = client.post("/v1/questions", json={"question": "Help"})
    assert response.status_code == 200
    assert response.json() == result