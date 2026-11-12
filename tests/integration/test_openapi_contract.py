"""Contract checks for the public question API schemas."""

import hashlib
import json

from app.dependencies.question_jobs import get_question_job_store
from app.dependencies.questions import get_question_workflow
from app.domain.questions import QuestionResult as InternalQuestionResult
from app.main import app
from app.routers import health, questions
from app.schemas.questions import QuestionResultResponse


def test_full_openapi_contract_fingerprint():
    """Preserve the complete canonical OpenAPI document across runtime extraction."""
    canonical = json.dumps(app.openapi(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == (
        "d9fa569e566cfbf2b72c9ee5ebf95d896448424a404511f873c519afc20fdb68"
    )


def test_question_routes_share_the_workflow_dependency():
    """Question routes use the shared workflow dependency."""
    assert {route.path for route in questions.router.routes} == {
        "/v1/questions",
        "/v1/questions/{question_id}",
    }
    for route in questions.router.routes:
        assert [dependency.call for dependency in route.dependant.dependencies] == [
            get_question_workflow
        ]


def test_readiness_uses_the_job_store_dependency():
    """Readiness checks the job store, and liveness has no service dependency."""
    ready = next(route for route in health.router.routes if route.path == "/ready")
    liveness = next(route for route in health.router.routes if route.path == "/health")
    assert [dependency.call for dependency in ready.dependant.dependencies] == [
        get_question_job_store
    ]
    assert liveness.dependant.dependencies == []


def test_public_result_is_independent_of_internal_model():
    """The HTTP contract has its own model with the same current JSON shape."""
    payload = {
        "question_id": "job-1",
        "answer": "Answer",
        "citations": [{"source_id": "doc", "chunk_id": "part", "text": "excerpt"}],
        "retrieval": {"documents_considered": 1, "chunks_returned": 1},
        "model": {
            "provider": "local",
            "name": "stub",
            "input_tokens": 1,
            "output_tokens": 2,
        },
    }
    public = QuestionResultResponse.model_validate(payload)
    internal = InternalQuestionResult.model_validate(payload)
    assert type(public) is not type(internal)
    assert not issubclass(QuestionResultResponse, InternalQuestionResult)
    assert public.model_dump() == internal.model_dump() == payload


def test_question_openapi_contract():
    """Question and error responses retain their documented schema references."""
    document = app.openapi()
    schemas = document["components"]["schemas"]
    question = schemas["QuestionCreate"]["properties"]["question"]
    assert question["minLength"] == 1
    assert question["maxLength"] == 4000
    assert set(schemas["QuestionResultResponse"]["required"]) == {
        "question_id",
        "answer",
        "citations",
        "retrieval",
        "model",
    }
    metrics = schemas["ModelMetricsResponse"]
    assert metrics["required"] == ["provider", "name", "input_tokens", "output_tokens"]
    for token_field in ("input_tokens", "output_tokens"):
        assert metrics["properties"][token_field]["anyOf"] == [
            {"type": "integer"},
            {"type": "null"},
        ]
    assert schemas["QuestionJobResponse"]["properties"]["status"]["enum"] == [
        "pending", "completed", "failed"
    ]
    assert schemas["HealthResponse"]["properties"]["status"]["const"] == "ok"
    assert schemas["ReadinessResponse"]["properties"]["status"]["const"] == "ready"

    responses = document["paths"]["/v1/questions"]["post"]["responses"]
    for code, schema in [("200", "QuestionResultResponse"), ("422", "ValidationErrorResponse"),
                         ("500", "ServerErrorResponse")]:
        assert responses[code]["content"]["application/json"]["schema"] == {
            "$ref": f"#/components/schemas/{schema}"
        }
    assert "409" in responses
    assert document["paths"]["/health"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/HealthResponse"}
    assert document["paths"]["/ready"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ReadinessResponse"}
