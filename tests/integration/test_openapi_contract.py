"""Contract checks for the public question API schemas."""

import hashlib
import json

from app.dependencies.questions import get_question_workflow
from app.main import app
from app.routers import health, questions
from app.schemas.questions import QuestionResult as PublicQuestionResult
from app.services.models import QuestionResult as InternalQuestionResult


def test_full_openapi_contract_fingerprint():
    """Preserve the complete canonical OpenAPI document across runtime extraction."""
    canonical = json.dumps(app.openapi(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == (
        "8be038145d6de8b1b3e18a5c3a9cbaf8271ba9bd565c1c4788045eb07a51ef42"
    )


def test_routes_share_the_overridable_workflow_dependency():
    """All workflow-backed routes use the same dependency object."""
    routes = [*health.router.routes, *questions.router.routes]
    workflow_routes = [route for route in routes if route.path != "/health"]
    assert {route.path for route in workflow_routes} == {
        "/ready", "/v1/questions", "/v1/questions/{request_id}"
    }
    for route in workflow_routes:
        assert [dependency.call for dependency in route.dependant.dependencies] == [
            get_question_workflow
        ]


def test_public_result_is_independent_of_internal_model():
    """The HTTP contract has its own model with the same current JSON shape."""
    assert PublicQuestionResult is not InternalQuestionResult
    assert not issubclass(PublicQuestionResult, InternalQuestionResult)
    assert PublicQuestionResult.model_json_schema() == InternalQuestionResult.model_json_schema()


def test_question_openapi_contract():
    """Question and error responses retain their documented schema references."""
    document = app.openapi()
    schemas = document["components"]["schemas"]
    question = schemas["QuestionRequest"]["properties"]["question"]
    assert question["minLength"] == 1
    assert question["maxLength"] == 4000
    assert set(schemas["QuestionResult"]["required"]) == {
        "request_id", "answer", "citations", "retrieval", "model"
    }
    assert schemas["QuestionJobResponse"]["properties"]["status"]["enum"] == [
        "pending", "completed", "failed"
    ]

    responses = document["paths"]["/v1/questions"]["post"]["responses"]
    for code, schema in [("200", "QuestionResult"), ("422", "ValidationErrorResponse"),
                         ("500", "ServerErrorResponse")]:
        assert responses[code]["content"]["application/json"]["schema"] == {
            "$ref": f"#/components/schemas/{schema}"
        }
    assert "409" in responses