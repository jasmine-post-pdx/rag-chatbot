"""Unit tests for shared logging policy helpers."""

import logging
from uuid import UUID

import pytest

from app.core.logging import RequestFormatter, normalize_request_id, sanitize_exception_summary


def test_request_formatter_emits_approved_fields_and_omits_unknown_fields():
    """Verify the formatter includes approved fields and omits unknown fields."""
    record = logging.LogRecord(
        name="app.api.middleware",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Request completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-1"
    record.method = "POST"
    record.route = "/v1/questions"
    record.status = 200
    record.elapsed_ms = 12.5
    record.private_detail = "must-not-appear"

    formatted = RequestFormatter().format(record)

    assert formatted == (
        "INFO:     Request completed request_id=req-1 method=POST "
        "route=/v1/questions status=200 elapsed_ms=12.5"
    )
    assert "must-not-appear" not in formatted


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("request failed with api_key=abc123", "request failed with api_key=<redacted>"),
        (
            "Authorization: Bearer abc123",
            "Authorization: <redacted>",
        ),
        ("provider returned Bearer abc123", "provider returned Bearer <redacted>"),
        (
            "request failed at https://user:pass@example.com/path",
            "request failed at https://<redacted>@example.com/path",
        ),
    ],
)
def test_exception_summary_redacts_common_credentials(message, expected):
    """Verify common credential forms are redacted from exception summaries."""
    assert sanitize_exception_summary(RuntimeError(message)) == expected


def test_exception_summary_is_single_line_and_bounded():
    """Verify exception summaries remove control characters and stay bounded."""
    summary = sanitize_exception_summary(
        RuntimeError("line one\r\n\t\x00\x7f" + "x" * 400)
    )

    assert not any(ord(char) < 32 or ord(char) == 127 for char in summary)
    assert len(summary) == 300
    assert summary.startswith("line one ")
    assert summary.endswith("…")


def test_empty_exception_summary_uses_fallback():
    """Verify an empty exception summary uses the fallback message."""
    assert sanitize_exception_summary(RuntimeError()) == "No exception message"


def test_normalize_request_id_preserves_safe_ids():
    """Verify a safe caller-provided request ID is preserved."""
    assert normalize_request_id("request-123.a_b:c") == "request-123.a_b:c"


@pytest.mark.parametrize("request_id", [None, "bad id", "line\nbreak", "x" * 129])
def test_normalize_request_id_replaces_unsafe_or_oversized_values(request_id):
    """Verify unsafe or oversized request IDs are replaced with UUIDs."""
    normalized = normalize_request_id(request_id)

    assert normalized != request_id
    parsed_id = UUID(normalized)
    assert str(parsed_id) == normalized
