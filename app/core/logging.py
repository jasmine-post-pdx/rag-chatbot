"""Application logging configuration.

The configuration uses the standard library today so that request logging is
self-contained and dependency-free. The formatter emits key/value pairs for
correlation IDs, HTTP method, route, status, and timing, which map cleanly onto
OpenTelemetry span/log attributes if the project later adopts OTel.
"""

import logging
import re
import sys
from typing import ClassVar
from uuid import uuid4

from app.core.config import settings


class RequestFormatter(logging.Formatter):
    """Format log records with ordered request context fields.

    Extra fields such as ``request_id``, ``method``, ``route``, ``status``,
    ``elapsed_ms``, and ``error_category`` are appended as ``key=value`` pairs
    when present. Unknown extra fields are ignored to keep request logs
    predictable and free of accidental sensitive data.
    """

    _ordered_keys: ClassVar[list[str]] = [
        "request_id",
        "question_id",
        "method",
        "route",
        "status",
        "elapsed_ms",
        "provider",
        "model",
        "input_tokens",
        "output_tokens",
        "error_category",
        "exception_type",
    ]

    def format(self, record: logging.LogRecord) -> str:
        """Return a log line with optional request context fields."""
        base = f"{record.levelname}:     {record.getMessage()}"
        extras: list[str] = []
        for key in self._ordered_keys:
            value = getattr(record, key, None)
            if value is not None:
                extras.append(f"{key}={value}")
        if extras:
            base += " " + " ".join(extras)
        return base


def configure_logging() -> None:
    """Configure root logger level and stdout formatting.

    The level is read from ``settings.log_level`` so it can be controlled per
    environment without code changes. Existing handlers (such as pytest's
    ``caplog`` handler) are left in place to avoid breaking test log capture.
    """
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate application stdout handlers across reloads.
    for handler in root.handlers:
        if isinstance(handler.formatter, RequestFormatter):
            handler.setLevel(level)
            return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RequestFormatter())
    root.addHandler(handler)


_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b(?:password|passwd|secret|token|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|credential|authorization)\b\s*[:=]\s*)"
    r"(Bearer\s+[^\s,;&]+|\"[^\"]*\"|'[^']*'|[^\s,;&]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_URL_CREDENTIAL_PATTERN = re.compile(r"(?i)(https?://)[^/@\s]+@")


def normalize_request_id(value: str | None) -> str:
    """Return a safe bounded correlation ID, generating one for invalid input."""
    if value and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return str(uuid4())


def sanitize_exception_summary(
    exc: BaseException,
    max_length: int = 300,
    sensitive_values: tuple[str, ...] = (),
) -> str:
    """Redact common credentials and keep exception summaries to one short line."""
    summary = str(exc)
    for value in sensitive_values:
        if value:
            summary = summary.replace(value, "<redacted>")
    summary = _SECRET_ASSIGNMENT_PATTERN.sub(r"\1<redacted>", summary)
    summary = _BEARER_PATTERN.sub("Bearer <redacted>", summary)
    summary = _URL_CREDENTIAL_PATTERN.sub(r"\1<redacted>@", summary)
    summary = re.sub(r"[\x00-\x1f\x7f]+", " ", summary).strip()
    if len(summary) > max_length:
        summary = summary[: max_length - 1].rstrip() + "…"
    return summary or "No exception message"


def log_request_summary(
    logger: logging.Logger,
    *,
    request_id: str,
    method: str,
    route: str,
    status: int,
    elapsed_ms: float,
    error_category: str | None = None,
) -> None:
    """Emit the standard one-line request summary."""
    fields: dict[str, object] = {
        "request_id": request_id,
        "method": method,
        "route": route,
        "status": status,
        "elapsed_ms": elapsed_ms,
    }
    if error_category is not None:
        fields["error_category"] = error_category
    logger.info("Request completed", extra=fields)


def log_exception_summary(
    logger: logging.Logger,
    exc: BaseException,
    *,
    message: str,
    request_id: str,
    error_category: str,
    question_id: str | None = None,
) -> None:
    """Emit one safe error summary with stable context fields."""
    fields: dict[str, object] = {
        "request_id": request_id,
        "error_category": error_category,
        "exception_type": type(exc).__name__,
    }
    if question_id is not None:
        fields["question_id"] = question_id
    summary = sanitize_exception_summary(exc)
    logger.error("%s: %s", message, summary, extra=fields)


def log_validation_failure(logger: logging.Logger, request_id: str) -> None:
    """Emit a validation warning without logging submitted field values."""
    logger.warning(
        "Request validation failed",
        extra={"request_id": request_id, "error_category": "validation"},
    )
