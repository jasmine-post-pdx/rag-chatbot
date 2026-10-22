"""HTTP middleware for request tracing and timing.

This small middleware is intentionally self-contained. It generates or accepts
a correlation ID, times the request, and emits a single structured log line
after the response is produced. If the project later adopts OpenTelemetry, this
module can be replaced by ``opentelemetry.instrumentation.fastapi`` while the
rest of the application (routes, exception handlers, service layer) remains
unchanged.
"""

import logging
import time
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


REQUEST_ID_HEADER = "x-request-id"


def _error_category_for_status(status_code: int) -> str | None:
    """Return a default error category based on common HTTP status codes."""
    if status_code == 422:
        return "validation"
    if status_code == 500:
        return "server"
    return None


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Log one line per request with correlation ID, route, status, and timing."""

    async def dispatch(self, request: Request, call_next):
        """Generate a request ID, time the request, and log the result.

        Args:
            request: The incoming HTTP request.
            call_next: Callable that runs the rest of the request handler stack.

        Returns:
            The response produced by the handler stack, with ``x-request-id``
            attached.
        """
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "route": request.url.path,
                    "status": 500,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "error_category": "server",
                },
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        error_category = getattr(request.state, "error_category", None)
        if error_category is None:
            error_category = _error_category_for_status(response.status_code)

        log_extra: dict[str, object] = {
            "request_id": request_id,
            "method": request.method,
            "route": request.url.path,
            "status": response.status_code,
            "elapsed_ms": round(elapsed_ms, 3),
        }
        if error_category is not None:
            log_extra["error_category"] = error_category

        logger.info("Request completed", extra=log_extra)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
