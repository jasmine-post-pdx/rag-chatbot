"""Application logging configuration.

The configuration uses the standard library today so that request logging is
self-contained and dependency-free. The formatter emits key/value pairs for
correlation IDs, HTTP method, route, status, and timing, which map cleanly onto
OpenTelemetry span/log attributes if the project later adopts OTel.
"""

import logging
import sys
from typing import ClassVar

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
        "method",
        "route",
        "status",
        "elapsed_ms",
        "error_category",
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
