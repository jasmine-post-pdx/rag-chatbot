FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv==0.12.23 \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/data \
    && chown -R app:app /app

COPY --chown=app:app pyproject.toml uv.lock README.md ./
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts

# Development/CI stage with test dependencies and test suite.
FROM base AS dev

RUN uv sync --frozen --extra dev --no-cache

COPY --chown=app:app tests ./tests

USER app

CMD ["pytest"]

# Production/runtime stage (default target).
FROM base AS runtime

RUN uv sync --frozen --no-dev --no-cache

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
