# rag-chatbot

A laboratory for building and evaluating retrieval-augmented generation systems.

## Layout

```text
rag-chatbot/
  app/
    api/          # request entrypoints
    services/     # orchestration
    models/       # LLM and embedding adapters
    retrieval/    # indexing, search, ranking
    tools/        # optional generation-time tools
  tests/
    unit/
    integration/
    evals/
  data/
    raw/          # unmodified source documents
    processed/    # derived chunks and embeddings
  scripts/        # ingest, index, and evaluation jobs
  docs/
    architecture.md
    decisions/    # architecture decision records
  docker-compose.yml
  pyproject.toml
```

See [docs/architecture.md](docs/architecture.md) for how these pieces fit together.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the API

```bash
uvicorn app.api.main:app --reload
```

The API includes `GET /health`, `GET /ready`, `POST /v1/questions`, and
`GET /v1/questions/{request_id}`. Question jobs currently use in-memory storage
and return a placeholder answer; restarting the process clears their state.

Local Postgres with pgvector:

```bash
docker compose up -d
```

## Tests

```bash
pytest
```
