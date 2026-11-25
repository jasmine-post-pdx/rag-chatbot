"""Centralized application settings.

Settings are loaded from environment variables and an optional ``.env`` file.
All values have sensible local-development defaults so the application runs
out of the box, but each value can be overridden per environment.
"""

from urllib.parse import quote

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration.

    :ivar log_level: Python logging level for the application.
    :ivar questions_db_path: Path to the SQLite database used for question jobs.
    :ivar questions_archive_dir: Directory for monthly terminal-job archives.
    :ivar questions_backup_dir: Directory for SQLite backup snapshots.
    :ivar questions_live_retention_days: Days jobs and idempotency keys stay live.
    :ivar questions_archive_retention_days: Days archived jobs are retained.
    :ivar postgres_host: Hostname of the Postgres vector store.
    :ivar postgres_port: Port of the Postgres vector store.
    :ivar postgres_user: Postgres username.
    :ivar postgres_password: Postgres password.
    :ivar postgres_db: Postgres database name.
    :ivar embedding_provider: Adapter name for embeddings (e.g., "local", "fake").
    :ivar embedding_model: Name or identifier of the embedding model.
    :ivar embedding_dimensions: Vector width for the fake embedder (ignored by
        local sentence-transformers when that adapter reports its own size).
    :ivar vector_store: Vector backend name (``memory`` or ``pgvector``).
    :ivar vector_collection: Logical collection/index name for stored chunks.
    :ivar similarity_metric: Documented similarity metric label (``cosine``).
    :ivar chunk_size: Character window size for recursive splitting.
    :ivar chunk_overlap: Character overlap between adjacent chunks.
    :ivar chunking_strategy: Default chunking strategy name
        (``recursive_character``, ``token_aware``, or ``structure_aware``).
    :ivar search_default_top_k: Default ``top_k`` for ``POST /v1/search``.
    :ivar search_max_top_k: Hard maximum ``top_k`` accepted by search.
    :ivar corpus_raw_dir: Directory of raw markdown sources for ingestion.
    :ivar generation_provider: Adapter name for answer generation.
    :ivar generation_model: Name or identifier of the generation model.
    :ivar generation_timeout_seconds: Per-request generation client timeout.
    :ivar generation_max_retries: Extra attempts after a failed generation request.
    :ivar generation_max_output_tokens: Completion-token cap sent as
        ``max_output_tokens``. Defaults to ``1024``. Also used as the output
        reserve when allocating the input context budget.
    :ivar generation_context_window_tokens: Model context-window size used for
        input budgeting. Defaults to ``128000``.
    :ivar generation_budget_safety_margin_tokens: Extra tokens reserved when
        estimating available retrieval capacity. Defaults to ``256``.
    :ivar generation_budget_debug: When ``True``, log extra budget metadata
        fields. Defaults to ``False``.
    :ivar generation_instructions: Optional instructions sent separately from the
        question. Unset means no extra instructions.
    :ivar openai_api_key: Optional OpenAI API key loaded from the environment.
    :ivar tool_search_max_limit: Hard maximum for ``search_documents.limit``.
    :ivar tool_call_timeout_seconds: Per-call deadline for document tools.
    :ivar tool_max_result_chars: Maximum characters returned in a tool observation.
    :ivar tool_max_turns: Maximum model↔tool iterations per question.
    :ivar tool_max_calls: Maximum total tool executions per question."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    log_level: str = "INFO"

    # Job store
    questions_db_path: str = "data/questions.db"
    questions_archive_dir: str = "data/question_archives"
    questions_backup_dir: str = "data/question_backups"
    questions_live_retention_days: int = Field(default=30, gt=0)
    questions_archive_retention_days: int = Field(default=365, gt=0)

    # Vector store (used once retrieval is wired up)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "raglab"
    postgres_password: str = "raglab"
    postgres_db: str = "raglab"

    # Model adapters and retrieval
    embedding_provider: str = "fake"
    embedding_model: str = "fake-embed"
    embedding_dimensions: int = Field(default=32, gt=0)
    vector_store: str = "memory"
    vector_collection: str = "documents"
    similarity_metric: str = "cosine"
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    chunking_strategy: str = "recursive_character"
    search_default_top_k: int = Field(default=5, ge=1)
    search_max_top_k: int = Field(default=20, ge=1)
    corpus_raw_dir: str = "data/raw"
    generation_provider: str = "local"
    generation_model: str = "stub"
    generation_timeout_seconds: float = Field(default=30.0, gt=0)
    generation_max_retries: int = Field(default=2, ge=0)
    generation_max_output_tokens: int = Field(default=1024, gt=0)
    generation_context_window_tokens: int = Field(default=128000, gt=0)
    generation_budget_safety_margin_tokens: int = Field(default=256, ge=0)
    generation_budget_debug: bool = False
    generation_instructions: str | None = None
    openai_api_key: SecretStr | None = None

    # Read-only document tools (Step 5)
    tool_search_max_limit: int = Field(default=5, ge=1)
    tool_call_timeout_seconds: float = Field(default=2.0, gt=0)
    tool_max_result_chars: int = Field(default=4000, ge=1)
    tool_max_turns: int = Field(default=3, ge=1)
    tool_max_calls: int = Field(default=6, ge=1)

    @field_validator("vector_store")
    @classmethod
    def validate_vector_store(cls, value: str) -> str:
        """Allow only supported vector store backends.

        :param value: Configured backend name.
        :returns: The normalized backend name.
        :raises ValueError: If the backend is unsupported."""
        normalized = value.strip().casefold()
        if normalized not in {"memory", "pgvector"}:
            raise ValueError("VECTOR_STORE must be 'memory' or 'pgvector'")
        return normalized

    @field_validator("similarity_metric")
    @classmethod
    def validate_similarity_metric(cls, value: str) -> str:
        """Allow only documented similarity metrics.

        :param value: Configured metric name.
        :returns: The normalized metric name.
        :raises ValueError: If the metric is unsupported."""
        normalized = value.strip().casefold()
        if normalized != "cosine":
            raise ValueError("SIMILARITY_METRIC must be 'cosine'")
        return normalized

    @field_validator("chunking_strategy")
    @classmethod
    def validate_chunking_strategy(cls, value: str) -> str:
        """Allow only supported chunking strategy names.

        :param value: Configured strategy name.
        :returns: The normalized strategy name.
        :raises ValueError: If the strategy is blank or unsupported."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("CHUNKING_STRATEGY must not be empty")
        normalized = value.strip().casefold()
        allowed = {"recursive_character", "token_aware", "structure_aware"}
        if normalized not in allowed:
            raise ValueError(
                "CHUNKING_STRATEGY must be one of: "
                "recursive_character, token_aware, structure_aware"
            )
        return normalized

    @field_validator("generation_instructions", mode="before")
    @classmethod
    def strip_generation_instructions(cls, value: object) -> object:
        """Reject a blank instructions value and strip surrounding whitespace.

        :param value: Raw settings value before coercion.
        :returns: The stripped string, or the original non-string value unchanged."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("GENERATION_INSTRUCTIONS must not be empty")
            return stripped
        return value

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def reject_empty_openai_api_key(cls, value: object) -> object:
        """Reject an explicitly configured empty or whitespace-only key.

        :param value: Raw settings value before coercion.
        :returns: The stripped string, or the original non-string value unchanged."""
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("OPENAI_API_KEY must not be empty")
            return value.strip()
        return value

    @property
    def postgres_dsn(self) -> str:
        """Return a SQLAlchemy-style Postgres connection URL.

        :returns: A ``postgresql://`` DSN with URL-encoded credentials and database name."""
        return (
            f"postgresql://{quote(self.postgres_user, safe='')}"
            f":{quote(self.postgres_password, safe='')}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{quote(self.postgres_db, safe='')}"
        )


settings = Settings()
