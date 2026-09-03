from functools import lru_cache
from urllib.parse import quote, quote_plus

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _compose_url(
    scheme: str,
    host: str,
    port: int,
    user: str | None = None,
    password: str | None = None,
    path: str = "",
) -> str:
    """Assemble a URL from its parts. User and password are
    percent-encoded so a password containing `@`, `/` or `:` cannot
    corrupt the result."""
    auth = ""
    if user is not None:
        auth = quote_plus(user)
        if password is not None:
            auth += f":{quote_plus(password)}"
        auth += "@"
    return f"{scheme}://{auth}{host}:{port}{path}"


class Settings(BaseSettings):
    """Application settings, read from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # The database, in parts. None rather than a default: a silently-wrong
    # database is worse than a refusal to start.
    db_host: str | None = None
    db_port: int | None = None
    db_user: str | None = None
    db_password: str | None = None
    db_name: str | None = None

    # Escape hatch for anything that hands out one ready-made DSN — a managed
    # Postgres, or the test suite pointing Alembic at a scratch database.
    # Wins over the parts above when set.
    database_url_override: str | None = Field(default=None, validation_alias="DATABASE_URL")

    app_name: str = "rag-beginner"
    environment: str = "development"
    log_level: str = "INFO"

    # Comma-separated, not list[str]: pydantic-settings JSON-decodes list fields,
    # which would reject a plain URL like "http://localhost:3000".
    cors_origins: str = "http://localhost:3000"

    host: str = "0.0.0.0"
    port: int = 8000

    # --- Infrastructure ---
    # RabbitMQ in parts, for the same reason as the database: the broker image
    # takes RABBITMQ_DEFAULT_USER / _PASS as its own variables, so credentials
    # buried in a URL would have to be maintained in two places and would
    # drift. Splitting also makes the vhost explicit rather than a trailing
    # slash nobody notices.
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_vhost: str = "/"
    rabbitmq_url_override: str | None = Field(default=None, validation_alias="RABBITMQ_URL")

    # Redis and Docling stay whole URLs on purpose. Neither carries a
    # credential compose needs separately, and both fail loudly -- connection
    # refused -- rather than quietly talking to the wrong place. Talking to the
    # wrong place silently is what splitting the database was guarding against.
    redis_url: str = "redis://redis:6379/0"

    docling_url: str = "http://docling:8100"
    docling_page_timeout_s: int = 90

    # Only the internal endpoint is assembled, because compose publishes
    # ${MINIO_PORT} and would otherwise repeat it. The public endpoint stays a
    # whole URL: in production it is a genuinely different address -- a CDN, a
    # proxy on 443 -- not this host under another name. A presigned URL is
    # signed for exactly one host, which is why there are two at all.
    minio_scheme: str = "http"
    minio_host: str = "minio"
    minio_port: int = 9000
    minio_endpoint_override: str | None = Field(default=None, validation_alias="MINIO_ENDPOINT")
    minio_public_endpoint: str | None = None
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_raw: str = "raw"
    minio_bucket_staging: str = "staging"

    # --- LLM ---
    # "stub" is the deliberate default: without a key the suite still runs.
    llm_provider: str = "stub"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    embed_dimensions: int = 1536

    # --- Limits ---
    max_file_size_mb: int = 50
    max_page_count: int = 500
    max_chunks_per_doc: int = 5000
    enrich_batch_size: int = 20
    embed_batch_size: int = 100
    rl_chat_rpm: int = 500
    rl_chat_tpm: int = 200_000
    rl_embed_rpm: int = 3_000
    rl_embed_tpm: int = 1_000_000

    @model_validator(mode="after")
    def _require_complete_database_config(self) -> "Settings":
        if self.database_url_override:
            return self
        missing = [
            name.upper()
            for name in ("db_host", "db_port", "db_user", "db_password", "db_name")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "missing database settings: "
                + ", ".join(missing)
                + " (or set DATABASE_URL to a full DSN)"
            )
        return self

    @property
    def database_url(self) -> str:
        """The SQLAlchemy DSN — the override when given, otherwise assembled
        from the parts. User and password are percent-encoded so a password
        containing `@`, `/` or `:` cannot corrupt the URL."""
        if self.database_url_override:
            return self.database_url_override
        return _compose_url(
            "postgresql+asyncpg",
            self.db_host,
            self.db_port,
            self.db_user,
            self.db_password,
            f"/{self.db_name}",
        )

    @property
    def rabbitmq_url(self) -> str:
        """The AMQP URI spec reads the vhost as the path with its leading
        slash removed, percent-decoded. So the default vhost "/" has to travel
        as "%2F" -- a bare "/" names the *empty* vhost, a different one."""
        if self.rabbitmq_url_override:
            return self.rabbitmq_url_override
        return _compose_url(
            "amqp",
            self.rabbitmq_host,
            self.rabbitmq_port,
            self.rabbitmq_user,
            self.rabbitmq_password,
            "/" + quote(self.rabbitmq_vhost, safe=""),
        )

    @property
    def minio_endpoint(self) -> str:
        """The override wins, since real S3 has no host:port to assemble."""
        if self.minio_endpoint_override:
            return self.minio_endpoint_override
        return _compose_url(self.minio_scheme, self.minio_host, self.minio_port)

    @property
    def worker_database_url(self) -> str:
        """The same database over a synchronous driver. The worker runs no
        event loop, so it cannot use asyncpg."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")

    @property
    def minio_public_url(self) -> str:
        """What to sign presigned URLs with. Falls back to the internal
        endpoint, which is correct for a single-host deployment. Read this
        rather than `minio_public_endpoint`, which may be None."""
        return self.minio_public_endpoint or self.minio_endpoint

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
