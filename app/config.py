from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        return (
            f"postgresql+asyncpg://{quote_plus(self.db_user)}:{quote_plus(self.db_password)}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
