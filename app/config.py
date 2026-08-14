from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required: a silently-wrong database is worse than a refusal to start.
    database_url: str

    app_name: str = "rag-beginner"
    environment: str = "development"
    log_level: str = "INFO"

    # Comma-separated, not list[str]: pydantic-settings JSON-decodes list fields,
    # which would reject a plain URL like "http://localhost:3000".
    cors_origins: str = "http://localhost:3000"

    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
