import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

DB_ENV = {
    "DB_HOST": "localhost",
    "DB_PORT": "5433",
    "DB_USER": "u",
    "DB_PASSWORD": "p",
    "DB_NAME": "db",
}


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """test_get_settings_is_cached below clears the lru_cache and repopulates
    it from monkeypatched (fake) database settings. monkeypatch undoes the env
    vars afterwards, but not the cache -- without this, every later test in
    the suite reads a Settings object built from those fake values instead of
    the real .env, since get_settings() never gets called with the env
    vars actually applied again."""
    yield
    get_settings.cache_clear()


@pytest.fixture
def db_env(monkeypatch):
    """A complete set of database settings in the environment."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name, value in DB_ENV.items():
        monkeypatch.setenv(name, value)


def _clear_db_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name in DB_ENV:
        monkeypatch.delenv(name, raising=False)


def test_database_config_is_required(monkeypatch):
    """The app must refuse to start without an explicit database."""
    _clear_db_env(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_partial_database_config_is_rejected(monkeypatch):
    """Four of five parts is the silently-wrong case the defaults avoid."""
    _clear_db_env(monkeypatch)
    for name, value in DB_ENV.items():
        if name != "DB_HOST":
            monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match="DB_HOST"):
        Settings(_env_file=None)


def test_database_url_is_composed_from_the_parts(db_env):
    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+asyncpg://u:p@localhost:5433/db"


def test_password_with_special_characters_is_percent_encoded(db_env, monkeypatch):
    monkeypatch.setenv("DB_PASSWORD", "p@ss/word")
    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+asyncpg://u:p%40ss%2Fword@localhost:5433/db"


def test_database_url_env_var_overrides_the_parts(db_env, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://other:pw@managed.test:6543/prod")
    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+asyncpg://other:pw@managed.test:6543/prod"


def test_database_url_alone_is_enough(monkeypatch):
    """A full DSN stands in for the five parts."""
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5433/db")
    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+asyncpg://u:p@localhost:5433/db"


def test_defaults_apply_when_only_the_database_is_set(db_env):
    settings = Settings(_env_file=None)

    assert settings.app_name == "rag-beginner"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000


def test_cors_origins_parses_comma_separated_string(db_env, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test ,")
    settings = Settings(_env_file=None)

    assert settings.cors_origins_list == ["http://a.test", "http://b.test"]


def test_get_settings_is_cached(db_env):
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_worker_database_url_uses_the_sync_driver(db_env):
    """The worker speaks psycopg (sync); the API speaks asyncpg. Two
    engines, one set of models."""
    settings = Settings(_env_file=None)

    assert settings.worker_database_url.startswith("postgresql+psycopg://")
    assert settings.database_url.startswith("postgresql+asyncpg://")
    assert settings.worker_database_url.endswith(settings.database_url.split("/")[-1])


def test_llm_provider_defaults_to_stub(db_env):
    """Stub by default on purpose: a forgotten key must leave the suite
    runnable rather than break it."""
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "stub"


def test_openai_key_is_not_required_when_provider_is_stub(db_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None


def test_minio_public_url_falls_back_to_the_internal_endpoint(db_env, monkeypatch):
    """A presigned URL is signed for one specific host, so a URL signed
    with the internal name is useless to a browser. That is the whole
    reason the two endpoints are separate."""
    monkeypatch.setenv("MINIO_ENDPOINT", "http://minio:9000")
    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    settings = Settings(_env_file=None)
    assert settings.minio_public_url == "http://minio:9000"


def test_rabbitmq_url_is_composed_from_the_parts(db_env):
    settings = Settings(_env_file=None)

    assert settings.rabbitmq_url == "amqp://guest:guest@rabbitmq:5672/%2F"


def test_rabbitmq_vhost_slash_is_encoded(db_env, monkeypatch):
    """The default vhost is literally "/", and the AMQP URI spec says the
    vhost is the path with its leading slash removed, percent-decoded. A bare
    "/" would mean the *empty* vhost, which is a different one."""
    monkeypatch.setenv("RABBITMQ_VHOST", "a/b")
    settings = Settings(_env_file=None)

    assert settings.rabbitmq_url.endswith("/a%2Fb")


def test_rabbitmq_password_with_special_characters_is_percent_encoded(db_env, monkeypatch):
    monkeypatch.setenv("RABBITMQ_PASSWORD", "p@ss/word")
    settings = Settings(_env_file=None)

    assert settings.rabbitmq_url.startswith("amqp://guest:p%40ss%2Fword@")


def test_rabbitmq_url_env_var_overrides_the_parts(db_env, monkeypatch):
    monkeypatch.setenv("RABBITMQ_URL", "amqps://u:p@broker.test:5671/prod")
    settings = Settings(_env_file=None)

    assert settings.rabbitmq_url == "amqps://u:p@broker.test:5671/prod"


def test_minio_endpoint_is_composed_from_the_parts(db_env):
    settings = Settings(_env_file=None)

    assert settings.minio_endpoint == "http://minio:9000"


def test_minio_endpoint_env_var_overrides_the_parts(db_env, monkeypatch):
    """Pointing at real S3, where there is no host:port to assemble."""
    monkeypatch.setenv("MINIO_ENDPOINT", "https://s3.eu-west-1.amazonaws.com")
    settings = Settings(_env_file=None)

    assert settings.minio_endpoint == "https://s3.eu-west-1.amazonaws.com"
