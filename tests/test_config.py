import pytest

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """test_get_settings_is_cached below clears the lru_cache and repopulates
    it from a monkeypatched (fake) DATABASE_URL. monkeypatch undoes the env
    var afterwards, but not the cache -- without this, every later test in
    the suite reads a Settings object built from that fake URL instead of
    the real .env, since get_settings() never gets called with the env
    var actually applied again."""
    yield
    get_settings.cache_clear()


def test_database_url_is_required(monkeypatch):
    """The app must refuse to start without an explicit database URL."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_defaults_apply_when_only_database_url_is_set(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5433/db")
    settings = Settings(_env_file=None)

    assert settings.app_name == "rag-beginner"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000


def test_cors_origins_parses_comma_separated_string(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5433/db")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test ,")
    settings = Settings(_env_file=None)

    assert settings.cors_origins_list == ["http://a.test", "http://b.test"]


def test_get_settings_is_cached(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5433/db")
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_settings() is get_settings()
