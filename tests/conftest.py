import asyncio
import os
import subprocess

import asyncpg
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.app import create_app
from app.config import get_settings
from app.utils.database import get_session

TEST_DB_NAME = "rag_beginner_test"
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    get_settings().database_url.rsplit("/", 1)[0] + f"/{TEST_DB_NAME}",
)


async def _create_test_database_if_missing() -> None:
    admin_dsn = (
        TEST_DATABASE_URL.rsplit("/", 1)[0].replace("postgresql+asyncpg://", "postgresql://")
        + "/postgres"
    )
    connection = await asyncpg.connect(admin_dsn)
    try:
        exists = await connection.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB_NAME
        )
        if not exists:
            await connection.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    finally:
        await connection.close()


@pytest.fixture(scope="session", autouse=True)
def prepare_test_database() -> None:
    """Create the test database and bring it to head.

    Deliberately a *sync* fixture: Alembic's async env.py calls asyncio.run(),
    which cannot run inside pytest-asyncio's event loop. Running it in a
    subprocess sidesteps that and exercises the real migration rather than
    metadata.create_all.
    """
    asyncio.run(_create_test_database_if_missing())

    subprocess.run(
        ["alembic", "upgrade", "head"],
        check=True,
        env={**os.environ, "DATABASE_URL": TEST_DATABASE_URL},
    )


@pytest_asyncio.fixture
async def db_session():
    """A session inside a transaction that is always rolled back.

    join_transaction_mode="create_savepoint" nests this session's own
    begin/commit/rollback under the outer transaction opened above via
    Connection.begin_nested(), which is what keeps that outer transaction
    intact for the final rollback regardless of what the session does.

    Note this is *not* the real get_session (app/utils/database.py): the
    `client` fixture below overrides get_session with one that wraps this
    same db_session without ever calling commit(), so nothing in this file
    exercises get_session's own commit/except/rollback contract. See
    tests/test_transaction_boundary.py for coverage of that.
    """
    engine = create_async_engine(TEST_DATABASE_URL)
    connection = await engine.connect()
    transaction = await connection.begin()
    session = AsyncSession(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield session
    finally:
        await session.close()
        await transaction.rollback()
        await connection.close()
        await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session):
    """An HTTP client whose requests use the rolled-back test session."""
    app = create_app()

    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
