import asyncio
import hashlib
import os
import subprocess
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.app import create_app
from app.config import get_settings
from app.utils.database import get_session
from shared.storage import get_public_store

# Shared by any test that needs a complete, valid database config in the
# environment -- e.g. one exercising Settings() directly rather than through
# the real .env.
DB_ENV = {
    "DB_HOST": "localhost",
    "DB_PORT": "5433",
    "DB_USER": "u",
    "DB_PASSWORD": "p",
    "DB_NAME": "db",
}

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

    # worker/db.py builds its engine from settings at import time, and those
    # point at the development database the running containers use. Seeded
    # documents and every chain a test launches would land there instead --
    # colliding on the sha256 unique key with whatever the dev stack has
    # ingested, and keeping whatever a failed test never cleaned up.
    import worker.db

    worker.db.engine.dispose()
    worker.db.engine = create_engine(
        TEST_DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg://"),
        pool_pre_ping=True,
    )
    worker.db.SessionLocal.configure(bind=worker.db.engine)


@pytest.fixture(autouse=True)
def stub_llm_provider(request, monkeypatch):
    """Every test runs on StubProvider whatever .env selects, unless marked
    real_llm. Switching .env to openai for a real run would otherwise send
    every chain-launching test to the paid API on each `pytest`, and break
    the ones that rely on the stub answering the same way twice."""
    if request.node.get_closest_marker("real_llm") is None:
        monkeypatch.setattr(get_settings(), "llm_provider", "stub")


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


@pytest.fixture
def launched(monkeypatch):
    """The document ids the API launched a chain for -- recorded here
    instead of published. The real launch() sends to the dev broker, whose
    workers look the id up in the dev database, where a test's rolled-back
    row never existed: a handful of dead tasks per test run."""
    ids: list[str] = []
    monkeypatch.setattr("worker.stages.launch", ids.append)
    return ids


@pytest_asyncio.fixture
async def client(db_session, launched):
    """An HTTP client whose requests use the rolled-back test session, and
    whose launched chains go nowhere -- see `launched`."""
    app = create_app()

    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def db_env(monkeypatch):
    """A complete set of database settings in the environment."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name, value in DB_ENV.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def uploaded_pdf():
    """Puts a real fixture PDF straight into MinIO, bypassing the presigned
    URL, and returns its register() payload. That isolates register()'s own
    tests from upload-url's checksum mechanics, which get their own coverage
    in test_storage.py and test_ingestion_api.py.

    raw/ has no lifecycle rule (kept forever by design, for re-ingestion),
    so the object is deleted here rather than left for cleanup that never
    comes.
    """
    data = (Path(__file__).parent / "fixtures" / "clean_text.pdf").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    key = f"raw/{digest}.pdf"
    store = get_public_store()
    store.put(key, data)
    yield {
        "object_key": key,
        "filename": "clean_text.pdf",
        "sha256": digest,
        "size_bytes": len(data),
    }
    store.delete(key)


@pytest.fixture
def seeded_document(uploaded_pdf):
    """A real, committed Document row, seeded through the worker's own sync
    engine rather than the async `db_session` the rest of the suite uses.

    The chain a stage-chain test launches runs outside this test's request
    cycle entirely -- worker/db.py's session_scope commits for real, so the
    row has to already be visible there before the chain starts, and this
    fixture has to clean up for real afterward since nothing rolls it back.
    """
    from types import SimpleNamespace

    from app.models.document import Document
    from worker.db import session_scope

    with session_scope() as session:
        doc = Document(
            sha256_hash=uploaded_pdf["sha256"],
            filename=uploaded_pdf["filename"],
            object_key=uploaded_pdf["object_key"],
            size_bytes=uploaded_pdf["size_bytes"],
        )
        session.add(doc)
        session.flush()
        doc_id = doc.id

    yield SimpleNamespace(id=doc_id)

    with session_scope() as session:
        obj = session.get(Document, doc_id)
        if obj is not None:
            session.delete(obj)

    # Chunk rows go with the document by cascade; staged artifacts are in
    # MinIO, shared with the dev stack, and nothing else removes them short
    # of the 7-day expiry. Every test that seeds a document can run the
    # chain, so the cleanup lives here rather than in each such test.
    from shared.storage import get_store
    from worker.pipeline.state import ARTIFACTS

    store = get_store()
    for name in (n for names in ARTIFACTS.values() for n in names):
        store.delete(f"staging/{doc_id}/{name}")

    # Likewise the Redis keys a stage leaves when the test ends it mid-way
    # -- a failed parse keeps its progress for an hour, by design.
    import redis

    from app.config import get_settings

    redis.from_url(get_settings().redis_url).delete(f"progress:{doc_id}", f"ocr:outage:{doc_id}")


@pytest.fixture
def store():
    from app.config import get_settings
    from shared.storage import ObjectStore

    return ObjectStore(endpoint=get_settings().minio_public_url)


@pytest.fixture
def uploaded(store):
    """Uploads a named fixture PDF to MinIO and returns its object_key,
    cleaning up every key it created afterward -- raw/ has no lifecycle
    rule, so nothing else ever removes these."""
    uploaded_keys: list[str] = []

    def _upload(name: str) -> str:
        data = (Path(__file__).parent / "fixtures" / name).read_bytes()
        key = f"raw/{hashlib.sha256(data).hexdigest()}.pdf"
        store.put(key, data)
        uploaded_keys.append(key)
        return key

    yield _upload

    for key in uploaded_keys:
        store.delete(key)
