"""Integration tests for the transaction boundary owned by
app.utils.database.get_session.

Every other test in this suite goes through the `client` fixture in
conftest.py, which overrides get_session with a version that neither
commits nor rolls back — so the real get_session function never runs
anywhere else in the suite. A regression as small as deleting
`await session.commit()`, or swapping the except/finally order, would pass
every other test and still stay invisible.

These tests call the real, unmodified get_session directly (via
contextlib.asynccontextmanager, which drives an async-generator dependency
exactly the way FastAPI does: run to the yield, then either resume normally
or throw the raised exception back in at the yield point) and only
repoint its module-level AsyncSessionLocal at rag_beginner_test instead of
the dev database get_session is bound to via .env — get_session's own
commit/except/rollback/finally code is untouched.

They commit real rows, so each one deletes the row it wrote in a finally
block. The suite-wide rolled-back `db_session` fixture does not protect
these tests.
"""

from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.utils.database as db_module
from app.exceptions import AppException, ErrorCode
from app.repositories.document_repository import DocumentRepository
from app.utils.database import get_session
from tests.conftest import TEST_DATABASE_URL

session_scope = asynccontextmanager(get_session)


@pytest_asyncio.fixture(autouse=True)
async def _point_get_session_at_the_test_database(monkeypatch):
    """Repoints the real get_session at rag_beginner_test for this module
    only. get_session's own body is never modified or bypassed."""
    test_engine = create_async_engine(TEST_DATABASE_URL)
    test_session_local = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_module, "AsyncSessionLocal", test_session_local)
    yield
    await test_engine.dispose()


async def _title_exists(title: str) -> bool:
    """Checks visibility from a brand-new connection — proof a row was
    actually committed, not merely flushed inside a still-open transaction."""
    engine = create_async_engine(TEST_DATABASE_URL)
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text("SELECT 1 FROM documents WHERE title = :title"),
                {"title": title},
            )
            return result.first() is not None
    finally:
        await engine.dispose()


async def _delete_title(title: str) -> None:
    engine = create_async_engine(TEST_DATABASE_URL)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text("DELETE FROM documents WHERE title = :title"),
                {"title": title},
            )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_successful_operation_commits_and_is_visible_from_another_connection():
    title = "txn-boundary-commit"
    try:
        async with session_scope() as session:
            await DocumentRepository(session).create(title=title, content="x")

        assert await _title_exists(title) is True
    finally:
        await _delete_title(title)


@pytest.mark.asyncio
async def test_app_exception_after_a_flush_is_rolled_back():
    title = "txn-boundary-app-exception"
    try:
        with pytest.raises(AppException):
            async with session_scope() as session:
                await DocumentRepository(session).create(title=title, content="x")
                raise AppException(ErrorCode.DOCUMENT_NOT_FOUND, "forced for test")

        assert await _title_exists(title) is False
    finally:
        await _delete_title(title)


@pytest.mark.asyncio
async def test_unhandled_exception_after_a_flush_is_rolled_back():
    title = "txn-boundary-unhandled-exception"
    try:
        with pytest.raises(RuntimeError):
            async with session_scope() as session:
                await DocumentRepository(session).create(title=title, content="x")
                raise RuntimeError("forced for test")

        assert await _title_exists(title) is False
    finally:
        await _delete_title(title)
