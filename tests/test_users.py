"""The seeded admin, and the dependency that says who is asking."""

import logging

import pytest
from sqlalchemy import delete

from app.auth import ADMIN_ID
from app.models import User

pytestmark = pytest.mark.asyncio


async def test_me_is_the_seeded_admin(query_client):
    client, _ = query_client

    response = await client.get("/me")

    assert response.json()["data"] == {
        "id": str(ADMIN_ID),
        "username": "admin",
        "display_name": "Admin",
        "is_admin": True,
    }


async def test_a_database_without_the_admin_is_an_error_not_a_new_admin(query_client, db_session):
    """A missing admin means the migrations have not run. Creating one on
    the fly would hide that, and own every conversation made meanwhile."""
    client, _ = query_client
    await db_session.execute(delete(User).where(User.id == ADMIN_ID))

    response = await client.get("/me")

    assert response.status_code == 500
    assert "admin" in response.json()["message"].lower()


def test_starting_without_authentication_says_so(caplog):
    from app.app import create_app

    with caplog.at_level(logging.WARNING):
        create_app()

    assert any(
        "AUTH_MODE=auto_admin" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
