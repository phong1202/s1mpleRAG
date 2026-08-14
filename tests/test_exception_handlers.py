import logging

import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from app.exceptions import AppException, ErrorCode
from app.exceptions.handlers import register_exception_handlers
from app.middleware import register_middleware
from app.middleware.request_id import REQUEST_ID_HEADER
from app.utils.logging import configure_logging


class Payload(BaseModel):
    title: str


def build_app() -> FastAPI:
    """A throwaway app exposing one route per failure mode."""
    app = FastAPI()
    register_middleware(app)
    register_exception_handlers(app)

    @app.get("/not-found")
    async def not_found():
        raise AppException(ErrorCode.DOCUMENT_NOT_FOUND, "Document 9 not found")

    @app.get("/conflict")
    async def conflict():
        raise AppException(ErrorCode.DOCUMENT_TITLE_EXISTS)

    @app.get("/http-error")
    async def http_error():
        raise HTTPException(status_code=403, detail="Forbidden")

    @app.get("/boom")
    async def boom():
        raise RuntimeError("internal detail that must not leak")

    @app.post("/validate")
    async def validate(payload: Payload):
        return {"ok": True}

    @app.get("/paginate")
    async def paginate(limit: int):
        return {"limit": limit}

    return app


@pytest_asyncio.fixture
async def client():
    app = build_app()
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_app_exception_uses_its_error_code_status_and_custom_message(client):
    response = await client.get("/not-found")

    assert response.status_code == 404
    assert response.json() == {"code": 404, "message": "Document 9 not found", "data": None}


@pytest.mark.asyncio
async def test_app_exception_falls_back_to_the_default_message(client):
    response = await client.get("/conflict")

    assert response.status_code == 409
    assert response.json()["message"] == "A document with this title already exists"


@pytest.mark.asyncio
async def test_http_exception_is_wrapped_in_the_envelope(client):
    response = await client.get("/http-error")

    assert response.status_code == 403
    assert response.json() == {"code": 403, "message": "Forbidden", "data": None}


@pytest.mark.asyncio
async def test_unhandled_exception_returns_a_generic_500_without_leaking(client):
    response = await client.get("/boom")

    assert response.status_code == 500
    body = response.json()
    assert body == {"code": 500, "message": "Internal server error", "data": None}
    assert "internal detail" not in response.text


@pytest.mark.asyncio
async def test_validation_error_reports_the_offending_field(client):
    response = await client.post("/validate", json={})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == 422
    assert body["message"] == "Validation failed"
    assert body["data"]["errors"][0]["field"] == "title"


@pytest.mark.asyncio
async def test_validation_error_on_a_query_param_has_no_location_prefix(client):
    response = await client.get("/paginate")

    assert response.status_code == 422
    body = response.json()
    assert body["data"]["errors"][0]["field"] == "limit"


@pytest.mark.asyncio
async def test_unhandled_exception_response_carries_the_request_id_header(client, caplog):
    # Exercise the real "console" handler + RequestIDFilter pipeline, not
    # just a bare LogRecord, so the filter's defer-if-already-set behavior is
    # actually covered - not merely the value passed via `extra`.
    configure_logging("INFO")

    with caplog.at_level(logging.ERROR):
        response = await client.get("/boom", headers={REQUEST_ID_HEADER: "test-request-id"})

    assert response.status_code == 500
    assert response.headers[REQUEST_ID_HEADER] == "test-request-id"

    # The log line for this same failure must carry the same id - a
    # correlation id a client can quote but that never shows up in the logs
    # is worse than no id at all.
    unhandled = [r for r in caplog.records if "Unhandled exception" in r.getMessage()]
    assert unhandled, "expected an 'Unhandled exception' log record"
    assert unhandled[-1].request_id == "test-request-id"
