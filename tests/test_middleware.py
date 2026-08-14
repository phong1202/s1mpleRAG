import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.middleware import register_middleware
from app.middleware.request_id import get_request_id


def build_app() -> FastAPI:
    app = FastAPI()
    register_middleware(app)

    @app.get("/echo-id")
    async def echo_id():
        return {"request_id": get_request_id()}

    return app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=build_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_generates_a_request_id_when_none_is_supplied(client):
    response = await client.get("/echo-id")

    header_id = response.headers["X-Request-ID"]
    assert header_id
    assert header_id != "-"
    assert response.json()["request_id"] == header_id


@pytest.mark.asyncio
async def test_honours_an_inbound_request_id(client):
    response = await client.get("/echo-id", headers={"X-Request-ID": "caller-supplied"})

    assert response.headers["X-Request-ID"] == "caller-supplied"
    assert response.json()["request_id"] == "caller-supplied"


@pytest.mark.asyncio
async def test_cors_preflight_is_answered(client):
    response = await client.options(
        "/echo-id",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
