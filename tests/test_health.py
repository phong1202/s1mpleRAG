import pytest

from app.config import get_settings


@pytest.mark.asyncio
async def test_health_returns_an_ok_envelope(client):
    response = await client.get("/health")

    assert response.status_code == 200

    body = response.json()
    settings = get_settings()

    assert body["code"] == 200
    assert body["message"] == "Success"
    assert body["data"]["status"] == "ok"
    assert body["data"]["app"] == settings.app_name
    assert body["data"]["environment"] == settings.environment


@pytest.mark.asyncio
async def test_health_response_carries_a_request_id(client):
    response = await client.get("/health")

    assert response.headers["X-Request-ID"]


@pytest.mark.asyncio
async def test_unknown_route_returns_the_envelope_not_fastapis_default(client):
    response = await client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"code": 404, "message": "Not Found", "data": None}
