from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_create_returns_201_envelope(client):
    response = await client.post("/documents", json={"title": "Intro", "content": "hello"})

    assert response.status_code == 201
    body = response.json()
    assert body["code"] == 201
    assert body["message"] == "Document created"
    assert body["data"]["id"] > 0
    assert body["data"]["title"] == "Intro"
    # embedding is never exposed
    assert "embedding" not in body["data"]


@pytest.mark.asyncio
async def test_get_round_trips_a_created_document(client):
    created = await client.post("/documents", json={"title": "Intro", "content": "hello"})
    document_id = created.json()["data"]["id"]

    response = await client.get(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["data"]["title"] == "Intro"


@pytest.mark.asyncio
async def test_get_missing_document_returns_404_with_null_data(client):
    response = await client.get("/documents/999999")

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == 404
    assert body["data"] is None
    assert "999999" in body["message"]


@pytest.mark.asyncio
async def test_out_of_range_id_returns_422_not_500(client):
    """The id column is Postgres int4; a value beyond its range must be
    rejected by the Path bound before it reaches asyncpg, which would
    otherwise raise DataError -> an unhandled 500 (see DocumentId in
    app/controllers/document_controller.py)."""
    response = await client.get("/documents/99999999999")

    assert response.status_code == 422
    assert response.json()["code"] == 422


@pytest.mark.asyncio
async def test_zero_id_returns_422(client):
    response = await client.get("/documents/0")

    assert response.status_code == 422
    assert response.json()["code"] == 422


@pytest.mark.asyncio
async def test_list_reports_pagination_inside_data(client):
    for index in range(3):
        await client.post("/documents", json={"title": f"Doc {index}", "content": "x"})

    response = await client.get("/documents", params={"limit": 2, "offset": 0})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 3
    assert data["limit"] == 2
    assert data["offset"] == 0
    assert len(data["items"]) == 2


@pytest.mark.asyncio
async def test_update_changes_only_supplied_fields(client):
    created = await client.post("/documents", json={"title": "Before", "content": "body"})
    document_id = created.json()["data"]["id"]

    response = await client.patch(f"/documents/{document_id}", json={"title": "After"})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["title"] == "After"
    assert data["content"] == "body"


@pytest.mark.asyncio
async def test_delete_returns_200_and_the_document_is_gone(client):
    created = await client.post("/documents", json={"title": "Doomed", "content": "x"})
    document_id = created.json()["data"]["id"]

    deleted = await client.delete(f"/documents/{document_id}")

    assert deleted.status_code == 200
    assert deleted.json() == {"code": 200, "message": "Document deleted", "data": None}

    follow_up = await client.get(f"/documents/{document_id}")
    assert follow_up.status_code == 404


@pytest.mark.asyncio
async def test_duplicate_title_returns_409(client):
    await client.post("/documents", json={"title": "Same", "content": "a"})

    response = await client.post("/documents", json={"title": "Same", "content": "b"})

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == 409
    assert body["data"] is None


@pytest.mark.asyncio
async def test_duplicate_title_race_returns_409_not_500(client):
    """Force the race the pre-check can't close: get_by_title misses (as it
    would for a concurrent request) while a row with that title already
    exists, so the INSERT itself hits the unique index."""
    await client.post("/documents", json={"title": "RaceCondition", "content": "a"})

    with patch(
        "app.repositories.document_repository.DocumentRepository.get_by_title",
        return_value=None,
    ):
        response = await client.post("/documents", json={"title": "RaceCondition", "content": "b"})

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == 409
    assert body["data"] is None


@pytest.mark.asyncio
async def test_validation_error_names_the_offending_field(client):
    response = await client.post("/documents", json={"content": "no title"})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == 422
    assert body["message"] == "Validation failed"
    assert body["data"]["errors"][0]["field"] == "title"
