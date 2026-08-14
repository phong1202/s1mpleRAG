from pydantic import BaseModel

from app.schemas.response import ApiResponse, PaginatedData


class Item(BaseModel):
    id: int
    name: str


def test_ok_builds_a_200_envelope():
    response = ApiResponse.ok(Item(id=1, name="a"))

    assert response.code == 200
    assert response.message == "Success"
    assert response.data.id == 1


def test_created_builds_a_201_envelope_with_custom_message():
    response = ApiResponse.created(Item(id=2, name="b"), message="Document created")

    assert response.code == 201
    assert response.message == "Document created"


def test_data_may_be_null():
    response = ApiResponse[Item](code=404, message="Not found", data=None)

    assert response.model_dump() == {"code": 404, "message": "Not found", "data": None}


def test_paginated_data_carries_pagination_inside_data():
    page = PaginatedData[Item](items=[Item(id=1, name="a")], total=1, limit=20, offset=0)
    response = ApiResponse.ok(page)

    dumped = response.model_dump()
    assert dumped["code"] == 200
    assert dumped["data"]["total"] == 1
    assert dumped["data"]["limit"] == 20
    assert dumped["data"]["offset"] == 0
    assert dumped["data"]["items"][0]["name"] == "a"
