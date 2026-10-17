"""GET /conversations, GET /conversations/{id}, DELETE /conversations/{id}."""

import uuid

import pytest
from sqlalchemy import select

from app.core.generator import Draft
from app.models import Message, QueryTrace
from tests.test_conversation_owners import other_users_conversation

pytestmark = pytest.mark.asyncio


async def _conversation(client, *questions) -> str:
    conversation_id = None
    for question in questions:
        body = {"question": question} | (
            {"conversation_id": conversation_id} if conversation_id else {}
        )
        conversation_id = (await client.post("/query", json=body)).json()["data"]["conversation_id"]
    return conversation_id


async def test_a_conversation_reads_back_whole_and_in_order(query_client, seeded_corpus):
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tai lieu [1]."))
    first = await client.post("/query", json={"question": "câu một"})
    conversation_id = first.json()["data"]["conversation_id"]
    await client.post("/query", json={"question": "câu hai", "conversation_id": conversation_id})

    data = (await client.get(f"/conversations/{conversation_id}")).json()["data"]

    assert data["title"] == "câu một"
    assert [(m["role"], m["text"]) for m in data["messages"]] == [
        ("user", "câu một"),
        ("assistant", "Theo tai lieu [1]."),
        ("user", "câu hai"),
        ("assistant", "Theo tai lieu [1]."),
    ]
    assert data["messages"][1]["citations"] == first.json()["data"]["citations"]
    assert data["messages"][0]["citations"] is None


async def test_the_list_is_newest_first_with_titles_and_counts(query_client, seeded_corpus):
    client, provider = query_client
    provider.script(Draft, Draft(answer="x"))
    older = await _conversation(client, "cũ")
    newer = await _conversation(client, "mới", "mới nữa")

    data = (await client.get("/conversations?limit=10&offset=0")).json()["data"]

    assert [i["id"] for i in data["items"]] == [newer, older]
    assert [(i["title"], i["message_count"]) for i in data["items"]] == [("mới", 4), ("cũ", 2)]
    assert (data["total"], data["limit"], data["offset"]) == (2, 10, 0)


async def test_answering_in_an_older_conversation_moves_it_to_the_top(query_client, seeded_corpus):
    client, provider = query_client
    provider.script(Draft, Draft(answer="x"))
    older = await _conversation(client, "cũ")
    await _conversation(client, "mới")
    await client.post("/query", json={"question": "tiếp", "conversation_id": older})

    data = (await client.get("/conversations")).json()["data"]

    assert data["items"][0]["id"] == older


async def test_deleting_removes_the_messages_and_keeps_the_traces(
    query_client, db_session, seeded_corpus
):
    client, provider = query_client
    provider.script(Draft, Draft(answer="x"))
    conversation_id = await _conversation(client, "câu")

    response = await client.delete(f"/conversations/{conversation_id}")

    assert response.status_code == 200 and response.json()["data"] is None
    assert (await client.get(f"/conversations/{conversation_id}")).status_code == 404
    assert (await db_session.execute(select(Message))).first() is None
    trace = (await db_session.execute(select(QueryTrace))).scalar_one()
    await db_session.refresh(trace)
    assert trace.conversation_id is None


async def test_another_users_conversation_is_not_found_anywhere(query_client, db_session):
    client, _ = query_client
    foreign = await other_users_conversation(db_session)

    assert (await client.get(f"/conversations/{foreign}")).status_code == 404
    assert (await client.delete(f"/conversations/{foreign}")).status_code == 404
    assert foreign not in [
        i["id"] for i in (await client.get("/conversations")).json()["data"]["items"]
    ]


async def test_an_unknown_conversation_is_not_found(query_client):
    client, _ = query_client

    assert (await client.get(f"/conversations/{uuid.uuid4()}")).status_code == 404
