"""R4: the server keeps the conversation, so a follow-up can be read in it."""

import uuid

import pytest
from sqlalchemy import func, select

from app.config import get_settings
from app.core.generator import Draft
from app.core.stages.rewrite import RewriteResult
from app.models import Conversation, Message
from app.repositories.conversation_repository import ConversationRepository

pytestmark = pytest.mark.asyncio


def _script(provider, answer="Phạt từ 400.000 đến 600.000 đồng [1].", standalone="q"):
    provider.script(
        RewriteResult, RewriteResult(intent="SEARCH_CORPUS", standalone=standalone, sub_queries=[])
    )
    provider.script(Draft, Draft(answer=answer))


async def test_a_first_question_starts_a_conversation(query_client, db_session, seeded_corpus):
    client, provider = query_client
    _script(provider)

    response = await client.post("/query", json={"question": "hoa don dien tu"})

    conversation_id = uuid.UUID(response.json()["data"]["conversation_id"])
    messages = (await db_session.execute(select(Message).order_by(Message.id))).scalars().all()
    assert [(m.conversation_id, m.role, m.text) for m in messages] == [
        (conversation_id, "user", "hoa don dien tu"),
        (conversation_id, "assistant", "Phạt từ 400.000 đến 600.000 đồng [1]."),
    ]


async def test_the_next_question_is_rewritten_with_the_conversation(
    query_client, db_session, seeded_corpus, monkeypatch
):
    monkeypatch.setattr(get_settings(), "retrieval_intent_enabled", True)
    client, provider = query_client
    _script(provider)
    first = await client.post(
        "/query", json={"question": "Đi xe máy không đội mũ bị phạt bao nhiêu?"}
    )
    conversation_id = first.json()["data"]["conversation_id"]

    second = await client.post(
        "/query",
        json={"question": "Thế người ngồi sau thì sao?", "conversation_id": conversation_id},
    )

    assert second.json()["data"]["conversation_id"] == conversation_id
    rewrites = [messages for schema, messages in provider.calls if schema is RewriteResult]
    sent = rewrites[-1][-1]["content"]
    assert "Đi xe máy không đội mũ bị phạt bao nhiêu?" in sent
    assert "Phạt từ 400.000 đến 600.000 đồng" in sent
    assert sent.rstrip().endswith("Thế người ngồi sau thì sao?")


async def test_an_unknown_conversation_is_not_found(query_client, seeded_corpus):
    client, provider = query_client
    _script(provider)

    response = await client.post(
        "/query", json={"question": "q", "conversation_id": str(uuid.uuid4())}
    )

    assert response.status_code == 404


async def test_a_failed_request_adds_nothing_to_the_conversation(
    query_client, db_session, seeded_corpus
):
    """The history holds only what the user was actually told."""
    client, provider = query_client
    _script(provider)
    first = await client.post("/query", json={"question": "q1"})
    conversation_id = first.json()["data"]["conversation_id"]
    provider.script(Draft)  # nothing scripted: the generator raises

    failed = await client.post(
        "/query", json={"question": "q2", "conversation_id": conversation_id}
    )

    assert failed.status_code == 500
    count = await db_session.execute(select(func.count()).select_from(Message))
    assert count.scalar_one() == 2


async def test_the_history_is_the_last_turns_in_order(db_session):
    repository = ConversationRepository(db_session)
    conversation = await repository.create()
    for i in range(5):
        await repository.add_exchange(conversation.id, f"hỏi {i}", f"đáp {i}")

    turns = await repository.recent_turns(conversation.id, limit=6)

    assert [t.text for t in turns] == ["hỏi 2", "đáp 2", "hỏi 3", "đáp 3", "hỏi 4", "đáp 4"]
    assert [t.role for t in turns[:2]] == ["user", "assistant"]


async def test_the_trace_keeps_the_question_that_was_searched(
    query_client, db_session, seeded_corpus, monkeypatch
):
    from app.models import QueryTrace

    monkeypatch.setattr(get_settings(), "retrieval_intent_enabled", True)
    client, provider = query_client
    _script(provider, standalone="Người ngồi sau không đội mũ bảo hiểm bị phạt bao nhiêu?")

    await client.post("/query", json={"question": "Thế người ngồi sau?"})

    row = (await db_session.execute(select(QueryTrace))).scalar_one()
    assert row.standalone == "Người ngồi sau không đội mũ bảo hiểm bị phạt bao nhiêu?"
    assert isinstance(
        await db_session.get(Conversation, uuid.UUID(str(row.conversation_id))), Conversation
    )
