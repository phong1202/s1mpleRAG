"""Conversations belong to a user, keep a title, and keep their answers' citations."""

import uuid

import pytest
from sqlalchemy import select

from app.auth import ADMIN_ID
from app.core.generator import Draft
from app.models import Conversation, Message, User

pytestmark = pytest.mark.asyncio


async def _ask(client, question, conversation_id=None):
    body = {"question": question}
    if conversation_id:
        body["conversation_id"] = conversation_id
    return await client.post("/query", json=body)


async def other_users_conversation(db_session) -> uuid.UUID:
    other = User(username=f"other-{uuid.uuid4().hex[:6]}", display_name="Other")
    db_session.add(other)
    await db_session.flush()
    conversation = Conversation(user_id=other.id)
    db_session.add(conversation)
    await db_session.flush()
    return conversation.id


async def test_a_conversation_belongs_to_the_asker_and_is_titled_by_its_first_question(
    query_client, db_session, seeded_corpus
):
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tai lieu [1]."))
    long_question = "hoa don " + "x" * 200

    first = await _ask(client, long_question)
    conversation_id = first.json()["data"]["conversation_id"]
    conversation = await db_session.get(Conversation, uuid.UUID(conversation_id))
    title, updated = conversation.title, conversation.updated_at
    await _ask(client, "câu thứ hai", conversation_id)
    await db_session.refresh(conversation)

    assert conversation.user_id == ADMIN_ID
    assert title == long_question[:120] and conversation.title == title
    assert conversation.updated_at > updated


async def test_an_answer_keeps_exactly_the_citations_it_was_returned_with(
    query_client, db_session, seeded_corpus
):
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tai lieu [1]."))

    response = await _ask(client, "hoa don")

    data = response.json()["data"]
    answer = (
        await db_session.execute(select(Message).where(Message.role == "assistant"))
    ).scalar_one()
    question = (
        await db_session.execute(select(Message).where(Message.role == "user"))
    ).scalar_one()
    assert answer.citations == data["citations"] and answer.citations
    assert question.citations is None


async def test_an_answer_without_citations_keeps_an_empty_list(
    query_client, db_session, seeded_corpus
):
    client, provider = query_client
    provider.script(Draft, Draft(answer="Không có thông tin."))

    await _ask(client, "hoa don")

    answer = (
        await db_session.execute(select(Message).where(Message.role == "assistant"))
    ).scalar_one()
    assert answer.citations == []


async def test_another_users_conversation_is_not_found(query_client, db_session, seeded_corpus):
    """404, not 403: the id's existence is not the asker's to learn."""
    client, provider = query_client
    provider.script(Draft, Draft(answer="x"))

    response = await _ask(client, "hoa don", str(await other_users_conversation(db_session)))

    assert response.status_code == 404
