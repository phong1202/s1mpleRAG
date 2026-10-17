"""R4: small talk asks for no information, so nothing is searched for it."""

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.core.generator import Draft
from app.core.stages.rewrite import RewriteResult
from app.models import QueryTrace

pytestmark = pytest.mark.asyncio


@pytest.fixture
def rewriting(monkeypatch):
    monkeypatch.setattr(get_settings(), "retrieval_intent_enabled", True)


async def test_thanks_are_answered_without_searching(
    query_client, db_session, seeded_corpus, rewriting, monkeypatch
):
    client, provider = query_client
    provider.script(
        RewriteResult,
        RewriteResult(intent="CHITCHAT", standalone="cảm ơn", sub_queries=[], document_hint=""),
    )
    provider.script(Draft, Draft(answer="Không có gì, bạn cứ hỏi tiếp nhé."))
    embedded: list[str] = []
    original = provider.embed_query

    async def embed_query(text):
        embedded.append(text)
        return await original(text)

    monkeypatch.setattr(provider, "embed_query", embed_query)

    response = await client.post("/query", json={"question": "cảm ơn"})

    data = response.json()["data"]
    assert data["answer"] == "Không có gì, bạn cứ hỏi tiếp nhé."
    assert data["citations"] == []
    assert embedded == []
    row = (await db_session.execute(select(QueryTrace))).scalar_one()
    detail = {n["node"]: n["detail"] for n in row.nodes}
    assert sum(detail["retrieve_detail"]["per_source"].values()) == 0


async def test_the_small_talk_reply_is_told_not_to_state_facts(
    query_client, seeded_corpus, rewriting
):
    client, provider = query_client
    provider.script(
        RewriteResult,
        RewriteResult(intent="CHITCHAT", standalone="xin chào", sub_queries=[], document_hint=""),
    )
    provider.script(Draft, Draft(answer="Chào bạn."))

    await client.post("/query", json={"question": "xin chào"})

    schema, messages = provider.calls[-1]
    assert schema is Draft
    assert "[TRUSTED SOURCES]" not in messages[-1]["content"]
    assert "facts" in messages[0]["content"].lower()
