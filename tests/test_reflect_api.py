"""R7 end to end, through POST /query: what is dropped is dropped everywhere."""

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.core.generator import Draft
from app.core.stages.reflect import Reflection
from app.models import Message, QueryTrace

pytestmark = pytest.mark.asyncio


async def test_the_checked_answer_is_the_one_returned_stored_and_cited(
    query_client, db_session, seeded_corpus, monkeypatch
):
    monkeypatch.setattr(get_settings(), "retrieval_reflect_enabled", True)
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tài liệu [1]. Tạm giữ xe 7 ngày [2]."))
    provider.script(
        Reflection,
        Reflection(
            grounded=False,
            unsupported=["Tạm giữ xe 7 ngày [2]."],
            note="Tài liệu không nói về tạm giữ xe.",
        ),
    )

    response = await client.post("/query", json={"question": "child 3"})

    data = response.json()["data"]
    assert data["answer"] == "Theo tài liệu [1].\n\nTài liệu không nói về tạm giữ xe."
    assert [c["ordinal"] for c in data["citations"]] == [1]
    stored = (
        await db_session.execute(select(Message).where(Message.role == "assistant"))
    ).scalar_one()
    assert stored.text == data["answer"] and stored.citations == data["citations"]
    row = (await db_session.execute(select(QueryTrace))).scalar_one()
    detail = next(n["detail"] for n in row.nodes if n["node"] == "reflect_detail")
    assert detail["dropped"] == 1


async def test_off_means_no_check(query_client, db_session, seeded_corpus, monkeypatch):
    monkeypatch.setattr(get_settings(), "retrieval_reflect_enabled", False)
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tài liệu [1]."))

    await client.post("/query", json={"question": "child 3"})

    assert Reflection not in {schema for schema, _ in provider.calls}
