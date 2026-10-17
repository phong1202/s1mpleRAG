"""R5: the router, in code, reading what the rewriter (R4) found."""

import uuid

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.core.contracts import Filters, Query, Trace
from app.core.generator import Draft
from app.core.stages.rewrite import RewriteResult
from app.core.stages.route import route
from app.models import QueryTrace
from tests.test_browse import _seed

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "query, strategy",
    [
        (Query(text="cảm ơn", intent="CHITCHAT"), "NO_RETRIEVAL"),
        (
            Query(
                text="Tóm tắt Luật Đường bộ", intent="SUMMARIZE_DOC", document_hint="Luật Đường bộ"
            ),
            "BROWSE_DOC",
        ),
        (
            Query(
                text="Tóm tắt tài liệu này",
                intent="SUMMARIZE_DOC",
                filters=Filters(document_id=uuid.uuid4()),
            ),
            "BROWSE_DOC",
        ),
        (Query(text="Tóm tắt đi", intent="SUMMARIZE_DOC"), "HYBRID"),
        (Query(text="Vượt đèn đỏ phạt bao nhiêu?", intent="SEARCH_CORPUS"), "HYBRID"),
        (
            Query(
                text="Nghị định 168 phạt gì?", intent="SEARCH_CORPUS", document_hint="Nghị định 168"
            ),
            "HYBRID",
        ),
        (Query(text="q"), "HYBRID"),
    ],
    ids=[
        "small-talk",
        "summary-named",
        "summary-open-document",
        "summary-unnamed",
        "search",
        "search-naming-a-document",
        "no-intent",
    ],
)
async def test_the_strategy_follows_the_intent_and_the_document(query, strategy):
    trace = Trace()

    plan = await route(query, trace)

    assert plan.strategy == strategy
    detail = {n.node: n.detail for n in trace.nodes}["route_detail"]
    assert detail["strategy"] == strategy and detail["reason"]


async def test_a_search_plan_keeps_the_rewritten_searches():
    plan = await route(Query(text="a", intent="SEARCH_CORPUS", sub_queries=("a", "b")), Trace())

    assert plan.queries == ("a", "b")


async def test_a_summary_is_answered_from_the_outline_and_the_trace_says_so(
    query_client, db_session, monkeypatch
):
    settings = get_settings()
    monkeypatch.setattr(settings, "retrieval_intent_enabled", True)
    monkeypatch.setattr(settings, "retrieval_router_enabled", True)
    await _seed(db_session)
    client, provider = query_client
    provider.script(
        RewriteResult,
        RewriteResult(
            intent="SUMMARIZE_DOC",
            standalone="Tóm tắt Luật Đường bộ",
            sub_queries=[],
            document_hint="Luật Đường bộ",
        ),
    )
    provider.script(Draft, Draft(answer="Luật gồm hai chương [1][2]."))

    response = await client.post("/query", json={"question": "Tóm tắt Luật Đường bộ"})

    assert len(response.json()["data"]["citations"]) == 2
    row = (await db_session.execute(select(QueryTrace))).scalar_one()
    assert row.strategy == "BROWSE_DOC"
    detail = {n["node"]: n["detail"] for n in row.nodes}
    assert detail["retrieve_detail"]["per_source"] == {"browse": 3}
