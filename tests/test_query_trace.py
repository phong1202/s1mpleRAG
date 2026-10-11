import uuid

import pytest
from sqlalchemy import func, select

from app.core.generator import Draft
from app.models import QueryTrace

pytestmark = pytest.mark.asyncio


async def test_a_trace_row_stores_the_per_node_detail(db_session):
    """Without the per-node breakdown, a poor answer can only be described as
    'it searched badly'. With it you can see the gate asked for a second pass
    and the rerank dropped the right parent at position 11."""
    trace = QueryTrace(
        question="hoa don dien tu",
        iterations=2,
        total_ms=6773,
        citation_count=4,
        nodes=[
            {"node": "retrieve", "ms": 95, "detail": {"vector": 50}},
            {"node": "gate", "ms": 520, "detail": {"sufficient": False}},
        ],
    )
    db_session.add(trace)
    await db_session.flush()

    stored = (await db_session.execute(select(QueryTrace))).scalar_one()

    assert stored.nodes[1]["detail"]["sufficient"] is False
    assert isinstance(stored.id, uuid.UUID)


async def test_optional_columns_stay_null_until_their_stage_exists(db_session):
    """`standalone` arrives at R4 and `strategy` at R5. R1 must not require
    them, or R1 cannot ship before them."""
    trace = QueryTrace(question="q", iterations=1, total_ms=1, citation_count=0, nodes=[])
    db_session.add(trace)
    await db_session.flush()

    assert trace.standalone is None and trace.strategy is None


async def test_every_query_writes_exactly_one_trace_row(query_client, db_session, seeded_corpus):
    client, provider = query_client
    provider.script(Draft, Draft(answer="Lap hoa don dieu chinh [1]."))

    response = await client.post("/query", json={"question": "hoa don dien tu"})

    rows = (await db_session.execute(select(QueryTrace))).scalars().all()

    assert response.status_code == 200
    assert len(rows) == 1
    # build_stages records the per-source counts as a node of their own, ahead
    # of the stage that produced them.
    recorded = [n["node"] for n in rows[0].nodes]
    assert recorded[:4] == ["rewrite", "plan", "retrieve_detail", "retrieve"]
    assert rows[0].iterations == 1


async def test_a_failed_trace_write_does_not_fail_the_answer(
    query_client, monkeypatch, seeded_corpus
):
    """Observability must never be able to take down the feature it observes."""
    client, provider = query_client
    provider.script(Draft, Draft(answer="Lap hoa don dieu chinh [1]."))

    def broken(*args, **kwargs):
        raise RuntimeError("trace table is gone")

    # Broken from the inside: _persist_trace imports QueryTrace when it runs.
    # Replacing _persist_trace itself would remove the very try/except this
    # test is about, and the request would fail for a reason of the test's own.
    monkeypatch.setattr("app.models.QueryTrace", broken)

    response = await client.post("/query", json={"question": "hoa don"})

    assert response.status_code == 200


async def test_a_trace_the_database_rejects_leaves_the_session_usable(
    query_client, db_session, monkeypatch, seeded_corpus
):
    """A failed INSERT aborts the transaction it ran in. Without a savepoint the
    answer is computed and then lost to the commit that ends the request."""
    client, provider = query_client
    provider.script(Draft, Draft(answer="Lap hoa don dieu chinh [1]."))

    def questionless(**columns):
        return QueryTrace(**{**columns, "question": None})  # violates NOT NULL

    monkeypatch.setattr("app.models.QueryTrace", questionless)

    response = await client.post("/query", json={"question": "hoa don"})

    assert response.status_code == 200
    # The session still answers: with a bare flush() this raises
    # PendingRollbackError, which is what the request's own commit would hit.
    count = await db_session.execute(select(func.count()).select_from(QueryTrace))
    assert count.scalar_one() == 0
