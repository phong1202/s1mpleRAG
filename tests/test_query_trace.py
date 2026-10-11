import uuid

import pytest
from sqlalchemy import select

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
