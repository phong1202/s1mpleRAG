"""R6 end to end, through POST /query: the gate sends the pipeline round again."""

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.core.generator import Draft
from app.core.stages.gate import GateResult
from app.models import QueryTrace

pytestmark = pytest.mark.asyncio


async def test_not_enough_once_means_a_second_round_that_adds_to_the_first(
    query_client, db_session, seeded_corpus, monkeypatch
):
    monkeypatch.setattr(get_settings(), "retrieval_max_iterations", 3)
    client, provider = query_client
    provider.script(
        GateResult,
        GateResult(sufficient=False, missing="thiếu", refined_query="child 7"),
        GateResult(sufficient=True, missing="", refined_query=""),
    )
    provider.script(Draft, Draft(answer="Theo tài liệu [1]."))

    response = await client.post("/query", json={"question": "child 3"})

    assert response.status_code == 200
    row = (await db_session.execute(select(QueryTrace))).scalar_one()
    assert row.iterations == 2
    gates = [n["detail"] for n in row.nodes if n["node"] == "gate_detail"]
    assert [g["sufficient"] for g in gates] == [False, True]
    shown = [n["detail"]["passages"] for n in row.nodes if n["node"] == "context_detail"]
    assert len(shown[1]) >= len(shown[0]), "the second round adds to the first"


async def test_one_round_means_no_gate_call(query_client, db_session, seeded_corpus, monkeypatch):
    monkeypatch.setattr(get_settings(), "retrieval_max_iterations", 1)
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tài liệu [1]."))

    await client.post("/query", json={"question": "child 3"})

    assert GateResult not in {schema for schema, _ in provider.calls}
