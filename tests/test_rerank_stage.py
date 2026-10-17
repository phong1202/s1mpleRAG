"""R3 end to end, through POST /query, with the reranker played by a fake: the
flag adds the step, a dead reranker costs nothing but the step, and no
database connection waits on it."""

import pytest
from sqlalchemy import select

import app.core.deps as deps
from app.config import get_settings
from app.core.generator import Draft
from app.core.rerank_client import RerankError
from app.models import QueryTrace

pytestmark = pytest.mark.asyncio


class FakeReranker:
    instances: list["FakeReranker"] = []
    fail = False
    session = None

    def __init__(self, url, model, api_key, timeout_s):
        self.url, self.model = url, model
        self.open_during_call: list[bool] = []
        FakeReranker.instances.append(self)

    async def scores(self, query, documents):
        self.open_during_call.append(FakeReranker.session.in_transaction())
        if FakeReranker.fail:
            raise RerankError("ConnectError: reranker down")
        return [float(i) for i in range(len(documents))]  # reverses the fused order


@pytest.fixture
def reranking(monkeypatch, db_session):
    monkeypatch.setattr(get_settings(), "retrieval_rerank_enabled", True)
    monkeypatch.setattr(deps, "RerankClient", FakeReranker)
    FakeReranker.instances, FakeReranker.fail, FakeReranker.session = [], False, db_session
    return FakeReranker


async def _detail(client, db_session) -> dict:
    response = await client.post("/query", json={"question": "child 3"})
    assert response.status_code == 200
    row = (await db_session.execute(select(QueryTrace))).scalar_one()
    return {node["node"]: node["detail"] for node in row.nodes}


async def test_with_the_flag_on_the_reranker_orders_the_context(
    query_client, db_session, seeded_corpus, reranking
):
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tai lieu [1]."))

    detail = await _detail(client, db_session)

    rerank = detail["rerank_detail"]
    assert rerank["fallback"] is False
    assert rerank["sent"] == 12, "the whole seeded corpus, not the top ten"
    assert rerank["from_rrf_ranks"][0] == 12, "the fake reverses the order"
    assert reranking.instances[0].model == get_settings().reranker_model


async def test_no_transaction_is_open_while_the_reranker_works(
    query_client, db_session, seeded_corpus, reranking
):
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tai lieu [1]."))

    await _detail(client, db_session)

    assert reranking.instances[0].open_during_call == [False]


async def test_a_dead_reranker_costs_the_step_not_the_answer(
    query_client, db_session, seeded_corpus, reranking
):
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tai lieu [1]."))
    reranking.fail = True

    detail = await _detail(client, db_session)

    assert "reranker down" in detail["rerank_detail"]["fallback"]
    assert len(detail["expand_detail"]["parents"]) >= 1


async def test_with_the_flag_off_there_is_no_rerank_step(
    query_client, db_session, seeded_corpus, monkeypatch
):
    monkeypatch.setattr(get_settings(), "retrieval_rerank_enabled", False)
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tai lieu [1]."))

    detail = await _detail(client, db_session)

    assert "rerank_detail" not in detail
