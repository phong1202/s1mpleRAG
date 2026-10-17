"""The CPU reranker's HTTP surface, with a fake scorer: no torch, no model."""

import httpx
import pytest

from reranker_service.app import create_app

pytestmark = pytest.mark.asyncio

MODEL = "some/cross-encoder"


def _client(scores: dict[str, float]):
    def scorer(query, documents):
        return [scores[d] for d in documents]

    app = create_app(scorer, model=MODEL)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_results_are_best_first_and_point_back_at_the_input():
    async with _client({"a": 0.1, "b": 0.9, "c": 0.5}) as client:
        response = await client.post("/rerank", json={"query": "q", "documents": ["a", "b", "c"]})

    assert response.json() == {
        "model": MODEL,
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.1},
        ],
    }


async def test_top_n_cuts_the_results():
    async with _client({"a": 0.1, "b": 0.9}) as client:
        response = await client.post(
            "/rerank", json={"query": "q", "documents": ["a", "b"], "top_n": 1}
        )

    assert [r["index"] for r in response.json()["results"]] == [1]


async def test_nothing_to_rank_is_an_empty_answer():
    async with _client({}) as client:
        response = await client.post("/rerank", json={"query": "q", "documents": []})

    assert response.json()["results"] == []


async def test_a_request_for_another_model_is_refused():
    """The API asks for the model in its own settings. If the container serves
    a different one, every score -- and every eval figure -- would belong to a
    model nobody chose."""
    async with _client({"a": 0.1}) as client:
        response = await client.post(
            "/rerank", json={"model": "other/model", "query": "q", "documents": ["a"]}
        )

    assert response.status_code == 400


async def test_the_server_says_which_model_it_serves():
    async with _client({}) as client:
        response = await client.get("/health")

    assert response.json() == {"status": "ok", "model": MODEL}


async def test_the_api_client_reads_this_server(monkeypatch):
    """Both ends of the contract in one test: the read path's client against
    this app, so neither can drift from the other unnoticed."""
    from app.core.rerank_client import RerankClient

    app = create_app(lambda q, docs: [len(d) for d in docs], model=MODEL)
    client = RerankClient(
        url="http://test",
        model=MODEL,
        api_key=None,
        timeout_s=5,
        transport=httpx.ASGITransport(app=app),
    )

    assert await client.scores("q", ["aaa", "a", "aa"]) == [3.0, 1.0, 2.0]


def _slow_app(max_waiting: int):
    import time

    def scorer(query, documents):
        time.sleep(0.3)
        return [1.0 for _ in documents]

    return create_app(scorer, model=MODEL, max_waiting=max_waiting)


async def test_a_burst_beyond_the_queue_is_refused_at_once():
    """Fifty requests at once all waited on the lock, every caller timed out,
    and the abandoned ones went on computing: the next request waited 17 s.
    Past the queue, a request is refused at once and its caller keeps the
    fused order without waiting for a timeout."""
    import asyncio

    app = _slow_app(max_waiting=1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            *(client.post("/rerank", json={"query": "q", "documents": ["a"]}) for _ in range(5))
        )

    codes = sorted(r.status_code for r in responses)
    assert 200 in codes and 503 in codes


async def test_health_answers_while_a_request_is_being_scored():
    import asyncio
    import time

    app = _slow_app(max_waiting=4)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        busy = asyncio.create_task(client.post("/rerank", json={"query": "q", "documents": ["a"]}))
        await asyncio.sleep(0.05)
        started = time.perf_counter()
        health = await client.get("/health")
        took = time.perf_counter() - started
        await busy

    assert health.status_code == 200 and took < 0.2


async def test_an_oversized_input_is_refused():
    """A 1 MB query held the lock for 12 s. Chunks are a few hundred tokens."""
    async with _client({"a": 0.1}) as client:
        huge_query = await client.post("/rerank", json={"query": "x" * 20_000, "documents": ["a"]})
        huge_doc = await client.post("/rerank", json={"query": "q", "documents": ["x" * 40_000]})

    assert huge_query.status_code == 422 and huge_doc.status_code == 422
