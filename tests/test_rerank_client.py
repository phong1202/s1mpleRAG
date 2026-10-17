"""The client speaks the Cohere shape of /rerank, which the repo's own reranker
service, OpenRouter, Cohere and vLLM all accept. No network: a MockTransport
plays the server."""

import json

import httpx
import pytest

from app.core.rerank_client import RerankClient, RerankError

pytestmark = pytest.mark.asyncio


def _client(handler, api_key: str | None = "k", timeout_s: float = 5.0) -> RerankClient:
    return RerankClient(
        url="http://reranker.test/v1",
        model="some/reranker",
        api_key=api_key,
        timeout_s=timeout_s,
        transport=httpx.MockTransport(handler),
    )


def _ranked(*pairs):
    return httpx.Response(
        200, json={"results": [{"index": i, "relevance_score": s} for i, s in pairs]}
    )


async def test_the_request_is_the_cohere_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _ranked((0, 0.1), (1, 0.2))

    await _client(handler).scores("câu hỏi", ["một", "hai"])

    assert seen["url"] == "http://reranker.test/v1/rerank"
    assert seen["auth"] == "Bearer k"
    assert seen["body"] == {
        "model": "some/reranker",
        "query": "câu hỏi",
        "documents": ["một", "hai"],
        "top_n": 2,
    }


async def test_scores_come_back_in_the_order_the_documents_went_out():
    """Servers return results best first. Reading them in that order would
    pin each score on the wrong chunk, and nothing would raise."""
    scores = await _client(lambda r: _ranked((2, 0.9), (0, 0.5), (1, 0.1))).scores(
        "q", ["a", "b", "c"]
    )

    assert scores == [0.5, 0.1, 0.9]


async def test_no_key_sends_no_authorization_header():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return _ranked((0, 1.0))

    await _client(handler, api_key=None).scores("q", ["a"])

    assert seen["auth"] is None


async def test_nothing_to_rank_asks_the_server_nothing():
    def handler(request):
        raise AssertionError("no request expected")

    assert await _client(handler).scores("q", []) == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.3}]}),
        httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.3}] * 2}),
        httpx.Response(200, json={"data": []}),
    ],
    ids=["http-500", "not-json", "an-index-missing", "an-index-twice", "wrong-shape"],
)
async def test_any_answer_that_cannot_be_trusted_is_an_error(response):
    with pytest.raises(RerankError):
        await _client(lambda r: response).scores("q", ["a", "b"])


async def test_a_timeout_is_an_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(RerankError):
        await _client(handler).scores("q", ["a"])


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": float("nan")}, {"index": 1, "relevance_score": 0.1}],
        [{"index": 0, "relevance_score": "inf"}, {"index": 1, "relevance_score": 0.1}],
        [{"index": 0.9, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.1}],
        [{"index": True, "relevance_score": 0.2}, {"index": 0, "relevance_score": 0.1}],
    ],
    ids=["nan", "infinity", "float-index", "bool-index"],
)
async def test_a_score_or_index_that_is_not_a_plain_number_is_an_error(results):
    """NaN sorts inconsistently -- the best chunk can lose its place to a worse
    one -- and Postgres refuses it in the trace, which then is not written."""
    body = json.dumps({"results": results}).encode()
    with pytest.raises(RerankError):
        await _client(lambda r: httpx.Response(200, content=body)).scores("q", ["a", "b"])


async def test_a_number_too_large_for_a_float_is_an_error():
    body = b'{"results": [{"index": 0, "relevance_score": 1' + b"0" * 400 + b"}]}"
    with pytest.raises(RerankError):
        await _client(lambda r: httpx.Response(200, content=body)).scores("q", ["a"])


async def test_a_malformed_url_is_an_error_not_a_crash():
    """httpx.InvalidURL is not an httpx.HTTPError: it escaped the client, and
    the pipeline's fallback for the stage dropped the keyword results too."""
    client = RerankClient(url="http://localhost:8200:1", model="m", api_key=None, timeout_s=1)

    with pytest.raises(RerankError):
        await client.scores("q", ["a"])


async def test_the_timeout_bounds_the_whole_call():
    """httpx times each read, not the call: a server sending a few bytes every
    second never trips it. The setting promises a deadline."""

    class Trickle(httpx.AsyncByteStream):
        async def __aiter__(self):
            import asyncio

            for _ in range(10):
                await asyncio.sleep(0.2)
                yield b" "

    def handler(request):
        return httpx.Response(200, stream=Trickle())

    with pytest.raises(RerankError):
        await _client(handler, timeout_s=0.5).scores("q", ["a"])


async def test_the_answer_says_which_model_served_it():
    def handler(request):
        return httpx.Response(
            200, json={"model": "served/model", "results": [{"index": 0, "relevance_score": 1.0}]}
        )

    client = _client(handler)
    await client.scores("q", ["a"])

    assert client.served_model == "served/model"


async def test_no_model_configured_sends_none_and_lets_the_server_choose():
    """The CPU container serves the model it was built with. Naming it again
    in the API's settings was a second place to keep in step by hand."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ranked((0, 1.0))

    client = RerankClient(
        url="http://reranker.test",
        model=None,
        api_key=None,
        timeout_s=5,
        transport=httpx.MockTransport(handler),
    )
    await client.scores("q", ["a"])

    assert "model" not in seen["body"]
