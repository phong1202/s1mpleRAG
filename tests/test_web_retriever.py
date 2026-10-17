"""R8: Tavily, read from a recorded response -- no network in the suite."""

import json
import pathlib

import httpx
import pytest

from app.core.contracts import WebRef
from app.core.retrievers.web import WebRetriever, WebSearchError

pytestmark = pytest.mark.asyncio

RECORDED = json.loads(
    (pathlib.Path(__file__).parent / "fixtures/tavily/search.json").read_text(encoding="utf-8")
)


def _retriever(handler, **kwargs) -> WebRetriever:
    return WebRetriever(
        api_key="tvly-test",
        max_results=kwargs.pop("max_results", 5),
        include_domains=kwargs.pop("include_domains", ()),
        timeout_s=kwargs.pop("timeout_s", 5.0),
        transport=httpx.MockTransport(handler),
    )


async def test_the_results_become_web_candidates_in_their_order():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=RECORDED)

    candidates = await _retriever(handler).search("Tỷ giá USD hôm nay?")

    assert [c.source for c in candidates] == ["web", "web"]
    assert [c.rank for c in candidates] == [1, 2]
    assert candidates[0].ref == WebRef(
        url="https://example.vn/ty-gia", title="Tỷ giá ngoại tệ hôm nay"
    )
    assert candidates[0].text.startswith("Tỷ giá USD niêm yết")
    assert candidates[0].score == pytest.approx(0.91)
    request = sent[0]
    assert str(request.url) == "https://api.tavily.com/search"
    assert request.headers["Authorization"] == "Bearer tvly-test"


async def test_tavilys_own_answer_is_never_asked_for():
    """Spec §7: an answer written by Tavily's model would bypass the
    generator, the citations and the trace."""
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=RECORDED)

    await _retriever(handler, max_results=3).search("q")

    body = bodies[0]
    assert body["include_answer"] is False and body["include_raw_content"] is False
    assert body["query"] == "q" and body["max_results"] == 3
    assert "include_domains" not in body


async def test_a_domain_list_narrows_the_search():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=RECORDED)

    await _retriever(handler, include_domains=("chinhphu.vn", "vbpl.vn")).search("q")

    assert bodies[0]["include_domains"] == ["chinhphu.vn", "vbpl.vn"]


@pytest.mark.parametrize(
    "respond",
    [
        lambda request: httpx.Response(429, json={"detail": "quota"}),
        lambda request: httpx.Response(200, json={"unexpected": True}),
    ],
)
async def test_any_failure_is_a_web_search_error(respond):
    """The pipeline turns it into no web results; it never fails the request."""
    with pytest.raises(WebSearchError):
        await _retriever(respond).search("q")


async def test_a_search_slower_than_its_deadline_fails():
    import asyncio

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, json=RECORDED)

    with pytest.raises(WebSearchError):
        await _retriever(slow, timeout_s=0.05).search("q")
