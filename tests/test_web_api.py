"""R8 end to end, through POST /query, with the search replayed."""

import pytest
from sqlalchemy import select

from app.config import Settings, get_settings
from app.core.contracts import Candidate, WebRef
from app.core.generator import Draft
from app.core.retrievers.web import WebRetriever
from app.core.stages.gate import GateResult
from app.models import Message, QueryTrace


@pytest.fixture
def web_on(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "web_search_enabled", True)
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test")
    monkeypatch.setattr(settings, "retrieval_max_iterations", 2)
    searched = []

    async def search(self, text):
        searched.append(text)
        return [
            Candidate(
                source="web",
                rank=1,
                score=0.9,
                text="Tỷ giá USD bán ra 25.460 đồng.",
                ref=WebRef(url="https://example.vn/ty-gia", title="Tỷ giá ngoại tệ"),
            )
        ]

    monkeypatch.setattr(WebRetriever, "search", search)
    return searched


@pytest.mark.asyncio
async def test_a_web_page_is_cited_returned_and_stored(
    query_client, db_session, seeded_corpus, web_on
):
    client, provider = query_client
    provider.script(
        GateResult, GateResult(sufficient=False, missing="tỷ giá", refined_query="tỷ giá USD")
    )
    # Every number a source could have: the ones not given are dropped as
    # phantoms, and the web page's is kept.
    cited = "".join(f"[{n}]" for n in range(1, 30))
    provider.script(Draft, Draft(answer=f"Theo một trang web, tỷ giá là 25.460 đồng {cited}."))

    response = await client.post("/query", json={"question": "child 3"})

    data = response.json()["data"]
    web = [c for c in data["citations"] if c["kind"] == "web"]
    assert web == [
        {
            "ordinal": web[0]["ordinal"],
            "kind": "web",
            "document_id": None,
            "filename": None,
            "page_number": None,
            "quote": None,
            "url": "https://example.vn/ty-gia",
            "title": "Tỷ giá ngoại tệ",
        }
    ]
    assert web_on == ["child 3"]
    stored = (
        await db_session.execute(select(Message).where(Message.role == "assistant"))
    ).scalar_one()
    assert web[0] in stored.citations
    row = (await db_session.execute(select(QueryTrace))).scalar_one()
    detail = next(n["detail"] for n in row.nodes if n["node"] == "web_detail")
    assert detail == {"query": "child 3", "urls": ["https://example.vn/ty-gia"]}


@pytest.mark.asyncio
async def test_off_means_no_search(query_client, db_session, seeded_corpus, web_on, monkeypatch):
    monkeypatch.setattr(get_settings(), "web_search_enabled", False)
    client, provider = query_client
    provider.script(GateResult, GateResult(sufficient=False, missing="x", refined_query="x"))
    provider.script(Draft, Draft(answer="Theo tài liệu [1]."))

    await client.post("/query", json={"question": "child 3"})

    assert web_on == []


def _settings(**overrides) -> Settings:
    base = {"DATABASE_URL": "postgresql+asyncpg://u:p@h/d"}
    return Settings(_env_file=None, **(base | overrides))


def test_web_search_without_a_key_does_not_start():
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        _settings(web_search_enabled=True, retrieval_max_iterations=3)


def test_web_search_with_one_round_does_not_start():
    """It would never run: the web waits for a second round of the corpus."""
    with pytest.raises(ValueError, match="RETRIEVAL_MAX_ITERATIONS"):
        _settings(web_search_enabled=True, tavily_api_key="k", retrieval_max_iterations=1)


def test_a_comma_separated_domain_list_is_read():
    settings = _settings(web_search_include_domains=" chinhphu.vn, vbpl.vn ,")

    assert settings.web_domains == ("chinhphu.vn", "vbpl.vn")
