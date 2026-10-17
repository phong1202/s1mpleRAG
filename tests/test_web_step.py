"""R8: the web is searched last, once, and only after the corpus was
searched again and still fell short (spec decision 13)."""

import pytest

from app.core.contracts import Candidate, GateVerdict, Query, Trace, WebRef
from app.core.pipeline import answer, gather
from tests.test_pipeline import _stages

pytestmark = pytest.mark.asyncio


def _page(text: str, n: int = 1) -> Candidate:
    return Candidate(
        source="web", rank=n, score=0.9, text=text, ref=WebRef(url=f"https://x.vn/{n}", title="X")
    )


def _searching(pages):
    searched = []

    async def web(query, trace):
        searched.append(query.text)
        return pages

    return web, searched


async def _never_enough(query, context, trace):
    return GateVerdict(sufficient=False, missing="tỷ giá", refined_query="tỷ giá USD")


async def test_enough_on_the_first_round_never_searches_the_web():
    web, searched = _searching([_page("x")])

    await gather(Query(text="q"), _stages(web=web), 3, Trace())

    assert searched == []


async def test_short_with_no_second_round_does_not_search_the_web():
    """The corpus must have been searched again first: the web is the last
    resort, not a second opinion."""
    web, searched = _searching([_page("x")])

    await gather(Query(text="q"), _stages(web=web, gate=_never_enough), 1, Trace())

    assert searched == []


async def test_short_after_a_second_round_searches_the_web_once_with_the_question():
    web, searched = _searching([_page("Tỷ giá 25.460 đồng.")])
    trace = Trace()

    gathered = await gather(
        Query(text="Tỷ giá USD?"), _stages(web=web, gate=_never_enough), 2, trace
    )

    # The question as asked, not the gate's search for the corpus.
    assert searched == ["Tỷ giá USD?"]
    assert [p.text for p in gathered.context.untrusted] == ["Tỷ giá 25.460 đồng."]
    held = [p.ordinal for p in gathered.context.passages]
    assert gathered.context.untrusted[0].ordinal == max(held) + 1
    assert gathered.asked.unanswered == "tỷ giá"
    assert [n.node for n in trace.nodes].count("web") == 1


async def test_without_a_web_stage_nothing_changes():
    trace = Trace()

    gathered = await gather(Query(text="q"), _stages(gate=_never_enough), 2, trace)

    assert gathered.context.untrusted == ()
    assert "web" not in [n.node for n in trace.nodes]


async def test_a_failing_search_answers_from_the_corpus():
    async def broken(query, trace):
        raise RuntimeError("quota exceeded")

    result = await answer(Query(text="q"), _stages(web=broken, gate=_never_enough), 2)

    assert result.text
    web = [n for n in result.trace.nodes if n.node == "web"]
    assert web and web[0].detail["fallback"]
