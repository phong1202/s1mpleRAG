"""R4: the question as it would read with no conversation before it."""

import pytest

from app.core.contracts import Query, Trace, Turn
from app.core.stages.rewrite import RewriteResult, make_rewrite
from shared.llm import AsyncStubProvider

pytestmark = pytest.mark.asyncio

HISTORY = (
    Turn(role="user", text="Đi xe máy không đội mũ bảo hiểm thì bị phạt bao nhiêu?"),
    Turn(role="assistant", text="Phạt từ 400.000 đến 600.000 đồng [1]."),
)


def _provider(**fields) -> AsyncStubProvider:
    provider = AsyncStubProvider()
    result = {"intent": "SEARCH_CORPUS", "standalone": "", "sub_queries": []} | fields
    provider.script(RewriteResult, RewriteResult(**result))
    return provider


async def test_a_follow_up_is_searched_as_a_standalone_question():
    standalone = "Người ngồi sau xe máy không đội mũ bảo hiểm bị phạt bao nhiêu?"
    query = Query(text="Thế nếu người ngồi sau không đội thì sao?", history=HISTORY)

    rewritten = await make_rewrite(_provider(standalone=standalone))(query, Trace())

    assert rewritten.text == standalone
    assert rewritten.search_texts == (standalone,)
    assert rewritten.history == HISTORY and rewritten.intent == "SEARCH_CORPUS"


async def test_the_model_sees_the_last_six_turns_and_the_question():
    turns = tuple(
        Turn(role="user" if i % 2 == 0 else "assistant", text=f"lượt {i}") for i in range(10)
    )
    provider = _provider(standalone="x")

    await make_rewrite(provider)(Query(text="câu hỏi mới", history=turns), Trace())

    sent = "\n".join(m["content"] for m in provider.calls[0][1])
    assert "lượt 3" not in sent and all(f"lượt {i}" in sent for i in range(4, 10))
    assert sent.index("lượt 4") < sent.index("lượt 9") < sent.index("câu hỏi mới")


async def test_one_search_question_means_no_sub_queries():
    rewritten = await make_rewrite(_provider(standalone="a", sub_queries=["a"]))(
        Query(text="q"), Trace()
    )

    assert rewritten.sub_queries == ()


async def test_several_things_asked_become_at_most_three_searches():
    subs = ["một", "hai", "ba", "bốn"]

    rewritten = await make_rewrite(_provider(standalone="a", sub_queries=subs))(
        Query(text="q"), Trace()
    )

    assert rewritten.search_texts == ("một", "hai", "ba")


async def test_an_empty_rewrite_keeps_the_question():
    rewritten = await make_rewrite(_provider(standalone="  "))(Query(text="câu gốc"), Trace())

    assert rewritten.text == "câu gốc"


async def test_the_trace_shows_what_was_searched_for():
    trace = Trace()

    await make_rewrite(_provider(standalone="câu đầy đủ", intent="CHITCHAT"))(
        Query(text="ok"), trace
    )

    detail = {n.node: n.detail for n in trace.nodes}["rewrite_detail"]
    assert detail == {"intent": "CHITCHAT", "standalone": "câu đầy đủ", "sub_queries": []}


async def test_a_refined_query_keeps_its_intent():
    query = Query(text="q", intent="SEARCH_CORPUS")

    assert query.refined("q2").intent == "SEARCH_CORPUS"


async def test_the_rewriter_asks_for_its_own_reasoning_effort():
    class Recording(AsyncStubProvider):
        async def complete(self, messages, schema, reasoning_effort=None):
            self.effort = reasoning_effort
            return await super().complete(messages, schema)

    provider = Recording()
    provider.script(
        RewriteResult, RewriteResult(intent="SEARCH_CORPUS", standalone="q", sub_queries=[])
    )

    await make_rewrite(provider, reasoning_effort="none")(Query(text="q"), Trace())

    assert provider.effort == "none"
