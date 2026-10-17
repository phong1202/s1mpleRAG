"""R7: does every sentence of the answer stand on a source?"""

import uuid

import pytest

from app.core.contracts import Answer, Context, DocRef, Passage, Query, Trace
from app.core.generator import map_citations
from app.core.pipeline import answer as run_pipeline
from app.core.stages.reflect import Reflection, make_reflect
from shared.llm import AsyncStubProvider

pytestmark = pytest.mark.asyncio


def _context(*texts: str) -> Context:
    passages = tuple(
        Passage(
            ordinal=i,
            text=text,
            ref=DocRef(
                document_id=uuid.UUID(int=1),
                parent_id=uuid.UUID(int=10 + i),
                child_id=uuid.UUID(int=20 + i),
                page_number=i,
                filename="nd-168.pdf",
            ),
            heading=f"Điều {i}",
        )
        for i, text in enumerate(texts, start=1)
    )
    return Context(passages=passages, token_count=100)


CONTEXT = _context("Phạt tiền 4 triệu đồng.", "Trừ 4 điểm giấy phép lái xe.")


def _draft(text: str, context: Context = CONTEXT) -> Answer:
    return Answer(text=text, citations=map_citations(text, context), trace=Trace())


def _provider(**fields) -> AsyncStubProvider:
    provider = AsyncStubProvider()
    provider.script(
        Reflection, Reflection(**({"grounded": True, "unsupported": [], "note": ""} | fields))
    )
    return provider


def _detail(trace: Trace) -> dict:
    return {n.node: n.detail for n in trace.nodes}["reflect_detail"]


async def test_a_grounded_answer_is_returned_untouched():
    draft = _draft("Phạt 4 triệu đồng [1]. Trừ 4 điểm [2].")

    result = await make_reflect(_provider())(Query(text="q"), draft, CONTEXT, Trace())

    assert result is draft


async def test_an_unsupported_sentence_is_dropped_with_its_marker_and_a_note_added():
    draft = _draft("Phạt 4 triệu đồng [1]. Tạm giữ xe 7 ngày [2]. Trừ 4 điểm [1].")
    provider = _provider(
        grounded=False,
        unsupported=["Tạm giữ xe 7 ngày [2]."],
        note="Tài liệu không nói về việc tạm giữ xe.",
    )
    trace = Trace()

    result = await make_reflect(provider)(Query(text="q"), draft, CONTEXT, trace)

    assert result.text == (
        "Phạt 4 triệu đồng [1]. Trừ 4 điểm [1].\n\nTài liệu không nói về việc tạm giữ xe."
    )
    # [2] was cited only by the dropped sentence; [1] is still used.
    assert [c.ordinal for c in result.citations] == [1]
    detail = _detail(trace)
    assert detail["grounded"] is False and detail["dropped"] == 1 and detail["unmatched"] == []


async def test_a_sentence_quoted_without_its_marker_or_with_other_spacing_still_matches():
    draft = _draft("Phạt 4 triệu đồng [1].\nTạm giữ   xe 7 ngày [2].")
    provider = _provider(grounded=False, unsupported=["Tạm giữ xe 7 ngày"], note="")

    result = await make_reflect(provider)(Query(text="q"), draft, CONTEXT, Trace())

    assert result.text == "Phạt 4 triệu đồng [1]."
    assert [c.ordinal for c in result.citations] == [1]


async def test_a_sentence_not_in_the_answer_is_ignored_and_recorded():
    """The model paraphrased instead of copying. Guessing which sentence it
    meant could drop the wrong one; leaving the answer alone cannot."""
    draft = _draft("Phạt 4 triệu đồng [1].")
    provider = _provider(grounded=False, unsupported=["Phạt 5 triệu."], note="Không có.")
    trace = Trace()

    result = await make_reflect(provider)(Query(text="q"), draft, CONTEXT, trace)

    assert result is draft
    assert _detail(trace)["unmatched"] == ["Phạt 5 triệu."]


async def test_a_bullet_left_empty_is_removed_with_its_line():
    draft = _draft(
        "Mức phạt:\n- Phạt 4 triệu đồng [1].\n- Tạm giữ xe 7 ngày [2].\n- Trừ 4 điểm [2]."
    )
    provider = _provider(grounded=False, unsupported=["Tạm giữ xe 7 ngày [2]."], note="Ghi chú.")

    result = await make_reflect(provider)(Query(text="q"), draft, CONTEXT, Trace())

    assert result.text == ("Mức phạt:\n- Phạt 4 triệu đồng [1].\n- Trừ 4 điểm [2].\n\nGhi chú.")


async def test_a_nested_list_keeps_its_indentation():
    draft = _draft("- Ô tô:\n  - Phạt 4 triệu đồng [1].\n  - Tạm giữ xe 7 ngày [2].")
    provider = _provider(grounded=False, unsupported=["Tạm giữ xe 7 ngày [2]."], note="")

    result = await make_reflect(provider)(Query(text="q"), draft, CONTEXT, Trace())

    assert result.text == "- Ô tô:\n  - Phạt 4 triệu đồng [1]."


async def test_when_every_sentence_goes_only_the_note_is_left():
    draft = _draft("Tạm giữ xe 7 ngày [2].")
    provider = _provider(
        grounded=False, unsupported=["Tạm giữ xe 7 ngày [2]."], note="Tài liệu không đề cập."
    )

    result = await make_reflect(provider)(Query(text="q"), draft, CONTEXT, Trace())

    assert (result.text, result.citations) == ("Tài liệu không đề cập.", ())


async def test_the_prompt_tells_an_omission_from_a_claim_and_reads_clause_heads():
    """Both over-drops measured at R7: an answer missing an exception, and a
    fine stated at the head of a clause for the point the answer named."""
    provider = _provider()

    await make_reflect(provider)(Query(text="q"), _draft("Phạt 4 triệu [1]."), CONTEXT, Trace())

    system = provider.calls[0][1][0]["content"]
    assert "incomplete, not unsupported" in system and "head of a clause" in system
    assert "never mentions the answer" in system


async def test_small_talk_is_not_checked():
    provider = _provider()
    draft = Answer(text="Không có gì!", citations=(), trace=Trace())
    trace = Trace()

    result = await make_reflect(provider)(
        Query(text="cảm ơn", intent="CHITCHAT"), draft, Context.empty(), trace
    )

    assert result is draft and provider.calls == []
    assert _detail(trace)["asked_model"] is False


async def test_the_model_reads_the_question_every_source_and_the_answer():
    provider = _provider()

    await make_reflect(provider)(
        Query(text="Vượt đèn đỏ phạt bao nhiêu?"), _draft("Phạt 4 triệu [1]."), CONTEXT, Trace()
    )

    system, user = (m["content"] for m in provider.calls[0][1])
    assert "copied exactly" in system
    assert "Vượt đèn đỏ phạt bao nhiêu?" in user and "Phạt 4 triệu [1]." in user
    assert "Phạt tiền 4 triệu đồng." in user and "Trừ 4 điểm giấy phép lái xe." in user
    assert "Điều 2" in user and "nd-168.pdf" in user


async def test_a_failing_check_leaves_the_draft_as_the_answer():
    """Spec §6: reflect fails → answer unchanged."""
    from tests.test_pipeline import _stages

    async def broken(query, answer, context, trace):
        raise RuntimeError("the model returned prose")

    result = await run_pipeline(Query(text="q"), _stages(reflect=broken), max_iterations=1)

    reflected = [n for n in result.trace.nodes if n.node == "reflect"]
    assert reflected and reflected[0].detail["fallback"]
    assert result.text


async def test_the_draft_and_the_checked_answer_come_from_one_generation():
    """The eval judges both. Two generations would differ by more than the
    check did, and the difference would be read as the check's."""
    from app.core.pipeline import gather, respond
    from tests.test_pipeline import _stages

    generated = []

    async def generate(query, context, trace):
        generated.append(query.text)
        return "Phạt 4 triệu [1]. Tạm giữ xe [1].", ()

    async def reflect(query, answer, context, trace):
        return Answer(text="Phạt 4 triệu [1].", citations=(), trace=trace)

    stages = _stages(generate=generate, reflect=reflect)
    trace = Trace()
    gathered = await gather(Query(text="q"), stages, 1, trace)

    draft, final = await respond(gathered, stages, trace)

    assert generated == ["q"]
    assert (draft.text, final.text) == ("Phạt 4 triệu [1]. Tạm giữ xe [1].", "Phạt 4 triệu [1].")
