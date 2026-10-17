"""R6: does the context answer the main part of the question?"""

import uuid

import pytest

from app.core.contracts import Context, DocRef, Passage, Query, Trace
from app.core.stages.gate import GateResult, make_gate
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


def _provider(**fields) -> AsyncStubProvider:
    provider = AsyncStubProvider()
    provider.script(
        GateResult,
        GateResult(**({"sufficient": True, "missing": "", "refined_query": ""} | fields)),
    )
    return provider


async def test_enough_is_sufficient():
    verdict = await make_gate(_provider())(Query(text="q"), _context("Phạt 4 triệu."), Trace())

    assert verdict.sufficient is True and verdict.refined_query is None


async def test_not_enough_says_what_is_missing_and_what_to_search():
    provider = _provider(
        sufficient=False,
        missing="mức phạt với người ngồi sau",
        refined_query="người ngồi sau không đội mũ bảo hiểm",
    )
    trace = Trace()

    verdict = await make_gate(provider)(Query(text="q"), _context("Người điều khiển xe..."), trace)

    assert (verdict.sufficient, verdict.missing, verdict.refined_query) == (
        False,
        "mức phạt với người ngồi sau",
        "người ngồi sau không đội mũ bảo hiểm",
    )
    detail = {n.node: n.detail for n in trace.nodes}["gate_detail"]
    assert detail["sufficient"] is False and detail["passages"] == 1


async def test_the_model_reads_the_question_every_passage_and_the_calibration():
    """Without the calibration clause the gate finds something missing every
    time, and every question costs three rounds (spec §6.3)."""
    provider = _provider()

    await make_gate(provider)(
        Query(text="Vượt đèn đỏ phạt bao nhiêu?"), _context("đoạn một", "đoạn hai"), Trace()
    )

    system, user = (m["content"] for m in provider.calls[0][1])
    assert "wrong or misleading" in system
    assert "Vượt đèn đỏ phạt bao nhiêu?" in user and "đoạn một" in user and "đoạn hai" in user
    assert "Điều 2" in user and "nd-168.pdf" in user


async def test_an_empty_context_is_not_enough_without_asking():
    provider = AsyncStubProvider()

    verdict = await make_gate(provider)(Query(text="câu hỏi"), Context.empty(), Trace())

    assert verdict.sufficient is False and verdict.refined_query == "câu hỏi"
    assert provider.calls == []


@pytest.mark.parametrize("intent", ["CHITCHAT", "SUMMARIZE_DOC"])
async def test_small_talk_and_summaries_need_no_second_search(intent):
    """Small talk searched nothing and a summary holds the whole outline: a
    second search cannot add to either."""
    provider = AsyncStubProvider()

    verdict = await make_gate(provider)(Query(text="q", intent=intent), _context("x"), Trace())

    assert verdict.sufficient is True and provider.calls == []


async def test_an_empty_refined_query_falls_back_to_the_question():
    provider = _provider(sufficient=False, missing="x", refined_query="  ")

    verdict = await make_gate(provider)(Query(text="câu hỏi"), _context("y"), Trace())

    assert verdict.refined_query == "câu hỏi"
