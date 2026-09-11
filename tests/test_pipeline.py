"""The five no-op seams are the whole point of R0. A seam nobody exercises
rots: the signature drifts and nothing notices until the stage that needs it
arrives, several months later. These tests exercise all nine seams from day
one, with fakes standing in for the stages that do not exist yet."""

import dataclasses
import uuid

import pytest

from app.core.contracts import (
    Answer,
    Candidate,
    DocRef,
    GateVerdict,
    Query,
)
from app.core.pipeline import Stages, answer
from app.core.stages import noops

pytestmark = pytest.mark.asyncio


def _candidate(text: str) -> Candidate:
    return Candidate(
        source="vector", rank=1, score=1.0, text=text,
        ref=DocRef(
            document_id=uuid.uuid4(), parent_id=uuid.uuid4(), child_id=uuid.uuid4(),
            page_number=1, filename="d.pdf",
        ),
    )


def _stages(**overrides) -> Stages:
    """R0 wiring with real no-ops, overridable per test."""
    async def retrieve(plan, trace):
        return [_candidate(f"passage for {plan.queries[0]}")]

    async def expand(candidates, trace):
        return candidates

    async def generate(query, context, trace):
        return "answer", ()

    base = Stages(
        rewrite=noops.rewrite,
        plan=noops.plan,
        retrieve=retrieve,
        fuse=noops.fuse,
        expand=expand,
        build_context=noops.build_context_stage,
        gate=noops.gate,
        generate=generate,
        reflect=noops.reflect,
    )
    return dataclasses.replace(base, **overrides)


async def test_r0_runs_exactly_one_iteration():
    """With every flag off the loop must behave as R0: one pass, no gate
    decision, no refinement."""
    seen = []

    async def counting_retrieve(plan, trace):
        seen.append(plan.queries[0])
        return [_candidate("x")]

    result = await answer(
        Query(text="cau hoi"), _stages(retrieve=counting_retrieve), max_iterations=1
    )

    assert seen == ["cau hoi"]
    assert result.trace.iterations == 1


async def test_the_loop_runs_again_when_the_gate_says_insufficient():
    """This is the test that keeps R6 cheap. If it passes at R0, R6 is a
    constant change and a function swap -- nothing structural."""
    verdicts = [
        GateVerdict(sufficient=False, missing="thieu phan thay the",
                    refined_query="hoa don thay the"),
        GateVerdict(sufficient=True),
    ]
    seen = []

    async def gate(query, context, trace):
        return verdicts.pop(0)

    async def retrieve(plan, trace):
        seen.append(plan.queries[0])
        return [_candidate(f"passage for {plan.queries[0]}")]

    result = await answer(
        Query(text="cau hoi goc"), _stages(gate=gate, retrieve=retrieve), max_iterations=3
    )

    assert seen == ["cau hoi goc", "hoa don thay the"], "pass two uses the refined query"
    assert result.trace.iterations == 2


async def test_the_rewrite_runs_once_however_many_iterations():
    """Re-entry is at `plan`, not `rewrite`: the gate writes the refined query
    itself, and running R4's rewriter over it again risks dropping the very
    term the gate added. Invisible at R0, where rewrite returns its input."""
    verdicts = [GateVerdict(sufficient=False, refined_query="second"), GateVerdict(sufficient=True)]
    rewritten = []

    async def counting_rewrite(query, trace):
        rewritten.append(query.text)
        return query

    async def gate(query, context, trace):
        return verdicts.pop(0)

    result = await answer(
        Query(text="first"),
        _stages(rewrite=counting_rewrite, gate=gate),
        max_iterations=3,
    )

    assert rewritten == ["first"]
    assert result.trace.iterations == 2


async def test_context_accumulates_across_iterations():
    """Iteration two must add to iteration one, not replace it."""
    verdicts = [GateVerdict(sufficient=False, refined_query="second"), GateVerdict(sufficient=True)]
    captured = {}

    async def gate(query, context, trace):
        return verdicts.pop(0)

    async def generate(query, context, trace):
        captured["passages"] = [p.text for p in context.passages]
        return "answer", ()

    await answer(
        Query(text="first"), _stages(gate=gate, generate=generate), max_iterations=3
    )

    assert captured["passages"] == ["passage for first", "passage for second"]


async def test_the_loop_answers_anyway_when_it_runs_out_of_iterations():
    """Out of budget is not an error. A partial answer beats a status code."""
    async def always_insufficient(query, context, trace):
        return GateVerdict(sufficient=False, refined_query="again")

    result = await answer(
        Query(text="q"), _stages(gate=always_insufficient), max_iterations=3
    )

    assert isinstance(result, Answer)
    assert result.trace.iterations == 3


async def test_every_stage_writes_one_trace_entry_per_pass():
    result = await answer(Query(text="q"), _stages(), max_iterations=1)

    recorded = [n.node for n in result.trace.nodes]
    assert recorded == [
        "rewrite", "plan", "retrieve", "fuse", "expand", "build_context", "gate",
        "generate", "reflect",
    ]


async def test_a_failing_stage_degrades_to_its_no_op():
    """A decision node that raises costs a capability, never the request."""
    async def broken_rewrite(query, trace):
        raise ValueError("model returned prose instead of json")

    result = await answer(
        Query(text="cau hoi"), _stages(rewrite=broken_rewrite), max_iterations=1
    )

    assert isinstance(result, Answer)
    assert any(n.detail.get("fallback") for n in result.trace.nodes)


async def test_a_stage_without_a_no_op_is_allowed_to_fail_the_request():
    """`retrieve`, `expand`, `build_context` and `generate` have no meaningful
    empty version. Swallowing their failures would return a confident answer
    built from nothing."""
    async def broken_retrieve(plan, trace):
        raise RuntimeError("the database is gone")

    with pytest.raises(RuntimeError):
        await answer(Query(text="q"), _stages(retrieve=broken_retrieve), max_iterations=1)
