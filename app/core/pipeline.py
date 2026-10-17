"""The read path, start to finish. Written at R0 and unchanged through R8.

R6 is `max_iterations` going from 1 to 3 plus a real function in `Stages.gate`.
Nothing here moves.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from app.core.contracts import Answer, Context, Query, Trace
from app.core.stages import noops


@dataclass(frozen=True)
class Stages:
    rewrite: Callable
    plan: Callable
    retrieve: Callable
    fuse: Callable
    expand: Callable
    build_context: Callable
    gate: Callable
    generate: Callable
    reflect: Callable
    # R8. None when web search is off: a stage that is not there leaves no
    # trace entry, so the flag off reproduces R7 to the last node.
    web: Callable | None = None


# Only the stages with a meaningful empty version. `retrieve`, `expand`,
# `build_context` and `generate` are absent on purpose: there is nothing they
# could return that is not a confident answer built from nothing.
_FALLBACKS = {
    "rewrite": noops.rewrite,
    "plan": noops.plan,
    "fuse": noops.fuse,
    "gate": noops.gate,
    "reflect": noops.reflect,
    "web": noops.web,
}


async def _run(name: str, stage: Callable, trace: Trace, *args) -> Any:
    """Times the stage, records it, and falls back to the no-op on failure.

    A decision node that returns prose instead of JSON costs the capability it
    provides. It must never cost the request -- the system degrades to R0.
    """
    started = time.perf_counter()
    try:
        result = stage(*args)
        # Stages may be sync: build_context is a pure function and gains
        # nothing from being a coroutine.
        result = await result if hasattr(result, "__await__") else result
        fallback = False
    except Exception as error:
        fallback_stage = _FALLBACKS.get(name)
        if fallback_stage is None:
            # No empty version of this stage means the request fails -- but
            # the trace still says which stage stopped it, and why.
            ms = int((time.perf_counter() - started) * 1000)
            trace.record(name, ms=ms, error=f"{type(error).__name__}: {error}")
            raise
        result = await fallback_stage(*args)
        fallback = str(error)
    trace.record(name, ms=int((time.perf_counter() - started) * 1000), fallback=fallback)
    return result


@dataclass(frozen=True)
class Gathered:
    """What the retrieval loop ends with. `asked` is the question to answer;
    `first_round` is what left `fuse` on the first round, which the eval has
    scored since R1."""

    asked: Query
    context: Context
    first_round: list
    iterations: int


async def gather(query: Query, stages: Stages, max_iterations: int, trace: Trace) -> Gathered:
    """Every stage up to the answer: rewrite once, then search, fuse, expand,
    build the context and ask the gate, as many rounds as it takes or may."""
    context = Context.empty()
    first_round: list = []

    # Outside the loop: re-entry is at `plan`. The gate writes the refined
    # query itself, and running the rewriter over it again risks dropping the
    # very term the gate added.
    query = await _run("rewrite", stages.rewrite, trace, query, trace)
    # What the user asked, as the rewriter put it. The gate's refined queries
    # are searches, not questions: the answer is always to this one.
    asked = query
    verdict = None

    for iteration in range(1, max_iterations + 1):
        plan = await _run("plan", stages.plan, trace, query, trace)
        candidates = await _run("retrieve", stages.retrieve, trace, plan, trace)
        candidates = await _run("fuse", stages.fuse, trace, query, candidates, trace)
        if iteration == 1:
            first_round = list(candidates)
        candidates = await _run("expand", stages.expand, trace, candidates, trace)
        context = await _run(
            "build_context", stages.build_context, trace, context, candidates, trace
        )
        verdict = await _run("gate", stages.gate, trace, query, context, trace)
        trace.iterations = iteration
        if verdict.sufficient or iteration == max_iterations:
            break
        query = query.refined(verdict.refined_query or query.text)

    if (
        verdict is not None
        and not verdict.sufficient
        and trace.iterations >= 2
        and stages.web is not None
    ):
        # The last resort (spec decision 13): the gate found the corpus short,
        # the corpus was searched again and is still short. Searched with the
        # question, not the gate's search, which is worded for the corpus.
        # What comes back goes into `untrusted` and on to the answer and the
        # check only: nothing after this point chooses a tool (§7).
        found = await _run("web", stages.web, trace, asked, trace)
        if found:
            context = await _run(
                "build_context", stages.build_context, trace, context, found, trace
            )

    if verdict is not None and not verdict.sufficient:
        # Out of rounds and still short: answer anyway (spec §6.3), saying
        # which part the documents may not cover rather than filling it in.
        asked = replace(asked, unanswered=verdict.missing or "part of the question")
    return Gathered(asked, context, first_round, trace.iterations)


async def respond(gathered: Gathered, stages: Stages, trace: Trace) -> tuple[Answer, Answer]:
    """Answer, then check the answer (R7): the draft, and what the check left
    of it. Apart from `answer` so the eval can judge both from one draft --
    the answering model does not write the same answer twice."""
    text, citations = await _run(
        "generate", stages.generate, trace, gathered.asked, gathered.context, trace
    )
    draft = Answer(text=text, citations=citations, trace=trace)
    # Checked against the question too: small talk has nothing to check.
    final = await _run(
        "reflect", stages.reflect, trace, gathered.asked, draft, gathered.context, trace
    )
    return draft, final


async def answer(
    query: Query, stages: Stages, max_iterations: int, trace: Trace | None = None
) -> Answer:
    # Passed in by a caller that wants the trace even when this raises.
    trace = Trace() if trace is None else trace
    gathered = await gather(query, stages, max_iterations, trace)
    _, final = await respond(gathered, stages, trace)
    return final
