"""The read path, start to finish. Written at R0 and unchanged through R8.

R6 is `max_iterations` going from 1 to 3 plus a real function in `Stages.gate`.
Nothing here moves.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
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


# Only the stages with a meaningful empty version. `retrieve`, `expand`,
# `build_context` and `generate` are absent on purpose: there is nothing they
# could return that is not a confident answer built from nothing.
_FALLBACKS = {
    "rewrite": noops.rewrite,
    "plan": noops.plan,
    "fuse": noops.fuse,
    "gate": noops.gate,
    "reflect": noops.reflect,
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
            raise
        result = await fallback_stage(*args)
        fallback = str(error)
    trace.record(name, ms=int((time.perf_counter() - started) * 1000), fallback=fallback)
    return result


async def answer(query: Query, stages: Stages, max_iterations: int) -> Answer:
    trace = Trace()
    context = Context.empty()

    # Outside the loop: re-entry is at `plan`. The gate writes the refined
    # query itself, and running the rewriter over it again risks dropping the
    # very term the gate added.
    query = await _run("rewrite", stages.rewrite, trace, query, trace)

    for iteration in range(1, max_iterations + 1):
        plan = await _run("plan", stages.plan, trace, query, trace)
        candidates = await _run("retrieve", stages.retrieve, trace, plan, trace)
        candidates = await _run("fuse", stages.fuse, trace, query, candidates, trace)
        candidates = await _run("expand", stages.expand, trace, candidates, trace)
        context = await _run("build_context", stages.build_context, trace,
                             context, candidates, trace)
        verdict = await _run("gate", stages.gate, trace, query, context, trace)
        trace.iterations = iteration
        if verdict.sufficient:
            break
        query = query.refined(verdict.refined_query or query.text)

    text, citations = await _run("generate", stages.generate, trace, query, context, trace)
    draft = Answer(text=text, citations=citations, trace=trace)
    return await _run("reflect", stages.reflect, trace, draft, context, trace)
