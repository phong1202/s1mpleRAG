"""Builds the stage set from the flags. This is the only place that knows
which rung of the ladder the deployment is standing on."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.contracts import DocRef
from app.core.generator import generate as _generate
from app.core.parent_expander import expand_parents
from app.core.pipeline import Stages
from app.core.retrievers.vector import VectorRetriever
from app.core.stages import noops
from app.repositories.chunk_repository import ChunkRepository
from shared.llm import AsyncLLMProvider


def build_stages(session: AsyncSession, provider: AsyncLLMProvider, settings: Settings) -> Stages:
    retrievers = [VectorRetriever(session, provider)]
    # R2 appends BM25Retriever here, R5 BrowseRetriever, R8 WebRetriever.

    async def retrieve(plan, trace):
        # NOTE for R2: these retrievers share one AsyncSession, and gathering
        # two that both query the database raises "another operation is in
        # progress". R0 has a single retriever, so the gather is safe today.
        results = await asyncio.gather(*(r.search(plan) for r in retrievers))
        per_source = {r.source: len(x) for r, x in zip(retrievers, results, strict=True)}
        trace.record("retrieve_detail", ms=0, per_source=per_source)
        return [candidate for group in results for candidate in group]

    # The three *_detail nodes are what lets a wrong answer be located: which
    # parents were ranked, which of them reached the model, which it cited.
    # Timings alone say a request was slow, never why it was wrong.
    async def expand(candidates, trace):
        expanded = await expand_parents(ChunkRepository(session), candidates)
        trace.record("expand_detail", ms=0, parents=[_where(c) for c in expanded])
        return expanded

    def build_context(previous, candidates, trace):
        context = noops.build_context_stage(previous, candidates, trace)
        shown = [
            {"n": p.ordinal, "file": p.ref.filename, "page": p.ref.page_number}
            for p in context.passages
        ]
        trace.record("context_detail", ms=0, passages=shown)
        return context

    async def generate(query, context, trace):
        # Retrieval is over and nothing has been written. Ending the
        # transaction here returns the connection to the pool for the seconds
        # the model takes; left open it sits "idle in transaction", and a
        # pool's worth of slow answers would stall every other endpoint.
        await session.commit()
        text, citations = await _generate(query, context, provider)
        trace.record("generate_detail", ms=0, cited=[c.ordinal for c in citations])
        return text, citations

    return Stages(
        rewrite=noops.rewrite,
        plan=noops.plan,
        retrieve=retrieve,
        fuse=noops.fuse,
        expand=expand,
        build_context=build_context,
        gate=noops.gate,
        generate=generate,
        reflect=noops.reflect,
    )


def _where(candidate) -> dict:
    """A candidate as the trace keeps it: enough to find the page, no text."""
    ref = candidate.ref
    place = (
        {"file": ref.filename, "page": ref.page_number}
        if isinstance(ref, DocRef)
        else {"url": ref.url}
    )
    return {**place, "score": round(candidate.score, 4)}
