"""Builds the stage set from the flags. This is the only place that knows
which rung of the ladder the deployment is standing on."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.contracts import DocRef
from app.core.generator import generate as _generate
from app.core.parent_expander import expand_parents
from app.core.pipeline import Stages
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.vector import VectorRetriever
from app.core.stages import fusion, noops
from app.repositories.chunk_repository import ChunkRepository
from shared.llm import AsyncLLMProvider


def build_stages(session: AsyncSession, provider: AsyncLLMProvider, settings: Settings) -> Stages:
    hybrid = settings.retrieval_bm25_enabled
    retrievers = [VectorRetriever(session, provider)]
    if hybrid:
        retrievers.append(BM25Retriever(session))
    # R5 appends BrowseRetriever here, R8 WebRetriever.

    async def retrieve(plan, trace):
        # One after the other, not gathered: the retrievers share one
        # AsyncSession, and a connection cannot run two statements at once.
        # The keyword query takes about 10 ms; there is nothing to win.
        results = [await r.search(plan) for r in retrievers]
        per_source = {r.source: len(x) for r, x in zip(retrievers, results, strict=True)}
        trace.record("retrieve_detail", ms=0, per_source=per_source)
        return [candidate for group in results for candidate in group]

    # The three *_detail nodes are what lets a wrong answer be located: which
    # parents were ranked, which of them reached the model, which it cited.
    # Timings alone say a request was slow, never why it was wrong.
    async def expand(candidates, trace):
        expanded = await expand_parents(ChunkRepository(session), candidates)
        trace.record("expand_detail", ms=0, parents=[_where(c) for c in expanded])
        # The last read of this pass, and nothing has been written. Ending the
        # transaction here returns the connection to the pool before the gate
        # or the generator calls a model: left open it sits "idle in
        # transaction" for as long as they take, and a pool's worth of slow
        # answers would stall every other endpoint. A reranker runs in `fuse`,
        # before this point, and will need the same care when R3 adds it.
        await session.commit()
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
        text, citations = await _generate(query, context, provider)
        trace.record("generate_detail", ms=0, cited=[c.ordinal for c in citations])
        return text, citations

    return Stages(
        rewrite=noops.rewrite,
        plan=noops.plan,
        retrieve=retrieve,
        # With one source there is nothing to fuse, and the no-op keeps R1's
        # scores and order to the last digit: the flag is also the switch the
        # eval flips to compare the two.
        fuse=fusion.fuse if hybrid else noops.fuse,
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
