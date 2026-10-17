"""Builds the stage set from the flags. This is the only place that knows
which rung of the ladder the deployment is standing on."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.contracts import DocRef
from app.core.generator import generate as _generate
from app.core.parent_expander import expand_parents
from app.core.pipeline import Stages
from app.core.rerank_client import RerankClient
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.browse import BrowseRetriever
from app.core.retrievers.vector import VectorRetriever
from app.core.stages import fusion, noops
from app.core.stages.gate import make_gate
from app.core.stages.reflect import make_reflect
from app.core.stages.rewrite import make_rewrite
from app.core.stages.route import route
from app.repositories.chunk_repository import ChunkRepository
from shared.llm import AsyncLLMProvider


def build_stages(session: AsyncSession, provider: AsyncLLMProvider, settings: Settings) -> Stages:
    hybrid = settings.retrieval_bm25_enabled
    reranker = (
        RerankClient(
            url=settings.reranker_url,
            model=settings.reranker_model,
            api_key=settings.reranker_api_key,
            timeout_s=settings.reranker_timeout_s,
        )
        if settings.retrieval_rerank_enabled
        else None
    )
    retrievers = [VectorRetriever(session, provider)]
    if hybrid:
        retrievers.append(BM25Retriever(session))
    # R5 appends BrowseRetriever here, R8 WebRetriever.

    browse = BrowseRetriever(session)

    async def retrieve(plan, trace):
        if plan.strategy == "BROWSE_DOC":
            # R5: a question about a whole document reads its outline.
            outline = await browse.search(plan)
            if outline:
                trace.record("retrieve_detail", ms=0, per_source={"browse": len(outline)})
                return outline
            # No single document is clearly the one named: search the corpus,
            # which can still answer, rather than browse the wrong one.
            trace.record(
                "browse_fallback",
                ms=0,
                reason=f"no single document matches {plan.document_hint!r}",
            )
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
        # answers would stall every other endpoint. The reranker runs in `fuse`,
        # before this point, and releases the connection the same way.
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
        # R4: one call that turns a follow-up into the question it stands for.
        rewrite=(
            make_rewrite(provider, reasoning_effort=settings.rewrite_reasoning_effort)
            if settings.retrieval_intent_enabled
            else noops.rewrite
        ),
        # R5: the rewriter's intent and named document choose the tool.
        plan=route if settings.retrieval_router_enabled else noops.plan,
        retrieve=retrieve,
        # With one source and no reranker there is nothing to fuse, and the
        # no-op keeps R1's scores and order to the last digit: the flags are
        # also the switches the eval flips to compare the rungs. The reranker
        # is another service: the transaction ends before it is called, as
        # it does before the model in `expand`.
        fuse=(
            fusion.make_fuse(reranker, release=session.commit) if hybrid or reranker else noops.fuse
        ),
        expand=expand,
        build_context=build_context,
        # R6: with room for more than one round, a real gate decides whether
        # to take it. With one round there is nothing to decide.
        gate=(
            make_gate(provider, reasoning_effort=settings.gate_reasoning_effort)
            if settings.retrieval_max_iterations > 1
            else noops.gate
        ),
        generate=generate,
        # R7: a second reading of the answer, sentence by sentence, against
        # the sources it was written from.
        reflect=(
            make_reflect(provider, reasoning_effort=settings.reflect_reasoning_effort)
            if settings.retrieval_reflect_enabled
            else noops.reflect
        ),
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
