"""Builds the stage set from the flags. This is the only place that knows
which rung of the ladder the deployment is standing on."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
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

    async def expand(candidates, trace):
        return await expand_parents(ChunkRepository(session), candidates)

    async def generate(query, context, trace):
        return await _generate(query, context, provider)

    return Stages(
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
