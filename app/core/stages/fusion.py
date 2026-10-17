"""R2: one list out of several.

Reciprocal rank fusion scores a candidate by where each source ranked it,
never by the sources' own scores: cosine similarity and ts_rank are on
different scales and cannot be added. With k = 60 the head of every list is
nearly flat (rank 1 is worth 1/61, rank 2 is 1/62), so a chunk two sources
both rank moderately beats one that a single source ranks first.
"""

import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Protocol

from app.config import get_settings
from app.core.contracts import Candidate, DocRef, Query, Trace
from app.core.rerank_client import RerankError

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    async def scores(self, query: str, documents: list[str]) -> list[float]: ...


def _key(candidate: Candidate) -> str:
    ref = candidate.ref
    return str(ref.child_id) if isinstance(ref, DocRef) else ref.url


def reciprocal_rank_fusion(
    candidates: list[Candidate], k: int, top_k: int
) -> tuple[list[Candidate], dict[str, int]]:
    """The fused list, cut to top_k, and where its members came from:
    {"bm25+vector": 4, "vector": 5, "bm25": 1}.

    Each source must list a chunk at most once; the retrievers see to that. A
    fused candidate is the copy from the source that ranked it highest, with
    its rank and score replaced by the fused ones.
    """
    totals: dict[str, float] = {}
    best: dict[str, Candidate] = {}
    sources: dict[str, set[str]] = {}
    for candidate in candidates:
        key = _key(candidate)
        totals[key] = totals.get(key, 0.0) + 1 / (k + candidate.rank)
        sources.setdefault(key, set()).add(candidate.source)
        # Equal ranks go to the source whose name sorts first, so the copy
        # kept does not depend on the order the retrievers were listed in.
        held = best.get(key)
        if held is None or (candidate.rank, candidate.source) < (held.rank, held.source):
            best[key] = candidate

    # Ties are broken by the better original rank, then by the key: the same
    # question gives the same context however the input was ordered.
    ordered = sorted(totals, key=lambda key: (-totals[key], best[key].rank, key))[:top_k]
    kept = Counter("+".join(sorted(sources[key])) for key in ordered)
    fused = [
        replace(best[key], rank=position, score=totals[key])
        for position, key in enumerate(ordered, start=1)
    ]
    return fused, dict(kept)


async def fuse(query: Query, candidates: list[Candidate], trace: Trace) -> list[Candidate]:
    """R2: the fused list, cut to top_k."""
    return await make_fuse(None)(query, candidates, trace)


def make_fuse(
    reranker: Reranker | None, release: Callable[[], Awaitable[None]] | None = None
) -> Callable:
    """The fuse stage. With a reranker (R3), RRF hands it a deeper pool and
    the reranker's order decides what is kept.

    `release` runs just before the reranker is called: the request's database
    transaction has nothing to wait for while another service works, and left
    open it holds a pooled connection for as long as the reranker takes.
    """

    async def stage(query: Query, candidates: list[Candidate], trace: Trace) -> list[Candidate]:
        settings = get_settings()
        depth = settings.retrieval_rerank_candidates if reranker else settings.retrieval_top_k
        fused, kept = reciprocal_rank_fusion(candidates, k=settings.retrieval_rrf_k, top_k=depth)
        trace.record("fuse_detail", ms=0, kept=kept)
        if reranker is None:
            return fused
        if release is not None:
            await release()
        return await _rerank(query, fused, reranker, trace)

    return stage


async def _rerank(
    query: Query, pool: list[Candidate], reranker: Reranker, trace: Trace
) -> list[Candidate]:
    settings = get_settings()
    started = time.perf_counter()
    try:
        scores = await reranker.scores(query.text, [candidate.text for candidate in pool])
        if len(scores) != len(pool):
            raise RerankError(f"{len(scores)} scores for {len(pool)} candidates")
    except Exception as error:  # noqa: BLE001
        # Anything at all. Left to the pipeline, an error here would take the
        # stage's no-op fallback -- the raw input cut to ten, keyword results
        # dropped -- instead of this one: the fused order, which is a good
        # order. Losing the reranker costs some MRR, never the request.
        logger.warning("rerank failed, keeping the fused order: %s", error)
        trace.record(
            "rerank_detail",
            ms=0,
            took_ms=int((time.perf_counter() - started) * 1000),
            sent=len(pool),
            fallback=f"{type(error).__name__}: {error}",
        )
        return pool[: settings.retrieval_top_k]
    # Equal scores keep the fused order, so a reranker that cannot tell two
    # chunks apart leaves them as the fusion had them.
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))[
        : settings.retrieval_rerank_keep
    ]
    trace.record(
        "rerank_detail",
        # The fuse node already times the whole stage: an ms here would be
        # counted twice in the trace's total.
        ms=0,
        took_ms=int((time.perf_counter() - started) * 1000),
        model=getattr(reranker, "served_model", None),
        sent=len(pool),
        fallback=False,
        # Where each kept chunk stood before: what the reranker changed.
        from_rrf_ranks=[pool[i].rank for i in order],
    )
    return [
        replace(pool[i], rank=position, score=scores[i])
        for position, i in enumerate(order, start=1)
    ]
