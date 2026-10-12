"""R2: one list out of several.

Reciprocal rank fusion scores a candidate by where each source ranked it,
never by the sources' own scores: cosine similarity and ts_rank are on
different scales and cannot be added. With k = 60 the head of every list is
nearly flat (rank 1 is worth 1/61, rank 2 is 1/62), so a chunk two sources
both rank moderately beats one that a single source ranks first.
"""

from collections import Counter
from dataclasses import replace

from app.config import get_settings
from app.core.contracts import Candidate, DocRef, Query, Trace


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
    settings = get_settings()
    fused, kept = reciprocal_rank_fusion(
        candidates, k=settings.retrieval_rrf_k, top_k=settings.retrieval_top_k
    )
    trace.record("fuse_detail", ms=0, kept=kept)
    return fused
