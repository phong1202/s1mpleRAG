"""The collapse from children to parents.

It runs AFTER fusion and reranking rather than inside the search. Those stages
score a candidate against the question, and a 150-token child is entirely about
one thing while the 700-token parent containing it is mostly about six others.
Collapsing first would hand the reranker the diluted version and throw away
every child that did not win its parent before anything had a chance to rescue
it.
"""

import uuid
from dataclasses import replace

from app.core.contracts import Candidate, DocRef
from app.repositories.chunk_repository import ChunkRepository


async def expand_parents(
    repository: ChunkRepository, candidates: list[Candidate]
) -> list[Candidate]:
    """One candidate per parent: the parent's text and heading, the winning
    child's ref.

    `candidates` arrives rank-ordered, so the first child of a parent to appear
    is its best one, and that child's page_number is what the citation points
    at. Keeping any other child cites a page the answer did not come from, and
    nothing raises.
    """
    # An outline from the browse tool (R5) already says what it means to: a
    # chapter and the titles under it. Replacing that with the text of the
    # chapter's first parent would throw the outline away.
    if candidates and all(c.source == "browse" for c in candidates):
        return candidates

    winners: dict[uuid.UUID, Candidate] = {}
    for candidate in candidates:
        if isinstance(candidate.ref, DocRef):
            winners.setdefault(candidate.ref.parent_id, candidate)

    parents = await repository.fetch_parents(list(winners))
    # Ranks are reassigned because the collapse removes entries: a list that
    # went 1, 2, 5, 9 would tell every later stage there are better candidates
    # it cannot see.
    return [
        replace(
            candidate,
            rank=position,
            text=parents[parent_id].content.strip(),
            heading=parents[parent_id].heading_path,
            matched=candidate.text,
        )
        for position, (parent_id, candidate) in enumerate(winners.items(), start=1)
    ]
