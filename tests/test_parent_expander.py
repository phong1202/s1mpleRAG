"""The collapse that Task 4's search deliberately does not do."""

from dataclasses import replace

import pytest

from app.core.contracts import Filters, Plan
from app.core.parent_expander import expand_parents
from app.core.retrievers.vector import VectorRetriever
from app.repositories.chunk_repository import ChunkRepository
from shared.llm import AsyncStubProvider
from tests.helpers import seed_chunks

pytestmark = pytest.mark.asyncio


async def _candidates(session, query: str = "hoa don"):
    retriever = VectorRetriever(session, AsyncStubProvider(dimensions=1536))
    return await retriever.search(
        Plan(strategy="VECTOR_ONLY", queries=(query,), filters=Filters())
    )


async def test_expansion_returns_one_candidate_per_parent(db_session):
    """Twelve children collapse to four parents. Without the collapse the same
    parent arrives three times and crowds out the rest of the context."""
    await seed_chunks(db_session)
    candidates = await _candidates(db_session)
    assert len(candidates) == 12

    expanded = await expand_parents(ChunkRepository(db_session), candidates)

    assert len(expanded) == 4
    assert len({c.ref.parent_id for c in expanded}) == 4


async def test_expansion_swaps_child_text_for_parent_text(db_session):
    """The child is what got scored; the parent is what the model reads."""
    await seed_chunks(db_session)
    candidates = await _candidates(db_session)

    expanded = await expand_parents(ChunkRepository(db_session), candidates)

    assert {c.text for c in expanded} == {"parent 0", "parent 1", "parent 2", "parent 3"}


async def test_the_best_ranked_child_wins_its_parent(db_session):
    """The winner must be the highest-ranked child of the parent, not whichever
    one happens to be first in insertion order. Taking any other child cites a
    page the answer did not come from, and nothing raises."""
    await seed_chunks(db_session)
    candidates = await _candidates(db_session)
    best_parent = candidates[0].ref.parent_id
    siblings = [c for c in candidates if c.ref.parent_id == best_parent]
    assert len(siblings) == 3, "the top parent must have losing siblings to discard"

    expanded = await expand_parents(ChunkRepository(db_session), candidates)

    assert expanded[0].ref.child_id == candidates[0].ref.child_id
    assert expanded[0].ref.child_id not in {c.ref.child_id for c in siblings[1:]}


async def test_the_citation_still_points_at_the_child(db_session):
    """Citation precision survives the collapse: the text widens to the parent
    while the reference stays on the child and its page."""
    await seed_chunks(db_session)
    candidates = await _candidates(db_session)

    expanded = await expand_parents(ChunkRepository(db_session), candidates)

    assert expanded[0].ref == candidates[0].ref
    assert expanded[0].text != candidates[0].text


async def test_expanded_ranks_are_contiguous_from_one(db_session):
    """The collapse removes entries. Leaving the original ranks would tell every
    later stage there are better candidates it cannot see.

    The input is reordered so all three children of one parent hold ranks 1-3:
    the surviving ranks are then 1, 4, ... and only a reassignment makes them
    contiguous. Left to the natural order the winners can land on 1, 2, 3, 4 by
    chance, and the test passes while asserting nothing."""
    await seed_chunks(db_session)
    candidates = await _candidates(db_session)
    crowded = candidates[0].ref.parent_id
    siblings = [c for c in candidates if c.ref.parent_id == crowded]
    others = [c for c in candidates if c.ref.parent_id != crowded]
    reordered = [
        replace(c, rank=rank) for rank, c in enumerate(siblings + others, start=1)
    ]

    expanded = await expand_parents(ChunkRepository(db_session), reordered)

    assert [c.rank for c in expanded] == [1, 2, 3, 4]
