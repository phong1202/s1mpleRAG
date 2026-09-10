import pytest

from app.core.contracts import DocRef, Filters, Plan
from app.core.retrievers.vector import VectorRetriever
from shared.llm import AsyncStubProvider
from tests.helpers import seed_chunks

pytestmark = pytest.mark.asyncio


def _retriever(session):
    return VectorRetriever(session, AsyncStubProvider(dimensions=1536))


async def test_candidates_are_ranked_from_one_within_the_source(db_session):
    """Rank is per source, starts at 1 and has no gaps. R2 divides by
    (k + rank); a rank starting at 0 or skipping a number silently skews every
    fused score."""
    await seed_chunks(db_session)

    candidates = await _retriever(db_session).search(
        Plan(strategy="VECTOR_ONLY", queries=("hoa don",), filters=Filters())
    )

    assert [c.rank for c in candidates] == list(range(1, len(candidates) + 1))
    assert {c.source for c in candidates} == {"vector"}


async def test_candidate_carries_a_doc_ref_with_the_child_page(db_session):
    """Citation accuracy comes from the child's page_number, not the parent's
    page range. A parent spanning two pages cites the wrong one."""
    await seed_chunks(db_session)

    candidates = await _retriever(db_session).search(
        Plan(strategy="VECTOR_ONLY", queries=("hoa don",), filters=Filters())
    )

    assert isinstance(candidates[0].ref, DocRef)
    assert candidates[0].ref.page_number >= 1
    assert candidates[0].text == candidates[0].text.strip()


async def test_scores_are_similarity_not_distance(db_session):
    """`<#>` returns NEGATIVE inner product. Reporting it unchanged would make
    any later sort by score descending return the worst matches first."""
    await seed_chunks(db_session)

    candidates = await _retriever(db_session).search(
        Plan(strategy="VECTOR_ONLY", queries=("hoa don",), filters=Filters())
    )

    assert candidates[0].score >= candidates[-1].score


async def test_sub_queries_are_merged_without_duplicating_a_child(db_session):
    """R4 emits up to three sub-queries that overlap heavily. Without the merge
    a chunk matched by two of them arrives twice from one source and scores
    twice in R2's fusion."""
    await seed_chunks(db_session)
    retriever = _retriever(db_session)

    one = await retriever.search(
        Plan(strategy="VECTOR_ONLY", queries=("a",), filters=Filters())
    )
    two = await retriever.search(
        Plan(strategy="VECTOR_ONLY", queries=("a", "b"), filters=Filters())
    )

    assert len(one) == len(two) == 12, "both queries reach the whole corpus"
    assert len({c.ref.child_id for c in two}) == 12
    assert [c.rank for c in two] == list(range(1, 13))
