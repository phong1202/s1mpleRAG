"""Against real pgvector. The similarity ordering, the DISTINCT ON collapse
and the over-fetch are precisely the parts most likely to be silently wrong,
so none of them is mocked."""

import uuid

import pytest

from app.core.contracts import Filters
from app.models import ChildChunk, Document, ParentChunk
from app.repositories.chunk_repository import ChunkRepository

pytestmark = pytest.mark.asyncio


def _unit(index: int, dimensions: int = 1536) -> list[float]:
    vector = [0.0] * dimensions
    vector[index] = 1.0
    return vector


async def _seed(session, children_per_parent: int = 3, parents: int = 4):
    document = Document(
        sha256_hash=uuid.uuid4().hex * 2,
        filename="decree.pdf",
        object_key="raw/x.pdf",
        size_bytes=1,
    )
    session.add(document)
    await session.flush()

    index = 0
    for p in range(parents):
        parent = ParentChunk(
            document_id=document.id,
            chunk_index=p,
            content=f"parent {p}",
            token_count=600,
            page_start=p + 1,
            page_end=p + 1,
        )
        session.add(parent)
        await session.flush()
        for _ in range(children_per_parent):
            session.add(
                ChildChunk(
                    document_id=document.id,
                    parent_id=parent.id,
                    chunk_index=index,
                    content=f"child {index}",
                    contextualized=f"ctx {index}",
                    page_number=p + 1,
                    token_count=150,
                    embedding=_unit(index),
                    category="LEGAL",
                )
            )
            index += 1
    await session.flush()
    return document


async def test_search_returns_one_row_per_parent(db_session):
    """Twelve children collapse to four parents. Without DISTINCT ON the same
    parent would arrive three times and crowd out the rest of the context."""
    await _seed(db_session)

    hits = await ChunkRepository(db_session).search(
        vector=_unit(0), over_fetch=50, top_k=10, filters=Filters()
    )

    assert len({hit.parent_id for hit in hits}) == len(hits) == 4


async def test_search_keeps_the_best_matching_child_of_each_parent(db_session):
    """DISTINCT ON keeps the FIRST row per group in ORDER BY order, so
    parent_id must lead that ORDER BY. If it does not, Postgres keeps an
    arbitrary child, the citation points at the wrong page, and nothing
    raises."""
    await _seed(db_session)

    # Child 2 is the third child of parent 0 -- the best match must be that
    # child, not child 0, which merely happens to be inserted first.
    hits = await ChunkRepository(db_session).search(
        vector=_unit(2), over_fetch=50, top_k=10, filters=Filters()
    )

    assert hits[0].child_content == "child 2"


async def test_search_orders_by_similarity_not_insertion(db_session):
    """`<#>` is NEGATIVE inner product: ASC is nearest first. Writing DESC
    reverses the entire result set and raises nothing."""
    await _seed(db_session)

    hits = await ChunkRepository(db_session).search(
        vector=_unit(6), over_fetch=50, top_k=10, filters=Filters()
    )

    assert hits[0].page_number == 3, "child 6 lives on parent 2, page 3"
    assert hits[0].distance <= hits[1].distance


async def test_over_fetch_takes_the_nearest_candidates_not_an_arbitrary_slice(db_session):
    """The over-fetch is a LIMIT on an ordered set, so with over_fetch wider
    than the corpus every ordering returns the same rows and the ordering of
    the candidates CTE is untested. Only a cut-off narrower than the corpus
    shows it sorts nearest-first."""
    await _seed(db_session)

    hits = await ChunkRepository(db_session).search(
        vector=_unit(6), over_fetch=1, top_k=10, filters=Filters()
    )

    assert [hit.child_content for hit in hits] == ["child 6"]


async def test_top_k_counts_parents_not_children(db_session):
    """Collapsing after LIMIT would return fewer than top_k parents. The
    over-fetch exists so the collapse still yields the requested count."""
    await _seed(db_session, parents=4)

    hits = await ChunkRepository(db_session).search(
        vector=_unit(0), over_fetch=50, top_k=2, filters=Filters()
    )

    assert len(hits) == 2


async def test_category_filter_narrows_the_candidate_pool(db_session):
    await _seed(db_session)

    hits = await ChunkRepository(db_session).search(
        vector=_unit(0), over_fetch=50, top_k=10, filters=Filters(category="FINANCIAL")
    )

    assert hits == []
