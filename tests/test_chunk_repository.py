"""Against real pgvector. The similarity ordering and the over-fetch are
precisely the parts most likely to be silently wrong, so neither is mocked."""

import pytest

from app.core.contracts import Filters
from app.repositories.chunk_repository import ChunkRepository
from tests.helpers import seed_chunks, unit_vector

pytestmark = pytest.mark.asyncio


async def test_search_returns_every_matching_child_not_one_per_parent(db_session):
    """Search is child-level. Collapsing here would hand the reranker 700-token
    parents and drop the children that lost their parent before fusion could
    rescue them."""
    await seed_chunks(db_session)

    hits = await ChunkRepository(db_session).search(
        vector=unit_vector(0), over_fetch=50, filters=Filters()
    )

    assert len(hits) == 12
    assert len({hit.parent_id for hit in hits}) == 4


async def test_search_orders_by_similarity_not_insertion(db_session):
    """`<#>` is NEGATIVE inner product: ASC is nearest first. Writing DESC
    reverses the entire result set and raises nothing."""
    await seed_chunks(db_session)

    hits = await ChunkRepository(db_session).search(
        vector=unit_vector(6), over_fetch=50, filters=Filters()
    )

    assert hits[0].child_content == "child 6"
    assert hits[0].page_number == 3, "child 6 lives on parent 2, page 3"
    assert hits[0].distance <= hits[1].distance


async def test_over_fetch_takes_the_nearest_candidates_not_an_arbitrary_slice(db_session):
    """The over-fetch is a LIMIT on an ordered set, so with over_fetch wider
    than the corpus every ordering returns the same rows and the ordering of
    the candidates CTE is untested. Only a cut-off narrower than the corpus
    shows it sorts nearest-first."""
    await seed_chunks(db_session)

    hits = await ChunkRepository(db_session).search(
        vector=unit_vector(6), over_fetch=1, filters=Filters()
    )

    assert [hit.child_content for hit in hits] == ["child 6"]


async def test_category_filter_narrows_the_candidate_pool(db_session):
    await seed_chunks(db_session)

    hits = await ChunkRepository(db_session).search(
        vector=unit_vector(0), over_fetch=50, filters=Filters(category="FINANCIAL")
    )

    assert hits == []


async def test_fetch_parents_returns_the_content_of_every_id(db_session):
    """One statement for the whole set: the expansion must not run a query per
    parent."""
    await seed_chunks(db_session)
    repository = ChunkRepository(db_session)
    hits = await repository.search(vector=unit_vector(0), over_fetch=50, filters=Filters())
    parent_ids = list({hit.parent_id for hit in hits})

    contents = await repository.fetch_parents(parent_ids)

    assert set(contents) == set(parent_ids)
    assert sorted(p.content for p in contents.values()) == [
        "parent 0",
        "parent 1",
        "parent 2",
        "parent 3",
    ]


async def test_fetch_parents_of_nothing_asks_the_database_nothing(db_session):
    assert await ChunkRepository(db_session).fetch_parents([]) == {}


async def test_fetch_parents_returns_each_parents_heading(db_session):
    """A parent is a page-bounded slice of an article. Its text often does not
    say which article -- the heading is the only place that does."""
    await seed_chunks(db_session)
    repository = ChunkRepository(db_session)
    hits = await repository.search(vector=unit_vector(0), over_fetch=50, filters=Filters())

    parents = await repository.fetch_parents([hits[0].parent_id])

    assert parents[hits[0].parent_id].heading_path == "Chuong I > Dieu 0"


async def test_search_sets_the_index_parameters_for_its_transaction(db_session):
    """ef_search widens the candidate queue to cover the over-fetch. Iterative
    scan keeps a filtered search complete: without it the index returns its
    nearest ef_search rows first and the filter is applied to those, so a
    filter matching few rows comes back short, or empty, with no error."""
    from sqlalchemy import text

    await seed_chunks(db_session)

    await ChunkRepository(db_session).search(unit_vector(0), over_fetch=50, filters=Filters())

    ef = await db_session.execute(text("SELECT current_setting('hnsw.ef_search')"))
    scan = await db_session.execute(text("SELECT current_setting('hnsw.iterative_scan')"))
    assert ef.scalar_one() == "100"
    assert scan.scalar_one() == "strict_order"
