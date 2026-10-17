import pytest

from app.core.contracts import DocRef, Filters, Plan
from app.core.retrievers.bm25 import BM25Retriever
from tests.helpers import seed_texts

pytestmark = pytest.mark.asyncio

TEXTS = [
    "Hóa đơn điện tử có mã của cơ quan thuế.",
    "Thuế giá trị gia tăng được khấu trừ.",
    "Hóa đơn giấy do cơ quan thuế đặt in.",
]


def _plan(*queries: str) -> Plan:
    return Plan(strategy="HYBRID", queries=queries, filters=Filters())


async def test_candidates_are_ranked_from_one_within_the_source(db_session):
    """RRF divides by (k + rank). A rank that starts at 0 or skips a number
    skews every fused score, silently."""
    await seed_texts(db_session, TEXTS)

    candidates = await BM25Retriever(db_session).search(_plan("hóa đơn thuế"))

    assert [c.rank for c in candidates] == [1, 2, 3]
    assert {c.source for c in candidates} == {"bm25"}
    assert candidates[0].score >= candidates[-1].score


async def test_sub_queries_are_merged_without_duplicating_a_child(db_session):
    """A chunk matched by two sub-queries must arrive once from this source,
    or it votes twice in the fusion for one source's opinion."""
    await seed_texts(db_session, TEXTS)

    candidates = await BM25Retriever(db_session).search(_plan("hóa đơn", "cơ quan thuế"))

    children = [c.ref.child_id for c in candidates]
    assert len(children) == len(set(children)) == 3
    assert [c.rank for c in candidates] == [1, 2, 3]


async def test_candidate_carries_a_doc_ref_with_the_child_page(db_session):
    await seed_texts(db_session, TEXTS)

    (candidate,) = await BM25Retriever(db_session).search(_plan("khấu trừ"))

    assert isinstance(candidate.ref, DocRef)
    assert candidate.ref.page_number == 2
    assert candidate.ref.filename == "notes.pdf"
