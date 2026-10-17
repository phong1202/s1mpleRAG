"""R5: a document's outline, as candidates the rest of the read path keeps."""

import uuid

import pytest

from app.core.contracts import Candidate, DocRef, Filters, Plan
from app.core.parent_expander import expand_parents
from app.core.retrievers.browse import BrowseRetriever
from app.core.stages.fusion import make_fuse
from app.core.stages.noops import fuse as noop_fuse
from app.models import ChildChunk, Document, ParentChunk
from app.repositories.chunk_repository import ChunkRepository
from tests.helpers import unit_vector

pytestmark = pytest.mark.asyncio

OUTLINE = [
    # (heading_path, page, content)
    (None, 1, "Căn cứ Hiến pháp. Luật này quy định về hoạt động đường bộ."),
    ("Chương I. QUY ĐỊNH CHUNG > Điều 1. Phạm vi điều chỉnh", 1, "Luật này quy định..."),
    ("Chương I. QUY ĐỊNH CHUNG > Điều 2. Giải thích từ ngữ", 2, "Trong Luật này..."),
    ("Chương I. QUY ĐỊNH CHUNG > Điều 2. Giải thích từ ngữ", 3, "tiếp theo của Điều 2"),
    ("Chương II. KẾT CẤU HẠ TẦNG > Mục 1. QUỸ ĐẤT > Điều 3. Đất dành cho đường bộ", 5, "..."),
    ("Chương II. KẾT CẤU HẠ TẦNG > Mục 2. CÔNG TRÌNH > Điều 4. Công trình đường bộ", 7, "..."),
]


async def _seed(session, title="Luật Đường bộ", status="COMPLETED", outline=OUTLINE):
    document = Document(
        sha256_hash=uuid.uuid4().hex * 2,
        filename="luat-duong-bo.pdf",
        object_key=f"raw/{uuid.uuid4().hex}.pdf",
        size_bytes=1,
        title=title,
        status=status,
    )
    session.add(document)
    await session.flush()
    for index, (heading, page, content) in enumerate(outline):
        parent = ParentChunk(
            document_id=document.id,
            chunk_index=index,
            content=content,
            heading_path=heading,
            token_count=600,
            page_start=page,
            page_end=page,
        )
        session.add(parent)
        await session.flush()
        session.add(
            ChildChunk(
                document_id=document.id,
                parent_id=parent.id,
                chunk_index=index,
                content=content,
                contextualized=content,
                page_number=page,
                token_count=150,
                embedding=unit_vector(index),
            )
        )
    await session.flush()
    return document


def _plan(document_id=None, hint=None) -> Plan:
    return Plan(
        strategy="BROWSE_DOC",
        queries=("q",),
        filters=Filters(document_id=document_id),
        document_hint=hint,
    )


async def test_the_outline_opens_the_document_then_lists_each_chapter(db_session):
    document = await _seed(db_session)

    candidates = await BrowseRetriever(db_session).search(_plan(document.id))

    assert [c.heading for c in candidates] == [
        "Luật Đường bộ",
        "Chương I. QUY ĐỊNH CHUNG",
        "Chương II. KẾT CẤU HẠ TẦNG",
    ]
    assert "Căn cứ Hiến pháp" in candidates[0].text
    assert candidates[1].text.splitlines() == [
        "Chương I. QUY ĐỊNH CHUNG",
        "- Điều 1. Phạm vi điều chỉnh",
        "- Điều 2. Giải thích từ ngữ",
    ]
    assert "- Mục 1. QUỸ ĐẤT > Điều 3. Đất dành cho đường bộ" in candidates[2].text
    assert [c.ref.page_number for c in candidates] == [1, 1, 5]
    assert {c.source for c in candidates} == {"browse"}
    assert [c.rank for c in candidates] == [1, 2, 3]


async def test_the_document_is_found_by_the_name_the_question_used(db_session):
    document = await _seed(db_session)

    candidates = await BrowseRetriever(db_session).search(_plan(hint="Luật Đường bộ"))

    assert candidates and all(c.ref.document_id == document.id for c in candidates)


async def test_a_name_that_finds_no_single_document_gives_nothing(db_session):
    """The caller then searches the corpus as usual."""
    await _seed(db_session)

    assert await BrowseRetriever(db_session).search(_plan(hint="Bộ luật Dân sự")) == []


async def test_a_document_without_headings_gives_its_opening_only(db_session):
    document = await _seed(db_session, outline=[(None, 1, "あいさつ")])

    candidates = await BrowseRetriever(db_session).search(_plan(document.id))

    assert len(candidates) == 1 and candidates[0].text.endswith("あいさつ")


def _browse(n: int) -> list[Candidate]:
    ref = lambda i: DocRef(  # noqa: E731
        document_id=uuid.UUID(int=1),
        parent_id=uuid.UUID(int=100 + i),
        child_id=uuid.UUID(int=200 + i),
        page_number=i,
        filename="x.pdf",
    )
    return [
        Candidate(
            source="browse",
            rank=i,
            score=1.0,
            text=f"Chương {i}",
            ref=ref(i),
            heading=f"Chương {i}",
        )
        for i in range(1, n + 1)
    ]


async def test_an_outline_is_neither_reordered_nor_cut_on_its_way_to_the_context():
    """Fourteen chapters are one outline. Cut to ten, or reordered by a
    reranker that scores "Chương 3" against a question, it stops being one."""

    class Reranker:
        async def scores(self, query, documents):
            raise AssertionError("an outline is not reranked")

    from app.core.contracts import Query, Trace

    outline = _browse(14)

    assert await make_fuse(Reranker())(Query(text="q"), outline, Trace()) == outline
    assert await noop_fuse(Query(text="q"), outline, Trace()) == outline


async def test_an_outline_keeps_its_text_through_parent_expansion(db_session):
    outline = _browse(3)

    expanded = await expand_parents(ChunkRepository(db_session), outline)

    assert [c.text for c in expanded] == ["Chương 1", "Chương 2", "Chương 3"]


async def test_a_browse_plan_reads_the_outline_instead_of_searching(db_session):
    from app.config import get_settings
    from app.core.contracts import Trace
    from app.core.deps import build_stages
    from shared.llm import AsyncStubProvider

    document = await _seed(db_session)
    stages = build_stages(db_session, AsyncStubProvider(), get_settings())
    trace = Trace()

    candidates = await stages.retrieve(_plan(hint="Luật Đường bộ"), trace)

    assert {c.source for c in candidates} == {"browse"}
    assert candidates[0].ref.document_id == document.id
    detail = {n.node: n.detail for n in trace.nodes}
    assert detail["retrieve_detail"]["per_source"] == {"browse": len(candidates)}


async def test_no_document_found_falls_back_to_searching_and_says_so(db_session):
    from app.config import get_settings
    from app.core.contracts import Trace
    from app.core.deps import build_stages
    from shared.llm import AsyncStubProvider

    await _seed(db_session)
    stages = build_stages(db_session, AsyncStubProvider(), get_settings())
    trace = Trace()

    candidates = await stages.retrieve(_plan(hint="Bộ luật Dân sự"), trace)

    assert candidates and "browse" not in {c.source for c in candidates}
    detail = {n.node: n.detail for n in trace.nodes}
    assert "no single document" in detail["browse_fallback"]["reason"]
