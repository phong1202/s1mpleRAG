"""The idempotency tests matter most here: a rerun must change NOTHING --
and a rerun on changed input must leave no trace of the old one."""

from sqlalchemy import func, select, text

from app.models import ChildChunk, Document, ParentChunk
from worker.db import session_scope
from worker.embedding import contextualize
from worker.persistence import persist_document

UNIT = [1.0] + [0.0] * 1535


def _payload(contents=("con",), language="vi"):
    """Shaped like real S2/S3/S4 output -- heading_path and language
    included, which persist_document reads."""
    chunks = {
        "parents": [
            {
                "chunk_index": 0,
                "content": "cha",
                "token_count": 600,
                "page_start": 1,
                "page_end": 1,
                "heading_path": "Chuong I",
                "language": language,
            }
        ],
        "children": [
            {
                "chunk_index": i,
                "parent_index": 0,
                "content": content,
                "token_count": 120,
                "page_number": 1,
                "language": language,
            }
            for i, content in enumerate(contents)
        ],
    }
    enriched = [
        {"id": i, "context": "ngữ cảnh", "category": "TECHNICAL"} for i in range(len(contents))
    ]
    return chunks, enriched, [UNIT for _ in contents]


def _children(document_id):
    """Scoped to this document and ordered: rows from other documents, or
    an order Postgres is free to change after an UPDATE, would make a
    before/after comparison pass or fail by accident."""
    with session_scope() as session:
        rows = session.scalars(
            select(ChildChunk)
            .where(ChildChunk.document_id == document_id)
            .order_by(ChildChunk.chunk_index)
        ).all()
        return [(c.id, c.chunk_index, c.parent_id, c.content, c.contextualized) for c in rows]


def _persist(document_id, chunks, enriched, vectors):
    with session_scope() as session:
        persist_document(document_id, chunks, enriched, vectors, session)


def test_persist_writes_parents_and_children(seeded_document):
    _persist(seeded_document.id, *_payload())

    with session_scope() as session:
        parents = session.scalar(
            select(func.count())
            .select_from(ParentChunk)
            .where(ParentChunk.document_id == seeded_document.id)
        )
    assert parents == 1
    assert len(_children(seeded_document.id)) == 1


def test_running_twice_changes_nothing(seeded_document):
    """ON CONFLICT (document_id, chunk_index) DO UPDATE. A rerun is boring,
    not destructive: same ids, same parents, same text."""
    _persist(seeded_document.id, *_payload())
    before = _children(seeded_document.id)

    _persist(seeded_document.id, *_payload())

    assert _children(seeded_document.id) == before


def test_children_resolve_their_parent_id(seeded_document):
    _persist(seeded_document.id, *_payload())

    with session_scope() as session:
        child = session.scalars(
            select(ChildChunk).where(ChildChunk.document_id == seeded_document.id)
        ).one()
        assert session.get(ParentChunk, child.parent_id).chunk_index == 0


def test_status_becomes_completed_with_a_language(seeded_document):
    _persist(seeded_document.id, *_payload(language="en"))

    with session_scope() as session:
        document = session.get(Document, seeded_document.id)
        assert document.status == "COMPLETED"
        assert document.completed_at is not None
        assert document.language == "en"


def test_contextualized_is_exactly_the_text_s4_embedded(seeded_document):
    _persist(seeded_document.id, *_payload(contents=("Q3 revenue was 41.7bn.",)))

    (_, _, _, content, contextualized) = _children(seeded_document.id)[0]
    assert contextualized == contextualize("ngữ cảnh", content)


def test_a_rerun_on_changed_content_rewrites_the_whole_row(seeded_document):
    """Re-ingesting after a chunking change must not leave a row half old:
    new embedding and contextualized text over the old content -- and the
    old content is what the generated tsv column, and so keyword search,
    reads."""
    _persist(seeded_document.id, *_payload(contents=("alpha invoices",)))
    _persist(seeded_document.id, *_payload(contents=("beta receipts",)))

    (_, _, _, content, contextualized) = _children(seeded_document.id)[0]
    assert content == "beta receipts"
    assert contextualized.endswith("beta receipts")

    with session_scope() as session:
        matches = session.execute(
            text(
                "SELECT plainto_tsquery('simple', 'beta') @@ tsv, "
                "plainto_tsquery('simple', 'alpha') @@ tsv "
                "FROM child_chunks WHERE document_id = :d"
            ),
            {"d": seeded_document.id},
        ).one()
    assert matches == (True, False)


def test_a_rerun_with_fewer_chunks_removes_the_leftovers(seeded_document):
    """An upsert alone never deletes: three chunks, then one, would leave
    chunks 1 and 2 of the old run searchable forever."""
    _persist(seeded_document.id, *_payload(contents=("one", "two", "three")))
    _persist(seeded_document.id, *_payload(contents=("only",)))

    assert [row[1] for row in _children(seeded_document.id)] == [0]
