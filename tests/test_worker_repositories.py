"""The worker's own sync repositories, against the real test database --
the stage tests exercise them too, but only through a whole task."""

from worker.db import session_scope
from worker.repositories.chunks import ChunkRepository
from worker.repositories.documents import DocumentStateRepository


def _document(document_id):
    with session_scope() as session:
        return DocumentStateRepository(session).get(document_id)


def test_advance_moves_the_status_and_only_moves_the_stage_when_given(seeded_document):
    with session_scope() as session:
        DocumentStateRepository(session).advance(seeded_document.id, "PARSING")
    assert (_document(seeded_document.id).status, _document(seeded_document.id).stage) == (
        "PARSING",
        None,
    )

    with session_scope() as session:
        DocumentStateRepository(session).advance(seeded_document.id, "PARSING", stage="PARSING")
    assert _document(seeded_document.id).stage == "PARSING"


def test_a_transient_failure_counts_until_the_ceiling(seeded_document):
    def fail():
        with session_scope() as session:
            return DocumentStateRepository(session).record_failure(
                seeded_document.id, "ENRICHING", "timeout", permanent=False, max_attempts=2
            )

    assert fail() is False
    assert _document(seeded_document.id).status == "RETRYING"
    assert fail() is True

    document = _document(seeded_document.id)
    assert (document.status, document.attempts) == ("DEAD_LETTER", 2)
    assert (document.failed_stage, document.last_error) == ("ENRICHING", "timeout")


def test_a_permanent_failure_dead_letters_at_once_without_counting(seeded_document):
    with session_scope() as session:
        dead = DocumentStateRepository(session).record_failure(
            seeded_document.id, "PARSING", "encrypted", permanent=True, max_attempts=3
        )

    document = _document(seeded_document.id)
    assert dead is True
    assert (document.status, document.attempts) == ("DEAD_LETTER", 0)


def test_parse_result_falls_back_to_the_filename_for_a_title(seeded_document):
    with session_scope() as session:
        DocumentStateRepository(session).set_parse_result(seeded_document.id, 3, None)

    document = _document(seeded_document.id)
    assert (document.page_count, document.title) == (3, "clean_text.pdf")


def test_complete_closes_the_document_out(seeded_document):
    with session_scope() as session:
        DocumentStateRepository(session).complete(seeded_document.id, "vi")

    document = _document(seeded_document.id)
    assert (document.status, document.stage, document.language) == ("COMPLETED", "PERSISTING", "vi")
    assert document.completed_at is not None


def test_chunks_upsert_resolve_parents_and_drop_leftovers(seeded_document):
    unit = [1.0] + [0.0] * 1535

    def parent(i):
        return {
            "document_id": seeded_document.id,
            "chunk_index": i,
            "content": f"parent {i}",
            "token_count": 600,
            "page_start": 1,
            "page_end": 1,
            "heading_path": None,
            "language": "vi",
        }

    def child(i, parent_id):
        return {
            "document_id": seeded_document.id,
            "parent_id": parent_id,
            "chunk_index": i,
            "content": f"child {i}",
            "contextualized": f"child {i}",
            "page_number": 1,
            "token_count": 120,
            "embedding": unit,
            "category": "OTHER",
            "language": "vi",
        }

    with session_scope() as session:
        chunks = ChunkRepository(session)
        chunks.upsert_parents([parent(0), parent(1)])
        ids = chunks.parent_ids(seeded_document.id)
        chunks.upsert_children([child(0, ids[0]), child(1, ids[1])])

    assert set(ids) == {0, 1}

    with session_scope() as session:
        chunks = ChunkRepository(session)
        chunks.delete_leftovers(seeded_document.id, parent_indices=[0], child_indices=[0])
        assert set(chunks.parent_ids(seeded_document.id)) == {0}
