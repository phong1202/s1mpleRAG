"""S5 -- Persist. One transaction, sync Session. Always safe to rerun --
with the same input it changes nothing, and with changed input it leaves no
trace of the old run.
"""

import uuid
from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import ChildChunk, Document, ParentChunk
from worker.embedding import contextualize

_NATURAL_KEY = ("document_id", "chunk_index")


def _upsert(session: Session, model: type, rows: list[dict]) -> None:
    """One multi-row INSERT .. ON CONFLICT per table rather than a round trip
    per row (3.9s for 2,000 children that way). Every column but the
    natural key is overwritten on conflict: a rerun after a chunking change
    must not leave a row half old -- a new embedding over old content, and
    old content is what the generated tsv column indexes."""
    if not rows:
        return
    statement = insert(model)
    overwritten = [column for column in rows[0] if column not in _NATURAL_KEY]
    statement = statement.on_conflict_do_update(
        index_elements=list(_NATURAL_KEY),
        set_={column: statement.excluded[column] for column in overwritten},
    )
    session.execute(statement, rows)


def persist_document(
    document_id: uuid.UUID,
    chunks: dict,
    enriched: list[dict],
    vectors: list[list[float]],
    session: Session,
) -> None:
    """`vectors` is row-aligned with chunks["children"]; the caller checks
    that against S4's manifest before getting here."""
    _upsert(session, ParentChunk, [{"document_id": document_id, **p} for p in chunks["parents"]])
    parent_ids = dict(
        session.execute(
            select(ParentChunk.chunk_index, ParentChunk.id).where(
                ParentChunk.document_id == document_id
            )
        ).all()
    )

    # Indexed, not .get() with a fallback: S3 gives every child an entry, so
    # a missing one means the staged artifacts came from different runs.
    context_by_id = {e["id"]: e for e in enriched}
    child_rows = []
    for child, vector in zip(chunks["children"], vectors, strict=True):
        meta = context_by_id[child["chunk_index"]]
        child_rows.append(
            {
                "document_id": document_id,
                "parent_id": parent_ids[child["parent_index"]],
                "chunk_index": child["chunk_index"],
                "content": child["content"],
                "contextualized": contextualize(meta["context"], child["content"]),
                "page_number": child["page_number"],
                "token_count": child["token_count"],
                "embedding": vector,
                "category": meta["category"],
                # Inherited from the parent in S2. It picks the text search
                # config for the generated tsv column, so a NULL here builds
                # the keyword index with the wrong analyser and no error.
                "language": child["language"],
            }
        )
    _upsert(session, ChildChunk, child_rows)

    # An upsert never deletes: three chunks, then one, would leave chunks 1
    # and 2 of the old run searchable forever. Children first -- a stale
    # parent's delete would cascade to them anyway, but kept children have
    # just been repointed above and must not be caught by it.
    session.execute(
        delete(ChildChunk).where(
            ChildChunk.document_id == document_id,
            ChildChunk.chunk_index.not_in([c["chunk_index"] for c in chunks["children"]]),
        )
    )
    session.execute(
        delete(ParentChunk).where(
            ParentChunk.document_id == document_id,
            ParentChunk.chunk_index.not_in([p["chunk_index"] for p in chunks["parents"]]),
        )
    )

    document = session.get(Document, document_id)
    document.status = "COMPLETED"
    document.stage = "PERSISTING"
    document.completed_at = datetime.now(UTC)
    # The dominant language, by parent count. A bilingual document gets the
    # majority side -- filtering at document level is only a coarse hint;
    # the per-chunk column is what retrieval actually reads.
    languages = [p["language"] for p in chunks["parents"] if p["language"]]
    document.language = Counter(languages).most_common(1)[0][0] if languages else None
