"""Parent and child chunk rows. Sync, and takes the caller's Session, for
the same reasons as documents.py."""

import uuid

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import ChildChunk, ParentChunk

_NATURAL_KEY = ("document_id", "chunk_index")


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_parents(self, rows: list[dict]) -> None:
        self._upsert(ParentChunk, rows)

    def upsert_children(self, rows: list[dict]) -> None:
        self._upsert(ChildChunk, rows)

    def parent_ids(self, document_id: uuid.UUID) -> dict[int, uuid.UUID]:
        """chunk_index -> id, for resolving a child's parent_index."""
        return dict(
            self.session.execute(
                select(ParentChunk.chunk_index, ParentChunk.id).where(
                    ParentChunk.document_id == document_id
                )
            ).all()
        )

    def delete_leftovers(
        self, document_id: uuid.UUID, parent_indices: list[int], child_indices: list[int]
    ) -> None:
        """An upsert never deletes: three chunks, then one, would leave
        chunks 1 and 2 of the old run searchable forever. Children first --
        a stale parent's delete would cascade to them anyway, but kept
        children have just been repointed and must not be caught by it."""
        self.session.execute(
            delete(ChildChunk).where(
                ChildChunk.document_id == document_id,
                ChildChunk.chunk_index.not_in(child_indices),
            )
        )
        self.session.execute(
            delete(ParentChunk).where(
                ParentChunk.document_id == document_id,
                ParentChunk.chunk_index.not_in(parent_indices),
            )
        )

    def _upsert(self, model: type, rows: list[dict]) -> None:
        """One multi-row INSERT .. ON CONFLICT per table rather than a round
        trip per row (3.9s for 2,000 children that way). Every column but
        the natural key is overwritten on conflict: a rerun after a chunking
        change must not leave a row half old -- a new embedding over old
        content, and old content is what the generated tsv column indexes."""
        if not rows:
            return
        statement = insert(model)
        overwritten = [column for column in rows[0] if column not in _NATURAL_KEY]
        statement = statement.on_conflict_do_update(
            index_elements=list(_NATURAL_KEY),
            set_={column: statement.excluded[column] for column in overwritten},
        )
        self.session.execute(statement, rows)
