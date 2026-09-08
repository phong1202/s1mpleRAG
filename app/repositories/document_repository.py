import uuid

from fastapi import Depends
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.utils.database import get_session


class DocumentRepository:
    """No business rules here; never commits -- the session owns the transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def insert_if_new(
        self, sha256_hash: str, filename: str, object_key: str, size_bytes: int
    ) -> Document | None:
        """INSERT ... ON CONFLICT DO NOTHING RETURNING id.

        One atomic statement. None means the file already exists. The
        caller publishes to the broker ONLY when a row comes back -- that
        rule makes enqueue exactly-once per file without a distributed lock.
        """
        statement = (
            insert(Document)
            .values(
                sha256_hash=sha256_hash,
                filename=filename,
                object_key=object_key,
                size_bytes=size_bytes,
            )
            .on_conflict_do_nothing(index_elements=["sha256_hash"])
            .returning(Document.id)
        )
        new_id = (await self.session.execute(statement)).scalar_one_or_none()
        if new_id is None:
            return None
        return await self.session.get(Document, new_id)

    async def get_by_hash(self, sha256_hash: str) -> Document | None:
        """Cheap pre-upload dedup check: tell the client "already ingested"
        before it spends bandwidth uploading, rather than only after."""
        result = await self.session.execute(
            select(Document).where(Document.sha256_hash == sha256_hash)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def list(self, limit: int, offset: int, status: str | None = None):
        query = select(Document)
        if status:
            query = query.where(Document.status == status)
        total = await self.session.scalar(select(func.count()).select_from(query.subquery()))
        rows = await self.session.execute(
            query.order_by(Document.created_at.desc()).limit(limit).offset(offset)
        )
        return list(rows.scalars().all()), total or 0


async def get_document_repository(
    session: AsyncSession = Depends(get_session),
) -> DocumentRepository:
    return DocumentRepository(session)
