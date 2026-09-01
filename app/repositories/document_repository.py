from fastapi import Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.utils.database import get_session


class DocumentRepository:
    """Data access for documents. No business rules; never commits — the
    request-scoped session owns the transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, title: str, content: str) -> Document:
        document = Document(title=title, content=content)
        self.session.add(document)
        await self.session.flush()
        await self.session.refresh(document)
        return document

    async def get_by_id(self, document_id: int) -> Document | None:
        return await self.session.get(Document, document_id)

    async def get_by_title(self, title: str) -> Document | None:
        result = await self.session.execute(select(Document).where(Document.title == title))
        return result.scalar_one_or_none()

    async def list(self, limit: int, offset: int) -> tuple[list[Document], int]:
        total = await self.session.scalar(select(func.count()).select_from(Document))
        result = await self.session.execute(
            select(Document).order_by(Document.id).limit(limit).offset(offset)
        )
        return list(result.scalars().all()), total or 0

    async def update(self, document: Document, **fields) -> Document:
        valid_columns = Document.__table__.columns.keys()
        for name in fields:
            if name not in valid_columns:
                raise ValueError(f"{name!r} is not a column of Document")
        for name, value in fields.items():
            setattr(document, name, value)
        await self.session.flush()
        await self.session.refresh(document)
        return document

    async def delete(self, document: Document) -> None:
        await self.session.delete(document)
        await self.session.flush()


async def get_document_repository(
    session: AsyncSession = Depends(get_session),
) -> DocumentRepository:
    return DocumentRepository(session)
