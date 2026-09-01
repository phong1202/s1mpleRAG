from fastapi import Depends
from sqlalchemy.exc import IntegrityError

from app.exceptions import AppException, ErrorCode
from app.models.document import Document
from app.repositories.document_repository import (
    DocumentRepository,
    get_document_repository,
)
from app.schemas.document import DocumentCreate, DocumentUpdate

# Name Postgres gave the unique constraint on documents.title (see the init
# migration). Used to tell "title already taken" apart from any other
# IntegrityError, which must not be swallowed as a 409.
_TITLE_UNIQUE_CONSTRAINT = "documents_title_key"


def _is_title_conflict(exc: IntegrityError) -> bool:
    """True when this IntegrityError is the documents.title unique violation.

    SQLAlchemy's asyncpg dialect wraps the driver error in its own DBAPI
    exception (exc.orig) and chains the original asyncpg error as its
    __cause__ — that's where the Postgres constraint name lives.
    """
    cause = getattr(exc.orig, "__cause__", None)
    return getattr(cause, "constraint_name", None) == _TITLE_UNIQUE_CONSTRAINT


class DocumentService:
    """Business rules for documents.

    Absence becomes an error here — the repository reports None, this layer
    decides what that means, and the exception handler renders it. No HTTP
    concepts appear in this class.
    """

    def __init__(self, repository: DocumentRepository) -> None:
        self.repository = repository

    async def create(self, data: DocumentCreate) -> Document:
        if await self.repository.get_by_title(data.title) is not None:
            raise AppException(
                ErrorCode.DOCUMENT_TITLE_EXISTS,
                f"A document titled '{data.title}' already exists",
            )
        try:
            return await self.repository.create(title=data.title, content=data.content)
        except IntegrityError as exc:
            # Fallback for the race the pre-check can't close: two requests
            # both pass get_by_title, then the second insert hits the unique
            # index. Without this, the client sees a bare 500.
            if not _is_title_conflict(exc):
                raise
            raise AppException(
                ErrorCode.DOCUMENT_TITLE_EXISTS,
                f"A document titled '{data.title}' already exists",
            ) from exc

    async def get(self, document_id: int) -> Document:
        document = await self.repository.get_by_id(document_id)
        if document is None:
            raise AppException(
                ErrorCode.DOCUMENT_NOT_FOUND,
                f"Document {document_id} not found",
            )
        return document

    async def list(self, limit: int, offset: int) -> tuple[list[Document], int]:
        return await self.repository.list(limit=limit, offset=offset)

    async def update(self, document_id: int, data: DocumentUpdate) -> Document:
        document = await self.get(document_id)

        fields = data.model_dump(exclude_unset=True, exclude_none=True)

        new_title = fields.get("title")
        # Kept nested rather than collapsed: the outer test asks whether the
        # title is changing at all, the inner whether the new one is taken.
        if new_title is not None and new_title != document.title:  # noqa: SIM102
            if await self.repository.get_by_title(new_title) is not None:
                raise AppException(
                    ErrorCode.DOCUMENT_TITLE_EXISTS,
                    f"A document titled '{new_title}' already exists",
                )

        if not fields:
            return document

        return await self.repository.update(document, **fields)

    async def delete(self, document_id: int) -> None:
        document = await self.get(document_id)
        await self.repository.delete(document)


async def get_document_service(
    repository: DocumentRepository = Depends(get_document_repository),
) -> DocumentService:
    return DocumentService(repository)
