import asyncio
import uuid
from collections.abc import Sequence

from fastapi import Depends

from app.config import get_settings
from app.exceptions import AppException, ErrorCode
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository, get_document_repository
from app.schemas.ingestion import (
    DocumentProgress,
    DocumentRegister,
    DocumentStatus,
    FileUrl,
    UploadTarget,
)
from shared import progress
from shared.storage import ObjectStore, get_public_store, get_store

_FILE_URL_EXPIRES_S = 300


class IngestionService:
    def __init__(self, repository: DocumentRepository, store: ObjectStore) -> None:
        self.repository = repository
        self.store = store

    async def create_upload_url(self, filename: str, sha256: str) -> UploadTarget:
        """Dedup runs here too, not only in register(): a client that
        already ingested this file should not have to spend upload
        bandwidth to find that out. This is a UX shortcut, not the
        correctness guarantee -- two concurrent uploads of the same file
        still race here, and insert_if_new's ON CONFLICT is what actually
        makes register() safe under that race.
        """
        existing = await self.repository.get_by_hash(sha256)
        if existing is not None:
            raise AppException(ErrorCode.DOCUMENT_ALREADY_INGESTED)

        # The key is derived from the hash so it can be bound into the
        # presigned URL's signature: MinIO then refuses any upload whose
        # bytes do not hash to it, which is what lets register() below
        # trust the hash without reading the object back.
        key = f"raw/{sha256}.pdf"
        return UploadTarget(
            upload_url=get_public_store().presigned_put(key, sha256_hex=sha256),
            object_key=key,
            expires_in=3600,
        )

    async def register(self, payload: DocumentRegister) -> Document:
        # Cheap and local: no I/O. Catches a payload where object_key and
        # sha256 were not both produced by the same create_upload_url call
        # -- the only way, given checksum-bound uploads, that the object at
        # object_key could actually mismatch sha256.
        expected_key = f"raw/{payload.sha256}.pdf"
        if payload.object_key != expected_key:
            raise AppException(
                ErrorCode.HASH_MISMATCH,
                f"object_key {payload.object_key!r} does not match sha256 {payload.sha256!r}",
            )

        # ObjectStore is sync (boto3); awaiting it directly would block the
        # event loop for the whole request. to_thread is enough here since
        # this is a lightweight HEAD, unlike a hash recompute over the file.
        exists = await asyncio.to_thread(self.store.exists, payload.object_key)
        if not exists:
            raise AppException(
                ErrorCode.DOCUMENT_NOT_FOUND, f"Object {payload.object_key} not found in storage"
            )

        limit = get_settings().max_file_size_mb * 1024 * 1024
        if payload.size_bytes > limit:
            raise AppException(ErrorCode.PDF_TOO_LARGE)

        document = await self.repository.insert_if_new(
            sha256_hash=payload.sha256,
            filename=payload.filename,
            object_key=payload.object_key,
            size_bytes=payload.size_bytes,
        )
        if document is None:
            raise AppException(ErrorCode.DOCUMENT_ALREADY_INGESTED)

        # Publish ONLY when a row came back. That rule is what makes
        # enqueue exactly-once per file, with no distributed lock.
        #
        # launch() publishes over kombu, a synchronous network call with no
        # async integration -- the same class of blocking call store.exists()
        # was in Task 9, wrapped the same way.
        from worker.stages import launch

        await asyncio.to_thread(launch, str(document.id))
        return document

    async def get(self, document_id: uuid.UUID) -> Document:
        document = await self.repository.get_by_id(document_id)
        if document is None:
            raise AppException(ErrorCode.DOCUMENT_NOT_FOUND, f"Document {document_id} not found")
        return document

    async def list(self, limit: int, offset: int, status: str | None) -> tuple[list[Document], int]:
        return await self.repository.list(limit=limit, offset=offset, status=status)

    # Sequence, not list: inside this class body `list` is the method above.
    async def statuses(self, documents: Sequence[Document]) -> Sequence[DocumentStatus]:
        """The rows as the API shows them, each running stage's live
        progress overlaid from Redis -- one round trip for the whole page."""
        items = [DocumentStatus.model_validate(d) for d in documents]
        live = await progress.read_many(
            [str(d.id) for d in documents if d.status not in progress.TERMINAL]
        )
        for item in items:
            if str(item.id) in live:
                item.progress = DocumentProgress.model_validate(live[str(item.id)])
        return items

    async def file_url(self, document_id: uuid.UUID) -> FileUrl:
        """Short-lived on purpose: the viewer fetches the bytes at once and
        never keeps the link, so a leaked one is worth five minutes. No
        check on status -- the file is there from upload on, and the FE
        decides what to show."""
        document = await self.get(document_id)
        url = get_public_store().presigned_get(
            document.object_key, expires_in=_FILE_URL_EXPIRES_S, filename=document.filename
        )
        return FileUrl(url=url, expires_in=_FILE_URL_EXPIRES_S)

    async def delete(self, document_id: uuid.UUID) -> None:
        """Cascades down to the chunks via ON DELETE CASCADE; raw/ is left
        alone -- it is the source of truth, and the only way to re-ingest."""
        document = await self.get(document_id)
        await self.repository.delete(document)


async def get_ingestion_service(
    repository: DocumentRepository = Depends(get_document_repository),
) -> IngestionService:
    return IngestionService(repository, get_store())
