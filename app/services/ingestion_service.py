import asyncio

from fastapi import Depends

from app.config import get_settings
from app.exceptions import AppException, ErrorCode
from app.models.document import Document
from app.repositories.document_repository import DocumentRepository, get_document_repository
from app.schemas.ingestion import DocumentRegister, UploadTarget
from shared.storage import ObjectStore, get_public_store, get_store


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
        return document


async def get_ingestion_service(
    repository: DocumentRepository = Depends(get_document_repository),
) -> IngestionService:
    return IngestionService(repository, get_store())
