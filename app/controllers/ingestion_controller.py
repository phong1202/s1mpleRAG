import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, status

from app.schemas.ingestion import (
    DocumentAccepted,
    DocumentRegister,
    DocumentStatus,
    UploadTarget,
    UploadUrlRequest,
)
from app.schemas.response import ApiResponse, ErrorResponse, PaginatedData
from app.services.ingestion_service import IngestionService, get_ingestion_service

router = APIRouter(prefix="/documents", tags=["ingestion"])

_CONFLICT = {409: {"model": ErrorResponse, "description": "Document already ingested"}}
_NOT_FOUND = {404: {"model": ErrorResponse, "description": "Document not found"}}

DocumentId = Annotated[uuid.UUID, Path()]


@router.post("/upload-url", response_model=ApiResponse[UploadTarget], responses=_CONFLICT)
async def create_upload_url(
    payload: UploadUrlRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[UploadTarget]:
    target = await service.create_upload_url(payload.filename, payload.sha256)
    return ApiResponse.ok(target)


@router.post(
    "",
    response_model=ApiResponse[DocumentAccepted],
    status_code=status.HTTP_202_ACCEPTED,
    responses=_CONFLICT,
)
async def register_document(
    payload: DocumentRegister,
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[DocumentAccepted]:
    document = await service.register(payload)
    return ApiResponse(
        code=202,
        message="Accepted",
        data=DocumentAccepted(document_id=str(document.id), status=document.status),
    )


@router.get(
    "/{document_id}/status", response_model=ApiResponse[DocumentStatus], responses=_NOT_FOUND
)
async def get_status(
    document_id: DocumentId,
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[DocumentStatus]:
    document = await service.get(document_id)
    return ApiResponse.ok(DocumentStatus.model_validate(document))


@router.get("", response_model=ApiResponse[PaginatedData[DocumentStatus]])
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[PaginatedData[DocumentStatus]]:
    documents, total = await service.list(limit=limit, offset=offset, status=status)
    return ApiResponse.ok(
        PaginatedData[DocumentStatus](
            items=[DocumentStatus.model_validate(d) for d in documents],
            total=total,
            limit=limit,
            offset=offset,
        )
    )


@router.delete("/{document_id}", response_model=ApiResponse[None], responses=_NOT_FOUND)
async def delete_document(
    document_id: DocumentId,
    service: IngestionService = Depends(get_ingestion_service),
) -> ApiResponse[None]:
    await service.delete(document_id)
    return ApiResponse(code=200, message="Document deleted", data=None)
