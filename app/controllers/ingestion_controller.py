from fastapi import APIRouter, Depends, status

from app.schemas.ingestion import (
    DocumentAccepted,
    DocumentRegister,
    UploadTarget,
    UploadUrlRequest,
)
from app.schemas.response import ApiResponse, ErrorResponse
from app.services.ingestion_service import IngestionService, get_ingestion_service

router = APIRouter(prefix="/documents", tags=["ingestion"])

_CONFLICT = {409: {"model": ErrorResponse, "description": "Document already ingested"}}


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
