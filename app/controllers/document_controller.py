from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, status

from app.schemas.document import DocumentCreate, DocumentOut, DocumentUpdate
from app.schemas.response import (
    ApiResponse,
    ErrorResponse,
    PaginatedData,
    ValidationErrorResponse,
)
from app.services.document_service import DocumentService, get_document_service

router = APIRouter(prefix="/documents", tags=["documents"])

# The `id` column is Postgres int4; bounding the path param here rejects an
# out-of-range id as a clean 422 instead of letting it reach the driver,
# which raises asyncpg.DataError -> an unhandled 500.
DocumentId = Annotated[int, Path(ge=1, le=2_147_483_647)]

# Shared `responses={...}` fragments, composed per route below, so the
# OpenAPI docs match what app/exceptions/handlers.py actually returns
# instead of FastAPI's default HTTPValidationError shape.
_VALIDATION_RESPONSE = {
    422: {"model": ValidationErrorResponse, "description": "Validation failed"}
}
_NOT_FOUND_RESPONSE = {
    404: {"model": ErrorResponse, "description": "Document not found"}
}
_TITLE_CONFLICT_RESPONSE = {
    409: {"model": ErrorResponse, "description": "A document with this title already exists"}
}


@router.post(
    "",
    response_model=ApiResponse[DocumentOut],
    status_code=status.HTTP_201_CREATED,
    responses={**_TITLE_CONFLICT_RESPONSE, **_VALIDATION_RESPONSE},
)
async def create_document(
    payload: DocumentCreate,
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse[DocumentOut]:
    document = await service.create(payload)
    return ApiResponse.created(
        DocumentOut.model_validate(document), message="Document created"
    )


@router.get(
    "",
    response_model=ApiResponse[PaginatedData[DocumentOut]],
    responses={**_VALIDATION_RESPONSE},
)
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse[PaginatedData[DocumentOut]]:
    documents, total = await service.list(limit=limit, offset=offset)
    return ApiResponse.ok(
        PaginatedData[DocumentOut](
            items=[DocumentOut.model_validate(document) for document in documents],
            total=total,
            limit=limit,
            offset=offset,
        )
    )


@router.get(
    "/{document_id}",
    response_model=ApiResponse[DocumentOut],
    responses={**_NOT_FOUND_RESPONSE, **_VALIDATION_RESPONSE},
)
async def get_document(
    document_id: DocumentId,
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse[DocumentOut]:
    document = await service.get(document_id)
    return ApiResponse.ok(DocumentOut.model_validate(document))


@router.patch(
    "/{document_id}",
    response_model=ApiResponse[DocumentOut],
    responses={**_NOT_FOUND_RESPONSE, **_TITLE_CONFLICT_RESPONSE, **_VALIDATION_RESPONSE},
)
async def update_document(
    document_id: DocumentId,
    payload: DocumentUpdate,
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse[DocumentOut]:
    document = await service.update(document_id, payload)
    return ApiResponse.ok(
        DocumentOut.model_validate(document), message="Document updated"
    )


@router.delete(
    "/{document_id}",
    response_model=ApiResponse[None],
    responses={**_NOT_FOUND_RESPONSE, **_VALIDATION_RESPONSE},
)
async def delete_document(
    document_id: DocumentId,
    service: DocumentService = Depends(get_document_service),
) -> ApiResponse[None]:
    # 200 with data: null rather than 204 — a 204 carries no body, which
    # would make this the one endpoint that breaks the envelope rule.
    await service.delete(document_id)
    return ApiResponse(code=200, message="Document deleted", data=None)
