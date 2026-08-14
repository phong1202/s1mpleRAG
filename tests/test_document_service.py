import pytest

from app.exceptions import AppException, ErrorCode
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentCreate, DocumentUpdate
from app.services.document_service import DocumentService


@pytest.fixture
def service(db_session):
    return DocumentService(DocumentRepository(db_session))


@pytest.mark.asyncio
async def test_create_returns_the_stored_document(service):
    document = await service.create(DocumentCreate(title="Intro", content="hello"))

    assert document.id is not None
    assert document.title == "Intro"


@pytest.mark.asyncio
async def test_create_rejects_a_duplicate_title(service):
    await service.create(DocumentCreate(title="Same", content="a"))

    with pytest.raises(AppException) as exc_info:
        await service.create(DocumentCreate(title="Same", content="b"))

    assert exc_info.value.error is ErrorCode.DOCUMENT_TITLE_EXISTS
    assert exc_info.value.error.status == 409


@pytest.mark.asyncio
async def test_get_raises_not_found_for_a_missing_id(service):
    with pytest.raises(AppException) as exc_info:
        await service.get(999_999)

    assert exc_info.value.error is ErrorCode.DOCUMENT_NOT_FOUND
    assert "999999" in exc_info.value.message


@pytest.mark.asyncio
async def test_update_applies_only_supplied_fields(service):
    created = await service.create(DocumentCreate(title="Before", content="body"))

    updated = await service.update(created.id, DocumentUpdate(title="After"))

    assert updated.title == "After"
    assert updated.content == "body"


@pytest.mark.asyncio
async def test_update_raises_not_found_for_a_missing_id(service):
    with pytest.raises(AppException) as exc_info:
        await service.update(999_999, DocumentUpdate(title="x"))

    assert exc_info.value.error is ErrorCode.DOCUMENT_NOT_FOUND


@pytest.mark.asyncio
async def test_update_rejects_a_title_taken_by_another_document(service):
    await service.create(DocumentCreate(title="Taken", content="a"))
    other = await service.create(DocumentCreate(title="Free", content="b"))

    with pytest.raises(AppException) as exc_info:
        await service.update(other.id, DocumentUpdate(title="Taken"))

    assert exc_info.value.error is ErrorCode.DOCUMENT_TITLE_EXISTS


@pytest.mark.asyncio
async def test_update_allows_setting_a_title_to_its_current_value(service):
    created = await service.create(DocumentCreate(title="Stable", content="a"))

    updated = await service.update(created.id, DocumentUpdate(title="Stable", content="b"))

    assert updated.content == "b"


@pytest.mark.asyncio
async def test_delete_raises_not_found_for_a_missing_id(service):
    with pytest.raises(AppException) as exc_info:
        await service.delete(999_999)

    assert exc_info.value.error is ErrorCode.DOCUMENT_NOT_FOUND
