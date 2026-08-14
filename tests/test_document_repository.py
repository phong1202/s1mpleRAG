import pytest

from app.repositories.document_repository import DocumentRepository


@pytest.mark.asyncio
async def test_create_assigns_an_id_and_leaves_embedding_null(db_session):
    repository = DocumentRepository(db_session)

    document = await repository.create(title="Intro", content="hello")

    assert document.id is not None
    assert document.title == "Intro"
    assert document.embedding is None
    assert document.created_at is not None


@pytest.mark.asyncio
async def test_get_by_id_returns_none_when_absent(db_session):
    repository = DocumentRepository(db_session)

    assert await repository.get_by_id(999_999) is None


@pytest.mark.asyncio
async def test_get_by_id_round_trips(db_session):
    repository = DocumentRepository(db_session)
    created = await repository.create(title="Intro", content="hello")

    found = await repository.get_by_id(created.id)

    assert found is not None
    assert found.id == created.id


@pytest.mark.asyncio
async def test_get_by_title_finds_an_existing_document(db_session):
    repository = DocumentRepository(db_session)
    await repository.create(title="Unique Title", content="x")

    assert await repository.get_by_title("Unique Title") is not None
    assert await repository.get_by_title("Absent Title") is None


@pytest.mark.asyncio
async def test_list_paginates_and_reports_the_total(db_session):
    repository = DocumentRepository(db_session)
    for index in range(5):
        await repository.create(title=f"Doc {index}", content="x")

    items, total = await repository.list(limit=2, offset=1)

    assert total == 5
    assert len(items) == 2
    assert items[0].title == "Doc 1"


@pytest.mark.asyncio
async def test_update_changes_only_the_supplied_fields(db_session):
    repository = DocumentRepository(db_session)
    document = await repository.create(title="Before", content="body")

    updated = await repository.update(document, title="After")

    assert updated.title == "After"
    assert updated.content == "body"


@pytest.mark.asyncio
async def test_update_rejects_a_key_that_is_not_a_column(db_session):
    repository = DocumentRepository(db_session)
    document = await repository.create(title="Before", content="body")

    with pytest.raises(ValueError):
        await repository.update(document, not_a_real_column="x", title="After")


@pytest.mark.asyncio
async def test_update_still_accepts_a_valid_key_after_a_rejection(db_session):
    repository = DocumentRepository(db_session)
    document = await repository.create(title="Before", content="body")

    with pytest.raises(ValueError):
        await repository.update(document, not_a_real_column="x")

    # The guard on the previous call doesn't leave the repository broken.
    updated = await repository.update(document, title="After")
    assert updated.title == "After"


@pytest.mark.asyncio
async def test_delete_removes_the_row(db_session):
    repository = DocumentRepository(db_session)
    document = await repository.create(title="Doomed", content="x")
    document_id = document.id

    await repository.delete(document)

    assert await repository.get_by_id(document_id) is None
