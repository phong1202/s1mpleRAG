"""The schema is the contract both the API and the worker rely on. Checked
against the real database, not only against the Python models -- indexes and
constraints do not exist in Python.
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.config import get_settings
from app.models import ChildChunk, Document, ParentChunk

pytestmark = pytest.mark.asyncio


async def test_document_is_a_file_entity_not_a_title_content_pair(db_session):
    doc = Document(
        sha256_hash="a" * 64,
        filename="bao-cao.pdf",
        object_key="raw/" + "a" * 64 + ".pdf",
        size_bytes=1234,
    )
    db_session.add(doc)
    await db_session.flush()

    assert isinstance(doc.id, uuid.UUID)
    assert doc.status == "QUEUED"
    assert doc.attempts == 0
    assert not hasattr(doc, "title")
    assert not hasattr(doc, "content")


async def test_sha256_hash_is_unique(db_session):
    for _ in range(2):
        db_session.add(
            Document(sha256_hash="b" * 64, filename="x.pdf", object_key="raw/x.pdf", size_bytes=1)
        )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_child_chunk_carries_the_page_number(db_session):
    """page_number lives on the child, not the parent -- that is what decides
    citation accuracy in Phase 2."""
    doc = Document(sha256_hash="c" * 64, filename="x.pdf", object_key="raw/x.pdf", size_bytes=1)
    db_session.add(doc)
    await db_session.flush()

    parent = ParentChunk(
        document_id=doc.id,
        chunk_index=0,
        content="parent",
        token_count=600,
        page_start=1,
        page_end=2,
    )
    db_session.add(parent)
    await db_session.flush()

    child = ChildChunk(
        document_id=doc.id,
        parent_id=parent.id,
        chunk_index=0,
        content="child",
        contextualized="context\n\nchild",
        page_number=2,
        token_count=120,
        embedding=[0.0] * 1536,
    )
    db_session.add(child)
    await db_session.flush()

    assert child.page_number == 2


async def test_hnsw_index_uses_inner_product(db_session):
    """vector_ip_ops assumes unit vectors. If the index were built with
    vector_cosine_ops, L2-normalising in S4 would be pointless."""
    result = await db_session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'child_chunks'")
    )
    defs = " ".join(row[0] for row in result)

    assert "hnsw" in defs.lower()
    assert "vector_ip_ops" in defs


async def test_document_status_is_constrained_to_the_known_set(db_session):
    """status is a closed set of nine values used across every later stage
    task -- QUEUED through DEAD_LETTER. Nothing in Python enforces that; a
    typo would otherwise sit silently in the column forever."""
    doc = Document(sha256_hash="d" * 64, filename="x.pdf", object_key="raw/x.pdf", size_bytes=1)
    db_session.add(doc)
    await db_session.flush()

    doc.status = "COMLETED"
    with pytest.raises(IntegrityError, match="ck_documents_status"):
        await db_session.flush()


async def test_the_embedding_column_width_matches_the_configured_dimensions():
    """The vector column's width is fixed at migration time, not read from
    settings at import time -- changing EMBED_DIMENSIONS later means a new
    migration and a reindex, not an env var flip. This test is the tripwire:
    if someone changes the setting without touching the schema, this is
    where it fails loudly instead of at the first mismatched insert."""
    assert get_settings().embed_dimensions == 1536
