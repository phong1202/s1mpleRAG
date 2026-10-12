"""Seed data shared by the retrieval tests.

It lives here rather than in one of the test modules because the search, the
parent expansion and the retriever all need the same corpus, and a test module
importing another test module's private helper breaks the moment either is
renamed.
"""

import uuid

from app.models import ChildChunk, Document, ParentChunk


def unit_vector(index: int, dimensions: int = 1536) -> list[float]:
    vector = [0.0] * dimensions
    vector[index] = 1.0
    return vector


async def seed_chunks(session, children_per_parent: int = 3, parents: int = 4):
    """`parents` parents, each with `children_per_parent` children, numbered
    consecutively so child `i` is the one and only match for unit_vector(i)."""
    document = Document(
        sha256_hash=uuid.uuid4().hex * 2,
        filename="decree.pdf",
        object_key="raw/x.pdf",
        size_bytes=1,
    )
    session.add(document)
    await session.flush()

    index = 0
    for p in range(parents):
        parent = ParentChunk(
            document_id=document.id,
            chunk_index=p,
            content=f"parent {p}",
            heading_path=f"Chuong I > Dieu {p}",
            token_count=600,
            page_start=p + 1,
            page_end=p + 1,
        )
        session.add(parent)
        await session.flush()
        for _ in range(children_per_parent):
            session.add(
                ChildChunk(
                    document_id=document.id,
                    parent_id=parent.id,
                    chunk_index=index,
                    content=f"child {index}",
                    contextualized=f"ctx {index}",
                    page_number=p + 1,
                    token_count=150,
                    embedding=unit_vector(index),
                    category="LEGAL",
                )
            )
            index += 1
    await session.flush()
    return document


async def seed_texts(session, texts: list[str], language: str | None = None):
    """One document, one parent, one child per text, each on its own page.

    For the keyword tests: the stub's vectors carry no meaning, the words do."""
    document = Document(
        sha256_hash=uuid.uuid4().hex * 2,
        filename="notes.pdf",
        object_key=f"raw/{uuid.uuid4().hex}.pdf",
        size_bytes=1,
    )
    session.add(document)
    await session.flush()
    parent = ParentChunk(
        document_id=document.id,
        chunk_index=0,
        content="\n".join(texts),
        token_count=600,
        page_start=1,
        page_end=len(texts),
        language=language,
    )
    session.add(parent)
    await session.flush()
    for index, content in enumerate(texts):
        session.add(
            ChildChunk(
                document_id=document.id,
                parent_id=parent.id,
                chunk_index=index,
                content=content,
                contextualized=content,
                page_number=index + 1,
                token_count=150,
                embedding=unit_vector(index),
                category="LEGAL",
                language=language,
            )
        )
    await session.flush()
    return document
