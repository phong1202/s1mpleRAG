import uuid

from sqlalchemy import ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ParentChunk(Base):
    """The unit of context used to generate an answer. Never embedded."""

    __tablename__ = "parent_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Deterministic from the parse output -- this natural key is what makes
    # S5 idempotent.
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    # "Chapter 2 > Article 19 > Clause 2". S2 already has this in hand while
    # splitting on headings; rebuilding it later means re-running Docling
    # over the whole corpus.
    heading_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Detected here, not on the child: 150 tokens is too little to classify
    # reliably, and Vietnamese prose mixed with English terms is exactly what
    # makes that misclassify.
    language: Mapped[str | None] = mapped_column(Text, nullable=True)
