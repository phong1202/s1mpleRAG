import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ChildChunk(Base):
    """The unit of retrieval. What gets embedded is `contextualized`, not
    `content`."""

    __tablename__ = "child_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("parent_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    contextualized: Mapped[str] = mapped_column(Text, nullable=False)
    # Citation accuracy lives HERE, not on the parent.
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Width fixed at 1536 by the migration, not read from settings: changing
    # it later is a new migration and a reindex, not an env var flip. See
    # test_the_embedding_column_width_matches_the_configured_dimensions.
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Inherited from the parent. It picks the text search config for `tsv`,
    # so it has to be set for the generated column to mean anything.
    language: Mapped[str | None] = mapped_column(Text, nullable=True)

    # `tsv` is deliberately NOT mapped here. The database owns it and Phase 2
    # reads it with raw SQL; mapping it would tempt SQLAlchemy to write to a
    # generated column, and Postgres refuses that.
