import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Document(Base):
    """An uploaded PDF file. Text lives in chunks, not here."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Unique here is what closes the dedup race: INSERT ... ON CONFLICT
    # DO NOTHING RETURNING id, publish only when a row comes back.
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # A closed set of nine values (see the ck_documents_status check
    # constraint in the migration): QUEUED, PARSING, STRUCTURING, ENRICHING,
    # EMBEDDING, PERSISTING, COMPLETED, RETRYING, DEAD_LETTER. Kept as a
    # plain string type here -- the constraint is what actually enforces it,
    # since a Python type hint cannot stop a raw SQL UPDATE or a typo from a
    # future task.
    status: Mapped[str] = mapped_column(Text, nullable=False, default="QUEUED")
    # Resume pointer: the stage that JUST completed, not the one running.
    stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
