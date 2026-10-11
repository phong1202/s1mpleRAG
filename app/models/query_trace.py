import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class QueryTrace(Base):
    """One row per answered question. This is what R2 through R8 argue from:
    without it, a regression can only be described, never located."""

    __tablename__ = "query_traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # Written from R4 onward; null before the rewriter exists.
    standalone: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Written from R5 onward. The distribution of this column is the only way
    # to catch a router that has quietly become a constant.
    strategy: Mapped[str | None] = mapped_column(Text, nullable=True)
    iterations: Mapped[int] = mapped_column(Integer, nullable=False)
    total_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    citation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    nodes: Mapped[list] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
