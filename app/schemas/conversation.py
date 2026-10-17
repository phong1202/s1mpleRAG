import uuid
from datetime import datetime

from pydantic import BaseModel

from app.schemas.query import CitationOut


class UserOut(BaseModel):
    id: uuid.UUID
    username: str
    display_name: str
    is_admin: bool


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int


class MessageOut(BaseModel):
    id: int
    role: str
    text: str
    # As POST /query returned them; null for questions and for answers from
    # before citations were kept.
    citations: list[CitationOut] | None
    created_at: datetime


class ConversationDetail(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]
