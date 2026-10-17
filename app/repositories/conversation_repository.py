"""The conversations the server remembers, and their turns."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.contracts import Turn
from app.models import Conversation, Message


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self) -> Conversation:
        conversation = Conversation()
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def get(self, conversation_id: uuid.UUID) -> Conversation | None:
        return await self._session.get(Conversation, conversation_id)

    async def recent_turns(self, conversation_id: uuid.UUID, limit: int) -> tuple[Turn, ...]:
        """The last `limit` turns, oldest first."""
        rows = await self._session.execute(
            select(Message.role, Message.text)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(limit)
        )
        return tuple(Turn(role=role, text=text) for role, text in reversed(rows.all()))

    async def add_exchange(self, conversation_id: uuid.UUID, question: str, answer: str) -> None:
        self._session.add_all(
            [
                Message(conversation_id=conversation_id, role="user", text=question),
                Message(conversation_id=conversation_id, role="assistant", text=answer),
            ]
        )
        await self._session.flush()
