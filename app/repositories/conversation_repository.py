"""The conversations the server remembers, and their turns. Every read is
scoped to one user: another user's conversation is not found."""

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.contracts import Turn
from app.models import Conversation, Message

_TITLE_CHARS = 120


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, user_id: uuid.UUID) -> Conversation:
        conversation = Conversation(user_id=user_id)
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def get(self, conversation_id: uuid.UUID, user_id: uuid.UUID) -> Conversation | None:
        conversation = await self._session.get(Conversation, conversation_id)
        if conversation is None or conversation.user_id != user_id:
            return None
        return conversation

    async def recent_turns(self, conversation_id: uuid.UUID, limit: int) -> tuple[Turn, ...]:
        """The last `limit` turns, oldest first."""
        rows = await self._session.execute(
            select(Message.role, Message.text)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(limit)
        )
        return tuple(Turn(role=role, text=text) for role, text in reversed(rows.all()))

    async def add_exchange(
        self,
        conversation: Conversation,
        question: str,
        answer: str,
        citations: list[dict] | None = None,
    ) -> None:
        self._session.add_all(
            [
                Message(conversation_id=conversation.id, role="user", text=question),
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    text=answer,
                    citations=citations if citations is not None else [],
                ),
            ]
        )
        if conversation.title is None:
            conversation.title = question[:_TITLE_CHARS]
        # clock_timestamp, not now(): now() is the transaction's start, the
        # same for every exchange written in one transaction.
        conversation.updated_at = func.clock_timestamp()
        await self._session.flush()
        await self._session.refresh(conversation, ["updated_at"])

    async def list_for(
        self, user_id: uuid.UUID, limit: int, offset: int
    ) -> tuple[list[tuple[Conversation, int]], int]:
        """The user's conversations, most recently updated first, each with
        its number of messages; and how many there are in all."""
        counts = (
            select(Message.conversation_id, func.count().label("n"))
            .group_by(Message.conversation_id)
            .subquery()
        )
        rows = await self._session.execute(
            select(Conversation, func.coalesce(counts.c.n, 0))
            .outerjoin(counts, counts.c.conversation_id == Conversation.id)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc(), Conversation.id)
            .limit(limit)
            .offset(offset)
        )
        total = await self._session.scalar(
            select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
        )
        return [(conversation, count) for conversation, count in rows.all()], total or 0

    async def messages(self, conversation_id: uuid.UUID) -> list[Message]:
        rows = await self._session.execute(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
        )
        return list(rows.scalars())

    async def delete(self, conversation: Conversation) -> None:
        # Messages go with it (ON DELETE CASCADE); traces keep their row with
        # conversation_id set to null (ON DELETE SET NULL).
        await self._session.execute(delete(Conversation).where(Conversation.id == conversation.id))
