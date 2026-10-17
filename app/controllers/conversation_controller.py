"""Who is asking, and their conversations. Everything here is scoped to the
current user: another user's conversation is a 404, as if it did not exist."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.exceptions.base import AppException
from app.exceptions.error_codes import ErrorCode
from app.models import Conversation, User
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.conversation import ConversationDetail, ConversationSummary, MessageOut, UserOut
from app.schemas.response import ApiResponse, ErrorResponse, PaginatedData
from app.utils.database import get_session

router = APIRouter(tags=["conversations"])


@router.get("/me", response_model=ApiResponse[UserOut])
async def me(user: User = Depends(get_current_user)) -> ApiResponse[UserOut]:
    return ApiResponse.ok(
        UserOut(
            id=user.id,
            username=user.username,
            display_name=user.display_name,
            is_admin=user.is_admin,
        )
    )


ConversationId = Annotated[uuid.UUID, Path()]
_NOT_FOUND = {404: {"model": ErrorResponse, "description": "Conversation not found"}}


async def _owned(
    conversation_id: uuid.UUID, user: User, repository: ConversationRepository
) -> Conversation:
    conversation = await repository.get(conversation_id, user.id)
    if conversation is None:
        raise AppException(ErrorCode.CONVERSATION_NOT_FOUND)
    return conversation


@router.get("/conversations", response_model=ApiResponse[PaginatedData[ConversationSummary]])
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[PaginatedData[ConversationSummary]]:
    rows, total = await ConversationRepository(session).list_for(user.id, limit, offset)
    items = [
        ConversationSummary(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=count,
        )
        for c, count in rows
    ]
    return ApiResponse.ok(
        PaginatedData[ConversationSummary](items=items, total=total, limit=limit, offset=offset)
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ApiResponse[ConversationDetail],
    responses=_NOT_FOUND,
)
async def get_conversation(
    conversation_id: ConversationId,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[ConversationDetail]:
    repository = ConversationRepository(session)
    conversation = await _owned(conversation_id, user, repository)
    messages = [
        MessageOut(
            id=m.id, role=m.role, text=m.text, citations=m.citations, created_at=m.created_at
        )
        for m in await repository.messages(conversation.id)
    ]
    return ApiResponse.ok(
        ConversationDetail(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=messages,
        )
    )


@router.delete(
    "/conversations/{conversation_id}", response_model=ApiResponse[None], responses=_NOT_FOUND
)
async def delete_conversation(
    conversation_id: ConversationId,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[None]:
    repository = ConversationRepository(session)
    await repository.delete(await _owned(conversation_id, user, repository))
    return ApiResponse(code=200, message="Conversation deleted", data=None)
