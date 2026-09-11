from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.response import ApiResponse
from app.services.query_service import QueryService
from app.utils.database import get_session
from shared.llm import AsyncLLMProvider, get_async_provider

router = APIRouter(tags=["query"])


@router.post("/query", response_model=ApiResponse[QueryResponse])
async def query(
    payload: QueryRequest,
    session: AsyncSession = Depends(get_session),
    provider: AsyncLLMProvider = Depends(get_async_provider),
) -> ApiResponse[QueryResponse]:
    return ApiResponse.ok(await QueryService(session, provider).ask(payload))
