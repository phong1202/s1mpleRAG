from fastapi import APIRouter

from app.config import get_settings
from app.schemas.health import HealthData
from app.schemas.response import ApiResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=ApiResponse[HealthData])
async def health() -> ApiResponse[HealthData]:
    settings = get_settings()
    return ApiResponse.ok(
        HealthData(
            status="ok",
            app=settings.app_name,
            environment=settings.environment,
        )
    )
