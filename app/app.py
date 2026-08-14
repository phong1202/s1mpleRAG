from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.controllers import register_controllers
from app.exceptions.handlers import register_exception_handlers
from app.middleware import register_middleware
from app.utils.database import engine
from app.utils.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Nothing to do on startup: the engine connects lazily on first use.
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )

    register_middleware(app)
    register_exception_handlers(app)
    register_controllers(app)

    return app


app = create_app()
