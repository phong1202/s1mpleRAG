from fastapi import FastAPI

from app.controllers import health_controller, ingestion_controller

__all__ = ["register_controllers"]


def register_controllers(app: FastAPI) -> None:
    app.include_router(health_controller.router)
    app.include_router(ingestion_controller.router)
