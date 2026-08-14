from fastapi import FastAPI

from app.controllers import document_controller, health_controller

__all__ = ["register_controllers"]


def register_controllers(app: FastAPI) -> None:
    app.include_router(health_controller.router)
    app.include_router(document_controller.router)
