import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.exceptions.base import AppException
from app.exceptions.error_codes import ErrorCode
from app.middleware.request_id import REQUEST_ID_HEADER, get_request_id

logger = logging.getLogger(__name__)

# Location segments FastAPI/Pydantic prefix onto `loc` to say where a
# validation error came from. Only the leading one is stripped, so a nested
# body field keeps its dotted path (`nested.name`) and a body field
# legitimately named e.g. "query" is left alone.
_LOCATION_PREFIXES = {"body", "query", "path", "header", "cookie"}


def _field_name(loc: tuple) -> str:
    parts = list(loc)
    if parts and parts[0] in _LOCATION_PREFIXES:
        parts = parts[1:]
    return ".".join(str(part) for part in parts)


def _envelope(status: int, message: str, data=None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"code": status, "message": message, "data": data},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers, NOT middleware.

    Starlette converts endpoint exceptions to responses in ExceptionMiddleware,
    which sits *inside* user middleware — a try/except in custom middleware
    would never observe them. The `Exception` handler below is installed into
    ServerErrorMiddleware at the outside of the stack, so it also catches
    failures raised inside middleware. Together they give full coverage.
    """

    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
        logger.warning("%s: %s", exc.error.name, exc.message)
        return _envelope(exc.error.status, exc.message, exc.data)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = [
            {
                "field": _field_name(error["loc"]),
                "message": error["msg"],
            }
            for error in exc.errors()
        ]
        return _envelope(
            ErrorCode.VALIDATION_FAILED.status,
            ErrorCode.VALIDATION_FAILED.message,
            {"errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _envelope(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # This handler runs in ServerErrorMiddleware, outside RequestIDMiddleware,
        # so the request id is read from request.state (see request_id.py) rather
        # than the ContextVar, which RequestIDMiddleware has already reset by the
        # time an unhandled exception reaches here. The same id is used for both
        # the log line and the response header so the two can be correlated.
        request_id = getattr(request.state, "request_id", None) or get_request_id()

        # Full traceback to the log; a generic message to the client. `extra`
        # sets record.request_id directly — RequestIDFilter defers to it
        # instead of overwriting it with the (by-now-reset) ContextVar value.
        logger.exception(
            "Unhandled exception on %s %s",
            request.method,
            request.url.path,
            extra={"request_id": request_id},
        )
        response = _envelope(
            ErrorCode.INTERNAL_ERROR.status,
            ErrorCode.INTERNAL_ERROR.message,
        )
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
