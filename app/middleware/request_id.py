import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

REQUEST_ID_HEADER = "X-Request-ID"


def get_request_id() -> str:
    """The current request's id, or "-" outside a request."""
    return request_id_ctx.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns each request an id, exposes it to the logging filter through a
    ContextVar, and echoes it back so a client can quote it in a bug report."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        # Also stash on request.state (backed by the ASGI scope, shared by
        # every layer including ServerErrorMiddleware's handler) because the
        # ContextVar below is reset, via the `finally`, while an exception
        # from `call_next` is still unwinding through this frame — before it
        # ever reaches a handler installed outside this middleware.
        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            request_id_ctx.reset(token)
