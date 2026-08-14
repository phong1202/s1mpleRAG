from typing import Any

from app.exceptions.error_codes import ErrorCode


class AppException(Exception):
    """Every expected application failure. The ErrorCode distinguishes the
    case, so no subclass hierarchy is needed."""

    def __init__(
        self,
        error: ErrorCode,
        message: str | None = None,
        data: Any = None,
    ) -> None:
        self.error = error
        self.message = message or error.message
        self.data = data
        super().__init__(self.message)
