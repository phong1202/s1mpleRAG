from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """The single response shape for this API — success and error alike.

    `code` always equals the HTTP status code of the response.
    """

    code: int = 200
    message: str = "Success"
    data: T | None = None

    @classmethod
    def ok(cls, data: T, message: str = "Success") -> "ApiResponse[T]":
        return cls(code=200, message=message, data=data)

    @classmethod
    def created(cls, data: T, message: str = "Created") -> "ApiResponse[T]":
        return cls(code=201, message=message, data=data)


class PaginatedData(BaseModel, Generic[T]):
    """Pagination travels inside `data`, keeping the envelope uniform."""

    items: list[T]
    total: int
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    """Documents the envelope shape for error responses whose `data` is
    always null (404, 409, ...). Not used at runtime — the exception
    handlers build the actual JSON directly — this exists purely so
    OpenAPI/generated clients see the real shape instead of FastAPI's
    default HTTPValidationError."""

    code: int
    message: str
    data: None = None


class ValidationErrorItem(BaseModel):
    field: str
    message: str


class ValidationErrorData(BaseModel):
    errors: list[ValidationErrorItem]


class ValidationErrorResponse(BaseModel):
    """Documents the 422 envelope shape actually returned by
    handle_validation_error in app/exceptions/handlers.py."""

    code: int
    message: str
    data: ValidationErrorData
