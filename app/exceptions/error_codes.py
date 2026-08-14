from enum import Enum


class ErrorCode(Enum):
    """Known failure conditions: (http_status, default_message).

    Status and message live together so a status can never drift away from
    the message that describes it, and adding an error is one line.
    """

    DOCUMENT_NOT_FOUND = (404, "Document not found")
    DOCUMENT_TITLE_EXISTS = (409, "A document with this title already exists")
    VALIDATION_FAILED = (422, "Validation failed")
    INTERNAL_ERROR = (500, "Internal server error")
    DATABASE_ERROR = (500, "Database operation failed")

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
