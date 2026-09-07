from enum import Enum


class ErrorCode(Enum):
    """Known failure conditions: (http_status, default_message).

    Status and message live together so a status can never drift away from
    the message that describes it, and adding an error is one line.
    """

    # --- General ---
    VALIDATION_FAILED = (422, "Validation failed")
    INTERNAL_ERROR = (500, "Internal server error")
    DATABASE_ERROR = (500, "Database operation failed")

    # --- Document ---
    DOCUMENT_NOT_FOUND = (404, "Document not found")
    DOCUMENT_ALREADY_INGESTED = (409, "Document already ingested")
    PDF_ENCRYPTED = (422, "PDF is encrypted and cannot be parsed")
    PDF_TOO_LARGE = (413, "PDF exceeds the size or page limit")
    HASH_MISMATCH = (400, "Uploaded object does not match the supplied hash")

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
