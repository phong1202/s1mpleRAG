"""Document state as the worker changes it.

Sync, unlike app/repositories: a Celery task runs no event loop. Takes the
Session from its caller, who owns the transaction -- S5 has to write chunks
and close the document out in one.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.models import Document


class DocumentGone(Exception):
    """The row was deleted -- typically by the user, while its chain was
    still queued. Not a failure: there is nothing left to process."""


class DocumentStateRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, document_id: uuid.UUID) -> Document:
        document = self.session.get(Document, document_id)
        if document is None:
            raise DocumentGone(str(document_id))
        return document

    def advance(self, document_id: uuid.UUID, status: str, stage: str | None = None) -> None:
        document = self.get(document_id)
        document.status = status
        if stage:
            document.stage = stage

    def record_failure(
        self,
        document_id: uuid.UUID,
        stage: str,
        error: str,
        permanent: bool,
        max_attempts: int,
    ) -> bool:
        """Returns True if the document is now DEAD_LETTER. A permanent
        error is not counted as an attempt: it dead-letters at once."""
        document = self.get(document_id)
        if not permanent:
            document.attempts += 1
        document.failed_stage = stage
        document.last_error = error
        dead = permanent or document.attempts >= max_attempts
        document.status = "DEAD_LETTER" if dead else "RETRYING"
        return dead

    def set_page_count(self, document_id: uuid.UUID, page_count: int) -> None:
        self.get(document_id).page_count = page_count

    def set_parse_result(self, document_id: uuid.UUID, page_count: int, title: str | None) -> None:
        document = self.get(document_id)
        document.page_count = page_count
        # Filename is the last-resort fallback and it lives here, not in the
        # parser: raw/{sha256}.pdf is the only name the parser ever sees.
        document.title = title or document.filename

    def complete(self, document_id: uuid.UUID, language: str | None) -> None:
        document = self.get(document_id)
        document.status = "COMPLETED"
        document.stage = "PERSISTING"
        document.completed_at = datetime.now(UTC)
        document.language = language
