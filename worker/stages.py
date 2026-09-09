"""Five stages. In this task all of them are no-ops -- only the state
machine changes.

documents.stage holds the stage that JUST completed, and only advances
after its artifact is durably written. So the crash window always reduces
to "artifact written, pointer not yet advanced" -- i.e. rerunning exactly
one stage, harmlessly.
"""

import uuid
from datetime import UTC, datetime

from celery import chain

from app.exceptions import AppException, ErrorCode
from app.models.document import Document
from worker.celery_app import app
from worker.db import session_scope

STAGES = ["PARSING", "STRUCTURING", "ENRICHING", "EMBEDDING", "PERSISTING"]

MAX_ATTEMPTS = 3

# Permanent: the file will never parse, so retrying only burns time and
# hides the real reason.
PERMANENT = {ErrorCode.PDF_ENCRYPTED, ErrorCode.PDF_TOO_LARGE, ErrorCode.HASH_MISMATCH}


def _advance(
    document_id: str, status: str, stage: str | None = None, completed: bool = False
) -> None:
    with session_scope() as session:
        document = session.get(Document, uuid.UUID(document_id))
        document.status = status
        if stage:
            document.stage = stage
        if completed:
            document.completed_at = datetime.now(UTC)


def stage_failed(document_id: str, stage: str, exc: BaseException) -> bool:
    """Records a failure. Returns True once the document is DEAD_LETTER --
    a permanent error, or the attempt ceiling reached -- so that a task's
    except block can stop asking Celery to retry.

    That check has to exist somewhere: Celery's own max_retries differs per
    task (3 here, 5 there) and is not the same number as MAX_ATTEMPTS, which
    is shared across all five stages. A task that retried on max_retries
    alone would keep running its real body -- an OpenAI call, for
    enrich/embed -- for every retry Celery's own ceiling still permits on a
    document already given up on; and a retry that happened to succeed
    would silently overwrite DEAD_LETTER on the next stage's _advance(), as
    though the failure had never happened.

    DEAD_LETTER must always carry failed_stage and last_error -- a silent
    dead letter is one nobody can debug.
    """
    permanent = isinstance(exc, AppException) and exc.error in PERMANENT

    with session_scope() as session:
        document = session.get(Document, uuid.UUID(document_id))
        if not permanent:
            document.attempts += 1
        document.failed_stage = stage
        document.last_error = str(exc)
        dead = permanent or document.attempts >= MAX_ATTEMPTS
        document.status = "DEAD_LETTER" if dead else "RETRYING"
        return dead


@app.task(name="worker.stages.parse", bind=True, max_retries=3)
def parse(self, document_id: str) -> str:
    try:
        _advance(document_id, "PARSING")
        _advance(document_id, "PARSING", stage="PARSING")
        return document_id
    except AppException as exc:
        stage_failed(document_id, "PARSING", exc)
        raise  # never retry a permanent error
    except Exception as exc:
        if stage_failed(document_id, "PARSING", exc):
            raise  # already DEAD_LETTER; do not ask Celery to retry too
        raise self.retry(exc=exc, countdown=2**self.request.retries) from exc


@app.task(name="worker.stages.structure", bind=True, max_retries=5)
def structure(self, document_id: str) -> str:
    try:
        _advance(document_id, "STRUCTURING", stage="STRUCTURING")
        return document_id
    except AppException as exc:
        stage_failed(document_id, "STRUCTURING", exc)
        raise
    except Exception as exc:
        if stage_failed(document_id, "STRUCTURING", exc):
            raise
        raise self.retry(exc=exc, countdown=2**self.request.retries) from exc


@app.task(name="worker.stages.enrich", bind=True, max_retries=3)
def enrich(self, document_id: str) -> str:
    try:
        _advance(document_id, "ENRICHING", stage="ENRICHING")
        return document_id
    except AppException as exc:
        stage_failed(document_id, "ENRICHING", exc)
        raise
    except Exception as exc:
        if stage_failed(document_id, "ENRICHING", exc):
            raise
        raise self.retry(exc=exc, countdown=2**self.request.retries) from exc


@app.task(name="worker.stages.embed", bind=True, max_retries=3)
def embed(self, document_id: str) -> str:
    try:
        _advance(document_id, "EMBEDDING", stage="EMBEDDING")
        return document_id
    except AppException as exc:
        stage_failed(document_id, "EMBEDDING", exc)
        raise
    except Exception as exc:
        if stage_failed(document_id, "EMBEDDING", exc):
            raise
        raise self.retry(exc=exc, countdown=2**self.request.retries) from exc


@app.task(name="worker.stages.persist", bind=True, max_retries=5)
def persist(self, document_id: str) -> str:
    try:
        _advance(document_id, "COMPLETED", stage="PERSISTING", completed=True)
        return document_id
    except AppException as exc:
        stage_failed(document_id, "PERSISTING", exc)
        raise
    except Exception as exc:
        if stage_failed(document_id, "PERSISTING", exc):
            raise
        raise self.retry(exc=exc, countdown=2**self.request.retries) from exc


def launch(document_id: str) -> None:
    chain(parse.s(document_id), structure.s(), enrich.s(), embed.s(), persist.s()).apply_async()
