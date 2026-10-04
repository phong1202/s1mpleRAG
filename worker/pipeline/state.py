"""The stage state machine: which stages exist, what each leaves in staging,
when a failure is final, and how the documents row moves along.

documents.stage holds the stage that JUST completed, and only advances
after its artifact is durably written. So the crash window always reduces
to "artifact written, pointer not yet advanced" -- i.e. rerunning exactly
one stage, harmlessly.
"""

import uuid

from app.exceptions import AppException, ErrorCode
from worker.db import session_scope
from worker.repositories.documents import DocumentStateRepository

STAGES = ["PARSING", "STRUCTURING", "ENRICHING", "EMBEDDING", "PERSISTING"]

MAX_ATTEMPTS = 3

# request.retries also counts rate-limit deferrals in the stages that have
# them, so an uncapped 2**retries would make one transient error after a
# dozen deferrals wait over an hour.
MAX_BACKOFF_S = 300

# Permanent: the file will never parse, so retrying only burns time and
# hides the real reason.
PERMANENT = {
    ErrorCode.PDF_ENCRYPTED,
    ErrorCode.PDF_MALFORMED,
    ErrorCode.PDF_TOO_LARGE,
    ErrorCode.HASH_MISMATCH,
    ErrorCode.NO_EXTRACTABLE_TEXT,
    # Config-driven, not document-driven: a batch that costs more than the
    # whole rate limit bucket costs exactly as much on every retry.
    ErrorCode.RATE_LIMIT_UNSATISFIABLE,
    # A wrong key, an empty quota, a misspelled model: the same answer on
    # every retry.
    ErrorCode.LLM_PROVIDER_REJECTED,
}

# Each stage's staged artifacts, partials included: a resume that picked up
# a partial computed from an older input would mix two runs' results under
# the same ids.
ARTIFACTS = {
    "PARSING": ("parsed.json",),
    "STRUCTURING": ("chunks.json",),
    "ENRICHING": ("enriched.json", "enriched.partial.json"),
    "EMBEDDING": ("manifest.json", "embeddings.npy", "embeddings.partial.npy"),
}


def advance(document_id: str, status: str, stage: str | None = None) -> None:
    with session_scope() as session:
        DocumentStateRepository(session).advance(uuid.UUID(document_id), status, stage)


def invalidate_downstream(store, document_id: str, stage: str) -> None:
    """Called by a stage about to compute its artifact afresh. A checkpoint
    skip only proves an artifact exists, not that it was built from the
    current input: regenerate enriched.json and S4 would skip on its old
    embeddings.npy, leaving S5 to store the new text beside the vector of
    the old one. Deleted BEFORE this stage writes its own artifact -- the
    other way round, a crash between the two leaves the new artifact (so a
    retry skips) above the stale ones."""
    for later in STAGES[STAGES.index(stage) + 1 :]:
        for name in ARTIFACTS.get(later, ()):
            store.delete(f"staging/{document_id}/{name}")


def stage_failed(document_id: str, stage: str, exc: BaseException) -> bool:
    """Records a failure. Returns True once the document is DEAD_LETTER --
    a permanent error, or the attempt ceiling reached -- so that a task's
    except block can stop asking Celery to retry.

    That check has to exist somewhere: Celery's own max_retries differs per
    task and is not the same number as MAX_ATTEMPTS, which is shared across
    all five stages. A task that retried on max_retries alone would keep
    running its real body -- an OpenAI call, for enrich/embed -- for every
    retry Celery's own ceiling still permits on a document already given up
    on; and a retry that happened to succeed would silently overwrite
    DEAD_LETTER on the next stage's advance(), as though the failure had
    never happened.

    DEAD_LETTER must always carry failed_stage and last_error -- a silent
    dead letter is one nobody can debug.
    """
    permanent = isinstance(exc, AppException) and exc.error in PERMANENT
    with session_scope() as session:
        return DocumentStateRepository(session).record_failure(
            uuid.UUID(document_id), stage, str(exc), permanent, MAX_ATTEMPTS
        )
