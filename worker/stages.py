"""Five stages: parse, structure, enrich, embed, persist.

documents.stage holds the stage that JUST completed, and only advances
after its artifact is durably written. So the crash window always reduces
to "artifact written, pointer not yet advanced" -- i.e. rerunning exactly
one stage, harmlessly.
"""

import uuid

from celery import chain

from app.exceptions import AppException, ErrorCode
from app.models.document import Document

# Top-level, unlike each stage's own imports: an except clause naming it is
# evaluated whenever an exception reaches it, so it must already be bound.
from shared.rate_limiter import RateLimited
from worker.celery_app import app
from worker.db import session_scope

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
}


# Each stage's staged artifacts, partials included: a resume that picked up
# a partial computed from an older input would mix two runs' results under
# the same ids.
_ARTIFACTS = {
    "PARSING": ("parsed.json",),
    "STRUCTURING": ("chunks.json",),
    "ENRICHING": ("enriched.json", "enriched.partial.json"),
    "EMBEDDING": ("manifest.json", "embeddings.npy", "embeddings.partial.npy"),
}


def _advance(document_id: str, status: str, stage: str | None = None) -> None:
    with session_scope() as session:
        document = session.get(Document, uuid.UUID(document_id))
        document.status = status
        if stage:
            document.stage = stage


def _invalidate_downstream(store, document_id: str, stage: str) -> None:
    """Called by a stage about to compute its artifact afresh. A checkpoint
    skip only proves an artifact exists, not that it was built from the
    current input: regenerate enriched.json and S4 would skip on its old
    embeddings.npy, leaving S5 to store the new text beside the vector of
    the old one. Deleted BEFORE this stage writes its own artifact -- the
    other way round, a crash between the two leaves the new artifact (so a
    retry skips) above the stale ones."""
    for later in STAGES[STAGES.index(stage) + 1 :]:
        for name in _ARTIFACTS.get(later, ()):
            store.delete(f"staging/{document_id}/{name}")


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
        # Imports live inside the try, not above it: an import failure here
        # is exactly as real a stage failure as anything else in the body,
        # and outside the try it would bypass stage_failed() entirely --
        # observed directly while wiring this in, where a document stuck at
        # QUEUED forever with attempts=0 and no last_error, because the
        # exception never reached either except clause.
        from app.config import get_settings
        from shared.storage import get_store
        from worker.parsing import parse_document

        store = get_store()
        key = f"staging/{document_id}/parsed.json"
        if store.exists(key):  # checkpoint skip
            _advance(document_id, "PARSING", stage="PARSING")
            return document_id

        _advance(document_id, "PARSING")
        _invalidate_downstream(store, document_id, "PARSING")
        with session_scope() as session:
            document = session.get(Document, uuid.UUID(document_id))
            object_key = document.object_key

        result = parse_document(object_key, store, get_settings().docling_url)
        store.put_json(key, result)

        with session_scope() as session:
            document = session.get(Document, uuid.UUID(document_id))
            document.page_count = result["page_count"]
            # Filename is the last-resort fallback and it lives here, not
            # in the parser: raw/{sha256}.pdf is the only name the parser
            # ever sees.
            document.title = result["title"] or document.filename

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
        from shared.storage import get_store
        from worker.chunking import chunk_document

        store = get_store()
        key = f"staging/{document_id}/chunks.json"
        if store.exists(key):  # checkpoint skip
            _advance(document_id, "STRUCTURING", stage="STRUCTURING")
            return document_id

        _advance(document_id, "STRUCTURING")
        _invalidate_downstream(store, document_id, "STRUCTURING")
        parsed = store.get_json(f"staging/{document_id}/parsed.json")
        chunks = chunk_document(parsed)
        # Checked here, after chunking, rather than on S1's raw text: a page
        # of a few stray characters is not empty, yet chunks to nothing.
        if not chunks["children"]:
            raise AppException(
                ErrorCode.NO_EXTRACTABLE_TEXT,
                f"No extractable text: no chunk survived from {parsed['page_count']} "
                "page(s) -- blank, or an image OCR could read nothing from",
            )
        store.put_json(key, chunks)

        _advance(document_id, "STRUCTURING", stage="STRUCTURING")
        return document_id
    except AppException as exc:
        stage_failed(document_id, "STRUCTURING", exc)
        raise
    except Exception as exc:
        if stage_failed(document_id, "STRUCTURING", exc):
            raise
        raise self.retry(exc=exc, countdown=2**self.request.retries) from exc


# max_retries=None: Celery's retry budget would otherwise be spent by
# rate-limit deferrals, which are not failures. MAX_ATTEMPTS, enforced by
# stage_failed(), stays the ceiling for real ones.
@app.task(name="worker.stages.enrich", bind=True, max_retries=None)
def enrich(self, document_id: str) -> str:
    try:
        from app.config import get_settings
        from shared.llm import get_provider
        from shared.storage import get_store
        from worker.enrichment import enrich_chunks

        store = get_store()
        key = f"staging/{document_id}/enriched.json"
        if store.exists(key):  # checkpoint skip
            _advance(document_id, "ENRICHING", stage="ENRICHING")
            return document_id

        _advance(document_id, "ENRICHING")
        _invalidate_downstream(store, document_id, "ENRICHING")
        chunks = store.get_json(f"staging/{document_id}/chunks.json")
        # chunk_index plays the "id" role in the provider contract -- it is
        # deterministic, so a rerun asks about exactly the same id set.
        children = [{**c, "id": c["chunk_index"]} for c in chunks["children"]]

        # Left behind on success for the staging bucket's 7-day expiry:
        # deleting it here would add a failure point after the real
        # checkpoint is already written.
        partial_key = f"staging/{document_id}/enriched.partial.json"
        done = store.get_json(partial_key)["chunks"] if store.exists(partial_key) else []

        try:
            enriched = enrich_chunks(
                children,
                provider=get_provider(),
                batch_size=get_settings().enrich_batch_size,
                done=done,
            )
        except RateLimited as exc:
            # Saved here, inside the outer try, and not in its RateLimited
            # handler: a storage failure while saving must still reach
            # stage_failed() like any other, not escape from a handler.
            store.put_json(partial_key, {"chunks": exc.partial})
            raise
        store.put_json(key, {"chunks": enriched})

        _advance(document_id, "ENRICHING", stage="ENRICHING")
        return document_id
    except AppException as exc:
        stage_failed(document_id, "ENRICHING", exc)
        raise
    except RateLimited as exc:
        # Must come before `except Exception`. Not a failure: nothing is
        # recorded, and this is the one and only retry published for it.
        raise self.retry(exc=exc, countdown=exc.countdown) from exc
    except Exception as exc:
        if stage_failed(document_id, "ENRICHING", exc):
            raise
        raise self.retry(exc=exc, countdown=min(2**self.request.retries, MAX_BACKOFF_S)) from exc


# max_retries=None for the same reason as enrich.
@app.task(name="worker.stages.embed", bind=True, max_retries=None)
def embed(self, document_id: str) -> str:
    try:
        from app.config import get_settings
        from shared.llm import get_provider
        from shared.storage import get_store
        from worker.embedding import contextualize, embed_chunks, vectors_from_npy, vectors_to_npy

        store = get_store()
        key = f"staging/{document_id}/embeddings.npy"
        if store.exists(key):  # checkpoint skip
            _advance(document_id, "EMBEDDING", stage="EMBEDDING")
            return document_id

        _advance(document_id, "EMBEDDING")
        _invalidate_downstream(store, document_id, "EMBEDDING")
        chunks = store.get_json(f"staging/{document_id}/chunks.json")
        enriched = {
            e["id"]: e for e in store.get_json(f"staging/{document_id}/enriched.json")["chunks"]
        }

        texts, manifest = [], []
        for child in chunks["children"]:
            # Indexed, not .get() with an empty fallback: S3 gives every
            # child an entry, even one whose context failed, so a missing
            # one means chunks.json and enriched.json came from different
            # runs -- worth failing on, not embedding past in silence.
            context = enriched[child["chunk_index"]]["context"]
            # What gets embedded is the contextualized text, NOT the raw content.
            texts.append(contextualize(context, child["content"]))
            manifest.append(child["chunk_index"])

        partial_key = f"staging/{document_id}/embeddings.partial.npy"
        done = vectors_from_npy(store.get(partial_key)) if store.exists(partial_key) else []
        try:
            vectors = embed_chunks(
                texts,
                provider=get_provider(),
                batch_size=get_settings().embed_batch_size,
                done=done,
            )
        except RateLimited as exc:
            # Inside the outer try for the same reason as in enrich.
            store.put(partial_key, vectors_to_npy(exc.partial))
            raise

        # The manifest first, the file the skip above checks last: written
        # the other way round, a crash between the two leaves a run every
        # retry skips, and S5 without a manifest for good.
        store.put_json(f"staging/{document_id}/manifest.json", {"chunk_index_by_row": manifest})
        store.put(key, vectors_to_npy(vectors))

        _advance(document_id, "EMBEDDING", stage="EMBEDDING")
        return document_id
    except AppException as exc:
        stage_failed(document_id, "EMBEDDING", exc)
        raise
    except RateLimited as exc:
        raise self.retry(exc=exc, countdown=exc.countdown) from exc
    except Exception as exc:
        if stage_failed(document_id, "EMBEDDING", exc):
            raise
        raise self.retry(exc=exc, countdown=min(2**self.request.retries, MAX_BACKOFF_S)) from exc


# No checkpoint skip, unlike the other four: the upserts are idempotent in
# themselves, so a rerun is boring rather than destructive.
@app.task(name="worker.stages.persist", bind=True, max_retries=5)
def persist(self, document_id: str) -> str:
    try:
        from shared.storage import get_store
        from worker.embedding import vectors_from_npy
        from worker.persistence import persist_document

        store = get_store()
        _advance(document_id, "PERSISTING")

        prefix = f"staging/{document_id}"
        chunks = store.get_json(f"{prefix}/chunks.json")
        enriched = store.get_json(f"{prefix}/enriched.json")["chunks"]
        vectors = vectors_from_npy(store.get(f"{prefix}/embeddings.npy"))
        manifest = store.get_json(f"{prefix}/manifest.json")["chunk_index_by_row"]

        # Row i of the matrix belongs to manifest[i]. Pairing vectors with
        # children by position is only safe once that list is exactly the
        # children, in order; otherwise the files came from different runs
        # and every vector would land under the wrong chunk, silently.
        expected = [c["chunk_index"] for c in chunks["children"]]
        if manifest != expected:
            raise ValueError(
                f"manifest lists {len(manifest)} rows that do not match the "
                f"{len(expected)} children of chunks.json -- staged artifacts are from "
                "different runs"
            )

        with session_scope() as session:
            persist_document(uuid.UUID(document_id), chunks, enriched, vectors, session)
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
