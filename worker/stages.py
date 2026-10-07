"""The five Celery tasks: parse, structure, enrich, embed, persist.

Thin on purpose. The state machine lives in worker/pipeline, the failure
policy is the @stage_task decorator, the work is in worker/steps, and the
database is reached through worker/repositories. What stays here is the
order of operations inside each stage: checkpoint skip, advance,
invalidate what is downstream, compute, write the artifact, advance.

Task names are fixed strings: queued messages refer to them, so moving this
module must not rename them. Each stage imports its step inside the body,
which @stage_task wraps -- see worker/pipeline/errors.py for why.
"""

import time
import uuid

from celery import chain

from app.exceptions import AppException, ErrorCode
from shared.rate_limiter import RateLimited
from worker.celery_app import app
from worker.db import session_scope
from worker.pipeline.errors import stage_task
from worker.pipeline.state import advance, invalidate_downstream
from worker.repositories.documents import DocumentStateRepository


# max_retries=None: an OCR outage defers, like a rate limit, and must not
# spend Celery's retry budget; MAX_ATTEMPTS stays the ceiling for failures.
@app.task(name="worker.stages.parse", bind=True, max_retries=None)
@stage_task("PARSING")
def parse(self, document_id: str) -> str:
    from app.config import get_settings
    from shared import progress
    from shared.storage import get_store
    from worker.pipeline.outage import outage_over
    from worker.steps.parsing import parse_document

    store = get_store()
    key = f"staging/{document_id}/parsed.json"
    if store.exists(key):  # checkpoint skip
        advance(document_id, "PARSING", stage="PARSING")
        return document_id

    advance(document_id, "PARSING")
    invalidate_downstream(store, document_id, "PARSING")
    with session_scope() as session:
        object_key = DocumentStateRepository(session).get(uuid.UUID(document_id)).object_key

    # Written after every OCR batch, as it lands -- not on the way out, like
    # enrich's and embed's: what loses a parse is a worker killed or cut off
    # from its broker, and neither runs an except clause. Left behind on
    # success for the staging bucket's 7-day expiry, as theirs are.
    partial_key = f"staging/{document_id}/parsed.partial.json"
    done = (
        {int(n): markdown for n, markdown in store.get_json(partial_key)["ocr"].items()}
        if store.exists(partial_key)
        else {}
    )

    to_read = {"pages": 0}

    def started(page_count: int, read: int, pages: int) -> None:
        # Known the moment the PDF opens; on a scan, half an hour before
        # the result -- the FE has a page count to show from the start.
        with session_scope() as session:
            DocumentStateRepository(session).set_page_count(uuid.UUID(document_id), page_count)
        to_read["pages"] = pages
        progress.report(document_id, "PARSING", done=read, total=pages)

    def checkpoint(ocr: dict[int, str | None]) -> None:
        store.put_json(partial_key, {"ocr": {str(n): markdown for n, markdown in ocr.items()}})
        progress.report(document_id, "PARSING", done=len(ocr), total=to_read["pages"])
        # A batch landed, so the server is back: a later outage is timed
        # from its own start, not from this one's.
        outage_over(document_id)

    settings = get_settings()
    result = parse_document(
        object_key,
        store,
        settings.ocr_url,
        settings.ocr_all_pages,
        done=done,
        on_batch=checkpoint,
        deadline=time.monotonic() + settings.parse_slice_s,
        on_start=started,
    )
    store.put_json(key, result)

    with session_scope() as session:
        DocumentStateRepository(session).set_parse_result(
            uuid.UUID(document_id), result["page_count"], result["title"]
        )

    progress.clear(document_id)
    advance(document_id, "PARSING", stage="PARSING")
    return document_id


@app.task(name="worker.stages.structure", bind=True, max_retries=5)
@stage_task("STRUCTURING")
def structure(self, document_id: str) -> str:
    from shared.storage import get_store
    from worker.steps.chunking import chunk_document

    store = get_store()
    key = f"staging/{document_id}/chunks.json"
    if store.exists(key):  # checkpoint skip
        advance(document_id, "STRUCTURING", stage="STRUCTURING")
        return document_id

    advance(document_id, "STRUCTURING")
    invalidate_downstream(store, document_id, "STRUCTURING")
    parsed = store.get_json(f"staging/{document_id}/parsed.json")
    chunks = chunk_document(parsed)
    # Checked here, after chunking, rather than on S1's raw text: a page of
    # a few stray characters is not empty, yet chunks to nothing.
    if not chunks["children"]:
        raise AppException(
            ErrorCode.NO_EXTRACTABLE_TEXT,
            f"No extractable text: no chunk survived from {parsed['page_count']} "
            "page(s) -- blank, or an image OCR could read nothing from",
        )
    store.put_json(key, chunks)

    advance(document_id, "STRUCTURING", stage="STRUCTURING")
    return document_id


# max_retries=None: Celery's retry budget would otherwise be spent by
# rate-limit deferrals, which are not failures. MAX_ATTEMPTS, enforced by
# stage_failed(), stays the ceiling for real ones.
@app.task(name="worker.stages.enrich", bind=True, max_retries=None)
@stage_task("ENRICHING")
def enrich(self, document_id: str) -> str:
    from app.config import get_settings
    from shared.llm import get_provider
    from shared.storage import get_store
    from worker.steps.enrichment import enrich_chunks

    store = get_store()
    key = f"staging/{document_id}/enriched.json"
    if store.exists(key):  # checkpoint skip
        advance(document_id, "ENRICHING", stage="ENRICHING")
        return document_id

    advance(document_id, "ENRICHING")
    invalidate_downstream(store, document_id, "ENRICHING")
    chunks = store.get_json(f"staging/{document_id}/chunks.json")
    # chunk_index plays the "id" role in the provider contract -- it is
    # deterministic, so a rerun asks about exactly the same id set.
    children = [{**c, "id": c["chunk_index"]} for c in chunks["children"]]

    # Left behind on success for the staging bucket's 7-day expiry: deleting
    # it here would add a failure point after the real checkpoint is written.
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
        # Saved here, inside the body @stage_task wraps, not in its handler: a
        # storage failure while saving must still reach stage_failed() like
        # any other, not escape from a handler.
        store.put_json(partial_key, {"chunks": exc.partial})
        raise
    store.put_json(key, {"chunks": enriched})

    advance(document_id, "ENRICHING", stage="ENRICHING")
    return document_id


# max_retries=None for the same reason as enrich.
@app.task(name="worker.stages.embed", bind=True, max_retries=None)
@stage_task("EMBEDDING")
def embed(self, document_id: str) -> str:
    from app.config import get_settings
    from shared.llm import get_provider
    from shared.storage import get_store
    from worker.steps.embedding import (
        contextualize,
        embed_chunks,
        vectors_from_npy,
        vectors_to_npy,
    )

    store = get_store()
    key = f"staging/{document_id}/embeddings.npy"
    if store.exists(key):  # checkpoint skip
        advance(document_id, "EMBEDDING", stage="EMBEDDING")
        return document_id

    advance(document_id, "EMBEDDING")
    invalidate_downstream(store, document_id, "EMBEDDING")
    chunks = store.get_json(f"staging/{document_id}/chunks.json")
    enriched = {
        e["id"]: e for e in store.get_json(f"staging/{document_id}/enriched.json")["chunks"]
    }

    texts, manifest = [], []
    for child in chunks["children"]:
        # Indexed, not .get() with an empty fallback: S3 gives every child an
        # entry, even one whose context failed, so a missing one means
        # chunks.json and enriched.json came from different runs -- worth
        # failing on, not embedding past in silence.
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
        # Inside the wrapped body for the same reason as in enrich.
        store.put(partial_key, vectors_to_npy(exc.partial))
        raise

    # The manifest first, the file the skip above checks last: written the
    # other way round, a crash between the two leaves a run every retry
    # skips, and S5 without a manifest for good.
    store.put_json(f"staging/{document_id}/manifest.json", {"chunk_index_by_row": manifest})
    store.put(key, vectors_to_npy(vectors))

    advance(document_id, "EMBEDDING", stage="EMBEDDING")
    return document_id


# No checkpoint skip, unlike the other four: the upserts are idempotent in
# themselves, so a rerun is boring rather than destructive.
@app.task(name="worker.stages.persist", bind=True, max_retries=5)
@stage_task("PERSISTING")
def persist(self, document_id: str) -> str:
    from shared.storage import get_store
    from worker.steps.embedding import vectors_from_npy
    from worker.steps.persistence import persist_document

    store = get_store()
    advance(document_id, "PERSISTING")

    prefix = f"staging/{document_id}"
    chunks = store.get_json(f"{prefix}/chunks.json")
    enriched = store.get_json(f"{prefix}/enriched.json")["chunks"]
    vectors = vectors_from_npy(store.get(f"{prefix}/embeddings.npy"))
    manifest = store.get_json(f"{prefix}/manifest.json")["chunk_index_by_row"]

    # Row i of the matrix belongs to manifest[i]. Pairing vectors with
    # children by position is only safe once that list is exactly the
    # children, in order; otherwise the files came from different runs and
    # every vector would land under the wrong chunk, silently.
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


def launch(document_id: str) -> None:
    chain(parse.s(document_id), structure.s(), enrich.s(), embed.s(), persist.s()).apply_async()
