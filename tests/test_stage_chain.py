"""Runs the chain in-process via task_always_eager -- this checks chain
logic and the state machine, not delivery through the real broker (that is
checked by hand: `docker compose kill worker-cpu` mid-chain and confirm the
document gets redelivered rather than stuck).
"""

import uuid

import pytest

from worker.celery_app import app as celery_app
from worker.stages import STAGES, launch


@pytest.fixture(autouse=True)
def eager():
    celery_app.conf.task_always_eager = True
    yield
    celery_app.conf.task_always_eager = False


def reload(document_id):
    from app.models.document import Document
    from worker.db import session_scope

    with session_scope() as session:
        return session.get(Document, uuid.UUID(str(document_id)))


def test_task_routes_split_cpu_and_llm():
    routes = celery_app.conf.task_routes
    assert routes["worker.stages.parse"]["queue"] == "cpu"
    assert routes["worker.stages.structure"]["queue"] == "cpu"
    assert routes["worker.stages.enrich"]["queue"] == "llm"
    assert routes["worker.stages.embed"]["queue"] == "llm"
    assert routes["worker.stages.persist"]["queue"] == "cpu"


def test_acks_late_is_on():
    """A crash must lead to redelivery, not lost work."""
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_no_result_backend():
    """documents.status in Postgres is the durable state, not a backend."""
    assert celery_app.conf.result_backend is None


def test_chain_drives_a_document_to_completed(seeded_document):
    launch(str(seeded_document.id))

    refreshed = reload(seeded_document.id)
    assert refreshed.status == "COMPLETED"
    assert refreshed.stage == STAGES[-1]
    assert refreshed.completed_at is not None


def test_a_permanent_error_goes_straight_to_dead_letter(seeded_document):
    """An encrypted PDF will never parse. Burning five retries on it is
    wasted time, and worse, it hides the real reason."""
    from app.exceptions import AppException, ErrorCode
    from worker.stages import stage_failed

    stage_failed(str(seeded_document.id), "PARSING", AppException(ErrorCode.PDF_ENCRYPTED))

    document = reload(seeded_document.id)
    assert document.status == "DEAD_LETTER"
    assert document.failed_stage == "PARSING"
    assert "encrypted" in document.last_error.lower()


def test_a_transient_error_becomes_retrying_and_counts_an_attempt(seeded_document):
    from worker.stages import stage_failed

    stage_failed(str(seeded_document.id), "ENRICHING", ConnectionError("broker went away"))

    document = reload(seeded_document.id)
    assert document.status == "RETRYING"
    assert document.attempts == 1


def test_dead_letter_after_the_attempt_ceiling(seeded_document):
    from worker.stages import MAX_ATTEMPTS, stage_failed

    for _ in range(4):
        stage_failed(str(seeded_document.id), "ENRICHING", ConnectionError("flaky"))

    document = reload(seeded_document.id)
    assert document.status == "DEAD_LETTER"
    assert document.attempts == 4
    assert document.attempts > MAX_ATTEMPTS, "the ceiling was already crossed at attempt 3"
    assert document.failed_stage == "ENRICHING"


def test_stage_failed_reports_whether_the_document_is_now_dead(seeded_document):
    """The return value is what a task checks before asking Celery to retry
    -- see test_retries_stop_at_the_ceiling_not_at_max_retries below for why
    that check has to exist at all."""
    from worker.stages import stage_failed

    document_id = str(seeded_document.id)
    assert stage_failed(document_id, "ENRICHING", ConnectionError("1")) is False
    assert stage_failed(document_id, "ENRICHING", ConnectionError("2")) is False
    assert stage_failed(document_id, "ENRICHING", ConnectionError("3")) is True


def test_retries_stop_at_the_attempt_ceiling_not_at_max_retries(seeded_document, monkeypatch):
    """structure's own max_retries=5 would allow five retries by itself.
    MAX_ATTEMPTS=3 has to win: once stage_failed says the document is dead,
    the task must not ask Celery to retry again, or a document already
    given up on keeps running its real body -- an OpenAI call, for
    enrich/embed -- for every retry Celery's own ceiling still permits.
    """
    from worker import stages

    calls = {"n": 0}

    def flaky_advance(*args, **kwargs):
        calls["n"] += 1
        raise ConnectionError("simulated transient failure")

    monkeypatch.setattr(stages, "_advance", flaky_advance)

    with pytest.raises(ConnectionError):
        stages.structure.apply(args=(str(seeded_document.id),)).get()

    assert calls["n"] == stages.MAX_ATTEMPTS

    document = reload(seeded_document.id)
    assert document.status == "DEAD_LETTER"
    assert document.failed_stage == "STRUCTURING"


@pytest.fixture
def staged_chunks(seeded_document):
    """chunks.json for seeded_document, shaped exactly as S2 leaves it:
    three parents of 20 children each -- three batches at enrich's default
    size. seeded_document's teardown removes it, with everything downstream."""
    from shared.storage import get_store

    store = get_store()
    prefix = f"staging/{seeded_document.id}"
    parents = [
        {
            "chunk_index": p,
            "content": f"parent {p}",
            "token_count": 600,
            "page_start": 1,
            "page_end": 1,
            "heading_path": f"Chuong {p}",
            "language": "vi",
        }
        for p in range(3)
    ]
    children = [
        {
            "chunk_index": i,
            "parent_index": i // 20,
            "content": f"body {i}",
            "token_count": 50,
            "page_number": 1,
            "language": "vi",
        }
        for i in range(60)
    ]
    store.put_json(f"{prefix}/chunks.json", {"parents": parents, "children": children})
    return seeded_document


def test_a_rate_limited_enrich_is_deferred_not_counted_as_a_failure(staged_chunks, monkeypatch):
    """Being rate limited is the limiter doing its job, not an error. It
    used to land in the stage's generic `except Exception` -- Celery's Retry
    is an Exception -- so every deferral cost an attempt (three deferrals
    dead-lettered a perfectly healthy document) and published a second
    retry on top of the first, running the rest of the chain twice."""
    from worker import stages

    calls = {"n": 0}

    class WaitOnceBucket:
        def acquire(self, tokens: int = 1) -> tuple[bool, int]:
            calls["n"] += 1
            return (False, 50) if calls["n"] == 1 else (True, 0)

    monkeypatch.setattr("shared.rate_limiter.get_bucket", lambda name: WaitOnceBucket())

    stages.enrich.apply(args=(str(staged_chunks.id),)).get()

    document = reload(staged_chunks.id)
    assert document.attempts == 0
    assert document.last_error is None
    assert document.failed_stage is None
    assert document.stage == "ENRICHING"


def test_enrich_stops_at_the_attempt_ceiling_with_no_celery_cap(seeded_document, monkeypatch):
    """enrich runs with max_retries=None, so that rate-limit deferrals never
    eat into a retry budget. That leaves MAX_ATTEMPTS as the only thing
    stopping a real failure from retrying forever -- so it gets its own
    test rather than borrowing structure's."""
    from worker import stages

    calls = {"n": 0}

    def flaky_advance(*args, **kwargs):
        calls["n"] += 1
        raise ConnectionError("simulated transient failure")

    monkeypatch.setattr(stages, "_advance", flaky_advance)

    with pytest.raises(ConnectionError):
        stages.enrich.apply(args=(str(seeded_document.id),)).get()

    assert calls["n"] == stages.MAX_ATTEMPTS

    document = reload(seeded_document.id)
    assert document.status == "DEAD_LETTER"
    assert document.failed_stage == "ENRICHING"


def test_a_mid_document_deferral_resumes_instead_of_starting_over(staged_chunks, monkeypatch):
    """A deferral used to throw away every batch already enriched -- results
    lived only in memory until the end -- so the retry sent them all again,
    paying twice; and under steady contention, where each attempt only has
    room for the same first few batches, the document never finished at
    all. What a deferred run paid for is saved before the requeue, and the
    retry skips it."""
    from collections import Counter

    import shared.llm
    from shared.llm import StubProvider
    from shared.storage import get_store
    from worker import stages

    sent = Counter()

    class CountingStub(StubProvider):
        def enrich(self, chunks):
            sent.update(c["id"] for c in chunks)
            return super().enrich(chunks)

    monkeypatch.setattr(shared.llm, "get_provider", lambda: CountingStub())

    acquires = {"n": 0}

    class WaitAtTheThirdBatch:
        def acquire(self, tokens: int = 1) -> tuple[bool, int]:
            acquires["n"] += 1  # two per batch: chat_rpm, then chat_tpm
            return (False, 50) if acquires["n"] == 5 else (True, 0)

    monkeypatch.setattr("shared.rate_limiter.get_bucket", lambda name: WaitAtTheThirdBatch())

    stages.enrich.apply(args=(str(staged_chunks.id),)).get()

    assert len(sent) == 60
    assert set(sent.values()) == {1}, "a chunk was sent to the LLM more than once"

    enriched = get_store().get_json(f"staging/{staged_chunks.id}/enriched.json")["chunks"]
    assert [c["id"] for c in enriched] == list(range(60))


@pytest.fixture
def staged_enriched(staged_chunks):
    """enriched.json on top of staged_chunks, as S3 would have left it --
    embed's input."""
    from shared.storage import get_store

    store = get_store()
    prefix = f"staging/{staged_chunks.id}"
    enriched = [{"id": i, "context": f"context {i}", "category": "TECHNICAL"} for i in range(60)]
    store.put_json(f"{prefix}/enriched.json", {"chunks": enriched})
    return staged_chunks


def test_a_mid_document_embed_deferral_resumes_with_rows_aligned(staged_enriched, monkeypatch):
    """S4 had S3's three rate-limit bugs copied in. Deferred at the third
    batch, it must count no attempt, embed every text exactly once, and
    still leave row i of the matrix matching manifest row i."""
    import io
    from collections import Counter

    import numpy as np

    import shared.llm
    from app.config import get_settings
    from shared.llm import StubProvider
    from shared.storage import get_store
    from worker import stages
    from worker.embedding import contextualize

    monkeypatch.setattr(get_settings(), "embed_batch_size", 20)
    sent = Counter()

    class CountingStub(StubProvider):
        def embed(self, texts):
            sent.update(texts)
            return super().embed(texts)

    monkeypatch.setattr(shared.llm, "get_provider", lambda: CountingStub())
    acquires = {"n": 0}

    class WaitAtTheThirdBatch:
        def acquire(self, tokens: int = 1) -> tuple[bool, int]:
            acquires["n"] += 1  # two per batch: embed_rpm, then embed_tpm
            return (False, 50) if acquires["n"] == 5 else (True, 0)

    monkeypatch.setattr("shared.rate_limiter.get_bucket", lambda name: WaitAtTheThirdBatch())

    stages.embed.apply(args=(str(staged_enriched.id),)).get()

    assert reload(staged_enriched.id).attempts == 0
    assert len(sent) == 60
    assert set(sent.values()) == {1}, "a text was embedded more than once"

    prefix = f"staging/{staged_enriched.id}"
    store = get_store()
    vectors = np.load(io.BytesIO(store.get(f"{prefix}/embeddings.npy")))
    manifest = store.get_json(f"{prefix}/manifest.json")["chunk_index_by_row"]
    expected = StubProvider().embed([contextualize(f"context {i}", f"body {i}") for i in range(60)])
    assert vectors.shape == (60, 1536)
    assert manifest == list(range(60))
    assert np.allclose(vectors, np.asarray(expected, dtype=np.float32))


def test_a_crash_between_the_two_embed_writes_is_recovered_not_skipped(
    staged_enriched, monkeypatch
):
    """embeddings.npy is the checkpoint this stage skips on. Written before
    manifest.json, a crash between the two leaves a run every retry skips
    -- and S5 without a manifest, for good."""
    from shared.storage import ObjectStore, get_store
    from worker import stages

    real_put_json = ObjectStore.put_json
    crashed = {"yet": False}

    def put_json_crashing_on_the_first_manifest(self, key, obj):
        if key.endswith("/manifest.json") and not crashed["yet"]:
            crashed["yet"] = True
            raise ConnectionError("simulated crash between the two writes")
        return real_put_json(self, key, obj)

    monkeypatch.setattr(ObjectStore, "put_json", put_json_crashing_on_the_first_manifest)

    stages.embed.apply(args=(str(staged_enriched.id),)).get()

    assert crashed["yet"]
    assert get_store().exists(f"staging/{staged_enriched.id}/manifest.json")


@pytest.fixture
def staged_embeddings(staged_enriched):
    """embeddings.npy + manifest.json on top of staged_enriched, as S4 would
    have left them -- persist's input."""
    from shared.llm import StubProvider
    from shared.storage import get_store
    from worker.embedding import vectors_to_npy

    store = get_store()
    prefix = f"staging/{staged_enriched.id}"
    store.put_json(f"{prefix}/manifest.json", {"chunk_index_by_row": list(range(60))})
    store.put(
        f"{prefix}/embeddings.npy",
        vectors_to_npy(StubProvider().embed([f"text {i}" for i in range(60)])),
    )
    return staged_enriched


def _child_count(document_id):
    from sqlalchemy import func, select

    from app.models import ChildChunk
    from worker.db import session_scope

    with session_scope() as session:
        return session.scalar(
            select(func.count())
            .select_from(ChildChunk)
            .where(ChildChunk.document_id == document_id)
        )


def test_persist_completes_a_document_from_its_staged_artifacts(staged_embeddings):
    from worker import stages

    stages.persist.apply(args=(str(staged_embeddings.id),)).get()

    document = reload(staged_embeddings.id)
    assert document.status == "COMPLETED"
    assert document.language == "vi"
    assert _child_count(staged_embeddings.id) == 60


def test_persist_refuses_vectors_built_for_a_different_chunks_json(staged_embeddings):
    """Vectors pair with children by row. If the manifest S4 wrote does not
    list exactly these children in this order, the two files came from
    different runs, and persisting would file every vector under the wrong
    chunk -- with nothing anywhere to say so."""
    from shared.storage import get_store
    from worker import stages

    get_store().put_json(
        f"staging/{staged_embeddings.id}/manifest.json",
        {"chunk_index_by_row": list(reversed(range(60)))},
    )

    with pytest.raises(ValueError):
        stages.persist.apply(args=(str(staged_embeddings.id),)).get()

    document = reload(staged_embeddings.id)
    assert document.status == "DEAD_LETTER"
    assert document.failed_stage == "PERSISTING"
    assert "manifest" in document.last_error
    assert _child_count(staged_embeddings.id) == 0


def test_a_document_with_no_extractable_text_is_dead_lettered(seeded_document):
    """A blank or image-only PDF that OCR could get nothing from used to go
    all the way to COMPLETED with zero chunks: a success that nothing can
    ever find, with language left NULL. It is a permanent failure, with a
    reason someone can act on."""
    from shared.storage import get_store
    from worker import stages

    blank = {"page": 1, "markdown": "", "source": "docling", "confidence": 0.0}
    get_store().put_json(
        f"staging/{seeded_document.id}/parsed.json",
        {"title": None, "page_count": 1, "pages": [blank]},
    )

    # Eager mode hands AppException back wrapped (it does not pickle), so
    # this matches on the message; the document's state is the contract.
    with pytest.raises(Exception, match="No extractable text"):
        stages.structure.apply(args=(str(seeded_document.id),)).get()

    document = reload(seeded_document.id)
    assert document.status == "DEAD_LETTER"
    assert document.failed_stage == "STRUCTURING"
    assert "no extractable text" in document.last_error.lower()
    assert not get_store().exists(f"staging/{seeded_document.id}/chunks.json")
