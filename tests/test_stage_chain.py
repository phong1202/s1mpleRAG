"""Runs the chain in-process via task_always_eager -- this checks chain
logic and the state machine, not delivery through the real broker (that is
checked by hand: `docker compose kill worker-cpu` mid-chain and confirm the
document gets redelivered rather than stuck).
"""

import uuid

import pytest

from worker.celery_app import app as celery_app
from worker.pipeline.state import STAGES
from worker.stages import launch


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


def test_task_routes_split_ocr_cpu_and_llm():
    """parse has a queue of its own so its worker's concurrency can be set
    by the GPU, not the CPU: on 2026-10-06, 8 parses at once queued 46 page
    requests on a server that runs 8, and no document finished early."""
    routes = celery_app.conf.task_routes
    assert routes["worker.stages.parse"]["queue"] == "ocr"
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
    from worker.pipeline.state import stage_failed

    stage_failed(str(seeded_document.id), "PARSING", AppException(ErrorCode.PDF_ENCRYPTED))

    document = reload(seeded_document.id)
    assert document.status == "DEAD_LETTER"
    assert document.failed_stage == "PARSING"
    assert "encrypted" in document.last_error.lower()


def test_a_transient_error_becomes_retrying_and_counts_an_attempt(seeded_document):
    from worker.pipeline.state import stage_failed

    stage_failed(str(seeded_document.id), "ENRICHING", ConnectionError("broker went away"))

    document = reload(seeded_document.id)
    assert document.status == "RETRYING"
    assert document.attempts == 1


def test_dead_letter_after_the_attempt_ceiling(seeded_document):
    from worker.pipeline.state import MAX_ATTEMPTS, stage_failed

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
    from worker.pipeline.state import stage_failed

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
    from worker.pipeline.state import MAX_ATTEMPTS

    calls = {"n": 0}

    def flaky_advance(*args, **kwargs):
        calls["n"] += 1
        raise ConnectionError("simulated transient failure")

    monkeypatch.setattr(stages, "advance", flaky_advance)

    with pytest.raises(ConnectionError):
        stages.structure.apply(args=(str(seeded_document.id),)).get()

    assert calls["n"] == MAX_ATTEMPTS

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
    from worker.pipeline.state import MAX_ATTEMPTS

    calls = {"n": 0}

    def flaky_advance(*args, **kwargs):
        calls["n"] += 1
        raise ConnectionError("simulated transient failure")

    monkeypatch.setattr(stages, "advance", flaky_advance)

    with pytest.raises(ConnectionError):
        stages.enrich.apply(args=(str(seeded_document.id),)).get()

    assert calls["n"] == MAX_ATTEMPTS

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
    from worker.steps.embedding import contextualize

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
    from worker.steps.embedding import vectors_to_npy

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

    blank = {"page": 1, "markdown": "", "source": "chandra", "confidence": 0.0}
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


def _delete(document_id):
    from app.models import Document
    from worker.db import session_scope

    with session_scope() as session:
        session.delete(session.get(Document, document_id))


def test_a_task_for_a_deleted_document_stops_quietly(seeded_document):
    """A user can delete a document while its chain is still queued. The
    next stage used to crash on the missing row -- AttributeError, logged as
    an error, chain dead anyway. It should end the chain, quietly: there is
    nothing left to process and nothing to retry."""
    from worker import stages

    _delete(seeded_document.id)

    result = stages.structure.apply(args=(str(seeded_document.id),))

    assert result.state == "IGNORED"


def test_a_failure_on_a_deleted_document_stops_quietly_too(seeded_document, monkeypatch):
    """The row can also vanish between a stage's failure and recording it --
    stage_failed() then crashed inside the except handler, past every
    clause that could have dealt with it."""
    from worker import stages

    def failing_advance(*args, **kwargs):
        _delete(seeded_document.id)
        raise ConnectionError("simulated transient failure")

    monkeypatch.setattr(stages, "advance", failing_advance)

    result = stages.structure.apply(args=(str(seeded_document.id),))

    assert result.state == "IGNORED"


@pytest.fixture
def ocr_every_page(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "ocr_url", "http://ocr")
    monkeypatch.setattr(get_settings(), "ocr_all_pages", True)


def test_parse_resumes_from_a_saved_ocr_checkpoint(seeded_document, ocr_every_page, monkeypatch):
    from shared.storage import get_store
    from worker import stages

    store = get_store()
    prefix = f"staging/{seeded_document.id}"
    store.put_json(f"{prefix}/parsed.partial.json", {"ocr": {"1": "# Read before the crash"}})
    sent = []

    def recording_ocr(data, page_numbers, ocr_url, on_batch=None, deadline=None):
        sent.extend(page_numbers)
        return {}

    monkeypatch.setattr("worker.steps.parsing._ocr", recording_ocr)

    stages.parse.apply(args=(str(seeded_document.id),)).get()

    assert sent == []
    parsed = store.get_json(f"{prefix}/parsed.json")
    assert parsed["pages"][0]["markdown"] == "# Read before the crash"


def test_parse_saves_each_ocr_batch_before_anything_can_fail(
    seeded_document, ocr_every_page, monkeypatch
):
    """Saved as each batch lands, not on the way out: a worker that is
    OOM-killed or loses its broker channel runs no except clause."""
    from shared.storage import get_store
    from worker import stages

    def ocr_then_die(data, page_numbers, ocr_url, on_batch=None, deadline=None):
        on_batch({1: "# Paid for"})
        raise RuntimeError("worker lost")

    monkeypatch.setattr("worker.steps.parsing._ocr", ocr_then_die)

    stages.parse.apply(args=(str(seeded_document.id),))

    partial = get_store().get_json(f"staging/{seeded_document.id}/parsed.partial.json")
    assert partial == {"ocr": {"1": "# Paid for"}}


@pytest.fixture
def outage_clock(seeded_document):
    """The Redis key the outage ceiling is timed by -- deleted afterwards,
    since a test that ends mid-outage never reaches the batch that clears
    it."""
    import redis

    from app.config import get_settings

    key = f"ocr:outage:{seeded_document.id}"
    client = redis.from_url(get_settings().redis_url)
    yield key, client
    client.delete(key)


def test_an_ocr_outage_is_deferred_without_counting_an_attempt(
    seeded_document, ocr_every_page, outage_clock, monkeypatch
):
    """2026-10-06: vLLM stalled 17 s on a burst of images, three parses
    timed out on /health, and each burned one of the three attempts every
    stage shares -- for a server that never went down. An outage is the
    server's, not the document's: it waits, like a rate limit."""
    from worker import stages
    from worker.steps.parsing import OcrUnavailable

    calls = {"n": 0}

    def down_once(data, page_numbers, ocr_url, on_batch=None, deadline=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OcrUnavailable("OCR server not answering", countdown=20)
        on_batch({1: "# Read once it was back"})
        return {1: "# Read once it was back"}

    monkeypatch.setattr("worker.steps.parsing._ocr", down_once)

    stages.parse.apply(args=(str(seeded_document.id),)).get()

    document = reload(seeded_document.id)
    assert calls["n"] == 2
    assert (document.stage, document.attempts, document.last_error) == ("PARSING", 0, None)
    key, client = outage_clock
    assert not client.exists(key), "a batch that landed must end the outage"


def test_an_outage_past_the_ceiling_counts_like_any_failure(
    seeded_document, ocr_every_page, outage_clock, monkeypatch
):
    """Deferring forever would hide a server that is never coming back:
    past OCR_OUTAGE_MAX_S, each further outage costs an attempt, and the
    document dead-letters with the reason on it."""
    from app.config import get_settings
    from worker import stages
    from worker.pipeline.state import MAX_ATTEMPTS
    from worker.steps.parsing import OcrUnavailable

    def always_down(data, page_numbers, ocr_url, on_batch=None, deadline=None):
        raise OcrUnavailable("OCR server unreachable (ConnectError)", countdown=60)

    monkeypatch.setattr("worker.steps.parsing._ocr", always_down)
    monkeypatch.setattr(get_settings(), "ocr_outage_max_s", 0)

    stages.parse.apply(args=(str(seeded_document.id),))

    document = reload(seeded_document.id)
    assert (document.status, document.attempts) == ("DEAD_LETTER", MAX_ATTEMPTS)
    assert "unreachable" in document.last_error


def test_an_outage_with_redis_down_is_counted_not_deferred_forever(
    seeded_document, ocr_every_page, monkeypatch
):
    import redis

    from worker import stages
    from worker.steps.parsing import OcrUnavailable

    def down(data, page_numbers, ocr_url, on_batch=None, deadline=None):
        raise OcrUnavailable("OCR server unreachable (ConnectError)", countdown=60)

    def no_redis(document_id):
        raise redis.ConnectionError("redis is down too")

    monkeypatch.setattr("worker.steps.parsing._ocr", down)
    monkeypatch.setattr("worker.pipeline.errors.outage_seconds", no_redis)

    stages.parse.apply(args=(str(seeded_document.id),))

    assert reload(seeded_document.id).status == "DEAD_LETTER"


def test_a_parse_slice_that_runs_out_continues_without_costing_an_attempt(
    seeded_document, ocr_every_page, monkeypatch
):
    """The end of a slice is not a failure: the task requeues itself -- to
    the back of the queue, so documents take turns and a short one is not
    stuck behind a long one -- and picks up from the checkpoint."""
    from worker import stages
    from worker.steps.parsing import ParseContinues

    calls = []

    def one_batch_per_slice(data, page_numbers, ocr_url, on_batch=None, deadline=None):
        calls.append(list(page_numbers))
        if len(calls) == 1:
            on_batch({1: "# Read in the first slice"})
            raise ParseContinues("slice over")
        return {}

    monkeypatch.setattr("worker.steps.parsing._ocr", one_batch_per_slice)

    stages.parse.apply(args=(str(seeded_document.id),)).get()

    document = reload(seeded_document.id)
    assert calls == [[1], []]
    assert (document.stage, document.attempts) == ("PARSING", 0)


def test_parse_reports_page_progress_and_the_page_count_up_front(
    seeded_document, ocr_every_page, monkeypatch
):
    """The FE showed "reading pages (OCR)..." for an hour on 2026-10-06:
    page_count arrived only once S1 was done, and nothing else was said.
    page_count is known the moment the PDF opens; pages read, after every
    batch."""
    from shared import progress
    from worker import stages

    seen = {}

    def ocr_and_look(data, page_numbers, ocr_url, on_batch=None, deadline=None):
        seen["page_count"] = reload(seeded_document.id).page_count
        seen["before"] = progress.read(str(seeded_document.id))
        on_batch({1: "# Page 1"})
        seen["after"] = progress.read(str(seeded_document.id))
        return {1: "# Page 1"}

    monkeypatch.setattr("worker.steps.parsing._ocr", ocr_and_look)

    stages.parse.apply(args=(str(seeded_document.id),)).get()

    assert seen["page_count"] == 1
    assert (seen["before"]["stage"], seen["before"]["done"], seen["before"]["total"]) == (
        "PARSING",
        0,
        1,
    )
    assert (seen["after"]["done"], seen["after"]["total"]) == (1, 1)
    assert progress.read(str(seeded_document.id)) is None, "a finished stage reports nothing"


def test_progress_lost_to_redis_does_not_fail_the_parse(
    seeded_document, ocr_every_page, monkeypatch
):
    """Progress is a courtesy to the FE, not part of the work."""
    import redis

    from worker import stages

    class DownRedis:
        def __getattr__(self, name):
            def fail(*args, **kwargs):
                raise redis.ConnectionError("redis is down")

            return fail

    monkeypatch.setattr("shared.progress._client", lambda: DownRedis())
    monkeypatch.setattr(
        "worker.steps.parsing._ocr",
        lambda data, page_numbers, ocr_url, on_batch=None, deadline=None: (
            on_batch({1: "# P"}) or {1: "# P"}
        ),
    )

    stages.parse.apply(args=(str(seeded_document.id),)).get()

    assert reload(seeded_document.id).stage == "PARSING"
