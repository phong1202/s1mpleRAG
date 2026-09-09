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
