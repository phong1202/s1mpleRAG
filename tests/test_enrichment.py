"""The first three tests are the reason StubProvider exists -- there is no
way to make a real API reliably fail in the exact shape a test needs."""

from collections import Counter

import pytest

from app.exceptions import AppException, ErrorCode
from shared.llm import CATEGORIES, EnrichedChunk, StubProvider
from shared.rate_limiter import RateLimited
from worker.enrichment import enrich_chunks


class CountingStub(StubProvider):
    """Fails the whole batch on the first call, succeeds when retried alone,
    and records which ids got retried solo."""

    def __init__(self, drop_ids=None):
        super().__init__(drop_ids=drop_ids)
        self.solo_calls = []

    def enrich(self, chunks):
        if len(chunks) == 1:
            self.solo_calls.append(chunks[0]["id"])
            return [EnrichedChunk(id=chunks[0]["id"], context="recovered", category="TECHNICAL")]
        return super().enrich(chunks)


@pytest.fixture
def children():
    return [
        {"id": i, "chunk_index": i, "content": f"content {i}", "token_count": 120}
        for i in range(25)
    ]


def test_every_chunk_gets_a_context_and_a_category(children):
    result = enrich_chunks(children, provider=StubProvider(), batch_size=20)

    assert len(result) == len(children)
    assert all(r["context"] for r in result)
    assert all(r["category"] in CATEGORIES for r in result)


def test_a_missing_id_is_retried_alone_not_the_whole_batch(children):
    """Acceptance criterion for this step: 19 out of 20 must trigger a retry
    of the EXACT missing id, not the whole batch."""
    provider = CountingStub(drop_ids=[7])

    result = enrich_chunks(children, provider=provider, batch_size=20)

    assert len(result) == len(children)
    assert provider.solo_calls == [7], "must retry exactly one id, alone"


def test_a_chunk_that_fails_twice_gets_an_empty_context_and_proceeds(children):
    """A missing context sentence makes retrieval a little worse; a
    dead-lettered document helps nobody."""
    provider = StubProvider(drop_ids=[3])

    result = enrich_chunks(children, provider=provider, batch_size=20)

    stubborn = next(r for r in result if r["id"] == 3)
    assert stubborn["context"] == ""


def test_an_off_enum_category_is_rejected_and_replaced_with_other(children):
    result = enrich_chunks(children, provider=StubProvider(bad_category_ids=[5]), batch_size=20)

    assert next(r for r in result if r["id"] == 5)["category"] == "OTHER"


def test_a_document_over_the_chunk_cap_is_refused_before_spending(children, monkeypatch):
    """The rate limiter caps SPEED, not TOTAL. This cap is what caps total."""
    monkeypatch.setattr("worker.enrichment.MAX_CHUNKS_PER_DOC", 10)

    with pytest.raises(AppException):
        enrich_chunks(children, provider=StubProvider(), batch_size=20)


def test_a_batch_bigger_than_the_bucket_fails_instead_of_retrying_forever(children, monkeypatch):
    """acquire() answers wait_ms=-1 for a request that can NEVER succeed --
    one larger than the bucket's whole capacity -- as opposed to one that
    merely has to wait. Handing that to retry(countdown=wait_ms/1000) makes
    a negative countdown, which Celery turns into an ETA already in the
    past and therefore runs immediately; with max_retries=None that is an
    endless tight loop on a document that cannot ever get through, with no
    error recorded anywhere. It has to fail loudly instead."""

    class ImpossibleBucket:
        def acquire(self, tokens: int = 1) -> tuple[bool, int]:
            return False, -1

    monkeypatch.setattr("shared.rate_limiter.get_bucket", lambda name: ImpossibleBucket())

    with pytest.raises(AppException) as exc:
        enrich_chunks(children, provider=StubProvider(), batch_size=20)

    assert exc.value.error is ErrorCode.RATE_LIMIT_UNSATISFIABLE


def test_a_bucket_that_says_wait_raises_rate_limited_with_its_wait(children, monkeypatch):
    """The work itself never touches Celery's retry machinery -- it only
    reports how long to wait, plus up to 30% jitter so a crowd of deferred
    tasks does not wake up in lockstep. The stage task decides what to do."""

    class WaitingBucket:
        def acquire(self, tokens: int = 1) -> tuple[bool, int]:
            return False, 500

    monkeypatch.setattr("shared.rate_limiter.get_bucket", lambda name: WaitingBucket())

    with pytest.raises(RateLimited) as exc:
        enrich_chunks(children, provider=StubProvider(), batch_size=20)

    assert 0.5 <= exc.value.countdown <= 0.65


def test_resuming_from_partials_sends_each_chunk_once_and_always_finishes(monkeypatch):
    """Each attempt here only has room for two batches -- the steady
    contention that used to livelock: every retry restarted at batch one,
    re-paid the same two batches, and never reached the third. Resuming from
    what each deferred attempt hands back, it finishes, and no chunk is ever
    sent twice."""
    children = [
        {"id": i, "chunk_index": i, "content": f"content {i}", "token_count": 120}
        for i in range(100)
    ]
    sent = Counter()

    class CountingProvider(StubProvider):
        def enrich(self, chunks):
            sent.update(c["id"] for c in chunks)
            return super().enrich(chunks)

    class RoomForTwoBatches:
        def __init__(self):
            self.acquires = 0

        def acquire(self, tokens: int = 1) -> tuple[bool, int]:
            self.acquires += 1  # two per batch: chat_rpm, then chat_tpm
            return (True, 0) if self.acquires <= 4 else (False, 500)

    provider = CountingProvider()
    done: list[dict] = []
    for _attempt in range(10):
        bucket = RoomForTwoBatches()
        monkeypatch.setattr("shared.rate_limiter.get_bucket", lambda name, b=bucket: b)
        try:
            result = enrich_chunks(children, provider=provider, batch_size=20, done=done)
            break
        except RateLimited as exc:
            done = exc.partial
    else:
        pytest.fail("never finished in 10 attempts")

    assert [r["id"] for r in result] == list(range(100))
    assert set(sent.values()) == {1}, "a chunk was sent to the LLM more than once"
