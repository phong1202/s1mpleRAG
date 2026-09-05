"""Runs against the real Redis from `docker compose up -d redis`.

The point of this module is atomicity under contention, which no mock can
demonstrate: a fake would serialise the calls and pass regardless.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
import redis as redis_lib

from app.config import get_settings
from shared.rate_limiter import TokenBucket, get_bucket


@pytest.fixture
def bucket():
    return TokenBucket(key=f"test:{uuid.uuid4()}", capacity=3, refill_per_sec=1.0)


def _rewind(bucket: TokenBucket, seconds: float) -> None:
    """Move the bucket's stored timestamp into the past.

    Couples to the field name and the millisecond unit on purpose: the
    alternative is sleeping, which makes the suite slower and flakier for no
    extra confidence.
    """
    client = redis_lib.from_url(get_settings().redis_url)
    stored = float(client.hget(bucket.key, "ts"))
    client.hset(bucket.key, "ts", stored - seconds * 1000)


def test_allows_up_to_capacity(bucket):
    assert [bucket.acquire()[0] for _ in range(3)] == [True, True, True]


def test_refuses_once_exhausted_and_reports_a_wait(bucket):
    for _ in range(3):
        bucket.acquire()

    allowed, wait_ms = bucket.acquire()

    assert allowed is False
    assert 0 < wait_ms <= 1000, "wait must be the time until a token is due"


def test_acquiring_more_than_one_token_at_a_time(bucket):
    assert bucket.acquire(tokens=3)[0] is True
    assert bucket.acquire(tokens=1)[0] is False


def test_a_request_larger_than_capacity_is_refused_not_hung(bucket):
    """A request bigger than the bucket can never be satisfied, so it has to
    be refused outright rather than handed an ever-growing wait."""
    allowed, wait_ms = bucket.acquire(tokens=99)

    assert allowed is False
    assert wait_ms == -1


def test_two_buckets_do_not_share_state():
    a = TokenBucket(key=f"a:{uuid.uuid4()}", capacity=1, refill_per_sec=1.0)
    b = TokenBucket(key=f"b:{uuid.uuid4()}", capacity=1, refill_per_sec=1.0)

    a.acquire()

    assert b.acquire()[0] is True


def test_the_bucket_refills_over_time(bucket):
    """Nothing else here covers refill_per_sec: every other test runs
    instantaneously, so a bucket that never recovered would pass all of them
    while starving the pipeline in production."""
    for _ in range(3):
        bucket.acquire()
    assert bucket.acquire()[0] is False

    _rewind(bucket, seconds=2)

    assert bucket.acquire(tokens=2)[0] is True, "2 seconds at 1/sec is 2 tokens"
    assert bucket.acquire()[0] is False, "and not a third"


def test_refill_never_exceeds_capacity(bucket):
    """An idle bucket must not accumulate an unbounded burst."""
    bucket.acquire()
    _rewind(bucket, seconds=3600)

    assert bucket.acquire(tokens=3)[0] is True
    assert bucket.acquire()[0] is False


def test_concurrent_acquires_never_exceed_capacity():
    """This is why the check and the decrement live in one Lua script. Read
    and write as separate commands and two clients both see "one left" and
    both spend it."""
    bucket = TokenBucket(key=f"race:{uuid.uuid4()}", capacity=10, refill_per_sec=0.01)

    with ThreadPoolExecutor(max_workers=20) as pool:
        granted = list(pool.map(lambda _: bucket.acquire()[0], range(40)))

    assert sum(granted) == 10


def test_get_bucket_is_cached_and_derives_its_rate_from_settings():
    settings = get_settings()

    chat = get_bucket("chat_rpm")

    assert get_bucket("chat_rpm") is chat
    assert chat.capacity == settings.rl_chat_rpm
    assert chat.refill_per_sec == pytest.approx(settings.rl_chat_rpm / 60)
    assert get_bucket("embed_tpm").capacity == settings.rl_embed_tpm
