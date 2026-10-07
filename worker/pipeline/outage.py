"""How long the OCR server has been unavailable to one document.

In Redis, not the documents row: it is a clock the stage reads on every
deferral, and losing it to a Redis restart costs one more full wait at
worst. Started by the first OcrUnavailable, ended by the first batch that
lands, so an outage hours later is timed from its own start.
"""

import time
from functools import lru_cache

import redis

from app.config import get_settings


@lru_cache
def _client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url)


def _key(document_id: str) -> str:
    return f"ocr:outage:{document_id}"


def outage_seconds(document_id: str) -> float:
    """Seconds since this document's current outage began -- starting the
    clock if this is its first deferral."""
    now = time.time()
    ceiling = get_settings().ocr_outage_max_s
    # The TTL only cleans up after a document deleted mid-outage: anything
    # older than a few ceilings has long since been counted as a failure.
    _client().set(_key(document_id), now, nx=True, ex=max(3 * ceiling, 60))
    started = _client().get(_key(document_id))
    return now - float(started) if started else 0.0


def outage_over(document_id: str) -> None:
    _client().delete(_key(document_id))
