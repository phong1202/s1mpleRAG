"""Live progress of a document's running stage, kept in Redis.

Not in Postgres, which stays the source of truth for status: progress is
written after every OCR'd page, read on every FE poll, and worth nothing
once the stage is over. Losing it to a Redis restart loses a progress bar,
never a document's state -- so every failure here is logged and swallowed.

One JSON string per document, so the API reads a whole page of documents
with a single MGET.
"""

import json
import logging
import time
from datetime import UTC, datetime
from functools import lru_cache

import redis
import redis.asyncio

from app.config import get_settings

logger = logging.getLogger(__name__)

# Long enough to outlive the worker that wrote it: a key left behind by a
# stage that stopped reporting is how the FE learns it stopped.
_TTL_S = 3600
# Past this with no report, a running stage reads as stalled. Not tighter:
# a page can take ~2.5 min with the GPU full, and between time slices a
# document waits its turn behind another's (worker/steps/parsing.py).
STALLED_AFTER_S = 600
# Statuses with nothing running, whose leftover key is not shown.
TERMINAL = {"COMPLETED", "DEAD_LETTER"}


def _key(document_id: str) -> str:
    return f"progress:{document_id}"


@lru_cache
def _client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url)


def report(document_id: str, stage: str, done: int, total: int) -> None:
    value = json.dumps({"stage": stage, "done": done, "total": total, "at": time.time()})
    try:
        _client().set(_key(document_id), value, ex=_TTL_S)
    except redis.RedisError as exc:
        logger.warning("could not report progress for %s: %s", document_id, exc)


def clear(document_id: str) -> None:
    try:
        _client().delete(_key(document_id))
    except redis.RedisError as exc:
        logger.warning("could not clear progress for %s: %s", document_id, exc)


def _shown(raw: bytes | None, now: float) -> dict | None:
    if raw is None:
        return None
    value = json.loads(raw)
    return {
        "stage": value["stage"],
        "done": value["done"],
        "total": value["total"],
        "updated_at": datetime.fromtimestamp(value["at"], UTC),
        "stalled": now - value["at"] > STALLED_AFTER_S,
    }


def read(document_id: str) -> dict | None:
    """The progress as the API shows it, read synchronously."""
    return _shown(_client().get(_key(document_id)), time.time())


async def _read_raw(document_ids: list[str]) -> list[bytes | None]:
    # A client per call: an async client is bound to the event loop it was
    # made on, and caching one would outlive it.
    async with redis.asyncio.from_url(get_settings().redis_url) as client:
        return await client.mget([_key(i) for i in document_ids])


async def read_many(document_ids: list[str]) -> dict[str, dict]:
    """Each document's progress, by id; absent where there is none -- and
    for all of them when Redis cannot be read."""
    if not document_ids:
        return {}
    try:
        raw = await _read_raw(document_ids)
    except redis.RedisError as exc:
        logger.warning("could not read progress: %s", exc)
        return {}
    now = time.time()
    shown = {i: _shown(r, now) for i, r in zip(document_ids, raw, strict=True)}
    return {i: p for i, p in shown.items() if p}
