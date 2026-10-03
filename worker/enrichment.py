"""S3 -- Enrich. The most expensive stage, and the one most worth
checkpointing.

Validate BEFORE writing anything. A broken batch is retried by its exact
missing ids, one at a time -- never the whole batch of 20 again.
"""

import random

from app.config import get_settings
from app.exceptions import AppException, ErrorCode
from shared.llm import CATEGORIES, EnrichedChunk, LLMProvider
from shared.rate_limiter import RateLimited, get_bucket

_settings = get_settings()
MAX_CHUNKS_PER_DOC = _settings.max_chunks_per_doc
SOLO_ATTEMPTS = 2


def validate_batch(batch: list[dict], results: list[EnrichedChunk]) -> list[int]:
    """Returns the ids that need a retry. Three checks, none skipped."""
    wanted = {c["id"] for c in batch}
    seen: dict[int, EnrichedChunk] = {}
    for item in results:
        if item.id in wanted and item.id not in seen:
            seen[item.id] = item
    return sorted(wanted - set(seen))


def _normalise(item: EnrichedChunk) -> dict:
    """A category outside the enum is rejected, not stored, and logged as a
    proposal. Tolerating a free-form tag accumulates Finance/Financial/
    financial-reports within a week."""
    category = item.category if item.category in CATEGORIES else "OTHER"
    if category != item.category:
        print(f"[taxonomy] rejected off-enum proposal: {item.category!r}")
    return {"id": item.id, "context": item.context, "category": category}


def _acquire_or_defer(estimated_tokens: int) -> None:
    for bucket_name, cost in (("chat_rpm", 1), ("chat_tpm", estimated_tokens)):
        allowed, wait_ms = get_bucket(bucket_name).acquire(cost)
        if allowed:
            continue

        if wait_ms < 0:
            # Not "wait and it clears" but "no amount of waiting clears it":
            # the batch costs more than the bucket holds in total. Deferring
            # on this would compute a negative countdown, which Celery turns
            # into an ETA already in the past and so runs immediately --
            # with max_retries=None below, an endless tight loop on a
            # document that can never get through, and nothing logged.
            raise AppException(
                ErrorCode.RATE_LIMIT_UNSATISFIABLE,
                f"{bucket_name}: batch costs {cost}, more than the bucket's whole capacity",
            )

        # Requeue, do NOT sleep: a sleeping worker still holds its slot and
        # reports itself busy, leaving the pool at 100% utilised while doing
        # nothing. The jitter keeps a crowd of deferred tasks from waking up
        # in lockstep and colliding on the bucket again.
        raise RateLimited(countdown=(wait_ms / 1000) * random.uniform(1.0, 1.3))


def enrich_chunks(
    children: list[dict],
    provider: LLMProvider,
    batch_size: int = 20,
    done: list[dict] | None = None,
) -> list[dict]:
    """`done` is what an earlier, deferred run already paid for -- the
    `partial` of its RateLimited. Those chunks are skipped, not sent again."""
    if len(children) > MAX_CHUNKS_PER_DOC:
        raise AppException(
            ErrorCode.PDF_TOO_LARGE,
            f"{len(children)} chunks exceed the cap of {MAX_CHUNKS_PER_DOC}",
        )

    out: dict[int, dict] = {r["id"]: r for r in done or []}
    pending = [c for c in children if c["id"] not in out]

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        estimated_tokens = (
            sum(c["token_count"] for c in batch) if "token_count" in batch[0] else len(batch) * 150
        )
        try:
            _acquire_or_defer(estimated_tokens)
        except RateLimited as exc:
            exc.partial = list(out.values())
            raise

        results = provider.enrich(batch)
        wanted = {c["id"] for c in batch}
        for item in results:
            if item.id in wanted:
                out[item.id] = _normalise(item)

        for missing_id in validate_batch(batch, results):
            single = next(c for c in batch if c["id"] == missing_id)
            for _ in range(SOLO_ATTEMPTS):
                retry = provider.enrich([single])
                if retry and retry[0].id == missing_id:
                    out[missing_id] = _normalise(retry[0])
                    break
            else:
                # Failed alone twice: empty context, and move on.
                out[missing_id] = {"id": missing_id, "context": "", "category": "OTHER"}

    return [out[c["id"]] for c in children]
