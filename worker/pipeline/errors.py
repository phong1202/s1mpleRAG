"""One failure policy for all five stages, applied as a decorator under
`@app.task(bind=True)`.

It wraps the whole stage body -- imports included. An import failure is as
real a stage failure as anything else: outside the try it would bypass
stage_failed() entirely, leaving a document stuck at QUEUED with attempts=0
and no last_error, as once happened.
"""

import functools
import logging

import redis
from celery.exceptions import Ignore

from app.config import get_settings
from app.exceptions import AppException
from shared.rate_limiter import RateLimited
from worker.pipeline.outage import outage_seconds
from worker.pipeline.state import MAX_BACKOFF_S, stage_failed
from worker.repositories.documents import DocumentGone
from worker.steps.parsing import OcrUnavailable

logger = logging.getLogger(__name__)


def stage_task(stage: str):
    def decorate(body):
        @functools.wraps(body)
        def run(self, document_id: str) -> str:
            try:
                return _with_failure_policy(self, body, document_id, stage)
            except DocumentGone:
                # Outermost on purpose: the row can vanish mid-failure, and
                # stage_failed() then raises this from inside an except
                # handler, past every clause below. Ignore ends the chain
                # with no retry -- there is nothing left to process.
                logger.warning(
                    "document %s no longer exists; dropping its %s task", document_id, stage
                )
                raise Ignore() from None

        return run

    return decorate


def _with_failure_policy(self, body, document_id: str, stage: str) -> str:
    try:
        return body(self, document_id)
    except AppException as exc:
        stage_failed(document_id, stage, exc)
        raise  # a permanent error is never retried
    except RateLimited as exc:
        # Before `except Exception`, which would otherwise catch it: being
        # rate limited is not a failure. Nothing is recorded, and this is the
        # one and only retry published for it.
        raise self.retry(exc=exc, countdown=exc.countdown) from exc
    except OcrUnavailable as exc:
        # Also before `except Exception`, for the same reason: the server is
        # busy or restarting, which is not this document's failure -- on
        # 2026-10-06, three parses each burned an attempt on a server that
        # never went down. Only an outage that outlasts the ceiling starts
        # costing attempts, so a server that is not coming back still ends
        # in DEAD_LETTER rather than in deferrals forever.
        if _outage_seconds(document_id) < get_settings().ocr_outage_max_s:
            raise self.retry(exc=exc, countdown=exc.countdown) from exc
        _count_and_retry(self, document_id, stage, exc)
    except Exception as exc:
        _count_and_retry(self, document_id, stage, exc)


def _outage_seconds(document_id: str) -> float:
    """Without its clock, an outage cannot be told from a long one, so it is
    counted like any failure -- erring toward a document that eventually
    dead-letters, never toward one deferred forever."""
    try:
        return outage_seconds(document_id)
    except redis.RedisError:
        logger.warning("no outage clock for %s; counting the outage as a failure", document_id)
        return float("inf")


def _count_and_retry(self, document_id: str, stage: str, exc: Exception):
    """Called from inside an except block, so the bare `raise` re-raises the
    exception being handled."""
    if stage_failed(document_id, stage, exc):
        raise  # already DEAD_LETTER; do not ask Celery to retry too
    countdown = min(2**self.request.retries, MAX_BACKOFF_S)
    raise self.retry(exc=exc, countdown=countdown) from exc
