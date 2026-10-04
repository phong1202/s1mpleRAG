"""One failure policy for all five stages, applied as a decorator under
`@app.task(bind=True)`.

It wraps the whole stage body -- imports included. An import failure is as
real a stage failure as anything else: outside the try it would bypass
stage_failed() entirely, leaving a document stuck at QUEUED with attempts=0
and no last_error, as once happened.
"""

import functools

from app.exceptions import AppException
from shared.rate_limiter import RateLimited
from worker.pipeline.state import MAX_BACKOFF_S, stage_failed


def stage_task(stage: str):
    def decorate(body):
        @functools.wraps(body)
        def run(self, document_id: str) -> str:
            try:
                return body(self, document_id)
            except AppException as exc:
                stage_failed(document_id, stage, exc)
                raise  # a permanent error is never retried
            except RateLimited as exc:
                # Before `except Exception`, which would otherwise catch it:
                # being rate limited is not a failure. Nothing is recorded,
                # and this is the one and only retry published for it.
                raise self.retry(exc=exc, countdown=exc.countdown) from exc
            except Exception as exc:
                if stage_failed(document_id, stage, exc):
                    raise  # already DEAD_LETTER; do not ask Celery to retry too
                countdown = min(2**self.request.retries, MAX_BACKOFF_S)
                raise self.retry(exc=exc, countdown=countdown) from exc

        return run

    return decorate
