from celery import Celery

from app.config import get_settings

settings = get_settings()

app = Celery("rag", broker=settings.rabbitmq_url)
app.conf.update(
    result_backend=None,  # Postgres is the source of truth for status
    task_acks_late=True,  # crash -> redelivered, not lost
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # don't hoard tasks behind a slow stage
    task_routes={
        "worker.stages.parse": {"queue": "cpu"},
        "worker.stages.structure": {"queue": "cpu"},
        "worker.stages.enrich": {"queue": "llm"},
        "worker.stages.embed": {"queue": "llm"},
        "worker.stages.persist": {"queue": "cpu"},
    },
)
# related_name defaults to "tasks" -- it would look for worker/tasks.py,
# which does not exist. Our tasks live in worker/stages.py, so the name
# has to be given explicitly, or a standalone `celery worker` process
# never imports the module at all and registers zero tasks; only pytest
# runs happened to work, because the test files import worker.stages
# themselves.
app.autodiscover_tasks(["worker"], related_name="stages")
