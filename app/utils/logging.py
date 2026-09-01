import logging
from logging.config import dictConfig

from app.middleware.request_id import get_request_id


class RequestIDFilter(logging.Filter):
    """Attaches the current request id to every record, so one request's
    lines can be correlated in the log.

    Defers to an id already present on the record (e.g. supplied via
    `extra={"request_id": ...}`) instead of overwriting it. This matters for
    the generic Exception handler in app/exceptions/handlers.py: it runs in
    ServerErrorMiddleware, outside RequestIDMiddleware, after that
    middleware's ContextVar has already been reset — so get_request_id()
    would return "-" there. That handler passes the real id explicitly via
    `extra`; without this guard, this filter would clobber it back to "-".
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id()
        return True


def configure_logging(level: str = "INFO") -> None:
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "request_id": {"()": RequestIDFilter},
            },
            "formatters": {
                "default": {
                    "format": (
                        "%(asctime)s | %(levelname)-8s | %(name)s | [%(request_id)s] | %(message)s"
                    ),
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["request_id"],
                },
            },
            # The root logger is attached to below, imperatively, instead of
            # via a "root" key here: dictConfig replaces a logger's handlers
            # wholesale, which would silently detach any handler already on
            # the root logger (e.g. pytest's `caplog` capture handler during
            # tests) and break log capture for anything configured earlier.
            "loggers": {
                # Align uvicorn's output with the application's so the two
                # do not look like separate programs.
                "uvicorn": {"handlers": ["console"], "level": level, "propagate": False},
                "uvicorn.error": {"handlers": ["console"], "level": level, "propagate": False},
                "uvicorn.access": {"handlers": ["console"], "level": level, "propagate": False},
            },
        }
    )

    root = logging.getLogger()
    root.setLevel(level)
    console_handler = logging.getHandlerByName("console")
    # dictConfig builds a brand new "console" handler object on every call,
    # so identity checks against the old one never match. Remove any handler
    # this function previously attached (tagged by the same .name, which
    # dictConfig sets from the "handlers" key) before adding the new one —
    # this keeps repeated calls idempotent without touching handlers this
    # function did not add (e.g. pytest's caplog handler).
    for existing in list(root.handlers):
        if existing is not console_handler and getattr(existing, "name", None) == "console":
            root.removeHandler(existing)
    if console_handler is not None and console_handler not in root.handlers:
        root.addHandler(console_handler)
