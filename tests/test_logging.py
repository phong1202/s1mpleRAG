import logging

from app.middleware.request_id import get_request_id, request_id_ctx
from app.utils.logging import configure_logging


def test_request_id_defaults_to_placeholder():
    assert get_request_id() == "-"


def test_request_id_reflects_context():
    token = request_id_ctx.set("abc-123")
    try:
        assert get_request_id() == "abc-123"
    finally:
        request_id_ctx.reset(token)


def test_configure_logging_sets_level_and_injects_request_id(caplog):
    configure_logging("DEBUG")
    logger = logging.getLogger("app.test")

    assert logger.getEffectiveLevel() == logging.DEBUG

    with caplog.at_level(logging.INFO, logger="app.test"):
        logger.info("hello")

    record = caplog.records[-1]
    assert record.request_id == "-"
