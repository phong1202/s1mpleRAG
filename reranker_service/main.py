"""uvicorn reranker_service.main:app -- loads the model once, at startup."""

from reranker_service import scorer
from reranker_service.app import create_app

_config = scorer.settings()
app = create_app(scorer.load(_config), model=_config["model"], max_waiting=_config["max_waiting"])
