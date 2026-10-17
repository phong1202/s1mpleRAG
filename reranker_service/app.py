"""The CPU reranker: one cross-encoder behind the Cohere shape of /rerank.

The read path's client (app/core/rerank_client.py) speaks only this shape, so
this container and a hosted reranker are interchangeable by configuration.

The scorer is passed in, not imported: the tests run this app with a fake
scorer, and neither torch nor a model file is needed to run them.
"""

import asyncio
from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, StringConstraints

Scorer = Callable[[str, list[str]], list[float]]

# Chunks are a few hundred tokens, questions a sentence or two. Past these a
# request is not a rerank request, and it would hold the model for seconds.
Query = Annotated[str, StringConstraints(max_length=8_000)]
Document = Annotated[str, StringConstraints(max_length=20_000)]


class RerankRequest(BaseModel):
    model: str | None = None
    query: Query
    documents: list[Document] = Field(max_length=1000)
    top_n: int | None = Field(default=None, ge=1)


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]


def create_app(scorer: Scorer, model: str, max_waiting: int = 2) -> FastAPI:
    """`max_waiting`: requests allowed to queue behind the one being scored.

    One request is scored at a time -- the model already uses every core it
    is given. A request that would queue past `max_waiting` is refused at
    once with 503: its caller keeps its fused order straight away, instead of
    waiting out its timeout while this server computes an answer nobody will
    read.
    """
    app = FastAPI(title="reranker")
    lock = asyncio.Lock()
    waiting = 0

    # async, on the event loop: it never needs a worker thread, so it answers
    # while a request is being scored.
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": model}

    @app.post("/rerank")
    async def rerank(body: RerankRequest, request: Request) -> RerankResponse:
        nonlocal waiting
        if body.model and body.model != model:
            # Silently scoring with another model would make every eval
            # figure belong to a model nobody asked for.
            raise HTTPException(400, f"this server serves {model!r}, not {body.model!r}")
        if not body.documents:
            return RerankResponse(model=model, results=[])
        if waiting >= max_waiting:
            raise HTTPException(503, "busy: the caller should keep its own order")
        waiting += 1
        try:
            await lock.acquire()
        finally:
            waiting -= 1
        try:
            if await request.is_disconnected():
                # Its caller has given up; scoring it would only delay the
                # requests behind it.
                raise HTTPException(499, "client went away")
            scores = await run_in_threadpool(scorer, body.query, body.documents)
        finally:
            lock.release()
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        results = [RerankResult(index=i, relevance_score=scores[i]) for i in ranked]
        return RerankResponse(model=model, results=results[: body.top_n])

    return app
