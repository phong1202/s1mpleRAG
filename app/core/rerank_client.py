"""R3: the one client the read path uses to rerank, whoever does the ranking.

It speaks the Cohere shape of /rerank -- {model, query, documents, top_n} in,
results[{index, relevance_score}] out -- which the repo's own reranker
service, OpenRouter, Cohere and vLLM all accept. Moving from the CPU container
to a hosted model is a change of RERANKER_URL, RERANKER_MODEL and
RERANKER_API_KEY, not of code.
"""

import asyncio
import math

import httpx


class RerankError(Exception):
    """The reranker gave no answer that can be trusted. The caller keeps the
    order it already had: a failed rerank must never fail the request."""


class RerankClient:
    def __init__(
        self,
        url: str,
        model: str | None,
        api_key: str | None,
        timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._endpoint = f"{url.rstrip('/')}/rerank"
        # None lets the server choose: the CPU container serves the one model
        # it was built with. Hosted rerankers need it named.
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout_s
        self._transport = transport
        # What the last answer said it was scored by, for the trace and the eval.
        self.served_model: str | None = model

    async def scores(self, query: str, documents: list[str]) -> list[float]:
        """One score per document, in the order the documents were given.

        Any failure at all is a RerankError: a URL httpx cannot parse, a
        number too large for a float, a deadline passed. The caller's promise
        is that a reranker can only ever cost the rerank."""
        if not documents:
            return []
        try:
            # httpx times each phase and each read; this bounds the whole call,
            # which is what the setting promises.
            async with asyncio.timeout(self._timeout):
                payload = await self._post(query, documents)
            return self._read(payload, len(documents))
        except RerankError:
            raise
        except Exception as error:  # noqa: BLE001 -- see the docstring
            raise RerankError(f"{type(error).__name__}: {error}") from error

    async def _post(self, query: str, documents: list[str]) -> dict:
        body = {"query": query, "documents": documents, "top_n": len(documents)}
        if self._model:
            body["model"] = self._model
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as http:
            response = await http.post(self._endpoint, json=body, headers=self._headers)
            response.raise_for_status()
            return response.json()

    def _read(self, payload: dict, expected: int) -> list[float]:
        results = payload["results"]
        by_index: dict[int, float] = {}
        for result in results:
            index, score = result["index"], float(result["relevance_score"])
            # bool is an int to Python; 0.9 would truncate to 0. Either puts a
            # score on the wrong chunk.
            if type(index) is not int:
                raise RerankError(f"index {index!r} is not an integer")
            # NaN sorts inconsistently and Postgres refuses it in the trace.
            if not math.isfinite(score):
                raise RerankError(f"score {score!r} for index {index} is not a finite number")
            by_index[index] = score
        # Results come back best first, and a server may drop or repeat one.
        if len(results) != expected or sorted(by_index) != list(range(expected)):
            raise RerankError(
                f"expected one score for each of {expected} documents, got {len(results)}"
            )
        self.served_model = payload.get("model") or self._model
        return [by_index[i] for i in range(expected)]
