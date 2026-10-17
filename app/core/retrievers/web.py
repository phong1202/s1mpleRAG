"""R8: Tavily, the last resort and the only untrusted input.

Not one of the retrievers `retrieve` runs side by side: the pipeline calls it
once, after the gate has found the corpus short twice (spec decision 13), and
what it returns only ever reaches `Context.untrusted` (§7).

Spoken to with httpx, as the reranker is, rather than through Tavily's SDK:
one POST, and the timeout and failure handling stay ours.
"""

import asyncio

import httpx

from app.core.contracts import Candidate, WebRef

_ENDPOINT = "https://api.tavily.com/search"


class WebSearchError(Exception):
    """No usable answer from the search. The pipeline answers from the
    corpus alone: a web search can only ever cost the web results."""


class WebRetriever:
    source = "web"

    def __init__(
        self,
        api_key: str,
        max_results: int,
        include_domains: tuple[str, ...],
        timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._max_results = max_results
        self._include_domains = list(include_domains)
        self._timeout = timeout_s
        self._transport = transport

    async def search(self, text: str) -> list[Candidate]:
        try:
            # httpx times each phase; this bounds the whole call.
            async with asyncio.timeout(self._timeout):
                payload = await self._post(text)
            return self._read(payload)
        except Exception as error:  # noqa: BLE001 -- see WebSearchError
            raise WebSearchError(f"{type(error).__name__}: {error}") from error

    async def _post(self, text: str) -> dict:
        body = {
            "query": text,
            "search_depth": "basic",
            "max_results": self._max_results,
            # Spec §7: Tavily's own model would write an answer that bypasses
            # the generator, the citations and the trace.
            "include_answer": False,
            # The excerpt Tavily matched is enough, and a whole page is a
            # whole page of someone else's text in the prompt.
            "include_raw_content": False,
        }
        if self._include_domains:
            body["include_domains"] = self._include_domains
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as http:
            response = await http.post(_ENDPOINT, json=body, headers=self._headers)
            response.raise_for_status()
            return response.json()

    def _read(self, payload: dict) -> list[Candidate]:
        candidates: list[Candidate] = []
        for result in payload["results"]:
            content = (result.get("content") or "").strip()
            if not content:
                continue
            candidates.append(
                Candidate(
                    source="web",
                    rank=len(candidates) + 1,
                    score=float(result.get("score") or 0.0),
                    text=content,
                    ref=WebRef(url=result["url"], title=result.get("title") or result["url"]),
                )
            )
        return candidates
