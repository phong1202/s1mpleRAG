"""The provider seam: one Protocol, two implementations.

Chosen two ways, deliberately:
  - the LLM_PROVIDER env var -- for a manual `docker compose up`
  - passed straight into a function (dependency injection) -- for tests,
    since the env var only turns the stub on or off, not what it returns.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.config import get_settings

CATEGORIES = frozenset(
    {"FINANCIAL", "LEGAL", "TECHNICAL", "MARKETING", "HR", "RESEARCH", "OPERATIONS", "OTHER"}
)


@dataclass(frozen=True)
class EnrichedChunk:
    id: int
    context: str
    category: str


class LLMProvider(Protocol):
    def enrich(self, chunks: list[dict]) -> list[EnrichedChunk]: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _unit_vector_from(text: str, dimensions: int) -> list[float]:
    """A deterministic, L2-normalised fake vector.

    Normalising here too is deliberate: if the stub returned an
    un-normalised vector, S4's normalisation assertion would fail whenever
    the suite runs on the stub, and the assertion -- the one that has to
    survive into production -- would get deleted to make the stub pass.
    """
    digest = hashlib.sha256(text.encode()).digest()
    raw = [(digest[i % len(digest)] - 128) / 128 for i in range(dimensions)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


class StubProvider:
    """Makes no network call. Can be told to fail in exactly the way a test
    needs it to."""

    def __init__(
        self,
        drop_ids: list[int] | None = None,
        bad_category_ids: list[int] | None = None,
    ) -> None:
        self.drop_ids = set(drop_ids or [])
        self.bad_category_ids = set(bad_category_ids or [])

    def enrich(self, chunks: list[dict]) -> list[EnrichedChunk]:
        out = []
        for chunk in chunks:
            cid = chunk["id"]
            if cid in self.drop_ids:
                continue
            category = "NOT_A_REAL_CATEGORY" if cid in self.bad_category_ids else "TECHNICAL"
            out.append(
                EnrichedChunk(
                    id=cid,
                    context=f"This chunk is about subject number {cid}.",
                    category=category,
                )
            )
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        dimensions = get_settings().embed_dimensions
        return [_unit_vector_from(t, dimensions) for t in texts]


class OpenAIProvider:
    def __init__(self) -> None:
        from openai import OpenAI

        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._chat_model = settings.openai_chat_model
        self._embed_model = settings.openai_embed_model
        self._dimensions = settings.embed_dimensions

    def enrich(self, chunks: list[dict]) -> list[EnrichedChunk]:
        prompt = (
            "For each chunk, write ONE sentence of context and choose exactly one "
            "category from: " + ", ".join(sorted(CATEGORIES)) + ". "
            'Return JSON {"chunks":[{"id":int,"context":str,"category":str}]}.'
        )
        response = self._client.chat.completions.create(
            model=self._chat_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(chunks, ensure_ascii=False)},
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        return [EnrichedChunk(**c) for c in payload["chunks"]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._embed_model, input=texts, dimensions=self._dimensions
        )
        return [item.embedding for item in response.data]


def get_provider() -> LLMProvider:
    return OpenAIProvider() if get_settings().llm_provider == "openai" else StubProvider()


# --- read path ---------------------------------------------------------------
# Added beside the synchronous half above, never replacing it: the worker
# imports LLMProvider, StubProvider and OpenAIProvider, and Phase 1's tests pin
# their behaviour. `_unit_vector_from` is reused rather than redefined.

T = TypeVar("T", bound=BaseModel)


class AsyncLLMProvider(Protocol):
    """The read path's provider. Async because it runs inside FastAPI, and
    schema-driven because every decision node returns a typed verdict rather
    than prose."""

    async def complete(self, messages: list[dict], schema: type[T]) -> T: ...
    async def embed_query(self, text: str) -> list[float]: ...


class AsyncStubProvider:
    """Deterministic, offline, scripted per schema type. This is what makes the
    whole R0-R8 ladder testable with no API key."""

    def __init__(
        self,
        responses: dict[type, list] | None = None,
        dimensions: int = 1536,
    ) -> None:
        self._responses = {k: list(v) for k, v in (responses or {}).items()}
        self._dimensions = dimensions
        self.calls: list[tuple[type, list[dict]]] = []

    async def complete(self, messages: list[dict], schema: type[T]) -> T:
        self.calls.append((schema, messages))
        # KeyError is deliberate: an unscripted schema must fail loudly rather
        # than hand back a default a test would silently pass against.
        queue = self._responses[schema]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    async def embed_query(self, text: str) -> list[float]:
        return _unit_vector_from(text, self._dimensions)


class AsyncOpenAIProvider:
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._chat_model = settings.openai_chat_model
        self._embed_model = settings.openai_embed_model
        self._dimensions = settings.embed_dimensions

    async def complete(self, messages: list[dict], schema: type[T]) -> T:
        response = await self._client.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return schema.model_validate(json.loads(response.choices[0].message.content))

    async def embed_query(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._embed_model, input=[text], dimensions=self._dimensions
        )
        vector = response.data[0].embedding
        # L2-normalise: vector_ip_ops treats inner product as cosine, and the
        # index returns wrong neighbours, with no error, if this is skipped.
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


def get_async_provider() -> AsyncLLMProvider:
    if get_settings().llm_provider == "openai":
        return AsyncOpenAIProvider()
    return AsyncStubProvider(dimensions=get_settings().embed_dimensions)
