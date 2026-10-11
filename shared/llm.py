"""The provider seam: one Protocol, two implementations.

Chosen two ways, deliberately:
  - the LLM_PROVIDER env var -- for a manual `docker compose up`
  - passed straight into a function (dependency injection) -- for tests,
    since the env var only turns the stub on or off, not what it returns.
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.config import get_settings
from app.exceptions import AppException, ErrorCode
from shared.rate_limiter import RateLimited

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
        self._reasoning_effort = settings.openai_reasoning_effort
        self._embed_model = settings.openai_embed_model
        self._dimensions = settings.embed_dimensions

    def enrich(self, chunks: list[dict]) -> list[EnrichedChunk]:
        prompt = (
            "For each chunk, write ONE sentence of context situating it, in the "
            "chunk's own language, and choose exactly one category."
        )
        # Only what the task reads: every other field of a chunk would be
        # tokens paid for and ignored.
        sent = [{"id": c["id"], "content": c["content"]} for c in chunks]
        response = _translating_errors(
            self._client.chat.completions.create,
            model=self._chat_model,
            reasoning_effort=self._reasoning_effort,
            response_format=_ENRICHED_SCHEMA,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(sent, ensure_ascii=False)},
            ],
        )
        return _parse_enriched(response.choices[0].message.content)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = _translating_errors(
            self._client.embeddings.create,
            model=self._embed_model,
            input=texts,
            dimensions=self._dimensions,
        )
        # By the index each item carries, not arrival order: a mismatch
        # there would file every vector under the wrong chunk, silently.
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


# Structured outputs, strict: the model can no longer return an off-enum
# category or an item missing a field -- JSON mode only promised valid JSON.
# _parse_enriched still checks, since the stub and future providers may not
# honour a schema.
_ENRICHED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "enriched_chunks",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "context": {"type": "string"},
                            "category": {"type": "string", "enum": sorted(CATEGORIES)},
                        },
                        "required": ["id", "context", "category"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["chunks"],
            "additionalProperties": False,
        },
    },
}


def _translating_errors(call, **kwargs):
    """Maps the provider's refusals onto what a stage knows how to handle.

    A 429 that outlasted the SDK's own retries means "wait", so it becomes
    RateLimited -- the stage defers without counting a failed attempt.
    Except insufficient_quota, which arrives as a 429 too but which no wait
    pays: that, a rejected key, a key without access and an unknown model
    are configuration, identical on every retry, so they fail permanently.
    """
    import openai

    try:
        return call(**kwargs)
    except openai.RateLimitError as exc:
        if exc.code == "insufficient_quota":
            raise AppException(ErrorCode.LLM_PROVIDER_REJECTED, f"quota exhausted: {exc}") from exc
        raise RateLimited(countdown=_retry_after_s(exc) * random.uniform(1.0, 1.3)) from exc
    except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as exc:
        raise AppException(ErrorCode.LLM_PROVIDER_REJECTED, str(exc)) from exc


def _retry_after_s(exc) -> float:
    """The provider's own estimate, when it sends one."""
    headers = exc.response.headers
    try:
        if "retry-after-ms" in headers:
            return float(headers["retry-after-ms"]) / 1000
        if "retry-after" in headers:
            return float(headers["retry-after"])
    except ValueError:
        pass
    return _DEFAULT_RETRY_AFTER_S


_DEFAULT_RETRY_AFTER_S = 10.0


def _parse_enriched(content: str | None) -> list[EnrichedChunk]:
    """Keeps every well-formed item and drops the rest, one by one. A dropped
    item is just a missing id to S3, which retries it alone; raising here
    instead would throw away the whole batch's good results with it.

    Invalid JSON -- typically a reply cut off at the token limit -- likewise
    yields nothing rather than an error: every id is then missing, and each
    is retried in a request small enough not to be cut off again."""
    try:
        payload = json.loads(content or "")
    except json.JSONDecodeError:
        return []
    items = payload.get("chunks") if isinstance(payload, dict) else None

    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            # Model JSON is untyped: "2" has to match the id 2 it names.
            chunk_id = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        context, category = item.get("context"), item.get("category")
        if isinstance(context, str) and isinstance(category, str):
            out.append(EnrichedChunk(id=chunk_id, context=context, category=category))
    return out


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

    def script(self, schema: type[T], *responses: T) -> None:
        """Queue responses for a schema after construction.

        The constructor copies its `responses` mapping, so a caller that builds
        the provider before it knows what the provider must return -- a FastAPI
        dependency override, for one -- has no other way in.
        """
        self._responses[schema] = list(responses)


class AsyncOpenAIProvider:
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._chat_model = settings.openai_chat_model
        self._reasoning_effort = settings.openai_reasoning_effort
        self._embed_model = settings.openai_embed_model
        self._dimensions = settings.embed_dimensions

    async def complete(self, messages: list[dict], schema: type[T]) -> T:
        # Structured outputs, strict -- the choice enrich() made above. JSON
        # mode never showed the model the schema, and the API refuses it
        # unless a message happens to say "json".
        response = await self._client.chat.completions.parse(
            model=self._chat_model,
            reasoning_effort=self._reasoning_effort,
            response_format=schema,
            messages=messages,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            # A refusal, or a reply cut off before the object closed.
            raise RuntimeError(f"the model returned no {schema.__name__}")
        return parsed

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
