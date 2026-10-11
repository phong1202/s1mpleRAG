# tests/test_async_llm_provider.py
"""The stub is what makes the whole R0-R8 ladder testable without an API key.
It is scripted per schema type, so a test can say 'the gate says INSUFFICIENT
once, then SUFFICIENT' without touching the network."""

import pytest
from pydantic import BaseModel

from shared.llm import AsyncStubProvider

pytestmark = pytest.mark.asyncio


class Verdict(BaseModel):
    sufficient: bool


async def test_stub_returns_scripted_responses_in_order():
    provider = AsyncStubProvider(
        responses={Verdict: [Verdict(sufficient=False), Verdict(sufficient=True)]}
    )

    first = await provider.complete([{"role": "user", "content": "?"}], Verdict)
    second = await provider.complete([{"role": "user", "content": "?"}], Verdict)

    assert first.sufficient is False
    assert second.sufficient is True


async def test_stub_repeats_its_last_response_when_the_script_runs_out():
    """A pipeline under test may call a node more times than the test scripted.
    Repeating beats raising: the test then fails on the behaviour it is about,
    not on the stub's bookkeeping."""
    provider = AsyncStubProvider(responses={Verdict: [Verdict(sufficient=True)]})

    for _ in range(3):
        assert (await provider.complete([], Verdict)).sufficient is True


async def test_stub_raises_for_an_unscripted_schema():
    """Silence here would let a test pass while a node it never scripted
    returned a default."""
    provider = AsyncStubProvider(responses={})

    with pytest.raises(KeyError):
        await provider.complete([], Verdict)


async def test_stub_embeddings_are_deterministic_and_normalised():
    """vector_ip_ops assumes unit vectors. A stub that returns un-normalised
    vectors would hide a bug that only appears against the real index."""
    provider = AsyncStubProvider(dimensions=8)

    first = await provider.embed_query("hoa don dien tu")
    second = await provider.embed_query("hoa don dien tu")

    assert first == second
    assert len(first) == 8
    assert abs(sum(x * x for x in first) - 1.0) < 1e-6


async def test_script_replaces_the_queue_rather_than_appending_to_it():
    """Two calls for the same schema must not leave the first response in front
    of the second: a test that scripts an answer would get the previous test's."""
    provider = AsyncStubProvider()
    provider.script(Verdict, Verdict(sufficient=False))
    provider.script(Verdict, Verdict(sufficient=True))

    assert (await provider.complete([], Verdict)).sufficient is True


def _async_openai_returning(monkeypatch, parsed=None, requests=None):
    """An AsyncOpenAIProvider whose client is a stand-in that records each
    request and answers with `parsed`, the way the SDK's parse() would. The
    async twin of `_openai_returning` in test_llm_provider.py."""
    from types import SimpleNamespace

    from app.config import get_settings
    from shared.llm import AsyncOpenAIProvider

    async def parse(**kwargs):
        if requests is not None:
            requests.append(kwargs)
        message = SimpleNamespace(parsed=parsed)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    monkeypatch.setattr(get_settings(), "openai_api_key", "sk-test")
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **_: client)
    return AsyncOpenAIProvider()


async def test_complete_asks_for_schema_bound_output_on_the_configured_model(monkeypatch):
    """JSON mode never showed the model the schema, and the API refuses it
    unless a message happens to say "json". Structured outputs bind the reply
    to the schema -- the choice enrich() already made on the write path."""
    from app.config import get_settings

    requests: list[dict] = []
    provider = _async_openai_returning(monkeypatch, Verdict(sufficient=True), requests)

    verdict = await provider.complete([{"role": "user", "content": "?"}], Verdict)

    (request,) = requests
    settings = get_settings()
    assert verdict.sufficient is True
    assert request["response_format"] is Verdict
    assert request["model"] == settings.openai_chat_model
    assert request["reasoning_effort"] == settings.openai_reasoning_effort


async def test_a_reply_with_no_object_raises_instead_of_returning_none(monkeypatch):
    """A refusal, or a reply cut off before the object closed, parses to None.
    Returned as it is, it would surface two calls later as an AttributeError
    with no hint of the cause."""
    provider = _async_openai_returning(monkeypatch, parsed=None)

    with pytest.raises(RuntimeError, match="Verdict"):
        await provider.complete([{"role": "user", "content": "?"}], Verdict)


async def test_openai_without_a_key_fails_at_construction(monkeypatch):
    from app.config import get_settings
    from shared.llm import AsyncOpenAIProvider

    monkeypatch.setattr(get_settings(), "openai_api_key", None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        AsyncOpenAIProvider()
