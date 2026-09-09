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
