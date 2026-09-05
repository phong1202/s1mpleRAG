"""The provider has two implementations. Stub exists NOT to save money --
the real cost is a few cents -- but because S3's acceptance criterion is
"return 19 of 20, retry exactly the missing id", and there is no way to make
a real API do that on demand.
"""

import pytest

from shared.llm import CATEGORIES, StubProvider, get_provider


@pytest.fixture
def chunks():
    return [{"id": i, "content": f"paragraph number {i}"} for i in range(5)]


def test_stub_returns_one_result_per_chunk(chunks):
    assert len(StubProvider().enrich(chunks)) == len(chunks)


def test_stub_is_deterministic(chunks):
    assert StubProvider().enrich(chunks) == StubProvider().enrich(chunks)


def test_stub_only_emits_categories_from_the_closed_enum(chunks):
    assert all(c.category in CATEGORIES for c in StubProvider().enrich(chunks))


def test_stub_can_be_told_to_drop_ids(chunks):
    """This is the whole reason the stub exists: reproduce exactly the
    failure S3 has to handle."""
    result = StubProvider(drop_ids=[2]).enrich(chunks)

    assert len(result) == 4
    assert 2 not in {c.id for c in result}


def test_stub_can_be_told_to_emit_an_off_enum_category(chunks):
    result = StubProvider(bad_category_ids=[1]).enrich(chunks)
    offender = next(c for c in result if c.id == 1)

    assert offender.category not in CATEGORIES


def test_stub_embeddings_have_the_right_shape_and_are_normalised():
    vectors = StubProvider().embed(["a", "b"])

    assert len(vectors) == 2
    assert all(len(v) == 1536 for v in vectors)
    for v in vectors:
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-6, "the stub must normalise too, or it hides a real bug"


def test_stub_embeddings_differ_for_different_text():
    """A stub that ignored its input would still pass every test above."""
    a, b = StubProvider().embed(["alpha", "something completely different"])

    assert a != b


def test_get_provider_returns_stub_by_default(db_env):
    assert isinstance(get_provider(), StubProvider)


def test_an_unknown_provider_name_fails_at_startup_not_silently(db_env, monkeypatch):
    """LLM_PROVIDER picks between real embeddings and normalised-noise ones.
    A typo here must not quietly downgrade a production run to the stub --
    it has to fail loud, the way the rest of this settings module does."""
    from pydantic import ValidationError

    from app.config import Settings

    monkeypatch.setenv("LLM_PROVIDER", "opneai")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
