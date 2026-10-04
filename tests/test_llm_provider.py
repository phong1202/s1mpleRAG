"""The provider has two implementations. Stub exists NOT to save money --
the real cost is a few cents -- but because S3's acceptance criterion is
"return 19 of 20, retry exactly the missing id", and there is no way to make
a real API do that on demand.
"""

import json

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


def _openai_returning(
    monkeypatch, chat_content=None, embedding_data=None, raises=None, requests=None
):
    """An OpenAIProvider whose client is a stand-in returning canned
    responses -- or raising `raises` -- and recording each request's
    arguments into `requests`: what the provider does with them is the
    point, and a real model cannot be made to misbehave on demand."""
    from types import SimpleNamespace

    from app.config import get_settings
    from shared.llm import OpenAIProvider

    def answering(response):
        def create(**kwargs):
            if requests is not None:
                requests.append(kwargs)
            if raises is not None:
                raise raises
            return response

        return create

    message = SimpleNamespace(content=chat_content)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=answering(SimpleNamespace(choices=[SimpleNamespace(message=message)]))
            )
        ),
        embeddings=SimpleNamespace(create=answering(SimpleNamespace(data=embedding_data))),
    )
    monkeypatch.setattr(get_settings(), "openai_api_key", "sk-test")
    monkeypatch.setattr("openai.OpenAI", lambda **_: client)
    return OpenAIProvider()


def _status_error(cls, status, code, headers=None):
    """A real openai SDK error, built the way the SDK builds one."""
    import httpx2

    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx2.Response(status, headers=headers or {}, request=request)
    return cls("rejected", response=response, body={"code": code, "message": "rejected"})


def test_one_malformed_item_drops_only_that_item(monkeypatch, chunks):
    """S3 retries a batch by its exact missing ids. One item without a
    category used to raise TypeError for the whole batch instead -- every
    good result in it thrown away, the whole stage retried."""
    payload = (
        '{"chunks": [{"id": 0, "context": "a", "category": "TECHNICAL"},'
        ' {"id": 1, "context": "b"}]}'
    )
    provider = _openai_returning(monkeypatch, chat_content=payload)

    assert [r.id for r in provider.enrich(chunks)] == [0]


def test_a_string_id_is_read_as_the_int_it_names(monkeypatch, chunks):
    """JSON from a model is not typed: "2" for 2 would otherwise never match
    the id that was asked about, and get retried for nothing."""
    payload = '{"chunks": [{"id": "2", "context": "c", "category": "LEGAL"}]}'
    provider = _openai_returning(monkeypatch, chat_content=payload)

    assert provider.enrich(chunks)[0].id == 2


def test_truncated_json_yields_nothing_rather_than_raising(monkeypatch, chunks):
    """A reply cut off at the token limit is invalid JSON. Every id is then
    simply missing, and each is retried alone -- small enough requests not
    to be cut off again."""
    provider = _openai_returning(monkeypatch, chat_content='{"chunks": [{"id": 0, "cont')

    assert provider.enrich(chunks) == []


def test_embeddings_are_returned_in_input_order(monkeypatch):
    """Each item carries its input index. Trusting arrival order instead
    would, if it ever differed, file every vector under the wrong chunk --
    with no error anywhere."""
    from types import SimpleNamespace

    data = [
        SimpleNamespace(index=1, embedding=[0.0, 1.0]),
        SimpleNamespace(index=0, embedding=[1.0, 0.0]),
    ]
    provider = _openai_returning(monkeypatch, embedding_data=data)

    assert provider.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_a_429_is_a_wait_of_the_providers_choosing_not_a_failure(monkeypatch, chunks):
    """Our limiter reserves an estimate; the provider's quota is the truth,
    and other consumers of the same key spend it too. A 429 that gets past
    the SDK's own retries means "wait" -- it used to count as a failed
    attempt, three of which dead-lettered a healthy document."""
    import openai

    from shared.rate_limiter import RateLimited

    error = _status_error(
        openai.RateLimitError, 429, "rate_limit_exceeded", {"retry-after-ms": "2000"}
    )
    provider = _openai_returning(monkeypatch, raises=error)

    with pytest.raises(RateLimited) as exc:
        provider.enrich(chunks)

    assert 2.0 <= exc.value.countdown <= 2.6


def test_an_exhausted_quota_is_permanent_not_a_wait(monkeypatch, chunks):
    """insufficient_quota also arrives as a 429, but no amount of waiting
    pays the bill: deferring on it would retry forever."""
    import openai

    from app.exceptions import AppException, ErrorCode

    error = _status_error(openai.RateLimitError, 429, "insufficient_quota")
    provider = _openai_returning(monkeypatch, raises=error)

    with pytest.raises(AppException) as exc:
        provider.embed(["a"])

    assert exc.value.error is ErrorCode.LLM_PROVIDER_REJECTED


@pytest.mark.parametrize(
    ("cls", "status"),
    [("AuthenticationError", 401), ("PermissionDeniedError", 403), ("NotFoundError", 404)],
)
def test_a_rejected_key_or_model_is_permanent_not_retried(monkeypatch, chunks, cls, status):
    """A wrong key, a key without access, a misspelled model: configuration,
    identical on every retry. They used to be retried as transient errors."""
    import openai

    from app.exceptions import AppException, ErrorCode

    provider = _openai_returning(
        monkeypatch, raises=_status_error(getattr(openai, cls), status, None)
    )

    with pytest.raises(AppException) as exc:
        provider.enrich(chunks)

    assert exc.value.error is ErrorCode.LLM_PROVIDER_REJECTED


def test_enrich_asks_for_schema_bound_output_on_the_configured_model(monkeypatch, chunks):
    """Structured outputs with the category enum in the schema: the model
    cannot emit an off-enum category or a malformed item at all, where JSON
    mode only promised valid JSON. Only id and content go out -- the rest of
    a chunk's fields would be paid-for tokens the task does not use."""
    from app.config import get_settings

    requests: list[dict] = []
    provider = _openai_returning(monkeypatch, chat_content='{"chunks": []}', requests=requests)

    provider.enrich([{**c, "token_count": 9, "page_number": 1} for c in chunks])

    (request,) = requests
    settings = get_settings()
    assert request["model"] == settings.openai_chat_model
    assert request["reasoning_effort"] == settings.openai_reasoning_effort
    schema = request["response_format"]
    assert schema["type"] == "json_schema" and schema["json_schema"]["strict"] is True
    item = schema["json_schema"]["schema"]["properties"]["chunks"]["items"]
    assert item["properties"]["category"]["enum"] == sorted(CATEGORIES)
    sent = json.loads(request["messages"][-1]["content"])
    assert {key for chunk in sent for key in chunk} == {"id", "content"}


def test_the_defaults_are_the_current_models():
    """gpt-4o-mini and text-embedding-3-small were the defaults; the current
    small reasoning model and the multilingual-stronger embedding replace
    them (MIRACL 54.9 vs 44.0 -- the corpus is Vietnamese)."""
    from app.config import Settings

    fields = Settings.model_fields
    assert fields["openai_chat_model"].default == "gpt-6-luna"
    assert fields["openai_embed_model"].default == "text-embedding-3-large"
    assert fields["openai_reasoning_effort"].default == "low"
