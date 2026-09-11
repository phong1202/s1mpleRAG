"""The HTTP edge of the read path. The LLM is injected rather than fetched, so
each test scripts the answer it needs."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.app import create_app
from app.core.generator import Draft
from app.utils.database import get_session
from shared.llm import AsyncStubProvider, get_async_provider
from tests.helpers import seed_chunks

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def query_client(db_session):
    """A client over the rolled-back session whose LLM is a scripted stub.

    `get_async_provider` is overridden, not monkeypatched: the route declares it
    with Depends, which is the only reason a test can replace it at all.
    """
    provider = AsyncStubProvider(dimensions=1536)
    app = create_app()

    async def _session():
        yield db_session

    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_async_provider] = lambda: provider

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, provider

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def seeded_corpus(db_session):
    return await seed_chunks(db_session)


async def test_query_returns_an_answer_with_citations(query_client, seeded_corpus):
    client, provider = query_client
    provider.script(Draft, Draft(answer="Lap hoa don dieu chinh [1]."))

    response = await client.post("/query", json={"question": "hoa don dien tu"})

    assert response.status_code == 200
    data = response.json()["data"]
    assert isinstance(data["answer"], str) and data["answer"]
    assert data["citations"][0]["kind"] == "document"
    assert data["citations"][0]["page_number"] >= 1
    assert data["citations"][0]["document_id"]
    assert data["latency_ms"] >= 0


async def test_an_empty_question_is_rejected_by_validation(query_client):
    client, _ = query_client

    response = await client.post("/query", json={"question": "   "})

    assert response.status_code == 422


async def test_a_question_with_no_matching_corpus_still_answers(query_client, seeded_corpus):
    """No results is not an error. The answer says so; the status stays 200."""
    client, provider = query_client
    provider.script(Draft, Draft(answer="Khong tim thay trong tai lieu."))

    response = await client.post(
        "/query", json={"question": "gi do", "category": "MARKETING"}
    )

    assert response.status_code == 200
    assert response.json()["data"]["citations"] == []


async def test_the_filters_reach_the_retrieval_query(query_client, seeded_corpus):
    """category and document_id are not decoration: they narrow the candidate
    pool inside the SQL. Dropping them on the way through would widen every
    filtered question to the whole corpus, silently."""
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tai lieu [1]."))

    matching = await client.post("/query", json={"question": "q", "category": "LEGAL"})
    other = await client.post("/query", json={"question": "q", "category": "MARKETING"})

    assert matching.json()["data"]["citations"] != []
    assert other.json()["data"]["citations"] == []
