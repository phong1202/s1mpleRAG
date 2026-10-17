"""R2 end to end, through POST /query: the flag decides whether keywords are
searched beside the vectors, and the trace says what each source contributed."""

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.core.generator import Draft
from app.models import QueryTrace
from tests.helpers import seed_texts

pytestmark = pytest.mark.asyncio


@pytest.fixture
def hybrid(monkeypatch):
    monkeypatch.setattr(get_settings(), "retrieval_bm25_enabled", True)


async def _trace_of(client, db_session, question: str) -> dict:
    await client.post("/query", json={"question": question})
    row = (await db_session.execute(select(QueryTrace))).scalar_one()
    return {node["node"]: node["detail"] for node in row.nodes}


async def test_with_the_flag_off_only_vectors_are_searched(
    query_client, db_session, seeded_corpus, monkeypatch
):
    """Off means R1 exactly. A before/after comparison is worth nothing if
    "before" quietly changed too."""
    # Set, not assumed: a developer's .env may have the flag on.
    monkeypatch.setattr(get_settings(), "retrieval_bm25_enabled", False)
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tai lieu [1]."))

    detail = await _trace_of(client, db_session, "child 3")

    assert set(detail["retrieve_detail"]["per_source"]) == {"vector"}
    assert "fuse_detail" not in detail


async def test_with_the_flag_on_both_sources_are_searched(
    query_client, db_session, seeded_corpus, hybrid
):
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tai lieu [1]."))

    detail = await _trace_of(client, db_session, "child 3")

    assert set(detail["retrieve_detail"]["per_source"]) == {"vector", "bm25"}
    assert sum(detail["fuse_detail"]["kept"].values()) == get_settings().retrieval_top_k


async def test_a_chunk_both_sources_agree_on_leads_the_context(
    query_client, db_session, seeded_corpus, hybrid
):
    """The stub's vectors carry no meaning, so every chunk gets a vector rank
    and none earns it. Only the chunk quoting the code has the keywords' vote
    as well, and two votes beat any single first place."""
    client, provider = query_client
    provider.script(Draft, Draft(answer="Theo tai lieu [1]."))
    await seed_texts(
        db_session,
        [
            "Mẫu số 01/TB-ĐKĐT dùng để thông báo tiếp nhận đăng ký.",
            "Mẫu số 04/SS-HĐĐT dùng để thông báo hóa đơn điện tử có sai sót.",
        ],
    )

    detail = await _trace_of(client, db_session, "04/SS-HĐĐT")

    first = detail["expand_detail"]["parents"][0]
    assert (first["file"], first["page"]) == ("notes.pdf", 2)
    assert detail["fuse_detail"]["kept"]["bm25+vector"] == 1
