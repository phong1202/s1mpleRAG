"""Phase 1's one end-to-end check that the stored data is USABLE, not just
stored. Wrong normalisation, a wrong dimension and a misconfigured index all
look exactly like "inserted successfully"; ranking real questions against
real embeddings is what tells them apart.

CANNOT run on the stub: its vectors carry no meaning, so which chunk comes
first is chance. Hence the real_llm marker -- without it, conftest swaps the
stub in whatever .env says.
"""

import hashlib
from pathlib import Path

import pytest
from sqlalchemy import text

from app.config import get_settings
from app.models import Document
from shared.llm import OpenAIProvider
from shared.storage import get_public_store, get_store
from worker.celery_app import app as celery_app
from worker.db import session_scope
from worker.stages import _ARTIFACTS, launch

pytestmark = [
    pytest.mark.real_llm,
    # Read through the settings, which load .env -- os.getenv would see
    # only exported variables, and stay None after the .env edit this test
    # asks for, skipping it forever without a word.
    pytest.mark.skipif(
        get_settings().llm_provider != "openai" or not get_settings().openai_api_key,
        reason="needs LLM_PROVIDER=openai and OPENAI_API_KEY in .env",
    ),
]

# One question per topic page of topics.pdf, with the keyword only that page
# contains. Written without diacritics, like the fixture, so that a failure
# points at the vectors and not at accent matching.
QUESTIONS = [
    ("doanh thu quy 3 nam 2024 dat bao nhieu", "doanh thu"),
    ("nhan vien duoc nghi bao nhieu ngay co luong moi nam", "nghi phep"),
    ("he thong may chu ngung hoat dong vao luc nao trong tuan", "bao tri"),
]


@pytest.fixture
def topics_document():
    """topics.pdf uploaded and seeded as a committed document, the chain run
    in-process -- it has to see the same test database this fixture writes
    to, which the broker-fed workers in the containers do not."""
    data = (Path(__file__).parent / "fixtures" / "topics.pdf").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    key = f"raw/{digest}.pdf"
    get_public_store().put(key, data)
    with session_scope() as session:
        document = Document(
            sha256_hash=digest, filename="topics.pdf", object_key=key, size_bytes=len(data)
        )
        session.add(document)
        session.flush()
        document_id = document.id

    celery_app.conf.task_always_eager = True
    yield document_id
    celery_app.conf.task_always_eager = False

    with session_scope() as session:
        session.delete(session.get(Document, document_id))
    for name in (n for names in _ARTIFACTS.values() for n in names):
        get_store().delete(f"staging/{document_id}/{name}")
    get_public_store().delete(key)


def test_each_question_ranks_its_own_topic_first(topics_document):
    launch(str(topics_document))

    with session_scope() as session:
        contents = (
            session.execute(
                text("SELECT content FROM child_chunks WHERE document_id = :d"),
                {"d": topics_document},
            )
            .scalars()
            .all()
        )
    assert contents, "no chunks at all -- the pipeline wrote nothing"
    # Without this the check below could pass for any ranking whatsoever.
    for _, keyword in QUESTIONS:
        assert sum(keyword in c.lower() for c in contents) == 1, f"{keyword!r} is not unique"

    provider = OpenAIProvider()
    for question, keyword in QUESTIONS:
        vector = provider.embed([question])[0]
        with session_scope() as session:
            # Through the HNSW index, not around it: with a handful of rows
            # Postgres would scan sequentially, and exact results would hide
            # an index that cannot serve this operator.
            session.execute(text("SET LOCAL enable_seqscan = off"))
            top = session.execute(
                text(
                    "SELECT content FROM child_chunks WHERE document_id = :d "
                    "ORDER BY embedding <#> CAST(:v AS vector) LIMIT 1"
                ),
                {"v": str(vector), "d": topics_document},
            ).scalar_one()
        assert keyword in top.lower(), (
            f"{question!r} did not rank its own topic first -- check, in order: L2 "
            "normalisation, EMBED_DIMENSIONS against vector(1536), and the HNSW opclass"
        )
