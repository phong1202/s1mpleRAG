"""The full chain, run on StubProvider. No OPENAI_API_KEY needed."""

import numpy as np
import pytest
from sqlalchemy import func, select, text

from app.models import ChildChunk, Document
from shared.llm import StubProvider
from shared.storage import get_store
from worker.celery_app import app as celery_app
from worker.db import session_scope
from worker.stages import launch


@pytest.fixture(autouse=True)
def eager():
    celery_app.conf.task_always_eager = True
    yield
    celery_app.conf.task_always_eager = False


def _fingerprint(document_id):
    """Row count plus one hash over every child's id, index, parent, text
    and vector: a rerun that rewrote anything at all changes it, which
    comparing counts alone would never catch."""
    with session_scope() as session:
        return session.execute(
            text(
                "SELECT count(*), md5(string_agg(id::text || chunk_index || parent_id::text"
                " || md5(contextualized) || md5(embedding::text), ',' ORDER BY chunk_index))"
                " FROM child_chunks WHERE document_id = :d"
            ),
            {"d": document_id},
        ).one()


class _Calls:
    """Counts how often each stage really did its work, rather than taking
    its checkpoint skip."""

    def __init__(self, monkeypatch):
        import worker.chunking
        import worker.embedding
        import worker.enrichment
        import worker.parsing

        self.counts = {}
        for module, name in (
            (worker.parsing, "parse_document"),
            (worker.chunking, "chunk_document"),
            (worker.enrichment, "enrich_chunks"),
            (worker.embedding, "embed_chunks"),
        ):
            monkeypatch.setattr(module, name, self._counting(name, getattr(module, name)))

    def _counting(self, name, real):
        def wrapper(*args, **kwargs):
            self.counts[name] = self.counts.get(name, 0) + 1
            return real(*args, **kwargs)

        return wrapper


def test_full_chain_produces_embedded_chunks(seeded_document):
    launch(str(seeded_document.id))

    count, _ = _fingerprint(seeded_document.id)
    assert count > 0

    with session_scope() as session:
        doc = session.get(Document, seeded_document.id)
        assert doc.status == "COMPLETED"
        # What Phase 2 builds on: null here after a full run means S1 or S2
        # skipped it in silence.
        assert doc.title is not None
        assert doc.language is not None
        dims = session.scalars(
            select(func.distinct(func.vector_dims(ChildChunk.embedding))).where(
                ChildChunk.document_id == seeded_document.id
            )
        ).all()
    assert dims == [1536]


def test_running_the_whole_chain_twice_changes_nothing(seeded_document):
    launch(str(seeded_document.id))
    before = _fingerprint(seeded_document.id)

    launch(str(seeded_document.id))

    assert _fingerprint(seeded_document.id) == before


def test_deleting_a_checkpoint_reruns_that_stage_and_downstream_only(seeded_document, monkeypatch):
    """Delete enriched.json: S1 and S2 skip on their checkpoints, S3 runs
    again -- and so does S4, whose input just changed. The end result is
    identical."""
    launch(str(seeded_document.id))
    before = _fingerprint(seeded_document.id)

    get_store().delete(f"staging/{seeded_document.id}/enriched.json")
    calls = _Calls(monkeypatch)
    launch(str(seeded_document.id))

    assert calls.counts == {"enrich_chunks": 1, "embed_chunks": 1}
    assert _fingerprint(seeded_document.id) == before


def test_a_regenerated_artifact_forces_everything_downstream_to_follow(
    seeded_document, monkeypatch
):
    """A real LLM does not answer the same way twice. Regenerate enriched.json
    with different contexts, and S4 used to skip anyway on its old
    embeddings.npy -- leaving S5 to store the NEW contextualized text beside
    the vector of the OLD one. Every stored vector must be the embedding of
    the text stored next to it."""
    import shared.llm

    launch(str(seeded_document.id))

    class RevisedStub(StubProvider):
        def enrich(self, chunks):
            return [
                type(item)(id=item.id, context="Revised: " + item.context, category=item.category)
                for item in super().enrich(chunks)
            ]

    monkeypatch.setattr(shared.llm, "get_provider", lambda: RevisedStub())
    get_store().delete(f"staging/{seeded_document.id}/enriched.json")
    launch(str(seeded_document.id))

    with session_scope() as session:
        rows = session.execute(
            select(ChildChunk.contextualized, ChildChunk.embedding)
            .where(ChildChunk.document_id == seeded_document.id)
            .order_by(ChildChunk.chunk_index)
        ).all()
    assert rows
    assert all(ctx.startswith("Revised: ") for ctx, _ in rows)
    for contextualized, stored in rows:
        expected = StubProvider().embed([contextualized])[0]
        assert np.allclose(np.asarray(stored), expected, atol=1e-6), (
            "stored vector is not the embedding of the stored text"
        )
