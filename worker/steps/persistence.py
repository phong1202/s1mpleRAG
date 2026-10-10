"""S5 -- Persist. One transaction, sync Session. Always safe to rerun --
with the same input it changes nothing, and with changed input it leaves no
trace of the old run.

The only step that writes to the database, so the only one handed a
Session; the rows it builds are plain data, the writes go through
worker/repositories.
"""

import uuid
from collections import Counter

from sqlalchemy.orm import Session

from worker.repositories.chunks import ChunkRepository
from worker.repositories.documents import DocumentStateRepository
from worker.steps.embedding import contextualize


def persist_document(
    document_id: uuid.UUID,
    chunks: dict,
    enriched: list[dict],
    vectors: list[list[float]],
    session: Session,
) -> None:
    """`vectors` is row-aligned with chunks["children"]; the caller checks
    that against S4's manifest before getting here."""
    rows = ChunkRepository(session)
    rows.upsert_parents([{"document_id": document_id, **p} for p in chunks["parents"]])
    rows.upsert_children(
        _child_rows(document_id, chunks, enriched, vectors, rows.parent_ids(document_id))
    )
    rows.delete_leftovers(
        document_id,
        parent_indices=[p["chunk_index"] for p in chunks["parents"]],
        child_indices=[c["chunk_index"] for c in chunks["children"]],
    )
    DocumentStateRepository(session).complete(document_id, _dominant_language(chunks))


def _child_rows(
    document_id: uuid.UUID,
    chunks: dict,
    enriched: list[dict],
    vectors: list[list[float]],
    parent_ids: dict[int, uuid.UUID],
) -> list[dict]:
    # Indexed, not .get() with a fallback: S3 gives every child an entry, so
    # a missing one means the staged artifacts came from different runs.
    context_by_id = {e["id"]: e for e in enriched}
    rows = []
    for child, vector in zip(chunks["children"], vectors, strict=True):
        meta = context_by_id[child["chunk_index"]]
        rows.append(
            {
                "document_id": document_id,
                "parent_id": parent_ids[child["parent_index"]],
                "chunk_index": child["chunk_index"],
                "content": child["content"],
                "contextualized": contextualize(meta["context"], child["content"]),
                "page_number": child["page_number"],
                "token_count": child["token_count"],
                "embedding": vector,
                "category": meta["category"],
                # Inherited from the parent in S2. It picks the text search
                # config for the generated tsv column, so a NULL here builds
                # the keyword index with the wrong analyser and no error.
                "language": child["language"],
            }
        )
    return rows


def _dominant_language(chunks: dict) -> str | None:
    """By parent count. A bilingual document gets the majority side --
    filtering at document level is only a coarse hint; the per-chunk column
    is what retrieval actually reads."""
    languages = [p["language"] for p in chunks["parents"] if p["language"]]
    return Counter(languages).most_common(1)[0][0] if languages else None
