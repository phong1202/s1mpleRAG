"""Re-runs S2 to S5 for documents that are already ingested.

    python -m worker.reingest --all
    python -m worker.reingest <document_id> [<document_id> ...]
    python -m worker.reingest --all --restore-from data/parsed-backup
    python -m worker.reingest --force <document_id>

--restore-from puts back parse checkpoints that storage has expired, from
files named <document_id>.parsed.json.

--force relaunches a document that is not finished. It is for a document
stuck in a running status because its task was lost. Make sure nothing is
running for it first: two chains over one document corrupt its chunks.
"""

import argparse
import pathlib

from sqlalchemy import select

from app.models import Document
from shared.storage import get_store
from worker.db import session_scope
from worker.stages import relaunch_from_structure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_ids", nargs="*")
    parser.add_argument("--all", action="store_true", help="every completed document")
    parser.add_argument("--restore-from", type=pathlib.Path)
    parser.add_argument("--force", action="store_true", help="also a document not finished")
    args = parser.parse_args()

    ids = list(args.document_ids)
    if args.all:
        with session_scope() as session:
            done = select(Document.id).where(Document.completed_at.is_not(None))
            ids = [str(row) for row in session.scalars(done)]

    store = get_store()
    failed: list[str] = []
    for document_id in ids:
        key = f"staging/{document_id}/parsed.json"
        backup = args.restore_from / f"{document_id}.parsed.json" if args.restore_from else None
        if not store.exists(key) and backup and backup.exists():
            store.put(key, backup.read_bytes())
            print(f"restored   {document_id}")
        try:
            relaunch_from_structure(document_id, force=args.force)
        except Exception as exc:
            # One document that cannot be relaunched must not strand the rest:
            # busy, its row gone, no checkpoint, or the broker refusing it.
            failed.append(document_id)
            print(f"skipped    {document_id}: {type(exc).__name__}: {exc}")
            continue
        print(f"relaunched {document_id}")
    if failed:
        raise SystemExit(f"{len(failed)} of {len(ids)} document(s) were not relaunched")


if __name__ == "__main__":
    main()
