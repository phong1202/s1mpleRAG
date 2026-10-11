"""Re-runs S2 to S5 for documents that are already ingested.

    python -m worker.reingest --all
    python -m worker.reingest <document_id> [<document_id> ...]
    python -m worker.reingest --all --restore-from data/parsed-backup

--restore-from puts back parse checkpoints that storage has expired, from
files named <document_id>.parsed.json.
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
    args = parser.parse_args()

    ids = list(args.document_ids)
    if args.all:
        with session_scope() as session:
            done = select(Document.id).where(Document.completed_at.is_not(None))
            ids = [str(row) for row in session.scalars(done)]

    store = get_store()
    for document_id in ids:
        key = f"staging/{document_id}/parsed.json"
        backup = args.restore_from / f"{document_id}.parsed.json" if args.restore_from else None
        if not store.exists(key) and backup and backup.exists():
            store.put(key, backup.read_bytes())
            print(f"restored   {document_id}")
        relaunch_from_structure(document_id)
        print(f"relaunched {document_id}")


if __name__ == "__main__":
    main()
