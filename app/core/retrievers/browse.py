"""R5: a document's outline, for a question about the whole of it.

"Tóm tắt Luật Đường bộ" cannot be answered from the ten passages a search
finds most similar to it: they come from wherever the words happen to fall.
The outline is in the database already -- every parent has its heading path
(Phase 1) -- so the tool reads it instead of searching:

1. the document's opening: its title and first parent, which for a law or a
   decree holds what it is based on and what it governs;
2. one candidate per top-level section (a chapter, or whatever the document's
   outline put at the top), listing the titles under it, without their text.

A section's candidate points at the section's first parent, so its citation
opens the page the chapter starts on.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.contracts import Candidate, DocRef, Plan, Source
from app.repositories.chunk_repository import ChunkRepository, OutlineRow
from app.repositories.document_lookup import find_document

_SEPARATOR = " > "


class BrowseRetriever:
    source: Source = "browse"

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repository = ChunkRepository(session)

    async def search(self, plan: Plan) -> list[Candidate]:
        """The outline of the document the plan names -- by id when the client
        sent one, else by the name the question used -- or nothing, when no
        single document is clearly the one named."""
        document_id = plan.filters.document_id
        if document_id is None and plan.document_hint:
            document_id = await find_document(self._session, plan.document_hint)
        if document_id is None:
            return []
        rows = [row for row in await self._repository.outline(document_id) if row.child_id]
        if not rows:
            return []
        return _outline(document_id, rows)


def _outline(document_id: uuid.UUID, rows: list[OutlineRow]) -> list[Candidate]:
    first = rows[0]
    title = first.title or first.filename
    entries: list[tuple[str, str, OutlineRow]] = [
        (title, f"{title}\n\n{first.content.strip()}", first)
    ]

    sections: dict[str, tuple[OutlineRow, list[str]]] = {}
    for row in rows:
        if not row.heading_path:
            continue
        top, _, below = row.heading_path.partition(_SEPARATOR)
        start, titles = sections.setdefault(top, (row, []))
        if below and below not in titles:
            titles.append(below)
    for top, (start, titles) in sections.items():
        entries.append((top, "\n".join([top] + [f"- {t}" for t in titles]), start))

    return [
        Candidate(
            source="browse",
            rank=position,
            score=1.0,
            text=text,
            heading=heading,
            ref=DocRef(
                document_id=document_id,
                parent_id=row.parent_id,
                child_id=row.child_id,
                page_number=row.page_start,
                filename=row.filename,
            ),
        )
        for position, (heading, text, row) in enumerate(entries, start=1)
    ]
