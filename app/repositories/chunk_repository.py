"""R0 retrieval. Read-only: this module never writes.

`search` returns CHILDREN, not parents. The collapse to parents is a separate
step (app/core/parent_expander.py) that runs after fusion and reranking,
because a cross-encoder scoring a 700-token parent has to find the one
relevant sentence buried in six irrelevant ones, while the same model scoring
the 150-token child reads nothing but the relevant text.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.contracts import Filters

# The query vector is bound as text and cast server-side. Binding it straight
# into CAST(:qvec AS vector) makes Postgres infer the parameter as `vector`,
# and the driver has no codec for that type -- the cast through text is what
# keeps the parameter a plain string.
#
# The ordering and the LIMIT stay inside a CTE over child_chunks alone: adding
# the documents join above them can cost the HNSW index scan.
_SEARCH = text("""
WITH candidates AS (
  SELECT c.id, c.parent_id, c.document_id, c.page_number, c.content,
         c.embedding <#> CAST(CAST(:qvec AS text) AS vector) AS distance
  FROM   child_chunks c
  WHERE  (CAST(:category AS text) IS NULL OR c.category = CAST(:category AS text))
    AND  (CAST(:document_id AS uuid) IS NULL OR c.document_id = CAST(:document_id AS uuid))
  -- ASC: <#> is NEGATIVE inner product, so nearest sorts first
  ORDER  BY c.embedding <#> CAST(CAST(:qvec AS text) AS vector)
  LIMIT  :over_fetch
)
SELECT c.id AS child_id, c.parent_id, c.document_id, c.page_number,
       c.content AS child_content, c.distance, d.filename
FROM   candidates c
JOIN   documents d ON d.id = c.document_id
ORDER  BY c.distance
""")

_PARENTS = text("""
SELECT id, content FROM parent_chunks WHERE id = ANY(CAST(:ids AS uuid[]))
""")


@dataclass(frozen=True)
class ChildHit:
    child_id: uuid.UUID
    parent_id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    child_content: str
    distance: float
    filename: str


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search(
        self, vector: list[float], over_fetch: int, filters: Filters
    ) -> list[ChildHit]:
        # pgvector's ef_search defaults to 40. Over-fetching 50 out of a queue
        # 40 wide degrades the tail silently, so the session sets it
        # explicitly. set_config rather than SET LOCAL: SET is a utility
        # statement and takes no bind parameters. The third argument is the
        # LOCAL flag -- it lasts until the end of this transaction.
        await self._session.execute(
            text("SELECT set_config('hnsw.ef_search', :ef, true)"),
            {"ef": str(get_settings().retrieval_ef_search)},
        )
        rows = await self._session.execute(
            _SEARCH,
            {
                "qvec": str(vector),
                "category": filters.category,
                "document_id": filters.document_id,
                "over_fetch": over_fetch,
            },
        )
        return [ChildHit(**row) for row in rows.mappings()]

    async def fetch_parents(self, parent_ids: list[uuid.UUID]) -> dict[uuid.UUID, str]:
        """One statement for the whole set. Fetching per parent inside the
        expansion loop is the classic N+1, and it hides well because every
        individual query is fast."""
        if not parent_ids:
            return {}
        rows = await self._session.execute(_PARENTS, {"ids": parent_ids})
        return {row.id: row.content for row in rows}
